"""Técnicas distribuídas (4a) e remédios de plasticidade no FedAvg (4b) da rodada 4.

Todas rodam sobre os mesmos clientes do `fedavg_iid` (10 clientes com 100 pontos, 15 passos locais, batch 20,
Adam recriado a cada rodada) e mudam UMA coisa:
  fedavg    FedAvg puro (aceita os remédios de 4b: L2, L2Init, ReDo local/federado, shrink-and-perturb por rodada)
  fedprox   termo proximal (mu/2)||w - w_global||^2 na perda local
  scaffold  variáveis de controle (opção I: c_k = gradiente cheio do cliente no modelo recebido)
  fedrep    corpo (l1, l2) na média, cabeça (l3) local de cada cliente; 5 passos na cabeça, depois 10 no corpo
  gossip    sem servidor: cada cliente mistura com os vizinhos (anel, grau 4 ou completo)
  diloco    H passos locais (Adam contínuo na rodada) e SGD externo com momento de Nesterov no pseudo-gradiente
`tec` é um dict (na condição, vai como JSON em `Condicao.tec`). `modelo` é o modelo REPRESENTATIVO: o global (FedAvg e
parentes), o corpo global com a cabeça média (FedRep) ou a média dos pesos dos clientes (gossip).
"""
import torch

from src.config import (
    BATCH, CLIENTES, DILOCO_LR_EXTERNA, DILOCO_MOMENTO, FEDREP_PASSOS_CABECA, FEDREP_PASSOS_CORPO, S_LOCAL, SP_RUIDO,
)
from src.federado import _vetor_estado, media_ponderada
from src.media_tempo import clonar_estado
from src.metricas import acuracia, cosseno_medio, perda_ce
from src.modelo import MLP
from src.tecnicas_local import (
    mascara_media, mascara_voto, montar_ajuste, redo_reiniciar, scores_neuronios, treinar_passos_ext,
)
from src.treino import shrink_and_perturb

TIPOS = ("fedavg", "fedprox", "scaffold", "fedrep", "gossip", "diloco")
TOPOLOGIAS = ("anel", "grau4", "completo")


def vizinhos(topologia: str, K: int) -> list[list[int]]:
    """Vizinhança de cada cliente (inclui ele mesmo); a mistura é a média simples sobre ela (matriz duplamente
    estocástica e simétrica). anel: k-1, k, k+1; grau4: k-2..k+2; completo: todos (= FedAvg)."""
    raio = {"anel": 1, "grau4": 2}.get(topologia)
    if topologia == "completo":
        return [list(range(K)) for _ in range(K)]
    if raio is None:
        raise ValueError(f"topologia desconhecida: {topologia}")
    return [sorted({(k + d) % K for d in range(-raio, raio + 1)}) for k in range(K)]


def matriz_mistura(topologia: str, K: int) -> torch.Tensor:
    W = torch.zeros(K, K, dtype=torch.float64)
    for k, viz in enumerate(vizinhos(topologia, K)):
        W[k, viz] = 1.0 / len(viz)
    return W


def iniciar_extra(tec: dict | None, modelo: MLP, K: int = CLIENTES) -> dict:
    """Estado que persiste entre tarefas fora do `modelo`: a init (L2Init), as cabeças (FedRep), os clientes (gossip)."""
    extra = {"theta0": clonar_estado(modelo), "metricas_rodada": []}
    tipo = (tec or {}).get("tipo")
    if tipo == "fedrep":
        extra["cabecas"] = [{k: v.clone() for k, v in modelo.l3.state_dict().items()} for _ in range(K)]
    if tipo == "gossip":
        extra["clientes"] = [clonar_estado(modelo) for _ in range(K)]
    return extra


def _clonar(local: MLP) -> dict:
    return {k: v.detach().clone() for k, v in local.state_dict().items()}


def _grad_cheio(local: MLP, X, y, p) -> dict:
    """Gradiente da entropia cruzada média nos pontos do cliente (SCAFFOLD, opção I)."""
    perda = torch.nn.functional.cross_entropy(local(X[p]), y[p])
    grads = torch.autograd.grad(perda, list(local.parameters()))
    return {n: g.detach().clone() for (n, _), g in zip(local.named_parameters(), grads)}


def _passo_externo(antes: dict, agg: dict, buf, lr: float, mom: float):
    """SGD externo com momento de Nesterov (semântica do torch.optim.SGD) sobre o pseudo-gradiente
    delta = antes - agg, em dupla precisão. Devolve (novo estado, buffer de momento)."""
    novo = {}
    novo_buf = {} if mom > 0 else None
    for k, a in antes.items():
        if not a.is_floating_point():
            novo[k] = agg[k].clone()
            continue
        delta = a.double() - agg[k].double()
        if mom > 0:
            b = delta.clone() if buf is None else mom * buf[k] + delta
            novo_buf[k] = b
            delta = delta + mom * b
        novo[k] = (a.double() - lr * delta).to(a.dtype)
    return novo, (novo_buf if mom > 0 else None)


def _metricas(estados: list[dict], vetor_ref: torch.Tensor | list, corpo: bool = False) -> dict:
    """Cosseno médio entre as atualizações dos clientes e fator de encolhimento da média:
    ||média das atualizações|| / média das ||atualizações|| (1 = atualizações iguais, ~0 = se cancelam)."""
    ups = [_vetor_estado(e) - r for e, r in zip(estados, vetor_ref)]
    normas = torch.stack([u.norm() for u in ups])
    enc = (torch.stack(ups).mean(dim=0).norm() / normas.mean().clamp_min(1e-12)).item()
    return {"cos": cosseno_medio(ups), "enc": enc}


def _rodada_media(modelo, extra, X, y, particoes, pesos, local, g, tec, aux, r, est) -> dict:
    tipo = tec["tipo"]
    antes = clonar_estado(modelo)
    rf = tec.get("redo_fed")
    if rf and r % rf["cada"] == 0:                       # ReDo federado: o servidor decide e reinicia no global
        scores = []
        for p in particoes:
            local.load_state_dict(antes)
            scores.append(scores_neuronios(local, X[p]))
        masc = (mascara_voto(scores, pesos, rf["tau"], rf["q"]) if rf.get("modo") == "voto"
                else mascara_media(scores, pesos, rf["tau"]))
        redo_reiniciar(modelo, masc, aux["redo"])
        antes = clonar_estado(modelo)
    vetor_global = _vetor_estado(antes)
    H = tec["H"] if tipo == "diloco" else S_LOCAL
    reset = H if tipo == "diloco" else S_LOCAL
    rl = tec.get("redo_local")
    estados, ck_novos = [], []
    for k, p in enumerate(particoes):
        local.load_state_dict(antes)
        if tipo == "scaffold":
            ck_novos.append(_grad_cheio(local, X, y, p))
        if rl and r % rl["cada"] == 0:                   # ReDo local: cada cliente reinicia o PRÓPRIO modelo
            s = scores_neuronios(local, X[p])
            redo_reiniciar(local, [s[0] <= rl["tau"], s[1] <= rl["tau"]], aux["redo"])
        prox = (tec["mu"], {n: antes[n] for n, _ in local.named_parameters()}) if tipo == "fedprox" else None
        corr = {n: est["c"][n] - est["ck"][k][n] for n, _ in local.named_parameters()} if tipo == "scaffold" else None
        l2 = (tec["l2init"], extra["theta0"]) if tec.get("l2init") else None
        ajuste = montar_ajuste(local, prox=prox, corr=corr, l2init=l2) if (prox or corr or l2) else None
        treinar_passos_ext(local, X, y, p, H, BATCH, reset, g, wd=tec.get("wd", 0.0), ajuste=ajuste)
        estados.append(_clonar(local))
    agg = media_ponderada(estados, pesos)
    if tipo == "diloco":
        nova, est["buf"] = _passo_externo(antes, agg, est.get("buf"), tec.get("lr_ext", DILOCO_LR_EXTERNA),
                                          tec.get("mom", DILOCO_MOMENTO))
    else:
        nova = agg
    modelo.load_state_dict(nova)
    if tec.get("sp_rodada"):                             # shrink-and-perturb do servidor, a cada rodada (exploratório)
        shrink_and_perturb(modelo, tec["sp_rodada"], SP_RUIDO, aux["sp"])
    if tipo == "scaffold":                               # atualiza c_k e c (participação plena)
        est["ck"] = ck_novos
        est["c"] = {n: sum(c[n] for c in ck_novos) / len(ck_novos) for n in ck_novos[0]}
    return _metricas(estados, [vetor_global] * len(estados))


def _rodada_fedrep(modelo, extra, X, y, particoes, pesos, local, g, tec, aux, r, est) -> dict:
    antes = clonar_estado(modelo)
    corpo_ref = _vetor_corpo(antes)
    estados = []
    for k, p in enumerate(particoes):
        local.load_state_dict(antes)
        local.l3.load_state_dict(extra["cabecas"][k])
        treinar_passos_ext(local, X, y, p, FEDREP_PASSOS_CABECA, BATCH, FEDREP_PASSOS_CABECA, g,
                           so_params=list(local.l3.parameters()))
        treinar_passos_ext(local, X, y, p, FEDREP_PASSOS_CORPO, BATCH, FEDREP_PASSOS_CORPO, g,
                           so_params=list(local.l1.parameters()) + list(local.l2.parameters()))
        extra["cabecas"][k] = {n: v.detach().clone() for n, v in local.l3.state_dict().items()}
        estados.append(_clonar(local))
    modelo.load_state_dict(media_ponderada(estados, pesos))          # corpo na média; l3 = cabeça média (só representativa)
    ups = [_vetor_corpo(e) - corpo_ref for e in estados]
    normas = torch.stack([u.norm() for u in ups])
    return {"cos": cosseno_medio(ups),
            "enc": (torch.stack(ups).mean(dim=0).norm() / normas.mean().clamp_min(1e-12)).item()}


def _vetor_corpo(estado: dict) -> torch.Tensor:
    return torch.cat([v.flatten().float() for k, v in estado.items() if k.startswith(("l1.", "l2."))])


def misturar_clientes(estados: list[dict], topologia: str) -> list[dict]:
    """Um passo de gossip: cada cliente vira a média simples (dupla precisão) dos estados da sua vizinhança."""
    vizs = vizinhos(topologia, len(estados))
    return [media_ponderada([estados[j] for j in vizs[k]], [1] * len(vizs[k])) for k in range(len(estados))]


def _rodada_gossip(modelo, extra, X, y, particoes, pesos, local, g, tec, aux, r, est) -> dict:
    clientes = extra["clientes"]
    novos = []
    for k, p in enumerate(particoes):
        local.load_state_dict(clientes[k])
        treinar_passos_ext(local, X, y, p, S_LOCAL, BATCH, S_LOCAL, g)
        novos.append(_clonar(local))
    met = _metricas(novos, [_vetor_estado(c) for c in clientes])
    extra["clientes"] = misturar_clientes(novos, tec["topologia"])
    modelo.load_state_dict(media_ponderada(extra["clientes"], [1] * len(novos)))
    return met


def treinar_fedx(modelo: MLP, extra: dict, X, y, particoes: list[torch.Tensor], gerador, rodadas: int, tec: dict,
                 aux: dict | None = None, ao_fim_de_rodada=None) -> list[float]:
    """`rodadas` rodadas da técnica `tec["tipo"]`. Devolve o cosseno médio entre as atualizações por rodada;
    o encolhimento por rodada vai em `extra["metricas_rodada"]`. `aux` = {"redo": Generator, "sp": Generator}.
    `ao_fim_de_rodada(modelo, extra, r)` só lê (calibração de R). O estado de SCAFFOLD (c, c_k) e o momento do
    DiLoCo valem só dentro de uma tarefa (recomeçam a cada chamada)."""
    tipo = tec["tipo"]
    if tipo not in TIPOS:
        raise ValueError(f"técnica desconhecida: {tipo}")
    particoes = [p for p in particoes if len(p) > 0]
    pesos = [len(p) for p in particoes]
    local = MLP().to(X.device)
    est = {}
    if tipo == "scaffold":
        zero = {n: torch.zeros_like(p) for n, p in modelo.named_parameters()}
        est = {"c": zero, "ck": [zero] * len(particoes)}
    rodada = {"fedrep": _rodada_fedrep, "gossip": _rodada_gossip}.get(tipo, _rodada_media)
    extra["metricas_rodada"] = []
    extra["estado_tecnica"] = est                        # só leitura (testes): c e c_k do SCAFFOLD, momento do DiLoCo
    for r in range(rodadas):
        extra["metricas_rodada"].append(rodada(modelo, extra, X, y, particoes, pesos, local, gerador, tec, aux, r, est))
        if ao_fim_de_rodada is not None:
            ao_fim_de_rodada(modelo, extra, r + 1)
    return [m["cos"] for m in extra["metricas_rodada"]]


# ---------------------------------------------------------------- avaliação
@torch.no_grad()
def avaliar_tecnica(modelo: MLP, extra: dict | None, tec: dict | None, X, y) -> tuple[float, float]:
    """(acurácia, perda CE) do modelo da condição. FedRep e gossip: média sobre os clientes, cada um com o próprio
    modelo, nos pontos da tarefa (o modelo médio, que não existe na prática, é só exploratório). Demais: o modelo."""
    tipo = (tec or {}).get("tipo")
    if tipo not in ("fedrep", "gossip"):
        return acuracia(modelo, X, y), perda_ce(modelo, X, y)
    tmp = MLP().to(X.device)
    accs, perdas = [], []
    for k in range(len(extra["cabecas"] if tipo == "fedrep" else extra["clientes"])):
        if tipo == "fedrep":
            tmp.load_state_dict(modelo.state_dict())
            tmp.l3.load_state_dict(extra["cabecas"][k])
        else:
            tmp.load_state_dict(extra["clientes"][k])
        accs.append(acuracia(tmp, X, y))
        perdas.append(perda_ce(tmp, X, y))
    return sum(accs) / len(accs), sum(perdas) / len(perdas)


@torch.no_grad()
def consenso(extra: dict) -> float:
    """Gossip: distância média dos clientes à média dos pesos (0 = consenso)."""
    vs = torch.stack([_vetor_estado(c) for c in extra["clientes"]]).double()
    return (vs - vs.mean(dim=0)).norm(dim=1).mean().item()
