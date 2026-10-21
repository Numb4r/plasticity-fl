"""Métricas diagnósticas do protocolo de Klein et al. (2026), Tabela 3 (Desvios v3.1).

Regras: tudo numa CÓPIA (deepcopy) em modo de avaliação; nada de passo de otimização;
nenhum parâmetro, estado de otimizador ou modo da rede original é tocado; o sorteio vem de
geradores próprios (papéis "diag_indices" e "diag_hess_v0").
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from src import config
from src.metricas import srank
from src.modelo import MLP
from src.sementes import gerador

NAN = float("nan")
CAMADAS = ("l1", "l2", "l3")
SUFIXOS = {"weight": "w", "bias": "b"}


def _nome(nome_param: str) -> str:
    camada, tipo = nome_param.split(".")
    return f"{camada}_{SUFIXOS[tipo]}"


def normas_parametros(modelo: MLP) -> dict:
    """Norma L2 de cada tensor (pesos e vieses separados) e a global."""
    ps = {_nome(n): p.detach() for n, p in modelo.named_parameters()}
    out = {f"norma_{k}": v.norm().item() for k, v in ps.items()}
    out["norma_global"] = math.sqrt(sum(v.norm().item() ** 2 for v in ps.values()))
    return out


def normas_gradiente(copia: MLP, X, y) -> dict:
    """Gradiente da entropia cruzada média sobre TODOS os pontos da tarefa, numa passada,
    sem passo de otimização. Normas L2 e L1 por tensor e globais."""
    params = dict(copia.named_parameters())
    perda = F.cross_entropy(copia(X), y)
    grads = torch.autograd.grad(perda, list(params.values()))
    out, soma_l2, soma_l1 = {}, 0.0, 0.0
    for (n, _), g in zip(params.items(), grads):
        l2, l1 = g.norm().item(), g.abs().sum().item()
        out[f"grad_l2_{_nome(n)}"], out[f"grad_l1_{_nome(n)}"] = l2, l1
        soma_l2, soma_l1 = soma_l2 + l2 ** 2, soma_l1 + l1
    out["grad_l2_global"], out["grad_l1_global"] = math.sqrt(soma_l2), soma_l1
    return out


def gradientes_por_amostra(copia: MLP, X, y) -> torch.Tensor:
    """Matriz G (pontos x parâmetros): gradiente de cada ponto, achatado, vetorizado com vmap."""
    params = {k: v.detach() for k, v in copia.named_parameters()}

    def perda(p, x, t):
        return F.cross_entropy(functional_call(copia, p, (x[None],)), t[None])

    g = vmap(grad(perda), in_dims=(None, 0, 0))(params, X, y)
    return torch.cat([g[k].reshape(len(X), -1) for k in params], dim=1)


def espectro_gradiente(G: torch.Tensor) -> dict:
    """srank de G, cosseno médio entre pares de linhas e fração de pares com cosseno negativo."""
    n = len(G)
    U = G / G.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C = U @ U.T
    pares = C[torch.triu_indices(n, n, offset=1).unbind()]
    return {"grad_srank": srank(G, config.SRANK_DELTA), "grad_cos_medio": pares.mean().item(),
            "grad_cos_neg": (pares < 0).float().mean().item()}


def autovalor_max(perda_fn, params: list[torch.Tensor], gen: torch.Generator,
                  iters: int = 30, tol: float = 0.01, v_ini=None):
    """Maior autovalor (em módulo) da Hessiana de `perda_fn()` em relação a `params`,
    por power iteration com produtos Hessiana-vetor (autograd duplo).

    Estimativa pelo quociente de Rayleigh. Convergiu se a variação relativa da estimativa
    ficou < tol nas últimas 5 iterações. Devolve (valor, convergiu, autovetor).
    """
    grads = torch.autograd.grad(perda_fn(), params, create_graph=True)
    dev = params[0].device
    v = ([torch.randn(p.shape, generator=gen).to(dev) for p in params] if v_ini is None
         else [x.clone() for x in v_ini])
    norma = math.sqrt(sum((x ** 2).sum().item() for x in v))
    if norma == 0.0:                       # partida inválida: sorteia de novo com o gerador próprio
        v = [torch.randn(p.shape, generator=gen).to(dev) for p in params]
        norma = math.sqrt(sum((x ** 2).sum().item() for x in v))
    v = [x / norma for x in v]
    lams = []
    for _ in range(iters):
        gv = sum((g * x).sum() for g, x in zip(grads, v))
        Hv = torch.autograd.grad(gv, params, retain_graph=True)
        lams.append(sum((h * x).sum().item() for h, x in zip(Hv, v)))
        norma = math.sqrt(sum((h ** 2).sum().item() for h in Hv))
        if norma == 0.0:
            break
        v = [h.detach() / norma for h in Hv]
    ult = lams[-6:]
    convergiu = len(ult) == 6 and all(
        abs(ult[i] - ult[i - 1]) / max(abs(ult[i]), 1e-12) < tol for i in range(1, 6))
    return lams[-1], convergiu, v


def chaves(prefixo: str) -> list[str]:
    """Todas as colunas de um conjunto de diagnósticos (para preencher NaN quando não há medição)."""
    base = [f"norma_{c}_{t}" for c in CAMADAS for t in "wb"] + ["norma_global"]
    base += [f"grad_{n}_{c}_{t}" for n in ("l2", "l1") for c in CAMADAS for t in "wb"]
    base += ["grad_l2_global", "grad_l1_global", "grad_srank", "grad_cos_medio", "grad_cos_neg",
             "hess_max", "hess_conv", "srank_feat"]
    return [prefixo + k for k in base]


def vazio(prefixo: str) -> dict:
    return {k: NAN for k in chaves(prefixo)}


def medir(modelo: MLP, X, y, sem: int, t: int, momento: int, quem: int, prefixo: str,
          v_ini=None, completo: bool = True):
    """Mede tudo numa cópia. momento: 0 = início, 1 = fim da tarefa; quem: 0 = contínuo, 1 = nova.

    completo=False pula o espectro do gradiente e a Hessiana (corte (c) dos Desvios v3.1);
    as colunas ficam NaN. Devolve (dict com prefixo, autovetor da Hessiana ou None).
    """
    copia = copy.deepcopy(modelo)
    copia.eval()
    out = {}
    with torch.enable_grad():
        out.update(normas_parametros(copia))
        out.update(normas_gradiente(copia, X, y))
        v = None
        if completo:
            perm = torch.randperm(len(X), generator=gerador(sem, "diag_indices", t)).to(X.device)
            idx_esp = perm[:config.DIAG_PONTOS_ESPECTRO]
            idx_h = perm if config.HESS_PONTOS is None else perm[:config.HESS_PONTOS]
            out.update(espectro_gradiente(gradientes_por_amostra(copia, X[idx_esp], y[idx_esp])))
            params = list(copia.parameters())
            lam, conv, v = autovalor_max(
                lambda: F.cross_entropy(copia(X[idx_h]), y[idx_h]), params,
                gerador(sem, "diag_hess_v0", t, 2 * momento + quem),
                config.HESS_ITERS, config.HESS_TOL, v_ini if config.HESS_PARTIDA_QUENTE else None)
            out["hess_max"], out["hess_conv"] = lam, float(conv)
        else:
            out.update({"grad_srank": NAN, "grad_cos_medio": NAN, "grad_cos_neg": NAN,
                        "hess_max": NAN, "hess_conv": NAN})
        with torch.no_grad():
            out["srank_feat"] = srank(copia.ativacoes(X)[1], config.SRANK_DELTA)
    return {prefixo + k: val for k, val in out.items()}, v
