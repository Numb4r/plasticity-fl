"""Condições da rodada 4 (4a: técnicas distribuídas; 4b: remédios de plasticidade no FedAvg).

Cada condição tem um hiperparâmetro opcional (escolhido na semente 1000) e um número de rodadas R (calibrado na
semente 1000 para a perda da rede nova ficar a até 10% da do `fedavg_iid` K=10). Os dois vêm de
`<dir>/calibracao/selecao_hp_<nome>_<tarefa>_m3.json`; `hp` e `rodadas` explícitos os substituem (testes, calibração).

O que foi unido por ser matematicamente igual (nada duplicado):
  gossip com grafo completo = fedavg_iid (mesmas contas, bit a bit); fica só como teste;
  shrink-and-perturb no cliente ~ no servidor: o encolhimento é linear e comuta com a média; se cada cliente o aplica
  ao próprio modelo logo ANTES de enviá-lo, o ruído independente de K clientes tem variância sigma^2/K depois da média,
  isto é, é o SP do servidor com sigma/sqrt(K) (essa versão não é uma condição e não foi testada; se o SP do cliente
  fosse aplicado ao RECEBER o modelo, os clientes partiriam de pontos diferentes e a equivalência some). Só existe a
  versão do servidor (por tarefa = primário, sigma = 0,01; por rodada = exploratório, aplicada antes de difundir nas
  rodadas 1, 2, ...). "Cliente e servidor" NÃO é igual a "servidor": seria lambda_c*lambda_s com variância
  sigma^2*(lambda_s^2/K + 1); essa combinação não entra.
"""
import json
from pathlib import Path

from src.condicoes import Condicao, pasta_calib
from src.config import (
    DILOCO_LR_EXTERNA, DILOCO_MOMENTO, GRADE_FEDPROX_MU, GRADE_L2_CENTRAL, GRADE_L2_FED, GRADE_L2INIT, GRADE_REDO_TAU,
    GRADE_REDO_VOTO_Q, GRADE_SP_RODADA, GRADE_SP_TAREFA, NIVEL_R2, R_BASE_R4, REDO_CADA, RESET_CENTRAL, S_LOCAL, T_PADRAO,
)

NIVEL = NIVEL_R2        # m* = 3 (nome dos arquivos)
DILOCO_H = {"diloco_h15": 15, "diloco_h50": 50, "diloco_h150": 150}
# nome -> (grade do hiperparâmetro ou None, exploratória?)
CATALOGO_4A = {
    "fedavg_iid": (None, False), "fedprox": (GRADE_FEDPROX_MU, False), "scaffold": (None, False),
    "fedrep": (None, False), "gossip_anel": (None, False), "gossip_grau4": (None, False),
    "diloco_h15": (None, True), "diloco_h50": (None, True), "diloco_h150": (None, True),
    "sem_media_k1": (None, False), "central_aprendizado": (None, False),
}
CATALOGO_4B = {
    "fedavg_iid": (None, False),
    "fed_l2": (GRADE_L2_FED, False), "fed_l2init": (GRADE_L2INIT, False), "fed_redo_local": (GRADE_REDO_TAU, False),
    "fed_sp": (GRADE_SP_TAREFA, False), "fed_redo_fed": (GRADE_REDO_TAU, False),
    "fed_l2_sp": (None, False), "fed_l2init_sp": (None, False), "fed_redo_ambos": (None, False),   # combinações
    "fed_sp_rodada": (GRADE_SP_RODADA, True), "fed_redo_fed_voto": (GRADE_REDO_VOTO_Q, True),
    "central_l2": (GRADE_L2_CENTRAL, False),
}
# combinações: nome -> componentes cujos hiperparâmetros escolhidos entram (não há busca nova)
COMBINACOES = {"fed_l2_sp": ("fed_l2", "fed_sp"), "fed_l2init_sp": ("fed_l2init", "fed_sp"),
               "fed_redo_ambos": ("fed_redo_local", "fed_redo_fed")}
DEPENDE = {"fed_redo_fed_voto": ("fed_redo_fed",)}      # lê o tau já escolhido do ReDo federado por média
CATALOGO = {"4a": CATALOGO_4A, "4b": CATALOGO_4B}


def tem_hp(nome: str) -> bool:
    """A condição tem hiperparâmetro (grade ou vindo de componentes)?"""
    return nome in COMBINACOES or any(cat.get(nome, (None,))[0] is not None for cat in CATALOGO.values())


def arq_selecao(dir_r4: Path, nome: str, tarefa: str) -> Path:
    return pasta_calib(Path(dir_r4)) / f"selecao_hp_{nome}_{tarefa}_m{NIVEL}.json"


def _sel(dir_r4: Path, nome: str, tarefa: str) -> dict:
    arq = arq_selecao(dir_r4, nome, tarefa)
    if not arq.exists():
        raise FileNotFoundError(f"{arq} não existe: rode a calibração e a seleção antes")
    return json.loads(arq.read_text())


def hp_de(nome: str, tarefa: str, dir_r4: Path, hp=None):
    """Hiperparâmetro da condição: o explícito, o escolhido na semente 1000 ou, nas combinações, o dos componentes."""
    if hp is not None:
        return hp
    if nome in COMBINACOES:
        a, b = COMBINACOES[nome]
        return {a: _sel(dir_r4, a, tarefa)["hp"], b: _sel(dir_r4, b, tarefa)["hp"]}
    return _sel(dir_r4, nome, tarefa)["hp"]


def tec_de(nome: str, hp, dir_r4: Path | None = None, tarefa: str = "A") -> tuple[dict | None, dict]:
    """(tec, extras): `tec` vai em Condicao.tec (JSON); `extras` são campos próprios da Condicao (wd, sp_lambda)."""
    if nome == "fedavg_iid":
        return {"tipo": "fedavg"}, {}
    if nome == "fedprox":
        return {"tipo": "fedprox", "mu": hp}, {}
    if nome in ("scaffold", "fedrep"):
        return {"tipo": nome}, {}
    if nome in ("gossip_anel", "gossip_grau4"):
        return {"tipo": "gossip", "topologia": nome.split("_")[1]}, {}
    if nome in DILOCO_H:
        return {"tipo": "diloco", "H": DILOCO_H[nome], "lr_ext": DILOCO_LR_EXTERNA, "mom": DILOCO_MOMENTO}, {}
    if nome == "fed_l2":
        return {"tipo": "fedavg", "wd": hp}, {}
    if nome == "fed_l2init":
        return {"tipo": "fedavg", "l2init": hp}, {}
    if nome == "fed_redo_local":
        return {"tipo": "fedavg", "redo_local": {"tau": hp, "cada": REDO_CADA}}, {}
    if nome == "fed_sp":
        return {"tipo": "fedavg"}, {"sp_lambda": hp}
    if nome == "fed_redo_fed":
        return {"tipo": "fedavg", "redo_fed": {"tau": hp, "cada": REDO_CADA, "modo": "media"}}, {}
    if nome == "fed_l2_sp":
        return {"tipo": "fedavg", "wd": hp["fed_l2"]}, {"sp_lambda": hp["fed_sp"]}
    if nome == "fed_l2init_sp":
        return {"tipo": "fedavg", "l2init": hp["fed_l2init"]}, {"sp_lambda": hp["fed_sp"]}
    if nome == "fed_redo_ambos":
        return {"tipo": "fedavg", "redo_local": {"tau": hp["fed_redo_local"], "cada": REDO_CADA},
                "redo_fed": {"tau": hp["fed_redo_fed"], "cada": REDO_CADA, "modo": "media"}}, {}
    if nome == "fed_sp_rodada":
        return {"tipo": "fedavg", "sp_rodada": hp}, {}
    if nome == "fed_redo_fed_voto":
        tau = _sel(dir_r4, "fed_redo_fed", tarefa)["hp"] if isinstance(hp, (int, float)) else hp["tau"]
        q = hp if isinstance(hp, (int, float)) else hp["q"]
        return {"tipo": "fedavg", "redo_fed": {"tau": tau, "cada": REDO_CADA, "modo": "voto", "q": q}}, {}
    raise ValueError(f"condição desconhecida na rodada 4: {nome}")


def condicao_r4(nome: str, tarefa: str, dir_r4: Path, *, hp=None, rodadas: int | None = None, T: int = T_PADRAO) -> Condicao:
    """Monta a condição da rodada 4. `rodadas` = R (para as centralizadas, os passos são 15·R)."""
    dir_r4 = Path(dir_r4)
    sel = None
    if rodadas is None and nome == "fedavg_iid":
        rodadas = R_BASE_R4                                  # o alvo da calibração: R = 75 por definição, sem arquivo
    if rodadas is None:
        sel = _sel(dir_r4, nome, tarefa)
        rodadas = sel["rodadas"]
    ok = True if sel is None else bool(sel.get("ok", True))
    comum = dict(nivel=NIVEL, T=T, sonda_versao=2, rodada=4, calibracao_ok=ok)
    if nome in ("sem_media_k1",):
        return Condicao(nome, "fl", "um", rodadas=rodadas, s_local=S_LOCAL, **comum)
    if nome == "central_aprendizado":
        return Condicao(nome, "central", passos=RESET_CENTRAL * rodadas, **comum)
    if nome == "central_l2":
        return Condicao(nome, "central", passos=RESET_CENTRAL * rodadas, wd=hp_de(nome, tarefa, dir_r4, hp), **comum)
    h = hp_de(nome, tarefa, dir_r4, hp) if tem_hp(nome) else None
    tec, extras = tec_de(nome, h, dir_r4, tarefa)
    return Condicao(nome, "fedx", "iid", rodadas=rodadas, s_local=S_LOCAL, tec=json.dumps(tec, sort_keys=True),
                    **extras, **comum)
