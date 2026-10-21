"""Sondas de plasticidade. Trabalham SEMPRE numa cópia; a rede original não muda."""
import copy

import torch
import torch.nn.functional as F

from src.config import LR, SONDA_ALVOS, SONDA_BATCH, SONDA_PASSOS
from src.modelo import MLP, criar_mlp
from src.sementes import gerador, semente


def _treinar_imitando(modelo: MLP, X, alvo, g: torch.Generator) -> float:
    """SONDA_PASSOS passos de Adam com MSE para imitar `alvo`; devolve a perda final em X."""
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    n = len(X)
    for _ in range(SONDA_PASSOS):
        idx = torch.randint(0, n, (SONDA_BATCH,), generator=g).to(X.device)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(modelo(X[idx]), alvo[idx]).backward()
        opt.step()
    with torch.no_grad():
        return F.mse_loss(modelo(X), alvo).item()


def sonda_alvos_aleatorios(modelo: MLP, X, sem: int, t: int) -> dict:
    """Sonda de alvos aleatórios (estilo Lyle et al., 2022).

    Para cada uma de 3 redes-alvo recém-inicializadas (semente nova): treina uma
    CÓPIA do modelo e uma rede nova, com os mesmos alvos, os mesmos lotes e os mesmos
    passos, e compara a perda final. razao = perda_cópia / perda_rede_nova
    (média das 3 razões). Razão > 1 = a rede perdeu capacidade de aprender alvos novos.
    """
    perdas_copia, perdas_nova, razoes = [], [], []
    for k in range(SONDA_ALVOS):
        alvo_net = criar_mlp(semente(sem, "sonda_alvo", t, k))
        with torch.no_grad():
            alvo = alvo_net(X)
        copia = copy.deepcopy(modelo)                      # a original não é tocada
        nova = criar_mlp(semente(sem, "sonda_nova", t, k))
        # mesmos lotes para a cópia e para a rede nova: só a inicialização difere
        pc = _treinar_imitando(copia, X, alvo, gerador(sem, "sonda_embaralha", t, k))
        pn = _treinar_imitando(nova, X, alvo, gerador(sem, "sonda_embaralha", t, k))
        perdas_copia.append(pc)
        perdas_nova.append(pn)
        razoes.append(pc / pn)
    n = len(razoes)
    return {
        "sonda_perda_copia": sum(perdas_copia) / n,
        "sonda_perda_nova": sum(perdas_nova) / n,
        "sonda_razao": sum(razoes) / n,
    }


def _imitar_v2(modelo: MLP, X, alvo, g: torch.Generator, passos: int) -> tuple[float, float]:
    """(perda no passo 0, perda após `passos` passos) de Adam/MSE imitando `alvo`, nos N pontos."""
    with torch.no_grad():
        l0 = F.mse_loss(modelo(X), alvo).item()
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    for _ in range(passos):
        idx = torch.randint(0, len(X), (SONDA_BATCH,), generator=g).to(X.device)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(modelo(X[idx]), alvo[idx]).backward()
        opt.step()
    with torch.no_grad():
        return l0, F.mse_loss(modelo(X), alvo).item()


def sonda_v2(modelo: MLP, X, sem: int, t: int, passos: int | None = None) -> dict:
    """Sonda da rodada 2: poucos passos (a de 300 satura), perdas POR ALVO nos passos 0 e `passos`.

    Estatísticas (PREREGISTRO.md, rodada 2): dlog_norm = mediana entre os alvos de
    log10(Lc(p)/Lc(0)) - log10(Ln(p)/Ln(0)) (normaliza pela escala das saídas) e
    dlog_raw = mediana entre os alvos de log10(Lc(p) + 1e-6) - log10(Ln(p) + 1e-6).
    A cópia e a rede nova usam os mesmos alvos e os mesmos lotes; o modelo original não é tocado.
    """
    import math
    import numpy as np
    from src.config import SONDA_PASSOS_R2
    passos = SONDA_PASSOS_R2 if passos is None else passos
    out, norm, raw = {}, [], []
    for k in range(SONDA_ALVOS):
        alvo_net = criar_mlp(semente(sem, "sonda_alvo", t, k))
        with torch.no_grad():
            alvo = alvo_net(X)
        copia = copy.deepcopy(modelo)
        nova = criar_mlp(semente(sem, "sonda_nova", t, k))
        lc0, lc = _imitar_v2(copia, X, alvo, gerador(sem, "sonda_embaralha", t, k), passos)
        ln0, ln = _imitar_v2(nova, X, alvo, gerador(sem, "sonda_embaralha", t, k), passos)
        out.update({f"sonda2_Lc0_{k}": lc0, f"sonda2_Lc20_{k}": lc, f"sonda2_Ln0_{k}": ln0, f"sonda2_Ln20_{k}": ln})
        norm.append(math.log10(lc / lc0) - math.log10(ln / ln0))
        raw.append(math.log10(lc + 1e-6) - math.log10(ln + 1e-6))
    out["sonda2_dlog_norm"], out["sonda2_dlog_raw"] = float(np.median(norm)), float(np.median(raw))
    return out


SONDA2_COLUNAS = ([f"sonda2_{p}_{k}" for k in range(SONDA_ALVOS) for p in ("Lc0", "Lc20", "Ln0", "Ln20")]
                  + ["sonda2_dlog_norm", "sonda2_dlog_raw"])
