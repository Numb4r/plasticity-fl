"""Calibra R (rodadas por tarefa) de uma condição da rodada 4 na semente 1000 (4a e 4b).

Alvo: a média da `loss_nova` do `fedavg_iid` K=10 (semente 1000, T=30, as 16 tarefas de referência), lida de
<dir>/calibracao/fedavg_iid_m3_<tarefa>_1000.csv. Grade: R inteiro de 50 a 200, passo 1. Escolhe o R de menor
|erro relativo| (empate: o menor R); ok se |erro relativo| <= 10%. Fora de 10%: ok=false e segue com o mais próximo
(a condição fica "não avaliável" na análise), sem forçar.

A rede nova é treinada pelo MESMO caminho da execução real (`treinar_na_tarefa`, com o shrink-and-perturb por
tarefa quando a condição o tem) por R_máx rodadas, e a perda é lida ao fim de cada rodada: a rede depois de R
rodadas é prefixo da de R_máx (mesmo gerador), então a leitura é exata. Nas centralizadas uma "rodada" são 15 passos.
"""
import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.condicoes import Condicao, _sp, aux_gerador, avaliar_cond, particoes_da_condicao, pasta_calib, treinar_na_tarefa
from src.config import R_GRADE_R4, RESET_CENTRAL, SEMENTE_CALIB, T_PADRAO, TOLERANCIA_R4, VERSAO_DADOS, tarefas_com_nova
from src.dados import Sequencia
from src.modelo import criar_mlp
from src.sementes import gerador, semente
from src.tecnicas_fed import iniciar_extra

logger = logging.getLogger(__name__)


def arq_alvo(dir_r4: Path, tarefa: str) -> Path:
    return pasta_calib(Path(dir_r4)) / f"fedavg_iid_m3_{tarefa}_{SEMENTE_CALIB}.csv"


def calibrar_R(cond: Condicao, tarefa: str, dir_r4: Path, saida: Path, *, grade: tuple[int, ...] = R_GRADE_R4,
               T: int = T_PADRAO, alvo_csv: Path | None = None) -> dict:
    """Devolve (e grava em `saida`) {rodadas, erro_rel, ok, ...}. `cond.rodadas`/`cond.passos` são ignorados."""
    alvo_csv = alvo_csv or arq_alvo(dir_r4, tarefa)
    tabela = pd.read_csv(alvo_csv)
    if len(tabela) != T:
        raise ValueError(f"{alvo_csv.name} tem {len(tabela)} tarefas, esperado {T}")
    ref = tabela["loss_nova"].dropna()
    alvo = float(ref.mean())
    tarefas = tarefas_com_nova(T)
    assert len(ref) == len(tarefas)
    r_max, conjunto = max(grade), set(grade)
    central = cond.familia == "central"
    cond_max = replace(cond, passos=RESET_CENTRAL * r_max) if central else replace(cond, rodadas=r_max)
    tec = json.loads(cond.tec) if cond.tec else None
    seq = Sequencia(tarefa, SEMENTE_CALIB, T)
    particoes = particoes_da_condicao(cond, SEMENTE_CALIB)
    por_r: dict[int, list[float]] = {}
    for i, t in enumerate(tarefas):
        X, y = seq.dados(t)
        nova = criar_mlp(semente(SEMENTE_CALIB, "calib_nova", i))
        extra = iniciar_extra(tec, nova) if cond.familia == "fedx" else None
        _sp(cond, nova, SEMENTE_CALIB, i, 1)                     # shrink-and-perturb por tarefa, se a condição o tem

        def registrar(modelo, ext, r, X=X, y=y) -> None:
            if r in conjunto:
                por_r.setdefault(r, []).append(avaliar_cond(cond, modelo, ext, X, y)[1])

        treinar_na_tarefa(cond_max, nova, X, y, particoes, gerador(SEMENTE_CALIB, "calib_embaralha", i), extra,
                          aux_gerador(cond, SEMENTE_CALIB, i, 1), registrar)
    medias = {r: (sum(v) / len(v) if all(np.isfinite(v)) else float("inf")) for r, v in sorted(por_r.items())}   # perda NaN/inf nunca vence
    melhor = min(medias, key=lambda r: (abs(medias[r] / alvo - 1), r))
    erro = medias[melhor] / alvo - 1
    res = {"condicao": cond.nome, "tarefa": tarefa, "alvo_loss_nova": alvo, "rodadas": melhor, "loss": medias[melhor],
           "erro_rel": erro, "ok": abs(erro) <= TOLERANCIA_R4, "tolerancia": TOLERANCIA_R4, "grade": [min(grade), r_max],
           "alvo_fora_do_alcance": bool(alvo < min(medias.values()) or alvo > max(v for v in medias.values() if np.isfinite(v))),
           "versao_dados": VERSAO_DADOS, "alvo_csv": str(alvo_csv),
           "alvo_sha256": hashlib.sha256(Path(alvo_csv).read_bytes()).hexdigest(), "tentativas": medias}
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    Path(saida).write_text(json.dumps(res, indent=1))
    logger.info("calibração R %s %s: alvo=%.4f R=%d loss=%.4f erro_rel=%+.3f ok=%s", cond.nome, tarefa, alvo, melhor,
                medias[melhor], erro, res["ok"])
    return res
