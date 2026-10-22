"""Calibra o número de rodadas R do central_HT_cal2 na semente 1000 (rodada 2b; PREREGISTRO.md).

Alvo: a média da `loss_nova` do media_sem_particao (T=30, semente 1000, 16 tarefas de referência), lida do CSV da
rodada 2 (results_rodada2/calibracao/media_sem_particao_m3_<A|B>_1000.csv). Grade fina: R inteiro de 75 a 150,
com S = 15 passos locais e o Adam recriado a cada rodada, como o central_HT. Escolhe o R de menor |erro relativo|
(empate: o menor R); ok se |erro relativo| <= 10%. Fora de 10%: grava ok=false e segue com o mais próximo
(H_tempo e H_dist da 2b ficam "não avaliável (calibração não atingida)").

A rede nova do HT é a lebre depois de R rodadas, treinada pelo MESMO código da execução real (`treinar_ht`);
como a lebre não depende de R (R só diz quando parar), cada rede é treinada uma vez, por 150 rodadas, e a perda
é lida ao fim de cada rodada de 75 em diante (o fluxo de lotes é o mesmo prefixo: exato, não aproximado).
Uso: python -m experiments.calibrar_ht2 --tarefa A --saida results_rodada2b
"""
import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from src.condicoes import pasta_calib
from src.config import (
    NIVEL_R2, R_CAL2, SEMENTE_CALIB, T_PADRAO, TOLERANCIA_HT_CAL, VERSAO_DADOS, tarefas_com_nova,
)
from src.dados import Sequencia, particionar
from src.media_tempo import clonar_estado, treinar_ht
from src.metricas import perda_ce
from src.modelo import criar_mlp
from src.sementes import gerador, semente

logger = logging.getLogger(__name__)
DIR_REFERENCIA = "results_rodada2"            # onde está o media_sem_particao da semente 1000 (rodada 2)


def calibrar_ht2(tarefa: str, dir_r2b: Path, *, T: int = T_PADRAO, grade: tuple[int, ...] = R_CAL2,
                 referencia: Path | None = None, saida: Path | None = None) -> dict:
    """Devolve (e grava) {rodadas, erro_rel, ok, ...}. `referencia`/`saida`/`T`/`grade` só mudam nos testes."""
    referencia = referencia or (pasta_calib(Path(DIR_REFERENCIA)) / f"media_sem_particao_m{NIVEL_R2}_{tarefa}_{SEMENTE_CALIB}.csv")
    saida = saida or (pasta_calib(Path(dir_r2b)) / f"calibracao_HT2_{tarefa}_m{NIVEL_R2}.json")
    tabela = pd.read_csv(referencia)
    if len(tabela) != T:
        raise ValueError(f"{referencia.name} tem {len(tabela)} tarefas, esperado {T}")
    ref = tabela["loss_nova"].dropna()
    alvo = float(ref.mean())
    tarefas = tarefas_com_nova(T)
    assert len(ref) == len(tarefas)
    seq = Sequencia(tarefa, SEMENTE_CALIB, T)
    particoes = particionar("um", SEMENTE_CALIB)
    r_min, r_max = min(grade), max(grade)
    grade_set = set(grade)
    por_r: dict[int, list[float]] = {}
    for i, t in enumerate(tarefas):
        X, y = seq.dados(t)
        nova = criar_mlp(semente(SEMENTE_CALIB, "calib_nova", i))
        extra = {"tartaruga": clonar_estado(nova)}                 # a tartaruga não age numa tarefa só

        def registrar(lebre, feitas: int, X=X, y=y) -> None:
            if feitas in grade_set:
                por_r.setdefault(feitas, []).append(perda_ce(lebre, X, y))

        treinar_ht(nova, extra, X, y, particoes, gerador(SEMENTE_CALIB, "calib_embaralha", i), r_max, 0.0,
                   ao_fim_de_rodada=registrar)
    medias = {r: sum(v) / len(v) for r, v in por_r.items()}
    melhor = min(medias, key=lambda r: (abs(medias[r] / alvo - 1), r))
    erro = medias[melhor] / alvo - 1
    res = {"tarefa": tarefa, "alvo_loss_nova_msp": alvo, "rodadas": melhor, "loss_ht": medias[melhor], "erro_rel": erro,
           "ok": abs(erro) <= TOLERANCIA_HT_CAL, "tolerancia": TOLERANCIA_HT_CAL, "grade": [r_min, r_max],
           "alvo_fora_do_alcance": bool(alvo < min(medias.values()) or alvo > max(medias.values())),
           "versao_dados": VERSAO_DADOS, "tentativas": medias}
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    Path(saida).write_text(json.dumps(res, indent=1))
    logger.info("calibração HT2 %s: alvo(msp)=%.4f R=%d loss=%.4f erro_rel=%+.3f ok=%s",
                tarefa, alvo, melhor, medias[melhor], erro, res["ok"])
    return res


if __name__ == "__main__":
    os.environ.setdefault("PLAST_DISPOSITIVO", "cpu")
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefa", required=True, choices=["A", "B"])
    ap.add_argument("--saida", default="results_rodada2b")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    calibrar_ht2(a.tarefa, Path(a.saida))
