"""Calibra os passos locais do central_HT_cal na semente 1000 (esclarecimento de desenho da rodada 2).

Objetivo: a perda de entropia cruzada de treino da REDE NOVA do Hare & Tortoise fica a até 10%
(relativo) da do media_sem_particao, na mesma lógica do central_aprendizado (Desvios v3, item b).
A rede nova do HT é uma rede treinada pelo caminho de um aprendiz só, com R = 75 rodadas de S passos
locais (S múltiplo de 15; o Adam é recriado a cada 15 passos); a tartaruga não muda a rede nova. Por isso
a calibração treina cada rede nova UMA vez, por 75·S_máx passos, e lê a perda a cada 75·15 passos
(o mesmo atalho de experiments/calibrar.py; equivalente em distribuição ao treino rodada a rodada).

Alvo = média da `loss_nova` do media_sem_particao (semente 1000, T=30, mesmas tarefas de referência).
Escolhe o S de menor |erro relativo| (empate: o menor S). Fora de 10%: grava ok=false e segue com o
mais próximo; a análise então reporta H_tempo e H_dist como "não avaliável (calibração não atingida)".
Uso: python -m experiments.calibrar_ht --tarefa A --saida results_rodada2
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
    BATCH, N, NIVEL_R2, PASSOS_MAX_CALIB, RESET_CENTRAL, RODADAS_BASE, SEMENTE_CALIB, T_PADRAO,
    TOLERANCIA_HT_CAL, VERSAO_DADOS, tarefas_com_nova,
)
from src.dados import Sequencia
from src.metricas import perda_ce
from src.modelo import criar_mlp
from src.sementes import gerador, semente
from src.treino import treinar_passos

logger = logging.getLogger(__name__)


def calibrar_ht(tarefa: str, dir_r2: Path, *, T: int = T_PADRAO, passos_max: int = PASSOS_MAX_CALIB,
                referencia: Path | None = None, saida: Path | None = None) -> dict:
    """Devolve (e grava) {passos_locais, erro_rel, ok, ...}. `referencia`/`saida`/`T`/`passos_max` só mudam nos testes."""
    pasta = pasta_calib(Path(dir_r2))
    referencia = referencia or pasta / f"media_sem_particao_m{NIVEL_R2}_{tarefa}_{SEMENTE_CALIB}.csv"
    saida = saida or pasta / f"calibracao_HT_{tarefa}_m{NIVEL_R2}.json"
    tabela = pd.read_csv(referencia)
    if len(tabela) != T:
        raise ValueError(f"{referencia.name} tem {len(tabela)} tarefas, esperado {T}")
    ref = tabela["loss_nova"].dropna()
    alvo = float(ref.mean())
    tarefas = tarefas_com_nova(T)
    assert len(ref) == len(tarefas)
    rodada_passos = RODADAS_BASE * NIVEL_R2 * RESET_CENTRAL          # passos de UMA unidade de S (S = 15·k)
    seq = Sequencia(tarefa, SEMENTE_CALIB, T)
    todos = torch.arange(N, device=seq.dados(0)[0].device)
    por_k: dict[int, list[float]] = {}
    for i, t in enumerate(tarefas):
        X, y = seq.dados(t)
        nova = criar_mlp(semente(SEMENTE_CALIB, "calib_nova", i))

        def registrar(feitos: int, nova=nova, X=X, y=y) -> None:
            if feitos % rodada_passos == 0:
                por_k.setdefault(feitos // rodada_passos, []).append(perda_ce(nova, X, y))

        treinar_passos(nova, X, y, todos, passos_max, BATCH, RESET_CENTRAL,
                       gerador(SEMENTE_CALIB, "calib_embaralha", i), ao_fim_de_bloco=registrar)
    medias = {k * RESET_CENTRAL: sum(v) / len(v) for k, v in por_k.items()}      # chave = S (passos locais por rodada)
    melhor = min(medias, key=lambda s: (abs(medias[s] / alvo - 1), s))
    erro = medias[melhor] / alvo - 1
    res = {"tarefa": tarefa, "alvo_loss_nova_msp": alvo, "passos_locais": melhor, "loss_ht": medias[melhor],
           "erro_rel": erro, "ok": abs(erro) <= TOLERANCIA_HT_CAL, "tolerancia": TOLERANCIA_HT_CAL,
           "alvo_fora_do_alcance": bool(alvo < min(medias.values())), "versao_dados": VERSAO_DADOS,
           "tentativas": medias}
    Path(saida).write_text(json.dumps(res, indent=1))
    logger.info("calibração HT %s: alvo(msp)=%.4f S=%d loss=%.4f erro_rel=%+.3f ok=%s",
                tarefa, alvo, melhor, medias[melhor], erro, res["ok"])
    return res


if __name__ == "__main__":
    os.environ.setdefault("PLAST_DISPOSITIVO", "cpu")
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefa", required=True, choices=["A", "B"])
    ap.add_argument("--saida", default="results_rodada2")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    calibrar_ht(a.tarefa, Path(a.saida))
