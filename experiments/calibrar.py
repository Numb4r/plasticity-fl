"""Calibra central_aprendizado na semente 0 (PREREGISTRO.md, escolha operacional 4).

Varre o número de passos em múltiplos de 15 e escolhe o que deixa a acurácia média
da rede nova centralizada mais perto da acurácia média das redes novas do fedavg_iid
(semente 0). Usa sementes de rede nova próprias (papel "calib_nova").
Uso: python -m experiments.calibrar --tarefa A
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch

from src.config import BATCH, FRESCA_CADA, N, RESET_CENTRAL, T_PADRAO
from src.dados import Sequencia
from src.metricas import acuracia
from src.modelo import criar_mlp
from src.sementes import gerador, semente
from src.treino import treinar_passos

logger = logging.getLogger(__name__)
TAREFAS = ["A", "B", "B_disjunta"]
K_MAX = 80        # até 80 * 15 = 1200 passos (24 épocas)
N_CALIB = 10      # redes novas por valor testado, uma por tarefa t%3==2 (como o alvo)
TOLERANCIA = 0.03


def calibrar(tarefa: str, dir_res: Path) -> dict:
    dir_res = Path(dir_res)
    tabela = pd.read_csv(dir_res / f"fedavg_iid_{tarefa}_0.csv")
    if len(tabela) != T_PADRAO:
        raise ValueError(f"fedavg_iid_{tarefa}_0.csv tem {len(tabela)} tarefas, esperado {T_PADRAO}")
    ref = tabela["acc_nova"].dropna()
    alvo = float(ref.mean())
    seq = Sequencia(tarefa, 0)
    todos = torch.arange(N, device=seq.dados(0)[0].device)
    # dados das MESMAS tarefas (t = 2, 5, 8, ...) em que o fedavg_iid tem rede nova
    dados = [seq.dados(FRESCA_CADA * i + FRESCA_CADA - 1) for i in range(N_CALIB)]
    tentativas = {}
    for k in range(1, K_MAX + 1):
        passos = k * RESET_CENTRAL
        accs = []
        for i in range(N_CALIB):
            nova = criar_mlp(semente(0, "calib_nova", i))
            Xi, yi = dados[i]
            treinar_passos(nova, Xi, yi, todos, passos, BATCH, RESET_CENTRAL,
                           gerador(0, "calib_embaralha", i))
            accs.append(acuracia(nova, Xi, yi))
        tentativas[passos] = sum(accs) / len(accs)
    melhor = min(tentativas, key=lambda p: (abs(tentativas[p] - alvo), p))
    erro = tentativas[melhor] - alvo
    res = {"tarefa": tarefa, "alvo_acc_nova_fedavg": alvo, "passos": melhor,
           "epocas": melhor / (N // BATCH), "acc_central": tentativas[melhor],
           "erro": erro, "ok": abs(erro) <= TOLERANCIA, "tentativas": tentativas}
    (dir_res / f"calibracao_{tarefa}.json").write_text(json.dumps(res, indent=1))
    logger.info("calibração %s: alvo=%.3f passos=%d (%.1f épocas) acc=%.3f erro=%+.3f ok=%s",
                tarefa, alvo, melhor, res["epocas"], tentativas[melhor], erro, res["ok"])
    return res


if __name__ == "__main__":
    os.environ.setdefault("PLAST_DISPOSITIVO", "cpu")
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefa", required=True, choices=TAREFAS)
    ap.add_argument("--saida", default="results")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.exit(0 if calibrar(a.tarefa, Path(a.saida))["ok"] else 2)
