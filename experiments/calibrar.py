"""Calibra central_aprendizado por nível m na semente 1000 (Desvios v3, item b).

Varre o número de passos em múltiplos de 15 (até PASSOS_MAX_CALIB) e escolhe o que deixa
a acurácia média da rede nova centralizada mais perto da acurácia média das redes novas
do fedavg_iid (semente 1000, mesmo nível m). As redes calibradas usam as MESMAS tarefas
em que o fedavg_iid tem rede nova (t mod 3 = 2, as 5 últimas e os índices 7 a 11: 16 tarefas com T=30) e sementes próprias.
Cada rede é treinada uma única vez, lendo a acurácia a cada 15 passos; isso é idêntico a
treinar separadamente para cada número de passos (o fluxo de lotes é o mesmo prefixo).
Fora de ±0,03 NÃO interrompe: grava ok=false e segue com o valor mais próximo.
Uso: python -m experiments.calibrar --tarefa A --nivel 1
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
    BATCH, NIVEIS, N, PASSOS_MAX_CALIB, RESET_CENTRAL, SEMENTE_CALIB, T_PADRAO, TOLERANCIA_CALIB, VERSAO_DADOS,
    tarefas_com_nova,
)
from src.dados import Sequencia
from src.metricas import acuracia
from src.modelo import criar_mlp
from src.sementes import gerador, semente
from src.treino import treinar_passos

logger = logging.getLogger(__name__)
TAREFAS = ["A", "B", "B_disjunta"]


def calibrar(tarefa: str, nivel: int, dir_res: Path) -> dict:
    pasta = pasta_calib(dir_res)
    origem = pasta / f"fedavg_iid_m{nivel}_{tarefa}_{SEMENTE_CALIB}"
    tabela = pd.read_csv(f"{origem}.csv")
    if len(tabela) != T_PADRAO:
        raise ValueError(f"fedavg_iid m{nivel} {tarefa} tem {len(tabela)} tarefas, esperado {T_PADRAO}")
    ref = tabela["acc_nova"].dropna()
    alvo = float(ref.mean())
    seq = Sequencia(tarefa, SEMENTE_CALIB)
    tarefas = tarefas_com_nova(T_PADRAO)
    assert len(ref) == len(tarefas)
    todos = torch.arange(N, device=seq.dados(0)[0].device)
    por_passos: dict[int, list[float]] = {}
    for i, t in enumerate(tarefas):
        X, y = seq.dados(t)
        nova = criar_mlp(semente(SEMENTE_CALIB, "calib_nova", i))

        def registrar(feitos: int, nova=nova, X=X, y=y) -> None:
            por_passos.setdefault(feitos, []).append(acuracia(nova, X, y))

        treinar_passos(nova, X, y, todos, PASSOS_MAX_CALIB, BATCH, RESET_CENTRAL,
                       gerador(SEMENTE_CALIB, "calib_embaralha", i), ao_fim_de_bloco=registrar)
    medias = {p: sum(v) / len(v) for p, v in por_passos.items()}
    melhor = min(medias, key=lambda p: (abs(medias[p] - alvo), p))
    erro = medias[melhor] - alvo
    res = {"tarefa": tarefa, "nivel": nivel, "alvo_acc_nova_fedavg": alvo, "passos": melhor,
           "epocas": melhor / (N // BATCH), "acc_central": medias[melhor], "erro": erro,
           "ok": abs(erro) <= TOLERANCIA_CALIB, "alvo_acima_de_0.90": alvo > 0.90,
           "versao_dados": VERSAO_DADOS,
           "fedavg_params": json.loads(Path(f"{origem}.params.json").read_text()) if Path(f"{origem}.params.json").exists() else None,
           "tentativas": medias}
    (pasta / f"calibracao_{tarefa}_m{nivel}.json").write_text(json.dumps(res, indent=1))
    logger.info("calibração %s m%d: alvo=%.3f passos=%d (%.1f épocas) acc=%.3f erro=%+.3f ok=%s",
                tarefa, nivel, alvo, melhor, res["epocas"], medias[melhor], erro, res["ok"])
    return res


if __name__ == "__main__":
    os.environ.setdefault("PLAST_DISPOSITIVO", "cpu")
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefa", required=True, choices=TAREFAS)
    ap.add_argument("--nivel", required=True, type=int, choices=NIVEIS)
    ap.add_argument("--saida", default="results")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    calibrar(a.tarefa, a.nivel, Path(a.saida))
