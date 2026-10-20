"""Roda uma condição, um tipo de tarefa e uma ou mais sementes (uma por vez, em série).

Uso: python -m experiments.rodar --cond fedavg_iid --tarefa A --sementes 0 1 2
Para paralelizar, rode um processo por semente (ou use experiments.pipeline).
"""
import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

import torch

from src.condicoes import CONDICOES, carregar_condicao, executar

TAREFAS = ["A", "B", "B_disjunta"]

if __name__ == "__main__":
    os.environ.setdefault("PLAST_DISPOSITIVO", "cpu")
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=CONDICOES)
    ap.add_argument("--tarefa", required=True, choices=TAREFAS)
    ap.add_argument("--sementes", type=int, nargs="+", default=[0])
    ap.add_argument("--saida", default="results")
    ap.add_argument("--T", type=int, default=None, help="só para testes rápidos; o padrão vem da condição")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cond = carregar_condicao(a.cond, a.tarefa, Path(a.saida))
    saida = Path(a.saida)
    if a.T is not None:
        cond = replace(cond, T=a.T)
        saida = saida / "teste"   # execução curta nunca se mistura com os resultados
    for s in a.sementes:
        executar(cond, a.tarefa, s, saida)
