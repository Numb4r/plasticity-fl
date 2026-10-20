"""Escolhe lambda de central_L2 e central_SP na semente 0 e congela (escolha operacional 5).

Critério: maior média da acurácia da rede contínua nas últimas 5 tarefas.
Uso: python -m experiments.selecionar --tarefa A --metodo L2
"""
import argparse
import json
import logging
import os
from dataclasses import replace
from pathlib import Path

import torch

from src.condicoes import Condicao, carregar_condicao, executar

logger = logging.getLogger(__name__)
TAREFAS = ["A", "B", "B_disjunta"]
GRADES = {"L2": (1e-4, 1e-3, 1e-2), "SP": (0.4, 0.6, 0.8)}


def selecionar(tarefa: str, metodo: str, dir_res: Path) -> dict:
    dir_res = Path(dir_res)
    base = carregar_condicao("central_aprendizado", tarefa, dir_res)
    notas = {}
    for lam in GRADES[metodo]:
        if metodo == "L2":
            cond = replace(base, nome="central_L2", wd=lam)
        else:
            cond = replace(base, nome="central_SP", sp_lambda=lam)
        df = executar(cond, tarefa, 0, dir_res / "selecao", sufixo=f"_lam{lam}")
        notas[lam] = float(df["acc_cont"].tail(5).mean())
        logger.info("seleção %s %s lambda=%g acc_cont(últimas 5)=%.3f", metodo, tarefa, lam, notas[lam])
    melhor = max(notas, key=notas.get)
    res = {"tarefa": tarefa, "metodo": metodo, "lambda": melhor, "notas": notas}
    (dir_res / f"selecao_{metodo}_{tarefa}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    os.environ.setdefault("PLAST_DISPOSITIVO", "cpu")
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefa", required=True, choices=TAREFAS)
    ap.add_argument("--metodo", required=True, choices=["L2", "SP"])
    ap.add_argument("--saida", default="results")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    selecionar(a.tarefa, a.metodo, Path(a.saida))
