"""Pipeline completo: fedavg_iid sem 0 -> calibra -> seleciona -> congela -> resto.

  python -m experiments.pipeline --quick      # 1 semente, só tarefa A, sem media_sem_particao
  python -m experiments.pipeline              # 5 sementes, tarefas A, B e B_disjunta, todas as condições

Cada job roda em um processo próprio com 1 thread (uma semente por processo).
Jobs cujo CSV já existe são pulados, então dá para retomar.
"""
import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from experiments.calibrar import calibrar
from experiments.selecionar import selecionar
from src.condicoes import CONDICOES, carregar_condicao, csv_valido, executar

logger = logging.getLogger(__name__)


def _preparar_processo() -> None:
    os.environ["PLAST_DISPOSITIVO"] = "cpu"  # CPU: reprodutível bit a bit
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO)


def job_rodar(nome: str, tarefa: str, sem: int, dir_res: str) -> str:
    dir_res = Path(dir_res)
    cond = carregar_condicao(nome, tarefa, dir_res)
    if csv_valido(cond, tarefa, sem, dir_res):
        return f"pulado {nome} {tarefa} {sem} (CSV completo, mesmos parâmetros)"
    executar(cond, tarefa, sem, dir_res)
    return f"ok {nome} {tarefa} {sem}"


def job_calibrar(tarefa: str, dir_res: str) -> dict:
    arq = Path(dir_res) / f"calibracao_{tarefa}.json"
    if arq.exists():                       # valor congelado: apague o JSON para recalibrar
        return json.loads(arq.read_text())
    return calibrar(tarefa, Path(dir_res))


def job_selecionar(tarefa: str, metodo: str, dir_res: str) -> dict:
    arq = Path(dir_res) / f"selecao_{metodo}_{tarefa}.json"
    if arq.exists():
        return json.loads(arq.read_text())
    return selecionar(tarefa, metodo, Path(dir_res))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sementes", type=int, default=None, help="número de sementes (padrão: 1 no quick, 5 no completo)")
    ap.add_argument("--tarefas", nargs="+", choices=["A", "B", "B_disjunta"], default=None,
                    help="padrão: A no quick; A, B e B_disjunta no completo")
    ap.add_argument("--saida", default="results")
    ap.add_argument("--processos", type=int, default=8)
    a = ap.parse_args()
    os.environ["PLAST_DISPOSITIVO"] = "cpu"   # Desvio 5: o pipeline sempre usa CPU
    logging.basicConfig(level=logging.INFO)
    n_sem = a.sementes or (1 if a.quick else 5)
    tarefas = a.tarefas or (["A"] if a.quick else ["A", "B", "B_disjunta"])
    conds = [c for c in CONDICOES if not (a.quick and c == "media_sem_particao")]
    dir_res = str(a.saida)
    Path(dir_res).mkdir(exist_ok=True)
    tempos, t0 = {}, time.time()
    with ProcessPoolExecutor(a.processos, initializer=_preparar_processo) as ex:
        # etapa 1: fedavg_iid semente 0 (alvo da calibração)
        t = time.time()
        for r in [ex.submit(job_rodar, "fedavg_iid", k, 0, dir_res) for k in tarefas]:
            logger.info(r.result())
        tempos["1_fedavg_iid_sem0"] = time.time() - t
        # etapa 2: calibrar, depois selecionar L2 e SP (valores congelados)
        t = time.time()
        for r in [ex.submit(job_calibrar, k, dir_res) for k in tarefas]:
            if not r.result()["ok"]:
                raise SystemExit("calibração fora de ±0,03: registrar em Desvios do PREREGISTRO.md")
        tempos["2_calibracao"] = time.time() - t
        t = time.time()
        for r in [ex.submit(job_selecionar, k, m, dir_res) for k in tarefas for m in ("L2", "SP")]:
            r.result()
        tempos["3_selecao"] = time.time() - t
        # etapa 3: todo o resto, os mais caros primeiro
        t = time.time()
        jobs = [(c, k, s) for c in conds for k in tarefas for s in range(n_sem)]
        jobs.sort(key=lambda j: j[0] != "media_sem_particao")
        for r in [ex.submit(job_rodar, c, k, s, dir_res) for c, k, s in jobs]:
            logger.info(r.result())
        tempos["4_resto"] = time.time() - t
    tempos["total"] = time.time() - t0
    modo = "quick" if a.quick else "completo"
    Path(dir_res, f"tempo_{modo}.json").write_text(json.dumps(tempos, indent=1))
    logger.info("tempos (s): %s", tempos)


if __name__ == "__main__":
    main()
