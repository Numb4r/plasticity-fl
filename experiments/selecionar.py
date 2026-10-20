"""Escolhe lambda de central_L2 e central_SP na semente 1000 e congela (Desvios v3, item d).

Roda só no nível do regime de teste. Critério: maior acurácia contínua média nas tarefas
26 a 30 (0-based 25 a 29) entre os lambdas que respeitam o limite de esquecimento do
Adendo 1 (esquecimento final <= o de central_aprendizado + 0,05). O limite só existe em
B_disjunta; em A e B só vale a acurácia. Se nenhum lambda respeitar o limite, escolhe o de
menor esquecimento e marca violacao=true.
Uso: python -m experiments.selecionar --tarefa A --metodo L2 --nivel 3
"""
import argparse
import json
import logging
import os
from dataclasses import replace
from pathlib import Path

import torch

from src.condicoes import carregar_condicao, csv_valido, executar, pasta_calib
from src.config import LIMITE_ESQUECIMENTO, NIVEIS, SEMENTE_CALIB, VERSAO_DADOS

logger = logging.getLogger(__name__)
TAREFAS = ["A", "B", "B_disjunta"]
GRADES = {"L2": (1e-4, 1e-3, 1e-2), "SP": (0.4, 0.6, 0.8)}


def _rodar(cond, tarefa: str, pasta: Path, sufixo: str = ""):
    import pandas as pd
    if csv_valido(cond, tarefa, SEMENTE_CALIB, pasta, sufixo):
        from src.condicoes import _base_arquivo
        return pd.read_csv(pasta / f"{_base_arquivo(cond, tarefa, SEMENTE_CALIB, sufixo)}.csv")
    return executar(cond, tarefa, SEMENTE_CALIB, pasta, sufixo=sufixo)


def referencia(tarefa: str, nivel: int, dir_res: Path) -> float:
    """Esquecimento final de central_aprendizado (semente 1000, B_disjunta): referência do limite.
    O pipeline roda isto uma vez antes dos jobs de L2 e SP, que depois só reaproveitam o CSV."""
    base = carregar_condicao("central_aprendizado", tarefa, nivel, Path(dir_res))
    return float(_rodar(base, tarefa, pasta_calib(Path(dir_res)) / "selecao")["esquecimento"].iloc[-1])


def selecionar(tarefa: str, metodo: str, nivel: int, dir_res: Path) -> dict:
    dir_res = Path(dir_res)
    pasta = pasta_calib(dir_res)
    pasta_sel = pasta / "selecao"
    base = carregar_condicao("central_aprendizado", tarefa, nivel, dir_res)
    esq_ref = referencia(tarefa, nivel, dir_res) if tarefa == "B_disjunta" else None
    notas, esq = {}, {}
    for lam in GRADES[metodo]:
        cond = (replace(base, nome="central_L2", wd=lam) if metodo == "L2"
                else replace(base, nome="central_SP", sp_lambda=lam))
        df = _rodar(cond, tarefa, pasta_sel, sufixo=f"_lam{lam}")
        notas[lam] = float(df["acc_cont"].tail(5).mean())
        if esq_ref is not None:
            esq[lam] = float(df["esquecimento"].iloc[-1])
        logger.info("seleção %s %s m%d lambda=%g acc_cont(últimas 5)=%.3f esq=%s", metodo, tarefa,
                    nivel, lam, notas[lam], f"{esq[lam]:.3f}" if esq else "n/d")
    validos = [lam for lam in notas if esq_ref is None or esq[lam] <= esq_ref + LIMITE_ESQUECIMENTO]
    violacao = not validos
    melhor = (min(esq, key=esq.get) if violacao else max(validos, key=notas.get))
    res = {"tarefa": tarefa, "metodo": metodo, "nivel": nivel, "lambda": melhor, "notas": notas,
           "esquecimento": esq, "esquecimento_ref": esq_ref, "violacao": violacao,
           "versao_dados": VERSAO_DADOS, "passos_centrais": base.passos}
    (pasta / f"selecao_{metodo}_{tarefa}_m{nivel}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    os.environ.setdefault("PLAST_DISPOSITIVO", "cpu")
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefa", required=True, choices=TAREFAS)
    ap.add_argument("--metodo", required=True, choices=["L2", "SP"])
    ap.add_argument("--nivel", required=True, type=int, choices=NIVEIS)
    ap.add_argument("--saida", default="results")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    selecionar(a.tarefa, a.metodo, a.nivel, Path(a.saida))
