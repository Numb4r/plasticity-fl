"""Pipeline por fases (Desvios v3).

  --fase teste_curto   semente 1000, tarefa A, m em {1,10}: fedavg_iid, sem_media_k1, central_passos.
                       Imprime os gaps; se central_passos em m=10 não perder plasticidade, PARE e olhe o log.
  --fase calibracao    semente 1000: fedavg_iid por nível e tarefa, calibração de central_aprendizado,
                       definição do regime de teste m*, seleção de lambda (L2, SP) em m*.
  --fase completo      calibracao (se faltar) + sementes de avaliação (1 a 5), tarefas A, B e B_disjunta,
                       todos os níveis (L2 e SP só em m*).
  --quick              atalho do completo: semente 1, tarefa A, sem media_sem_particao.

Cada job roda em um processo próprio com 1 thread. Jobs cujo CSV já é válido (mesmos parâmetros,
mesmo número de tarefas) são pulados, então dá para retomar.
"""
import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from experiments.calibrar import calibrar
from experiments.selecionar import referencia, selecionar
from src.condicoes import CONDICOES, _base_arquivo, carregar_condicao, csv_valido, executar, pasta_calib
from src.config import NIVEIS, SEMENTE_CALIB, SEMENTES_AVALIACAO, VERSAO_DADOS

logger = logging.getLogger(__name__)
LIMITE_PERDA = -0.10        # regime de teste: central_aprendizado com gap <= -0,10 na tarefa A
CUSTO = {"fedavg_iid": 30, "media_sem_particao": 12, "sem_media_k1": 3, "central_passos": 1,
         "central_aprendizado": 1, "central_L2": 1, "central_SP": 1}


def _preparar_processo() -> None:
    os.environ["PLAST_DISPOSITIVO"] = "cpu"  # CPU: reprodutível bit a bit
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO)


def job_rodar(nome: str, tarefa: str, nivel: int, sem: int, dir_res: str) -> str:
    dir_res = Path(dir_res)
    pasta = pasta_calib(dir_res) if sem == SEMENTE_CALIB else dir_res
    cond = carregar_condicao(nome, tarefa, nivel, dir_res)
    if csv_valido(cond, tarefa, sem, pasta):
        return f"pulado {nome} m{nivel} {tarefa} s{sem} (CSV completo, mesmos parâmetros)"
    executar(cond, tarefa, sem, pasta)
    return f"ok {nome} m{nivel} {tarefa} s{sem}"


def job_calibrar(tarefa: str, nivel: int, dir_res: str) -> dict:
    arq = pasta_calib(Path(dir_res)) / f"calibracao_{tarefa}_m{nivel}.json"
    if arq.exists():                       # valor congelado: só reaproveita se a origem não mudou
        antigo = json.loads(arq.read_text())
        origem = pasta_calib(Path(dir_res)) / f"fedavg_iid_m{nivel}_{tarefa}_{SEMENTE_CALIB}.params.json"
        atual = json.loads(origem.read_text()) if origem.exists() else None
        if antigo.get("versao_dados") == VERSAO_DADOS and antigo.get("fedavg_params") == atual:
            return antigo
    return calibrar(tarefa, nivel, Path(dir_res))


def job_selecionar(tarefa: str, metodo: str, nivel: int, dir_res: str) -> dict:
    arq = pasta_calib(Path(dir_res)) / f"selecao_{metodo}_{tarefa}_m{nivel}.json"
    if arq.exists():
        antigo = json.loads(arq.read_text())
        calib = json.loads((pasta_calib(Path(dir_res)) / f"calibracao_{tarefa}_m{nivel}.json").read_text())
        if antigo.get("versao_dados") == VERSAO_DADOS and antigo.get("passos_centrais") == calib["passos"]:
            return antigo
    return selecionar(tarefa, metodo, nivel, Path(dir_res))


def job_referencia(tarefa: str, nivel: int, dir_res: str) -> float:
    return referencia(tarefa, nivel, Path(dir_res))


def gap_ultimas5(csv: Path) -> float:
    """Métrica principal de um CSV: média dos gaps pareados nas 5 últimas tarefas."""
    d = pd.read_csv(csv)
    return float(d[d["tarefa"] >= d["tarefa"].max() - 4]["gap"].dropna().mean())


def _gap_cond(dir_res: Path, nome: str, nivel: int, tarefa: str) -> float:
    base = f"{nome}_m{nivel}_{tarefa}_{SEMENTE_CALIB}"
    return gap_ultimas5(pasta_calib(dir_res) / f"{base}.csv")


def _rodar_jobs(ex, jobs: list[tuple]) -> None:
    for r in [ex.submit(job_rodar, *j) for j in jobs]:
        logger.info(r.result())


def fase_teste_curto(ex, dir_res: Path) -> dict:
    jobs = [(c, "A", m, SEMENTE_CALIB, str(dir_res))
            for m in (1, 10) for c in ("fedavg_iid", "sem_media_k1", "central_passos")]
    _rodar_jobs(ex, jobs)
    gaps = {f"{c}_m{m}": _gap_cond(dir_res, c, m, "A")
            for m in (1, 10) for c in ("fedavg_iid", "sem_media_k1", "central_passos")}
    perde = gaps["central_passos_m10"] <= LIMITE_PERDA
    logger.info("TESTE CURTO (semente %d, tarefa A) gaps das 5 últimas tarefas: %s", SEMENTE_CALIB,
                {k: round(v, 3) for k, v in gaps.items()})
    logger.info("central_passos m=10 perde plasticidade (gap <= %.2f)? %s", LIMITE_PERDA, perde)
    if not perde:
        logger.warning("PARE: central_passos em m=10 NÃO perde plasticidade (gap=%.3f > %.2f). "
                       "Não siga para o completo; mostre este log ao autor.",
                       gaps["central_passos_m10"], LIMITE_PERDA)
    return {"gaps": gaps, "central_passos_m10_perde": perde}


def definir_regime(dir_res: Path) -> dict:
    """m* = menor nível em que central_aprendizado tem gap <= -0,10 na tarefa A (semente 1000)."""
    gaps = {m: _gap_cond(dir_res, "central_aprendizado", m, "A") for m in NIVEIS}
    m_star = next((m for m in NIVEIS if gaps[m] <= LIMITE_PERDA), None)
    res = {"m_star": m_star, "gap_central_aprendizado_A": gaps, "limite": LIMITE_PERDA,
           "nota": None if m_star else "regime sem perda; H_real, H_reg, H_part e H_orc não testáveis neste toy"}
    (pasta_calib(dir_res) / "regime.json").write_text(json.dumps(res, indent=1))
    logger.info("regime de teste: %s", res)
    return res


def fase_calibracao(ex, dir_res: Path, tarefas: list[str], niveis: list[int]) -> dict:
    """O regime de teste usa central_aprendizado na tarefa A em TODOS os níveis; por isso a
    calibração sempre inclui a tarefa A e todos os níveis, além do que foi pedido."""
    tarefas = sorted(set(tarefas) | {"A"})
    niveis = list(NIVEIS)
    pasta_calib(dir_res).mkdir(parents=True, exist_ok=True)
    tempos, t = {}, time.time()
    _rodar_jobs(ex, [("fedavg_iid", k, m, SEMENTE_CALIB, str(dir_res)) for m in niveis for k in tarefas])
    tempos["calib_1_fedavg_iid_sem1000"] = time.time() - t
    t = time.time()
    for r in [ex.submit(job_calibrar, k, m, str(dir_res)) for m in niveis for k in tarefas]:
        r.result()
    tempos["calib_2_calibracao"] = time.time() - t
    t = time.time()
    _rodar_jobs(ex, [("central_aprendizado", "A", m, SEMENTE_CALIB, str(dir_res)) for m in niveis])
    regime = definir_regime(dir_res)
    tempos["calib_3_regime"] = time.time() - t
    t = time.time()
    if regime["m_star"] is not None:
        if "B_disjunta" in tarefas:     # referência do limite de esquecimento, uma vez só
            ex.submit(job_referencia, "B_disjunta", regime["m_star"], str(dir_res)).result()
        for r in [ex.submit(job_selecionar, k, met, regime["m_star"], str(dir_res))
                  for k in tarefas for met in ("L2", "SP")]:
            r.result()
    tempos["calib_4_selecao"] = time.time() - t
    return tempos


def fase_completo(ex, dir_res: Path, tarefas, niveis, sementes, conds) -> dict:
    tempos = fase_calibracao(ex, dir_res, tarefas, niveis)
    m_star = json.loads((pasta_calib(dir_res) / "regime.json").read_text())["m_star"]
    jobs = []
    for c in conds:
        for k in tarefas:
            for m in niveis:
                if c in ("central_L2", "central_SP") and m != m_star:
                    continue                      # mitigação só no regime de teste
                jobs += [(c, k, m, s, str(dir_res)) for s in sementes]
    jobs.sort(key=lambda j: -CUSTO[j[0]] * j[2])  # mais caros primeiro
    t = time.time()
    _rodar_jobs(ex, jobs)
    tempos["completo_confirmatorio"] = time.time() - t
    return tempos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", choices=["teste_curto", "calibracao", "completo"], default="completo")
    ap.add_argument("--quick", action="store_true", help="completo com semente 1, tarefa A, sem media_sem_particao")
    ap.add_argument("--tarefas", nargs="+", choices=["A", "B", "B_disjunta"], default=None)
    ap.add_argument("--niveis", nargs="+", type=int, choices=NIVEIS, default=None)
    ap.add_argument("--sementes", nargs="+", type=int, default=None, help="padrão: 1 2 3 4 5 (1 no quick)")
    ap.add_argument("--saida", default="results")
    ap.add_argument("--processos", type=int, default=4, help="cada processo usa ~0,6 GB; confira a memória livre")
    a = ap.parse_args()
    os.environ["PLAST_DISPOSITIVO"] = "cpu"   # o pipeline sempre usa CPU
    logging.basicConfig(level=logging.INFO)
    fora = [x for x in (a.sementes or ()) if x not in SEMENTES_AVALIACAO]
    if fora:
        raise SystemExit(f"sementes {fora} fora das confirmatórias {SEMENTES_AVALIACAO} "
                         f"(a {SEMENTE_CALIB} é só de calibração; a 0 do desenho antigo é exploratória)")
    tarefas = a.tarefas or (["A"] if a.quick else ["A", "B", "B_disjunta"])
    niveis = a.niveis or list(NIVEIS)
    sementes = a.sementes or ([SEMENTES_AVALIACAO[0]] if a.quick else list(SEMENTES_AVALIACAO))
    conds = [c for c in CONDICOES if not (a.quick and c == "media_sem_particao")]
    dir_res = Path(a.saida)
    dir_res.mkdir(exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.processos, initializer=_preparar_processo) as ex:
        if a.fase == "teste_curto":
            tempos = {"resultado": fase_teste_curto(ex, dir_res)}
        elif a.fase == "calibracao":
            tempos = fase_calibracao(ex, dir_res, tarefas, niveis)
        else:
            tempos = fase_completo(ex, dir_res, tarefas, niveis, sementes, conds)
    tempos["total_s"] = time.time() - t0
    nome = "quick" if a.quick else a.fase
    (dir_res / f"tempo_{nome}.json").write_text(json.dumps(tempos, indent=1, default=str))
    logger.info("tempos: %s", tempos)


if __name__ == "__main__":
    main()
