"""Pipeline da rodada 2b (PREREGISTRO.md: "Rodada 2b · `central_HT_cal2`").

  --fase plano         imprime o plano e a estimativa de tempo (não roda nada)
  --fase teste_curto   semente 1000, T curto; cobre os caminhos novos e aplica os critérios de PARE;
                       escreve <saída>/calibracao/teste_curto/RESULTADO.txt (código de saída 3 se PARE)
  --fase completo      calibração de R (A e B) -> seleção de μ -> 30 execuções (sementes 11 a 15).
                       RECUSA começar se o teste curto não deu OK.

Resultados em results_rodada2b/ (a semente 1000 em results_rodada2b/calibracao/). O alvo da calibração (o
media_sem_particao da semente 1000) vem de results_rodada2/calibracao/ (rodada 2, somente leitura).
Um processo por job, 1 thread, CPU. Jobs com CSV válido são pulados (retomável).
"""
import argparse
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.calibrar_ht2 import calibrar_ht2
from experiments.pipeline_r2 import _preparar_processo, rodar
from src.condicoes import _base_arquivo, condicao_r2, pasta_calib
from src.config import MUS_2B, NIVEL_R2, R_CAL2, SEMENTE_CALIB, SEMENTES_R2B, T_PADRAO

logger = logging.getLogger(__name__)
DIR_PADRAO = "results_rodada2b"
CONDICOES_2B = ("central_HT_cal2", "media_sem_particao", "sem_media_k1")   # sem_media_k1: controle de P_vant
LIMITE_APRENDE = 0.3
# segundos de CPU de uma execução de T=30 (rodada 1/2; HT_cal2 cresce com R/75)
CUSTO_S = {"central_HT_cal2": 130, "media_sem_particao": 400, "sem_media_k1": 130}
R_ESTIMADO = (75, 150)
FALHAS: list[str] = []


def plano_confirmatorio() -> list[tuple]:
    """(condição, tarefa, semente, T) das sementes 11 a 15: 3 condições x 2 tarefas x 5 sementes = 30 execuções."""
    return [(c, tipo, sem, T_PADRAO) for tipo in ("A", "B") for sem in SEMENTES_R2B for c in CONDICOES_2B]


def nome_csv(job: tuple) -> str:
    cond, tipo, sem, _ = job
    return f"{cond}_m{NIVEL_R2}_{tipo}_{sem}.csv"


def _resultado(f) -> None:
    try:
        logger.info(f.result())
    except Exception as e:                      # noqa: BLE001 (qualquer erro de worker deve ser registrado)
        FALHAS.append(repr(e))
        logger.error("JOB FALHOU: %r", e)


def job_rodar(nome: str, tipo: str, sem: int, T: int, dir_r2b: str) -> str:
    pasta = pasta_calib(Path(dir_r2b)) if sem == SEMENTE_CALIB else Path(dir_r2b)
    return rodar(nome, tipo, sem, T, None, None, str(pasta), "", dir_r2b)


def job_calibrar(tarefa: str, dir_r2b: str) -> dict:
    arq = pasta_calib(Path(dir_r2b)) / f"calibracao_HT2_{tarefa}_m{NIVEL_R2}.json"
    if arq.exists():
        return json.loads(arq.read_text())
    return calibrar_ht2(tarefa, Path(dir_r2b))


def job_selecao_mu(tarefa: str, mu: float, rodadas: int, dir_r2b: str) -> str:
    pasta = pasta_calib(Path(dir_r2b)) / "selecao_mu"
    return rodar("central_HT_cal2", tarefa, SEMENTE_CALIB, T_PADRAO, None, mu, str(pasta), f"_mu{mu:g}", dir_r2b,
                 rodadas=rodadas)


def escolher_mu(tarefa: str, dir_r2b: Path, grade: tuple[float, ...] = MUS_2B) -> dict:
    """Maior acurácia contínua média nas 5 últimas tarefas (empate: o menor μ); borda registrada."""
    pasta = pasta_calib(dir_r2b) / "selecao_mu"
    notas = {}
    for mu in grade:
        df = pd.read_csv(pasta / f"central_HT_cal2_mu{mu:g}_m{NIVEL_R2}_{tarefa}_{SEMENTE_CALIB}.csv")
        notas[mu] = float(df["acc_cont"].tail(5).mean())
    melhor = min(m for m in grade if notas[m] == max(notas.values()))
    res = {"metodo": "HTcal2", "tarefa": tarefa, "mu": melhor, "notas": notas, "borda": melhor in (min(grade), max(grade))}
    (pasta_calib(dir_r2b) / f"selecao_HTcal2_{tarefa}_m{NIVEL_R2}.json").write_text(json.dumps(res, indent=1))
    logger.info("seleção de μ (HT_cal2, tarefa %s): μ=%g notas=%s%s", tarefa, melhor,
                {k: round(v, 3) for k, v in notas.items()}, " [BORDA]" if res["borda"] else "")
    return res


def estimar() -> dict:
    n = len(plano_confirmatorio())
    def total(r: int) -> float:
        runs = sum(CUSTO_S[j[0]] * (r / 75 if j[0] == "central_HT_cal2" else 1.0) for j in plano_confirmatorio())
        sel = len(MUS_2B) * 2 * CUSTO_S["central_HT_cal2"] * r / 75 * 0.6     # seleção sem diagnósticos
        cal = 2 * 400                                                          # calibração de R (2 tarefas)
        return runs + sel + cal
    lo, hi = (total(r) / 3600 for r in R_ESTIMADO)
    return {"execucoes": n, "R_supostos": list(R_ESTIMADO), "cpu_horas": [lo, hi],
            "parede_20_proc_min": [lo / 20 * 60 * 1.5, hi / 20 * 60 * 2.5], "parede_3_proc_h": [lo / 3, hi / 3 * 1.8]}


# ---------------------------------------------------------------- teste curto
R_TC = 80                                      # R de teste (≠ 75); a calibração real só roda no completo


def jobs_teste_curto() -> list[dict]:
    """Semente 1000. Sanidade (T=3): HT_cal2 com μ=0 e R=R_TC reproduz o sem_media_k1 com R_TC rodadas.
    T=12: HT_cal2 (R_TC, μ=0,99) e sem_media_k1 de referência; msp (T=12) alimenta a calibração em miniatura."""
    J = lambda nome, T, suf, mu=None, r=None: dict(nome=nome, tipo="A", T=T, sufixo=suf, mu=mu, rodadas=r)
    return [J("sem_media_k1", 3, "_sanidade", r=R_TC), J("central_HT_cal2", 3, "_sanidade_mu0", mu=0.0, r=R_TC),
            J("central_HT_cal2", 12, "_tc", mu=0.99, r=R_TC), J("sem_media_k1", 12, "_tc"),
            J("media_sem_particao", 12, "_tc")]


def _caminho_tc(pasta: Path, j: dict, dir_r2b: Path) -> Path:
    cond = condicao_r2(j["nome"], j["tipo"], dir_r2b, T=j["T"], mu=j["mu"], rodadas=j["rodadas"])
    return pasta / f"{_base_arquivo(cond, j['tipo'], SEMENTE_CALIB, j['sufixo'])}.csv"


def avaliar_teste_curto(dir_r2b: Path) -> tuple[list[str], list[str]]:
    pasta = pasta_calib(dir_r2b) / "teste_curto"
    pare, info = [], []
    jobs = {(j["nome"], j["sufixo"]): j for j in jobs_teste_curto()}
    df = {k: pd.read_csv(_caminho_tc(pasta, j, dir_r2b)) for k, j in jobs.items()}
    # 1) μ = 0 com R≠75 reproduz o sem_media_k1 com o mesmo R, bit a bit
    k1, ht0 = df[("sem_media_k1", "_sanidade")], df[("central_HT_cal2", "_sanidade_mu0")]
    cols = ["acc_cont", "loss_cont", "norma", "distancia", "rank_ef", "acc_nova", "loss_nova", "gap"]
    if not k1[cols].equals(ht0[cols]):
        pare.append(f"central_HT_cal2 com μ=0 e R={R_TC} NÃO reproduz o sem_media_k1 com R={R_TC} (colunas {cols})")
    info.append(f"sanidade μ=0 (R={R_TC}): " + ("idêntico ao sem_media_k1" if k1[cols].equals(ht0[cols]) else "FALHOU"))
    # 2) o HT_cal2 aprende, o R foi aplicado (params) e a rede nova responde ao R (≠ sem_media_k1 com 75 rodadas)
    ht, k1l, msp = df[("central_HT_cal2", "_tc")], df[("sem_media_k1", "_tc")], df[("media_sem_particao", "_tc")]
    acc = float(ht["acc_cont"].tail(3).mean())
    info.append(f"acc_cont média das 3 últimas tarefas (T=12): central_HT_cal2(R={R_TC}, μ=0,99)={acc:.3f}, "
                f"acc_tartaruga={float(ht['acc_tartaruga'].tail(3).mean()):.3f}, sem_media_k1(R=75)={float(k1l['acc_cont'].tail(3).mean()):.3f}")
    if acc < LIMITE_APRENDE:
        pare.append(f"central_HT_cal2 não aprende (acc_cont média das 3 últimas tarefas = {acc:.3f} < {LIMITE_APRENDE})")
    if ht[["acc_cont", "loss_cont", "loss_nova"]].isna().all(axis=None):
        pare.append("central_HT_cal2: colunas vazias")
    p = json.loads(Path(str(_caminho_tc(pasta, jobs[("central_HT_cal2", "_tc")], dir_r2b))[:-4] + ".params.json").read_text())
    if p.get("rodadas") != R_TC or p.get("s_local") != 15 or p.get("familia") != "ht":
        pare.append(f"params do central_HT_cal2 não registram R={R_TC}, S=15, familia=ht: {p}")
    if ht["loss_nova"].dropna().equals(k1l["loss_nova"].dropna()):
        pare.append(f"central_HT_cal2 (R={R_TC}) tem a mesma loss_nova do sem_media_k1 (R=75): o R não está agindo")
    # 3) calibração em miniatura: referência = o sem_media_k1 T=12 deste teste (só testa o código), grade R_TC..R_TC+4
    ref = _caminho_tc(pasta, jobs[("sem_media_k1", "_tc")], dir_r2b)
    c = calibrar_ht2("A", dir_r2b, T=12, grade=tuple(range(R_TC, R_TC + 5)), referencia=ref, saida=pasta / "calibracao_HT2_tc.json")
    if not (np.isfinite(c["erro_rel"]) and c["rodadas"] in range(R_TC, R_TC + 5) and len(c["tentativas"]) == 5):
        pare.append(f"calibração em miniatura malformada: {c['rodadas']=} {c['erro_rel']=} {len(c['tentativas'])=}")
    info.append(f"calibração HT2 em miniatura (T=12, R {R_TC}..{R_TC + 4}, alvo = loss_nova do k1): R={c['rodadas']} "
                f"erro_rel={c['erro_rel']:+.3f} (só testa o código; não decide nada)")
    info.append(f"loss_nova média (T=12, semente 1000): media_sem_particao={float(msp['loss_nova'].dropna().mean()):.4f}, "
                f"sem_media_k1={float(k1l['loss_nova'].dropna().mean()):.4f}, central_HT_cal2(R={R_TC})={float(ht['loss_nova'].dropna().mean()):.4f}")
    # 4) sonda v2 e perdas finitas
    for (nome, suf), d in df.items():
        if len(d) >= 10 and d.loc[[4, 9], "sonda2_dlog_norm"].isna().any():
            pare.append(f"{nome}{suf}: sonda v2 sem valor nas tarefas 4 e 9")
        if d["loss_nova"].isna().ne(d["acc_nova"].isna()).any():
            pare.append(f"{nome}{suf}: loss_nova e acc_nova com padrões de NaN diferentes")
    return pare, info


def fase_teste_curto(ex, dir_r2b: Path) -> int:
    pasta = pasta_calib(dir_r2b) / "teste_curto"
    if pasta.exists():                          # o teste curto sempre recomeça: não reaproveita CSV de código antigo
        shutil.rmtree(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    futs = [ex.submit(rodar, j["nome"], j["tipo"], SEMENTE_CALIB, j["T"], None, j["mu"], str(pasta), j["sufixo"],
                      str(dir_r2b), None, j["rodadas"]) for j in jobs_teste_curto()]
    for f in futs:
        _resultado(f)
    try:
        pare, info = avaliar_teste_curto(dir_r2b)
    except Exception as e:                      # noqa: BLE001 (CSV faltando por falha de job)
        pare, info = [f"não foi possível avaliar o teste curto: {e!r}"], []
    for i in info:
        logger.info("TESTE CURTO: %s", i)
    for p in pare:
        logger.warning("PARE: %s", p)
    linhas = ["PARE" if pare else "OK"] + [f"PARE: {p}" for p in pare] + [f"INFO: {i}" for i in info]
    (pasta / "RESULTADO.txt").write_text("\n".join(linhas) + "\n")
    logger.info("TESTE CURTO: %s", "PARE (veja as linhas PARE acima; NÃO solte o completo)" if pare else "OK, pode soltar o completo")
    return 3 if pare else 0


# ---------------------------------------------------------------- completo
def fase_selecao(ex, dir_r2b: Path) -> dict:
    """Semente 1000: (1) calibração de R (A e B); (2) seleção de μ com o R calibrado."""
    t0 = time.time()
    pasta_calib(dir_r2b).mkdir(parents=True, exist_ok=True)
    cal = {t: ex.submit(job_calibrar, t, str(dir_r2b)) for t in ("A", "B")}
    rodadas = {}
    for t, f in cal.items():
        c = f.result()
        rodadas[t] = c["rodadas"]
        logger.info("calibração central_HT_cal2 %s: R=%d erro_rel=%+.3f ok=%s%s", t, c["rodadas"], c["erro_rel"], c["ok"],
                    "" if c["ok"] else "  [FORA DA TOLERÂNCIA: H_tempo e H_dist ficarão 'não avaliável']")
    for f in [ex.submit(job_selecao_mu, t, mu, rodadas[t], str(dir_r2b)) for t in ("A", "B") for mu in MUS_2B]:
        _resultado(f)
    for t in ("A", "B"):
        escolher_mu(t, dir_r2b)
    return {"calibracao_e_selecao_s": time.time() - t0}


def _custo(job: tuple) -> float:
    return CUSTO_S[job[0]] * (112 / 75 if job[0] == "central_HT_cal2" else 1.0)


def fase_completo(ex, dir_r2b: Path, sem_teste_curto: bool = False) -> dict:
    res = dir_r2b / "calibracao" / "teste_curto" / "RESULTADO.txt"
    if not sem_teste_curto and not (res.exists() and res.read_text().startswith("OK")):
        raise SystemExit("o teste curto não deu OK (ou não rodou): rode `--fase teste_curto` e leia RESULTADO.txt. "
                         "Para ignorar, --sem-teste-curto.")
    tempos = fase_selecao(ex, dir_r2b)
    t0 = time.time()
    plano = sorted(plano_confirmatorio(), key=_custo, reverse=True)
    for f in [ex.submit(job_rodar, c, tipo, sem, T, str(dir_r2b)) for c, tipo, sem, T in plano]:
        _resultado(f)
    tempos["confirmatorio_s"] = time.time() - t0
    return tempos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", choices=["plano", "teste_curto", "completo"], required=True)
    ap.add_argument("--saida", default=DIR_PADRAO)
    ap.add_argument("--processos", type=int, default=3, help="cada processo usa ~0,3 GB")
    ap.add_argument("--sem-teste-curto", action="store_true", help="não exige RESULTADO.txt do teste curto")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.fase == "plano":
        e = estimar()
        print(json.dumps(e, indent=1))
        print(f"R de {R_CAL2[0]} a {R_CAL2[-1]} ({len(R_CAL2)} valores), μ em {MUS_2B}")
        return
    os.environ["PLAST_DISPOSITIVO"] = "cpu"
    dir_r2b = Path(a.saida)
    dir_r2b.mkdir(exist_ok=True)
    t0 = time.time()
    codigo = 0
    with ProcessPoolExecutor(a.processos, initializer=_preparar_processo) as ex:
        if a.fase == "teste_curto":
            codigo = fase_teste_curto(ex, dir_r2b)
            tempos = {}
        else:
            tempos = fase_completo(ex, dir_r2b, a.sem_teste_curto)
    tempos["total_s"] = time.time() - t0
    (dir_r2b / f"tempo_{a.fase}.json").write_text(json.dumps(tempos, indent=1))
    logger.info("tempos: %s", tempos)
    if FALHAS:
        logger.error("%d JOB(S) FALHARAM: %s (rode a mesma fase de novo para retomar)", len(FALHAS), FALHAS)
        codigo = codigo or 1
    raise SystemExit(codigo)


if __name__ == "__main__":
    main()
