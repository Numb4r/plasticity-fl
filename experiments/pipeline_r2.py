"""Pipeline da rodada 2 (PREREGISTRO.md: "Rodada 2 · Qual média protege a plasticidade, e por quê?").

  --fase plano         só imprime o plano e a estimativa de tempo (não roda nada)
  --fase teste_curto   semente 1000, T=12 (T=100 num caso), cobre todos os caminhos novos e
                       aplica os critérios de PARE; escreve <saída>/calibracao/teste_curto/RESULTADO.txt
  --fase completo      seleção de μ e calibração da B_fixa (semente 1000) -> confirmatório (sementes 6 a 10)
                       -> braço longo (T=100, sementes 6 a 8). RECUSA começar se o teste curto não deu OK.

Resultados em results_rodada2/ (a semente 1000 em results_rodada2/calibracao/). Um processo por job,
1 thread, CPU. Jobs com CSV válido são pulados (retomável).
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

from experiments.calibrar import calibrar
from experiments.calibrar_ht import calibrar_ht
from src.condicoes import _base_arquivo, condicao_r2, csv_valido, executar, pasta_calib
from src.config import (
    ALFAS, MUS, NIVEL_R2, SEMENTE_CALIB, SEMENTES_LONGO, SEMENTES_R2, T_LONGO, T_PADRAO, tarefas_com_nova,
)

logger = logging.getLogger(__name__)
DIR_PADRAO = "results_rodada2"
REFS = ("fedavg_iid", "media_sem_particao", "sem_media_k1", "central_aprendizado")
LIMITE_APRENDE = 0.3     # teste curto: PARE se a lebre do central_HT fica abaixo disto (não aprende)
# segundos de CPU de uma execução de T=30 (medidos na rodada 1, m=3 interpolado, com diagnósticos)
CUSTO_S = {"fedavg_iid": 400, "media_sem_particao": 400, "fedavg_dirichlet": 400, "sem_media_k1": 130,
           "central_HT": 130, "central_lookahead": 130, "central_aprendizado": 85, "central_HT_cal": 130}
# o custo do central_HT_cal cresce com os passos locais S calibrados (15 = custo do central_HT); S só se
# conhece depois da calibração, então a estimativa dá um intervalo
S_HT_CAL_ESTIMADO = (30, 120)
NOME_METODO = {"HT": "central_HT", "lookahead": "central_lookahead", "HTcal": "central_HT_cal"}


def plano_confirmatorio() -> list[tuple]:
    """(condição, tarefa, semente, T, alfa) das sementes 6 a 10: 100 + 25 = 125 execuções."""
    jobs = []
    for tipo in ("A", "B"):
        for sem in SEMENTES_R2:
            jobs += [(c, tipo, sem, T_PADRAO, None) for c in REFS + ("central_HT_cal", "central_HT", "central_lookahead")]
            jobs += [("fedavg_dirichlet", tipo, sem, T_PADRAO, a) for a in ALFAS]
    for sem in SEMENTES_R2:
        jobs += [(c, "B_fixa", sem, T_PADRAO, None) for c in REFS + ("central_HT",)]
    return jobs


def plano_longo() -> list[tuple]:
    """Braço de horizonte longo: T=100, sementes 6 a 8, tarefas A e B, 5 condições: 30 execuções."""
    return [(c, tipo, sem, T_LONGO, None) for tipo in ("A", "B") for sem in SEMENTES_LONGO
            for c in REFS + ("central_HT",)]


def sufixo_job(alfa: float | None, T: int) -> str:
    if alfa is not None:
        return f"_a{alfa:g}"
    return "_T100" if T == T_LONGO else ""


def nome_csv(job: tuple) -> str:
    cond, tipo, sem, T, alfa = job
    return f"{cond}{sufixo_job(alfa, T)}_m{NIVEL_R2}_{tipo}_{sem}.csv"


def _preparar_processo() -> None:
    os.environ["PLAST_DISPOSITIVO"] = "cpu"
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO)


def rodar(nome: str, tipo: str, sem: int, T: int, alfa: float | None, mu: float | None,
          pasta: str, sufixo: str, dir_r2: str, s_local: int | None = None, rodadas: int | None = None) -> str:
    """Roda UMA execução (ou pula se o CSV já é válido). `pasta` é onde o CSV vai."""
    cond = condicao_r2(nome, tipo, Path(dir_r2), T=T, alfa=alfa, mu=mu, s_local=s_local, rodadas=rodadas)
    pasta = Path(pasta)
    if csv_valido(cond, tipo, sem, pasta, sufixo):
        return f"pulado {nome}{sufixo} {tipo} s{sem} (CSV válido)"
    executar(cond, tipo, sem, pasta, sufixo=sufixo)
    return f"ok {nome}{sufixo} {tipo} s{sem} T{T}"


def job_rodar_r2(nome: str, tipo: str, sem: int, T: int, alfa: float | None, dir_r2: str) -> str:
    pasta = pasta_calib(Path(dir_r2)) if sem == SEMENTE_CALIB else Path(dir_r2)
    return rodar(nome, tipo, sem, T, alfa, None, str(pasta), sufixo_job(alfa, T), dir_r2)


def job_selecao_mu(metodo: str, tarefa: str, mu: float, dir_r2: str) -> str:
    nome = NOME_METODO[metodo]
    pasta = pasta_calib(Path(dir_r2)) / "selecao_mu"
    return rodar(nome, tarefa, SEMENTE_CALIB, T_PADRAO, None, mu, str(pasta), f"_mu{mu:g}", dir_r2)


def escolher_mu(metodo: str, tarefa: str, dir_r2: Path) -> dict:
    """Maior acurácia contínua média nas 5 últimas tarefas (empate: o menor μ); borda registrada."""
    nome = NOME_METODO[metodo]
    pasta = pasta_calib(dir_r2) / "selecao_mu"
    notas = {}
    for mu in MUS:
        df = pd.read_csv(pasta / f"{nome}_mu{mu:g}_m{NIVEL_R2}_{tarefa}_{SEMENTE_CALIB}.csv")
        notas[mu] = float(df["acc_cont"].tail(5).mean())
    melhor = min((m for m in MUS if notas[m] == max(notas.values())))      # empate: o menor μ
    res = {"metodo": metodo, "tarefa": tarefa, "mu": melhor, "notas": notas, "borda": melhor in (min(MUS), max(MUS))}
    (pasta_calib(dir_r2) / f"selecao_{metodo}_{tarefa}_m{NIVEL_R2}.json").write_text(json.dumps(res, indent=1))
    logger.info("seleção de μ (%s, tarefa %s): μ=%g notas=%s%s", metodo, tarefa, melhor,
                {k: round(v, 3) for k, v in notas.items()}, " [BORDA]" if res["borda"] else "")
    return res


def job_calibrar_b_fixa(dir_r2: str) -> dict:
    arq = pasta_calib(Path(dir_r2)) / f"calibracao_B_fixa_m{NIVEL_R2}.json"
    if arq.exists():
        return json.loads(arq.read_text())
    return calibrar("B_fixa", NIVEL_R2, Path(dir_r2))


def job_calibrar_ht(tarefa: str, dir_r2: str) -> dict:
    arq = pasta_calib(Path(dir_r2)) / f"calibracao_HT_{tarefa}_m{NIVEL_R2}.json"
    if arq.exists():
        return json.loads(arq.read_text())
    return calibrar_ht(tarefa, Path(dir_r2))


def _redes(T: int) -> int:
    """Redes treinadas numa execução: T da contínua mais as de referência."""
    return T + len(tarefas_com_nova(T, janela_hpart=T <= T_PADRAO))


def _custo(job: tuple) -> float:
    """Segundos de CPU estimados: o custo vai com o número de redes treinadas (T=30: 46; T=100: 139)."""
    fator = (sum(S_HT_CAL_ESTIMADO) / 2 / 15) if job[0] == "central_HT_cal" else 1.0
    return CUSTO_S[job[0]] * fator * _redes(job[3]) / _redes(T_PADRAO)


def estimar() -> dict:
    """Estimativa de CPU (s) do completo. O central_HT_cal depende do S calibrado: dá-se o intervalo."""
    conf, longo = plano_confirmatorio(), plano_longo()
    ht_cal = [j for j in conf if j[0] == "central_HT_cal"]

    def total(s_ht: int) -> float:
        resto = sum(_custo(j) for j in conf + longo if j[0] != "central_HT_cal")
        cal = sum(CUSTO_S["central_HT_cal"] * s_ht / 15 * _redes(j[3]) / _redes(T_PADRAO) for j in ht_cal)
        fixo = (12 * 130                                   # seleção de μ do HT e do lookahead
                + 6 * 130 * s_ht / 15                      # seleção de μ do HT_cal (com os S calibrados)
                + 2 * 400 + 2 * 300 + 300)                 # msp T=30 na semente 1000, calibração do HT, calibração B_fixa
        return resto + cal + fixo

    lo, hi = (total(s) / 3600 for s in S_HT_CAL_ESTIMADO)
    return {"execucoes_confirmatorias": len(conf), "execucoes_longo": len(longo), "selecao_mu": 12 + 6,
            "S_HT_cal_supostos": list(S_HT_CAL_ESTIMADO), "cpu_horas": [lo, hi],
            "parede_3_proc_h": [lo / 3, hi / 3 * 1.8]}


# ---------------------------------------------------------------- teste curto
def jobs_teste_curto() -> list[dict]:
    """Caminhos novos, semente 1000, T=12 (T=3 na sanidade, T=100 no caso longo). O central_HT_cal roda com
    S=30 passos locais por rodada (valor de teste; a calibração real só roda no completo)."""
    J = lambda nome, tipo, T, suf, alfa=None, mu=None, s=None: dict(nome=nome, tipo=tipo, T=T, sufixo=suf, alfa=alfa, mu=mu, s_local=s)
    jobs = [J("sem_media_k1", "A", 3, "_sanidade"), J("central_HT", "A", 3, "_sanidade_mu0", mu=0.0),
            J("central_lookahead", "A", 3, "_sanidade_mu0", mu=0.0),
            J("central_HT", "A", 12, "_tc", mu=0.9), J("central_lookahead", "A", 12, "_tc", mu=0.9),
            J("sem_media_k1", "A", 12, "_tc"), J("sem_media_k1", "B_fixa", 12, "_tc"),
            J("media_sem_particao", "A", 12, "_tc"), J("central_HT_cal", "A", 12, "_tc", mu=0.9, s=30),
            J("sem_media_k1", "A", T_LONGO, "_T100_tc")]
    for a in ALFAS:
        jobs += [J("fedavg_dirichlet", tipo, 12, f"_a{a:g}_tc", alfa=a) for tipo in ("A", "B")]
    return jobs


def _csv_tc(pasta: Path, j: dict) -> pd.DataFrame:
    cond = condicao_r2(j["nome"], j["tipo"], pasta.parent.parent, T=j["T"], alfa=j["alfa"], mu=j["mu"], s_local=j["s_local"])
    return pd.read_csv(pasta / f"{_base_arquivo(cond, j['tipo'], SEMENTE_CALIB, j['sufixo'])}.csv")


def avaliar_teste_curto(dir_r2: Path) -> tuple[list[str], list[str]]:
    """(PAREs, informações). Lê os CSVs do teste curto e aplica os critérios de PARE."""
    pasta = pasta_calib(dir_r2) / "teste_curto"
    pare, info = [], []
    jobs = {(j["nome"], j["tipo"], j["sufixo"]): j for j in jobs_teste_curto()}
    df = {k: _csv_tc(pasta, j) for k, j in jobs.items()}
    # 1) μ = 0 reproduz o sem_media_k1 bit a bit (HT e lookahead)
    k1 = df[("sem_media_k1", "A", "_sanidade")]
    cols = ["acc_cont", "loss_cont", "norma", "distancia", "rank_ef", "acc_nova", "loss_nova", "gap"]
    for nome in ("central_HT", "central_lookahead"):
        d = df[(nome, "A", "_sanidade_mu0")]
        if not k1[cols].equals(d[cols]):
            pare.append(f"{nome} com μ=0 NÃO reproduz o sem_media_k1 (colunas {cols})")
    info.append("sanidade μ=0: " + ("HT e lookahead idênticos ao sem_media_k1" if not pare else "FALHOU"))
    # 2) o central_HT aprende (a lebre)
    ht, la, k1l = df[("central_HT", "A", "_tc")], df[("central_lookahead", "A", "_tc")], df[("sem_media_k1", "A", "_tc")]
    acc_ht = float(ht["acc_cont"].tail(3).mean())
    info.append(f"acc_cont média das 3 últimas tarefas (T=12): central_HT(lebre, μ=0,9)={acc_ht:.3f}, "
                f"acc_tartaruga={float(ht['acc_tartaruga'].tail(3).mean()):.3f}, "
                f"central_lookahead(μ=0,9)={float(la['acc_cont'].tail(3).mean()):.3f}, sem_media_k1={float(k1l['acc_cont'].tail(3).mean()):.3f}")
    if acc_ht < LIMITE_APRENDE:
        pare.append(f"central_HT não aprende (acc_cont média das 3 últimas tarefas = {acc_ht:.3f} < {LIMITE_APRENDE})")
    # 3) Dirichlet: sem NaN e com partições registradas
    for (nome, tipo, suf), d in df.items():
        if nome != "fedavg_dirichlet":
            continue
        if d[["acc_cont", "loss_cont"]].isna().any().any():
            pare.append(f"fedavg_dirichlet{suf} {tipo}: NaN em acc_cont/loss_cont")
        info.append(f"dirichlet {suf} {tipo}: part_min={int(d['part_min'].min())} part_max={int(d['part_max'].max())} "
                    f"vazios(máx)={int(d['part_vazios'].max())} acc_cont final={d['acc_cont'].iloc[-1]:.3f}")
    # 4) B_fixa: reaprendizado gravado e maj_frac
    base = pasta / f"{_base_arquivo(condicao_r2('sem_media_k1', 'B_fixa', dir_r2, T=12), 'B_fixa', SEMENTE_CALIB, '_tc')}"
    rea = Path(f"{base}_reaprendizado.csv")
    if not rea.exists():
        pare.append("B_fixa: arquivo de reaprendizado não foi gravado")
    else:
        r = pd.read_csv(rea)
        if len(r) != 2 or not np.isfinite(r["passos_copia"]).all() or not np.isfinite(r["passos_nova"]).all():
            pare.append(f"B_fixa: reaprendizado malformado ({len(r)} linhas)")
        info.append(f"B_fixa reaprendizado: passos_copia={r['passos_copia'].tolist()} passos_nova={r['passos_nova'].tolist()} "
                    f"censurados={r['censurado_copia'].tolist()}")
    bf = df[("sem_media_k1", "B_fixa", "_tc")]
    frac = float((bf["maj_frac"] > 0.3).mean())
    info.append(f"B_fixa maj_frac máx={bf['maj_frac'].max():.3f}; fração de tarefas > 0,3 = {frac:.2f}"
                + (" (AVISO: gatilho de 20% do pré-registro: registrar e seguir)" if frac > 0.2 else ""))
    # 5) T=100: 100 linhas e rede nova nas tarefas certas (sem a janela 7 a 11)
    L = df[("sem_media_k1", "A", "_T100_tc")]
    esperado = set(tarefas_com_nova(T_LONGO, janela_hpart=False))
    obtido = set(L.index[L["acc_nova"].notna()])
    if len(L) != T_LONGO or obtido != esperado:
        pare.append(f"T={T_LONGO}: {len(L)} linhas; redes novas fora do esperado (faltam {sorted(esperado - obtido)[:5]}, sobram {sorted(obtido - esperado)[:5]})")
    info.append(f"T={T_LONGO}: {len(L)} linhas, {len(obtido)} redes novas (esperado {len(esperado)}), acc_cont final={L['acc_cont'].iloc[-1]:.3f}")
    # 6) sonda v2 e perdas finitas
    for (nome, tipo, suf), d in df.items():
        if len(d) >= 10 and d.loc[[4, 9], "sonda2_dlog_norm"].isna().any():
            pare.append(f"{nome}{suf} {tipo}: sonda v2 sem valor nas tarefas 4 e 9")
        if d["loss_nova"].isna().ne(d["acc_nova"].isna()).any():
            pare.append(f"{nome}{suf} {tipo}: loss_nova e acc_nova com padrões de NaN diferentes")
    # 7) central_HT_cal: roda, a rede nova responde ao S (≠ sem_media_k1) e a calibração em miniatura gravou algo finito
    cal = df[("central_HT_cal", "A", "_tc")]
    msp = df[("media_sem_particao", "A", "_tc")]
    if cal[["acc_cont", "loss_cont"]].isna().any().any():
        pare.append("central_HT_cal: NaN em acc_cont/loss_cont")
    if k1l["loss_nova"].equals(cal["loss_nova"]):
        pare.append("central_HT_cal (S=30) tem a mesma loss_nova do sem_media_k1 (S=15): o S local não está agindo")
    perda_media = lambda d: float(d["loss_nova"].dropna().mean())
    info.append(f"loss_nova média (T=12, semente 1000): media_sem_particao={perda_media(msp):.4f}, "
                f"sem_media_k1={perda_media(k1l):.4f}, central_HT_cal(S=30)={perda_media(cal):.4f}")
    arq = pasta / "calibracao_HT_tc.json"
    if not arq.exists():
        pare.append("calibração do central_HT_cal em miniatura não gravou calibracao_HT_tc.json")
    else:
        c = json.loads(arq.read_text())
        if not (np.isfinite(c["erro_rel"]) and np.isfinite(c["alvo_loss_nova_msp"])):
            pare.append("calibração do central_HT_cal em miniatura com valores não finitos")
        info.append(f"calibração HT em miniatura (T=12, S≤30): S={c['passos_locais']} erro_rel={c['erro_rel']:+.3f} "
                    f"ok={c['ok']} alvo_fora_do_alcance={c['alvo_fora_do_alcance']} (só testa o código; não decide nada)")
    return pare, info


def fase_teste_curto(ex, dir_r2: Path) -> int:
    pasta = pasta_calib(dir_r2) / "teste_curto"
    if pasta.exists():                          # o teste curto sempre recomeça: não reaproveita CSV de código antigo
        shutil.rmtree(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    futs = [ex.submit(rodar, j["nome"], j["tipo"], SEMENTE_CALIB, j["T"], j["alfa"], j["mu"], str(pasta), j["sufixo"],
                      str(dir_r2), j["s_local"]) for j in jobs_teste_curto()]
    for f in futs:
        _resultado(f)
    # calibração do central_HT_cal em miniatura: referência = media_sem_particao T=12 deste teste, S de 15 a 30
    ref_j = next(j for j in jobs_teste_curto() if j["nome"] == "media_sem_particao")
    ref = pasta / f"{_base_arquivo(condicao_r2('media_sem_particao', 'A', dir_r2, T=12), 'A', SEMENTE_CALIB, ref_j['sufixo'])}.csv"
    if ref.exists():
        calibrar_ht("A", dir_r2, T=12, passos_max=2 * 75 * 15, referencia=ref, saida=pasta / "calibracao_HT_tc.json")
    pare, info = avaliar_teste_curto(dir_r2)
    for i in info:
        logger.info("TESTE CURTO: %s", i)
    for p in pare:
        logger.warning("PARE: %s", p)
    linhas = ["PARE" if pare else "OK"] + [f"PARE: {p}" for p in pare] + [f"INFO: {i}" for i in info]
    (pasta / "RESULTADO.txt").write_text("\n".join(linhas) + "\n")
    logger.info("TESTE CURTO: %s", "PARE (veja as linhas PARE acima; NÃO solte o completo)" if pare else "OK, pode soltar o completo")
    return 3 if pare else 0


# ---------------------------------------------------------------- completo
FALHAS: list[str] = []


def _resultado(f) -> None:
    """Um job que falha não derruba o resto da noite: registra e segue; as falhas saem no fim."""
    try:
        logger.info(f.result())
    except Exception as e:                      # noqa: BLE001 (qualquer erro de worker deve ser registrado)
        FALHAS.append(repr(e))
        logger.error("JOB FALHOU: %r", e)


def fase_selecao(ex, dir_r2: Path) -> dict:
    """Semente 1000, em 3 etapas: (1) execuções de referência e seleção de μ do HT e do lookahead;
    (2) calibração dos passos locais do central_HT_cal (precisa do media_sem_particao da etapa 1) e da B_fixa;
    (3) seleção de μ do central_HT_cal (precisa dos passos locais calibrados)."""
    t0 = time.time()
    pasta_calib(dir_r2).mkdir(parents=True, exist_ok=True)
    fs = [ex.submit(job_rodar_r2, "fedavg_iid", "B_fixa", SEMENTE_CALIB, T_PADRAO, None, str(dir_r2))]
    fs += [ex.submit(job_rodar_r2, "media_sem_particao", t, SEMENTE_CALIB, T_PADRAO, None, str(dir_r2)) for t in ("A", "B")]
    fs += [ex.submit(job_selecao_mu, m, t, mu, str(dir_r2)) for m in ("HT", "lookahead") for t in ("A", "B") for mu in MUS]
    for f in fs:
        _resultado(f)
    for m in ("HT", "lookahead"):
        for t in ("A", "B"):
            escolher_mu(m, t, dir_r2)
    cal_b = ex.submit(job_calibrar_b_fixa, str(dir_r2))
    cal_ht = {t: ex.submit(job_calibrar_ht, t, str(dir_r2)) for t in ("A", "B")}
    cal = cal_b.result()
    logger.info("calibração B_fixa: passos=%s ok=%s", cal["passos"], cal["ok"])
    for t, f in cal_ht.items():
        c = f.result()
        logger.info("calibração central_HT_cal %s: S=%d erro_rel=%+.3f ok=%s%s", t, c["passos_locais"], c["erro_rel"], c["ok"],
                    "" if c["ok"] else "  [FORA DA TOLERÂNCIA: H_tempo e H_dist ficarão 'não avaliável']")
    for f in [ex.submit(job_selecao_mu, "HTcal", t, mu, str(dir_r2)) for t in ("A", "B") for mu in MUS]:
        _resultado(f)
    for t in ("A", "B"):
        escolher_mu("HTcal", t, dir_r2)
    return {"selecao_e_calibracao_s": time.time() - t0}


def fase_completo(ex, dir_r2: Path, sem_teste_curto: bool = False) -> dict:
    res = dir_r2 / "calibracao" / "teste_curto" / "RESULTADO.txt"
    if not sem_teste_curto and not (res.exists() and res.read_text().startswith("OK")):
        raise SystemExit("o teste curto não deu OK (ou não rodou): rode `--fase teste_curto` e leia RESULTADO.txt. "
                         "Para ignorar, --sem-teste-curto.")
    tempos = fase_selecao(ex, dir_r2)
    for nome, plano in (("confirmatorio", plano_confirmatorio()), ("longo", plano_longo())):
        t0 = time.time()
        plano = sorted(plano, key=_custo, reverse=True)
        fs = [ex.submit(job_rodar_r2, c, tipo, sem, T, alfa, str(dir_r2)) for c, tipo, sem, T, alfa in plano]
        for f in fs:
            _resultado(f)
        tempos[f"{nome}_s"] = time.time() - t0
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
        return
    os.environ["PLAST_DISPOSITIVO"] = "cpu"
    dir_r2 = Path(a.saida)
    dir_r2.mkdir(exist_ok=True)
    t0 = time.time()
    codigo = 0
    with ProcessPoolExecutor(a.processos, initializer=_preparar_processo) as ex:
        if a.fase == "teste_curto":
            codigo = fase_teste_curto(ex, dir_r2)
            tempos = {}
        else:
            tempos = fase_completo(ex, dir_r2, a.sem_teste_curto)
    tempos["total_s"] = time.time() - t0
    (dir_r2 / f"tempo_{a.fase}.json").write_text(json.dumps(tempos, indent=1))
    logger.info("tempos: %s", tempos)
    if FALHAS:
        logger.error("%d JOB(S) FALHARAM: %s (rode a mesma fase de novo para retomar)", len(FALHAS), FALHAS)
        codigo = codigo or 1
    raise SystemExit(codigo)


if __name__ == "__main__":
    main()
