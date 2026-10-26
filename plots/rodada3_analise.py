"""Análise de decisão da RODADA 3 (PREREGISTRO.md: "Rodada 3 · Por que a média entre clientes com dados diferentes
protege..." e bloco de 2026-10-26 sobre hash e leituras a confirmar). Só lê CSVs e JSONs; não roda nada.

Regras literais: δ = 0,05; IC pareado por semente (bootstrap percentil, 10 000, default_rng(0), de `analise.py`),
fechado; janela principal = tarefas 25 a 29 (0-based) com gap, média por semente; sementes 16 a 23.

DECISÕES DE LEITURA além do texto (também listadas na resposta do gerador):
 1. Dados: o nome do arquivo `<cond>_m3_<A|B>_<semente>.csv` manda (cond, tarefa, semente); sementes fora de 16..23
    (ex.: 1000) são descartadas. Semente "completa" = 5 linhas com gap na janela; senão a comparação é
    "não avaliável (falta ...)".
 2. Calibração: `calibracao/selecao_hp_<cond>_<A|B>_m3.json`, campo `ok` booleano obrigatório (ausente =
    "falta ..."; false = "calibração não atingida"). Vale para a condição nas DUAS tarefas (literal: "nas duas
    tarefas"); `fedavg_iid` é ok por definição (R=75), sem arquivo. As demais condições, inclusive `sem_media_k1`,
    precisam do arquivo.
 3. Elegibilidade pela perda: por tarefa, |média das médias por semente de loss_nova da condição − a do fedavg_iid|
    / a do fedavg_iid ≤ 0,10 (fechado); acc_nova (±0,03) só reportada. Ordem dos motivos: calibração, dados,
    nível.
 4. Por tarefa, H_dados/H_cópias é "não avaliável" se QUALQUER dos K = 5, 10, 20 o é (não se decide com parte).
 5. H_K: a elegibilidade exige as três condições da `particao_iid` (K = 5, 10, 20); "sem faixa dinâmica" é checada
    antes da ordem das médias.
 6. Veredito geral: "não avaliável" tem precedência sobre "parcial"; P_rep falha ou falta = todas as hipóteses
    "não avaliável" (os números por K continuam na tabela como descritivos). Dois "inconclusivo" ou um
    "inconclusivo" com "atendida" = "parcial" (regra dos "demais casos"); o geral nunca é "inconclusivo".
 7. Braço de épocas fixas: contraste d_K = gap(particao_epocas_kK) − gap(copias_kK) para K = 5, 20 (e 10 com o
    `fedavg_iid`, que é igual a particao_epocas_k10 por igualdade); K = 2 também. Sem veredito.
 8. Previsão: "H_cópias: não" = "não atendida" (leitura 2 do bloco de 2026-10-26); "H_K: sim" = H_K como registrada.
    Geral "não avaliável" não conta.
CLI: python plots/rodada3_analise.py --results results_rodada3
"""
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise as AN  # noqa: E402

SEMENTES = tuple(range(16, 24))
JANELA = (25, 29)
N_JANELA = 5
DELTA = 0.05
TOL_CE = 0.10
TOL_ACC = 0.03
ZERO = 0.01
TIPOS = ("A", "B")
KS = (5, 10, 20)
CALIB_FEDAVG = dict(rodadas=75, ok=True, erro_rel=0.0)
MSG_PREP = "não avaliável (a vantagem do FedAvg não se reproduziu)"
SEM_FAIXA = "inconclusivo (sem faixa dinâmica)"
CONDICOES = ("fedavg_iid", "particao_iid_k2", "particao_iid_k5", "particao_iid_k20", "copias_k2", "copias_k5",
             "copias_k10", "copias_k20", "sem_media_k1", "particao_epocas_k2", "particao_epocas_k5",
             "particao_epocas_k20")
PREVISAO = (("H_dados", "não"), ("H_cópias", "não"), ("H_K", "sim"))


def cond_part(k):
    return "fedavg_iid" if k == 10 else f"particao_iid_k{k}"


def cond_cop(k):
    return f"copias_k{k}"


def cond_ep(k):
    return "fedavg_iid" if k == 10 else f"particao_epocas_k{k}"


# ---------------------------------------------------------------- dados

def carregar(results="results_rodada3"):
    """Lê <results>/<cond>_m3_<A|B>_<semente>.csv (raiz; calibracao/ não entra). Sementes só 16 a 23."""
    dfs = []
    for a in sorted(glob.glob(os.path.join(results, "*.csv"))):
        m = re.match(r"^(.+)_m3_([AB])_(\d+)\.csv$", os.path.basename(a))
        if not m:
            continue
        s = int(m.group(3))
        if s not in SEMENTES:
            warnings.warn(f"semente {s} descartada (fora de 16..23): {a}", stacklevel=2)
            continue
        d = pd.read_csv(a)
        d["cond"], d["tipo"], d["semente"] = m.group(1), m.group(2), s
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def carregar_calibracao(results="results_rodada3"):
    """{(cond, tipo): dict|None}; fedavg_iid é fixo (R=75, ok, erro 0); arquivo ausente/ilegível = None."""
    out = {}
    for c in CONDICOES:
        for t in TIPOS:
            if c == "fedavg_iid":
                out[(c, t)] = dict(CALIB_FEDAVG)
                continue
            try:
                with open(os.path.join(results, "calibracao", f"selecao_hp_{c}_{t}_m3.json"), encoding="utf-8") as f:
                    out[(c, t)] = json.load(f)
            except (OSError, ValueError):
                out[(c, t)] = None
    return out


def _janela(df, cond, tipo):
    if df is None or df.empty:
        return pd.DataFrame(columns=["semente", "tarefa", "gap"])
    d = df[(df["cond"] == cond) & (df["tipo"] == tipo) & (df["tarefa"] >= JANELA[0]) & (df["tarefa"] <= JANELA[1])]
    return d[d["gap"].notna()]


def serie(df, cond, tipo, col="gap"):
    """{semente: média de `col` nas 5 tarefas da janela com gap}; só sementes completas (5 tarefas)."""
    d = _janela(df, cond, tipo)
    out = {}
    for s, g in d.groupby("semente"):
        if g["tarefa"].nunique() == N_JANELA and col in g.columns:
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            out[int(s)] = float(v.mean()) if len(v) else float("nan")
    return out


def faltas(df, cond, tipo):
    """Texto do que falta (sementes ausentes ou com tarefas da janela faltando); '' se completo."""
    d = _janela(df, cond, tipo)
    ausentes = [s for s in SEMENTES if s not in set(d["semente"])]
    incompl = [s for s in SEMENTES if s in set(d["semente"]) and d[d["semente"] == s]["tarefa"].nunique() < N_JANELA]
    partes = []
    if ausentes:
        partes.append("semente(s) " + ", ".join(map(str, ausentes)))
    if incompl:
        partes.append("tarefa(s) da janela da(s) semente(s) " + ", ".join(map(str, incompl)))
    return f"{cond} {tipo}: " + "; ".join(partes) if partes else ""


def resumo(df, cond, tipo, col="gap"):
    v = np.array([x for x in serie(df, cond, tipo, col).values() if x == x])
    m, lo, hi = AN.bootstrap_ic(v)
    return dict(media=m, inf=lo, sup=hi, n=len(v))


def contraste(df, a, b, tipo, col="gap"):
    """Diferença pareada por semente a − b; falta (str) se alguma das 8 sementes completas faltar."""
    f = " e ".join(x for x in (faltas(df, a, tipo), faltas(df, b, tipo)) if x)
    if f:
        return dict(falta=f, media=np.nan, inf=np.nan, sup=np.nan, n=0)
    sa, sb = serie(df, a, tipo, col), serie(df, b, tipo, col)
    m, lo, hi = AN.bootstrap_ic(np.array([sa[s] - sb[s] for s in SEMENTES]))
    return dict(falta="", media=m, inf=lo, sup=hi, n=len(SEMENTES))


# ---------------------------------------------------------------- calibração e elegibilidade

def calib_cond(calib, cond):
    """None se a calibração da condição está ok nas duas tarefas; senão motivo (sem o prefixo 'não avaliável')."""
    if cond == "fedavg_iid":
        return None
    falta, ruim = [], []
    for t in TIPOS:
        k = (calib or {}).get((cond, t))
        if k is None:
            falta.append(f"selecao_hp_{cond}_{t}_m3.json")
        elif not isinstance(k.get("ok"), bool):
            falta.append(f"campo ok em selecao_hp_{cond}_{t}_m3.json")
        elif not k["ok"]:
            ruim.append(t)
    if falta:
        return "falta " + ", ".join(falta)
    if ruim:
        return f"calibração não atingida ({cond}, tarefa {', '.join(ruim)})"
    return None


def perda_media(df, cond, tipo):
    v = [x for x in serie(df, cond, tipo, "loss_nova").values() if x == x]
    return float(np.mean(v)) if v else float("nan")


def razao_perda(df, cond, tipo):
    """|perda(cond) − perda(fedavg_iid)| / perda(fedavg_iid) (média das médias por semente)."""
    ref = perda_media(df, "fedavg_iid", tipo)
    return abs(perda_media(df, cond, tipo) - ref) / ref


def dif_acc_nova(df, cond, tipo):
    a, b = serie(df, cond, tipo, "acc_nova"), serie(df, "fedavg_iid", tipo, "acc_nova")
    va, vb = [x for x in a.values() if x == x], [x for x in b.values() if x == x]
    return float(np.mean(va) - np.mean(vb)) if va and vb else float("nan")


def comparavel(df, calib, conds, tipo):
    """None se a comparação é elegível; senão o texto 'não avaliável (...)'."""
    for c in dict.fromkeys(conds):
        m = calib_cond(calib, c)
        if m:
            return f"não avaliável ({m})" if m.startswith("falta") else "não avaliável (calibração não atingida)"
    f = " e ".join(x for x in (faltas(df, c, tipo) for c in dict.fromkeys(conds + ("fedavg_iid",))) if x)
    if f:
        return f"não avaliável (falta {f})"
    for c in dict.fromkeys(conds):
        if c != "fedavg_iid" and not razao_perda(df, c, tipo) <= TOL_CE + 1e-12:
            return "não avaliável (nível de aprendizado diferente)"
    return None


def elegibilidade(df, calib, cond, tipo):
    """Linha descritiva: calibração, razão da perda, elegível, diferença de acc_nova (só reportada)."""
    k = (calib or {}).get((cond, tipo))
    m = comparavel(df, calib, (cond,), tipo)
    ok_cal = calib_cond(calib, cond) is None
    ra = float("nan") if faltas(df, cond, tipo) or faltas(df, "fedavg_iid", tipo) else razao_perda(df, cond, tipo)
    da = dif_acc_nova(df, cond, tipo)
    return dict(rodadas=None if k is None else k.get("rodadas"), erro_rel=None if k is None else k.get("erro_rel"),
                ok_calib=ok_cal, razao=ra, elegivel=m is None, motivo=m or "", dacc=da,
                acc_ok=None if da != da else abs(da) <= TOL_ACC + 1e-12)


# ---------------------------------------------------------------- hipóteses

def p_rep(df):
    """P_rep: inf do IC de gap(fedavg_iid) − gap(sem_media_k1) > +δ em A e em B. 'ok' é True/False/None (falta)."""
    por, faltou = {}, []
    for t in TIPOS:
        c = contraste(df, "fedavg_iid", "sem_media_k1", t)
        por[t] = dict(c=c, ok=None if c["falta"] else bool(c["inf"] > DELTA))
        if c["falta"]:
            faltou.append(c["falta"])
    if faltou:
        return dict(por=por, ok=None, msg=f"não avaliável (falta {'; '.join(faltou)})")
    ok = all(por[t]["ok"] for t in TIPOS)
    return dict(por=por, ok=ok, msg="" if ok else MSG_PREP)


def _cmp(df, calib, a, b, tipo, regra):
    m = comparavel(df, calib, (a, b), tipo)
    if m:
        return dict(res=m, c=None)
    c = contraste(df, a, b, tipo)
    if c["inf"] != c["inf"]:
        return dict(res="não avaliável (IC indefinido)", c=c)
    return dict(res="vale" if regra(c) else "não vale", c=c)


def regra_dados(c):
    return c["inf"] > DELTA


def regra_copias(c):
    return c["inf"] >= -DELTA and c["sup"] <= DELTA


def veredito_k(porK):
    """atendida / parcial / não atendida / não avaliável (K = 5, 10, 20)."""
    nav = [f"K={k}: {porK[k]['res']}" for k in KS if porK[k]["res"].startswith("não avaliável")]
    if nav:
        return "não avaliável (" + "; ".join(nav) + ")"
    vale = {k for k in KS if porK[k]["res"] == "vale"}
    if vale == set(KS):
        return "atendida"
    if 10 in vale and len(vale) == 2:
        return "parcial"
    return "não atendida"


def h_dados(df, calib, tipo, familia=cond_part):
    porK = {k: _cmp(df, calib, familia(k), cond_cop(k), tipo, regra_dados) for k in (2,) + KS}
    return dict(porK=porK, veredito=veredito_k(porK))


def h_copias(df, calib, tipo):
    porK = {k: _cmp(df, calib, cond_cop(k), "sem_media_k1", tipo, regra_copias) for k in (2,) + KS}
    return dict(porK=porK, veredito=veredito_k(porK))


def tendencia(df, calib, tipo, familia=cond_part):
    """Médias por K (2, 5, 10, 20), ordem 20>10>5 e contraste gap(20) − gap(5) (sem veredito)."""
    medias = {k: resumo(df, familia(k), tipo) for k in (2,) + KS}
    c = contraste(df, familia(20), familia(5), tipo)
    m = [medias[k]["media"] for k in (20, 10, 5)]
    ordem = None if any(x != x for x in m) else bool(m[0] > m[1] > m[2])
    acima = None if c["falta"] or c["inf"] != c["inf"] else bool(c["inf"] > 0)
    return dict(medias=medias, ordem=ordem, c=c, ic_acima0=acima,
                elegivel=comparavel(df, calib, tuple(familia(k) for k in KS), tipo))


def h_k(df, calib, tipo):
    t = tendencia(df, calib, tipo)
    if t["elegivel"]:
        ver = t["elegivel"]
    elif all(abs(t["medias"][k]["media"]) < ZERO for k in KS):
        ver = SEM_FAIXA
    else:
        ver = "atendida" if (t["ordem"] and t["ic_acima0"]) else "não atendida"
    return dict(t=t, veredito=ver)


def geral(vers):
    """Veredito geral a partir dos vereditos por tarefa (A, B)."""
    nav = [v for v in vers if v.startswith("não avaliável")]
    if nav:
        return nav[0] if len(set(vers)) == 1 else "não avaliável (" + "; ".join(
            f"{t}: {v}" for t, v in zip(TIPOS, vers) if v.startswith("não avaliável")) + ")"
    if all(v == "atendida" for v in vers):
        return "atendida"
    if all(v == "não atendida" for v in vers):
        return "não atendida"
    return "parcial"


def decidir(df, calib):
    """Tudo: P_rep, hipóteses por tarefa e veredito geral (P_rep falho/ausente derruba todas)."""
    pr = p_rep(df)
    hip = {"H_dados": {t: h_dados(df, calib, t) for t in TIPOS},
           "H_cópias": {t: h_copias(df, calib, t) for t in TIPOS},
           "H_K": {t: h_k(df, calib, t) for t in TIPOS}}
    res = {}
    for nome, por in hip.items():
        vers = [pr["msg"] if pr["ok"] is not True else por[t]["veredito"] for t in TIPOS]
        res[nome] = dict(por=por, vers=dict(zip(TIPOS, vers)), geral=geral(vers))
    return dict(p_rep=pr, hip=res)


def julgar_previsao(dec):
    """Itens da previsão do autor × veredito geral (regra do pré-registro; não avaliável não conta)."""
    out = []
    for nome, prev in PREVISAO:
        g = dec["hip"][nome]["geral"]
        alvo = "não atendida" if prev == "não" else "atendida"
        oposto = "atendida" if prev == "não" else "não atendida"
        if g == alvo:
            j = "acerta"
        elif g == "parcial":
            j = "acerta em parte"
        elif g == oposto:
            j = "erra"
        else:
            j = "não conta"
        out.append(dict(item=nome, previsao=prev, calculado=g, julgamento=j))
    return out


# ---------------------------------------------------------------- braço de épocas e mecanismo (exploratórios)

def braco_epocas(df, calib, tipo):
    """Mesmos contrastes d_K com particao_epocas e a tendência em K; sem veredito (elegibilidade só informativa)."""
    dk = {}
    for k in (2,) + KS:
        a, b = cond_ep(k), cond_cop(k)
        dk[k] = dict(c=contraste(df, a, b, tipo), eleg=comparavel(df, calib, (a, b), tipo),
                     vale=None)
        c = dk[k]["c"]
        dk[k]["vale"] = None if c["falta"] or c["inf"] != c["inf"] else bool(c["inf"] > DELTA)
    return dict(dk=dk, tend=tendencia(df, calib, tipo, cond_ep))


FAMILIAS = {"particao_iid": cond_part, "copias": cond_cop, "particao_epocas": cond_ep}
MEDIDAS_MEC = ("cos_medio", "encolhimento", "rank_ef", "srank_feat")


def mecanismo(df, tipo):
    """{(familia, K): {medida: resumo}} para K = 2, 5, 10, 20, mais ('sem_media_k1', 1)."""
    out = {}
    for fam, f in FAMILIAS.items():
        for k in (2, 5, 10, 20):
            out[(fam, k)] = {m: resumo(df, f(k), tipo, m) for m in MEDIDAS_MEC}
    out[("sem_media_k1", 1)] = {m: resumo(df, "sem_media_k1", tipo, m) for m in MEDIDAS_MEC}
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Vereditos da rodada 3 (texto)")
    p.add_argument("--results", default="results_rodada3")
    a = p.parse_args(argv)
    df = carregar(a.results) if os.path.isdir(a.results) else pd.DataFrame()
    if df.empty:
        print(f"sem resultados em {a.results}/ (nenhum CSV válido das sementes 16 a 23); nada foi calculado",
              file=sys.stderr)
        return 1
    dec = decidir(df, carregar_calibracao(a.results))
    print("P_rep:", "ok" if dec["p_rep"]["ok"] else dec["p_rep"]["msg"])
    for n, h in dec["hip"].items():
        print(f"{n}: {h['geral']}  (A: {h['vers']['A']}; B: {h['vers']['B']})")
    for j in julgar_previsao(dec):
        print(f"previsão {j['item']} ({j['previsao']}): {j['julgamento']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
