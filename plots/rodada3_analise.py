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
 4. Por tarefa, H_dados/H_cópias é "não avaliável" se QUALQUER dos K = 5, 10, 25 o é (não se decide com parte).
 5. H_K: a elegibilidade exige as três condições da `particao_iid` (K = 5, 10, 25); "sem faixa dinâmica" é checada
    antes da ordem das médias.
 6. Veredito geral: "não avaliável" tem precedência sobre "parcial". P_rep é POR TAREFA (decisão do autor,
    2026-10-26): onde falha, o veredito da tarefa é "não avaliável (a vantagem do FedAvg não se reproduziu)" (os
    números continuam na tabela como descritivos); onde é não avaliável (falta, calibração, nível do
    sem_media_k1), o motivo. O geral usa só as tarefas com P_rep válido; as duas = regra de antes; uma só = veredito
    dela + " (só na tarefa X)"; nenhuma = MSG_PREP. P_rep não avaliável em alguma tarefa (falta/elegibilidade) conta
    como "não avaliável por outro motivo" e tem precedência (leitura nossa). Dois "inconclusivo" ou um
    "inconclusivo" com "atendida" = "parcial"; um "inconclusivo" sozinho (uma só tarefa válida) = "parcial (só na
    tarefa X)"; o geral nunca é "inconclusivo". K = 25 no lugar de 20 em tudo.
 7. Braço de épocas fixas: contraste d_K = gap(particao_epocas_kK) − gap(copias_kK) para K = 5, 25 (e 10 com o
    `fedavg_iid`, que é igual a particao_epocas_k10 por igualdade); K = 2 também. Sem veredito.
 8. Previsão: "H_cópias: não" = "não atendida" (leitura 2 do bloco de 2026-10-26); "H_K: sim" = H_K como registrada
    (maior K = 25); o julgamento usa o veredito-base, sem o rótulo "só na tarefa X", mostrado ao lado.
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
KS = (5, 10, 25)
CALIB_FEDAVG = dict(rodadas=75, ok=True, erro_rel=0.0)
MSG_PREP = "não avaliável (a vantagem do FedAvg não se reproduziu)"
SEM_FAIXA = "inconclusivo (sem faixa dinâmica)"
CONDICOES = ("fedavg_iid", "particao_iid_k2", "particao_iid_k5", "particao_iid_k25", "copias_k2", "copias_k5",
             "copias_k10", "copias_k25", "sem_media_k1", "particao_epocas_k2", "particao_epocas_k5",
             "particao_epocas_k25")
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

def p_rep(df, calib=None):
    """P_rep POR TAREFA (decisão do autor, 2026-10-26): em cada tarefa t, o inf do IC pareado de
    gap(fedavg_iid) − gap(sem_media_k1) > +δ, com a elegibilidade do sem_media_k1 (calibração ok nas duas tarefas e
    perda a até 10%) e as 8 sementes completas. por[t]['ok'] é True / False / None (não avaliável em t) e
    por[t]['msg'] é '' se True, MSG_PREP se False, ou o motivo ('não avaliável (...)') se None.
    'validas' = tarefas com P_rep True; 'ok' (global, só informativo) = valem nas duas; 'msg' = resumo textual."""
    por = {}
    for t in TIPOS:
        m = comparavel(df, calib, ("fedavg_iid", "sem_media_k1"), t) if calib is not None else None
        if m:
            por[t] = dict(c=None, ok=None, msg=m)
            continue
        c = contraste(df, "fedavg_iid", "sem_media_k1", t)
        if c["falta"]:
            por[t] = dict(c=c, ok=None, msg=f"não avaliável (falta {c['falta']})")
        else:
            ok = bool(c["inf"] > DELTA)
            por[t] = dict(c=c, ok=ok, msg="" if ok else MSG_PREP)
    validas = [t for t in TIPOS if por[t]["ok"] is True]
    ok = len(validas) == len(TIPOS)
    msg = "" if ok else "; ".join(f"{t}: {por[t]['msg']}" for t in TIPOS if por[t]["ok"] is not True)
    return dict(por=por, ok=ok, validas=validas, msg=msg)


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
    """atendida / parcial / não atendida / não avaliável (K = 5, 10, 25)."""
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
    """Médias por K (2, 5, 10, 25), ordem 25>10>5 e contraste gap(25) − gap(5) (sem veredito)."""
    medias = {k: resumo(df, familia(k), tipo) for k in (2,) + KS}
    c = contraste(df, familia(25), familia(5), tipo)
    m = [medias[k]["media"] for k in (25, 10, 5)]
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
    """Veredito geral a partir dos vereditos por tarefa (lista, uma por tarefa considerada)."""
    nav = [v for v in vers if v.startswith("não avaliável")]
    if nav:
        return nav[0] if len(set(vers)) == 1 else "não avaliável (" + "; ".join(
            f"{t}: {v}" for t, v in zip(TIPOS, vers) if v.startswith("não avaliável")) + ")"
    if all(v == "atendida" for v in vers):
        return "atendida"
    if all(v == "não atendida" for v in vers):
        return "não atendida"
    return "parcial"


def veredito_geral(pr, por):
    """(base, rótulo, vers): base = veredito geral sem rótulo; rótulo = '' ou ' (só na tarefa X)'; vers = veredito
    por tarefa (MSG_PREP onde P_rep falha; o motivo onde P_rep é não avaliável; senão o da hipótese).
    Leitura do registrador (2026-10-26): o geral usa só as tarefas com P_rep válido; 'não avaliável' por outro
    motivo (inclusive P_rep não avaliável em alguma tarefa, falta/elegibilidade) tem precedência; uma só tarefa
    válida dá o veredito dela com o rótulo ('inconclusivo' sozinho vira 'parcial'); nenhuma = MSG_PREP."""
    vers = {t: pr["por"][t]["msg"] if pr["por"][t]["ok"] is not True else por[t]["veredito"] for t in TIPOS}
    outros = [t for t in TIPOS if pr["por"][t]["ok"] is None
              or (pr["por"][t]["ok"] and vers[t].startswith("não avaliável"))]
    if outros:
        iguais = len(outros) == len(TIPOS) and len({vers[t] for t in TIPOS}) == 1
        return (vers[outros[0]] if iguais else _nav_parcial(outros, vers)), "", vers
    validas = [t for t in TIPOS if pr["por"][t]["ok"]]
    if not validas:
        return MSG_PREP, "", vers
    if len(validas) == len(TIPOS):
        return geral([vers[t] for t in TIPOS]), "", vers
    t = validas[0]
    v = vers[t] if vers[t] in ("atendida", "não atendida") else "parcial"
    return v, f" (só na tarefa {t})", vers


def _nav_parcial(outros, vers):
    """'não avaliável' em parte das tarefas: lista só as tarefas afetadas."""
    return "não avaliável (" + "; ".join(f"{t}: {vers[t]}" for t in outros) + ")"


def decidir(df, calib):
    """Tudo: P_rep por tarefa, hipóteses por tarefa e veredito geral (base, rótulo e texto final)."""
    pr = p_rep(df, calib)
    hip = {"H_dados": {t: h_dados(df, calib, t) for t in TIPOS},
           "H_cópias": {t: h_copias(df, calib, t) for t in TIPOS},
           "H_K": {t: h_k(df, calib, t) for t in TIPOS}}
    res = {}
    for nome, por in hip.items():
        base, rotulo, vers = veredito_geral(pr, por)
        res[nome] = dict(por=por, vers=vers, base=base, rotulo=rotulo, geral=base + rotulo)
    return dict(p_rep=pr, hip=res)


def julgar_previsao(dec):
    """Itens da previsão do autor × veredito-base (sem o rótulo 'só na tarefa X', que vai ao lado; regra do
    pré-registro; não avaliável não conta). H_K refere-se ao maior K (25)."""
    out = []
    for nome, prev in PREVISAO:
        h = dec["hip"][nome]
        g = h["base"]
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
        out.append(dict(item=nome, previsao=prev, calculado=h["geral"], base=g, rotulo=h["rotulo"].strip(),
                        julgamento=j))
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
    """{(familia, K): {medida: resumo}} para K = 2, 5, 10, 25, mais ('sem_media_k1', 1)."""
    out = {}
    for fam, f in FAMILIAS.items():
        for k in (2,) + KS:
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
