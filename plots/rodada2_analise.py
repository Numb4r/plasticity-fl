"""Biblioteca de análise da RODADA 2 (PREREGISTRO.md: "Rodada 2 · Qual média protege a plasticidade, e por quê?"
e o bloco datado de 2026-10-21 "Rodada 2: hash da previsão, aprovações e ajustes do autor").

Não altera nada da rodada 1: só importa `analise.bootstrap_ic` (bootstrap percentil, 10 000, default_rng(0)).
CLI: `python plots/rodada2_analise.py --results results_rodada2` imprime os vereditos (sem figuras).

AMBIGUIDADES do PREREGISTRO.md e como foram codificadas (texto literal onde existe; o resto está aqui).
A condição de H_tempo e H_dist é `central_HT_cal` (bloco de 2026-10-21 "esclarecimento de desenho"); `central_HT`
(15 passos, sem calibração) e `central_lookahead` são exploratórios, sem veredito:
 1. Elegibilidade do central_HT_cal (perda): L = média entre sementes das médias por semente de `loss_nova` nas
    tarefas de referência (gap definido) da janela principal; |L_HT_cal-L_msp|/L_msp <= 0,10 (fechado), por tipo
    de tarefa (A e B); falhar em qualquer tipo torna H_tempo/H_dist "não avaliável". Acurácia +-0,03: só reportada.
    Segundo filtro: vale depois da calibração (calibracao_HT_{A,B}_m3.json, ok=false = "calibração não atingida").
 2. Ordem das condições de "não avaliável": falta de dado (CSV do central_HT_cal, de media_sem_particao e de
    sem_media_k1, ou o JSON de calibração) > calibração não atingida > P_vant e elegibilidade. Todos os motivos
    que se aplicam são listados; com calibração não atingida e nada mais, o veredito é exatamente
    "não avaliável (calibração não atingida)" para H_tempo e para H_dist (são o mesmo veredito).
 3. Equivalência: "IC inteiro em [-d,+d]" = inf >= -d e sup <= +d (fechado); "HT acima" = inf > +d (estrito);
    H_dist = sup < -d (estrito). Por tarefa: dentro / acima / abaixo / intermediário; geral: os dois iguais ->
    esse veredito ("intermediário" quando as duas são intermediárias); diferentes -> "misto".
 4. "Falta dado": toda comparação exige, por (condição, tarefa), TODAS as sementes esperadas (6 a 10; 6 a 8 no
    braço longo) com as 5 tarefas da janela principal com gap; senão "não avaliável (falta X)". As tabelas
    descritivas mostram o n e não exigem completude.
 5. Janela principal: T=30 -> tarefas 25..29; T=100 -> 95..99; só tarefas com gap definido (referências).
 6. H_het, veredito geral: se uma das tarefas for "não avaliável (falta ...)" o geral é "não avaliável"; "parcial"
    = exatamente uma tarefa atendida (a outra pode ser inconclusiva, reversa ou não atendida).
 7. H_het (3) "oposta": gap(100) < gap(1) < gap(0,1) estrito e sup do IC pareado de gap(100)-gap(0,1) < 0.
    Qualquer outro caso = "não atendida" (inclui a ordem prevista ou a oposta com o IC pareado passando por 0).
 8. Rank (exploratório): o grupo "com média" tem 4 condições (fedavg_iid, media_sem_particao, central_HT_cal e
    fedavg_dirichlet, esta UMA vez, pooled nos 3 alfas); central_HT e central_lookahead aparecem à parte, FORA do
    grupo; valor de grupo por semente = média das condições do grupo presentes naquela semente. Rank vale para as
    tarefas A e B, 5 últimas tarefas (todas, não só referências).
 9. Sonda: a mediana de dlog_norm é recalculada dos 12 Lc/Ln; dlog_raw vem da coluna sonda2_dlog_raw (o CSV
    guarda a estatística exploratória pronta). Média por semente nas tarefas com sonda (colunas não-NaN).
10. Reaprendizado: mediana POOLED sobre (semente, tarefa_atual, tarefa_antiga); economia = 1 - mediana(cópia)
    / mediana(nova) (razão de medianas; se a nova é censurada e a cópia não, só um limite inferior; se a cópia
    é censurada, n/d). '> 1500' quando MAIS da metade é censurada (estrito). Fração "no limiar no passo 0" =
    passos == 0.
11. Colapso: média móvel de 3 referências (t%3==2) consecutivas <= -0,10 (com tolerância 1e-12); exige as 33
    referências da semente (senão "falta"). Mediana [mín, máx] só entre as sementes que colapsaram (DESCRITIVA);
    fração sem colapso entre todas.
12. Julgamento do horizonte longo (tarefa A), regra confirmada pelo autor: uma condição "colapsa" se pelo menos
    uma semente colapsa; "impede" exige que sem_media_k1 e central_aprendizado colapsem nesse sentido e que nenhuma
    semente do fedavg_iid colapse; "só atrasa" compara a mediana (entre sementes que colapsaram) do fedavg_iid com
    as das duas; se uma delas não colapsou, "outro". (`julgar_horizonte`.) A versão PROPOSTA (censura à direita,
    mediana nas 3 sementes) está no item 15.
13. Mesma semente em duas condições -> diferença pareada por semente na interseção das sementes.
14. Critério secundário literal: média do central_HT_cal dentro do IC de media_sem_particao (fechado) e
    inf(IC msp) > sup(IC HT_cal) (estrito); reportado, sem veredito. Exploratórios, sem veredito: (a) central_HT
    (15 passos) contra media_sem_particao; (b) central_HT - sem_media_k1 por tarefa e semente (a rede nova é a
    mesma, então a diferença de gap é a de acc_cont); (c) central_lookahead contra media_sem_particao.
15. PROPOSTA, aguarda confirmação do autor (`julgar_previsao`): (1) H_tempo/H_dist é um só item; a previsão acerta
    só se o veredito geral for "HT acima"; "misto" com "HT acima" em uma tarefa = acerto parcial; "não avaliável"
    não conta. (2) H_het: "atendida" = acerto (confirmado), "parcial" = acerto parcial (proposta), qualquer outro
    caso (inclui ordem certa com IC passando por 0) = erro. (3) Horizonte: "sem colapso" = censura à direita
    (ordenado como o último, > T); mediana nas 3 sementes de cada condição; empate não é "mais tarde".
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
import analise  # noqa: E402  (só bootstrap_ic; a rodada 1 não é alterada)

SEMENTES = (6, 7, 8, 9, 10)
SEMENTES_LONGO = (6, 7, 8)
T_CURTO, T_LONGO, N_JANELA = 30, 100, 5
DELTA, TOL_CE, TOL_ACC = 0.02, 0.10, 0.03
LIM_HET, LIM_COLAPSO, TETO = 0.01, -0.10, 1500
ALFAS = (0.1, 1.0, 100.0)
TIPOS_AB = ("A", "B")
COND_REF = ("fedavg_iid", "media_sem_particao", "sem_media_k1", "central_aprendizado")
COND_LONGO = COND_REF + ("central_HT",)
HT, HTCAL = "central_HT", "central_HT_cal"   # HTCAL decide H_tempo/H_dist; HT é exploratório
COND_T30 = COND_REF + ("central_HT", "central_HT_cal", "central_lookahead")
COND_BFIXA = COND_REF + ("central_HT",)
GRUPO_COM = ("fedavg_iid", "media_sem_particao", "central_HT_cal", "fedavg_dirichlet")
FORA_GRUPO = ("central_HT", "central_lookahead")   # rank: mostrados à parte, fora dos dois grupos
PREVISAO_LITERAL = "H_tempo não, H_dist não, H_het sim, FedAvg só atrasa"
GRUPO_SEM = ("sem_media_k1", "central_aprendizado")
COLUNAS_CHAVE = {"tarefa", "tipo", "cond", "semente"}
SUFIXOS_IGNORADOS = ("_matriz.csv", "_reaprendizado.csv", "_rodadas.csv")
SONDA_COLS = [f"sonda2_{p}_{k}" for p in ("Lc0", "Lc20", "Ln0", "Ln20") for k in range(3)]


def serie_nome(cond, alfa=None):
    return f"fedavg_dirichlet_a{alfa:g}" if cond == "fedavg_dirichlet" else cond


def rotulo(serie):
    m = re.match(r"fedavg_dirichlet_a(.+)$", serie)
    return f"fedavg_dirichlet (α={m.group(1)})" if m else serie


# ---------------------------------------------------------------- carga

def carregar(results="results_rodada2"):
    """Lê só results/*.csv (raiz, não recursivo; calibracao/ nunca entra). Mantém sementes 6..10 (T100: 6..8) e
    avisa as descartadas (ex.: 1000). Acrescenta `braco` (T30/T100, do nome do arquivo) e `serie`."""
    dfs = []
    for a in sorted(glob.glob(os.path.join(results, "*.csv"))):
        if a.endswith(SUFIXOS_IGNORADOS):
            continue
        d = pd.read_csv(a)
        if not COLUNAS_CHAVE <= set(d.columns):
            print(f"aviso: {a} sem as colunas {sorted(COLUNAS_CHAVE)}; ignorado", file=sys.stderr)
            continue
        nome = os.path.basename(a)
        d["braco"] = "T100" if "_T100_m3_" in nome else "T30"
        if "alfa" not in d.columns:
            d["alfa"] = np.nan
        m = re.search(r"_a([0-9.]+)_m3_", nome)
        if m and d["alfa"].isna().all():
            d["alfa"] = float(m.group(1))
        dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    fora = sorted(set(df["semente"].unique()) - set(SEMENTES))
    if fora:
        warnings.warn("sementes descartadas (fora de 6..10): " + ", ".join(str(x) for x in fora), stacklevel=2)
    df = df[df["semente"].isin(SEMENTES)]
    longo_fora = df[(df["braco"] == "T100") & ~df["semente"].isin(SEMENTES_LONGO)]
    if len(longo_fora):
        warnings.warn("braço T100: sementes fora de 6..8 descartadas: "
                      + ", ".join(str(x) for x in sorted(longo_fora["semente"].unique())), stacklevel=2)
        df = df.drop(longo_fora.index)
    df = df.reset_index(drop=True)
    alfa = df["alfa"].map(lambda a: f"{a:g}")
    df["serie"] = np.where(df["cond"] == "fedavg_dirichlet", "fedavg_dirichlet_a" + alfa, df["cond"])
    return df


def carregar_reaprendizado(results="results_rodada2"):
    """<cond>_m3_B_fixa_<semente>_reaprendizado.csv (raiz) -> DataFrame com cond e semente (6..10)."""
    dfs = []
    for a in sorted(glob.glob(os.path.join(results, "*_reaprendizado.csv"))):
        m = re.match(r"^(.+)_m3_B_fixa_(\d+)_reaprendizado\.csv$", os.path.basename(a))
        if not m or int(m.group(2)) not in SEMENTES:
            if m:
                warnings.warn(f"reaprendizado de semente {m.group(2)} descartado", stacklevel=2)
            continue
        d = pd.read_csv(a)
        d["cond"], d["semente"] = m.group(1), int(m.group(2))
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ---------------------------------------------------------------- métrica principal

def janela(braco="T30"):
    T = T_LONGO if braco == "T100" else T_CURTO
    return T - N_JANELA, T - 1


def sementes_esperadas(braco="T30"):
    return SEMENTES_LONGO if braco == "T100" else SEMENTES


def _sub(df, serie, tipo, braco="T30"):
    if df.empty:
        return df
    return df[(df["serie"] == serie) & (df["tipo"] == tipo) & (df["braco"] == braco)]


def _janela(df, serie, tipo, braco="T30"):
    d = _sub(df, serie, tipo, braco)
    lo, hi = janela(braco)
    return d[(d["tarefa"] >= lo) & (d["tarefa"] <= hi)]


def valores_semente(df, serie, tipo, coluna="gap", braco="T30", so_ref=True):
    """Média por semente de `coluna` nas tarefas da janela principal (so_ref: só as com gap definido)."""
    d = _janela(df, serie, tipo, braco)
    if d.empty or coluna not in d.columns:
        return pd.Series(dtype=float)
    if so_ref:
        d = d.dropna(subset=["gap"])
    return d.dropna(subset=[coluna]).groupby("semente")[coluna].mean()


def faltas(df, serie, tipo, braco="T30"):
    """Sementes esperadas sem as 5 tarefas da janela com gap (lista de inteiros)."""
    d = _janela(df, serie, tipo, braco)
    n = d.dropna(subset=["gap"]).groupby("semente").size() if len(d) else pd.Series(dtype=int)
    return [s for s in sementes_esperadas(braco) if n.get(s, 0) < N_JANELA]


def texto_falta(df, itens, braco="T30"):
    """itens [(serie, tipo)] -> None se tudo completo, senão 'falta serie/tipo (sementes ...)'."""
    partes = []
    for serie, tipo in itens:
        f = faltas(df, serie, tipo, braco)
        if f:
            partes.append(f"{serie}/{tipo} sementes {','.join(str(s) for s in f)}")
    return "falta " + "; ".join(partes) if partes else None


def resumir(v):
    """dict media/inf/sup/n; v é Series/array por semente."""
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    m, lo, hi = analise.bootstrap_ic(v)
    return dict(media=m, inf=lo, sup=hi, n=len(v))


def resumo(df, serie, tipo, coluna="gap", braco="T30", so_ref=True):
    return resumir(valores_semente(df, serie, tipo, coluna, braco, so_ref).values)


def dif_pareada(df, serie_a, serie_b, tipo, coluna="gap", braco="T30"):
    """IC da média de d_s = a_s - b_s sobre as sementes comuns."""
    a, b = valores_semente(df, serie_a, tipo, coluna, braco), valores_semente(df, serie_b, tipo, coluna, braco)
    return resumir((a - b).dropna().values)


def curva(df, serie, tipo, coluna="gap", braco="T30", so_ref=False):
    """Por tarefa: média e IC entre sementes (pontos com valor definido)."""
    cols = ["tarefa", "media", "inf", "sup", "n"]
    d = _sub(df, serie, tipo, braco)
    if d.empty or coluna not in d.columns:
        return pd.DataFrame(columns=cols)
    d = d.dropna(subset=[coluna])
    linhas = []
    for t, g in d.groupby("tarefa"):
        m, lo, hi = analise.bootstrap_ic(g[coluna].values)
        linhas.append(dict(tarefa=t, media=m, inf=lo, sup=hi, n=len(g)))
    return pd.DataFrame(linhas, columns=cols)


def _ok_ic(r):
    return not (np.isnan(r["inf"]) or np.isnan(r["sup"]))


# ---------------------------------------------------------------- H_tempo / H_dist

def classe_d(r, delta=DELTA):
    """Classe do IC da diferença d: dentro / acima / abaixo / intermediário (ambiguidade 3)."""
    if not _ok_ic(r):
        return "indefinido"
    if r["inf"] >= -delta and r["sup"] <= delta:
        return "dentro"
    if r["inf"] > delta:
        return "acima"
    if r["sup"] < -delta:
        return "abaixo"
    return "intermediário"


def carregar_calibracao(results="results_rodada2"):
    """Lê <results>/calibracao/{calibracao_HT,selecao_HTcal,selecao_HT,selecao_lookahead}_<A|B>_m3.json.
    Devolve {tipo: {"calibracao": dict|None, "sel_HTcal": ..., "sel_HT": ..., "sel_lookahead": ...}}; arquivo
    ausente ou ilegível = None (as CSVs de resultados nunca são lidos daqui)."""
    nomes = {"calibracao": "calibracao_HT_{t}_m3.json", "sel_HTcal": "selecao_HTcal_{t}_m3.json",
             "sel_HT": "selecao_HT_{t}_m3.json", "sel_lookahead": "selecao_lookahead_{t}_m3.json"}
    out = {}
    for t in TIPOS_AB:
        out[t] = {}
        for chave, modelo in nomes.items():
            arq = os.path.join(results, "calibracao", modelo.format(t=t))
            try:
                with open(arq, encoding="utf-8") as f:
                    out[t][chave] = json.load(f)
            except (OSError, ValueError):
                out[t][chave] = None
    return out


def _checar_calibracao(calib):
    """-> (arquivos ausentes/sem o campo `ok`, tipos com ok=false). calib None = checagem não pedida."""
    ausentes, falhas = [], []
    for t in TIPOS_AB:
        c = (calib or {}).get(t, {}).get("calibracao")
        if not isinstance(c, dict) or not isinstance(c.get("ok"), bool):
            ausentes.append(f"calibracao_HT_{t}_m3.json")
        elif not c["ok"]:
            falhas.append(t)
    return ausentes, falhas


def _expl_ht(df, a, b, t):
    """Exploratório: d = a - b pareado (gap) com IC, d por semente e a mesma diferença em acc_cont."""
    va, vb = valores_semente(df, a, t), valores_semente(df, b, t)
    return dict(d=dif_pareada(df, a, b, t), d_acc_cont=dif_pareada(df, a, b, t, "acc_cont"),
                por_semente=(va - vb).dropna(), falta=texto_falta(df, [(a, t), (b, t)]))


def h_tempo(df, calib=None):
    """H_tempo / 'HT acima' / H_dist / intermediário / misto / 'não avaliável (motivos)' + detalhes por tarefa.
    A condição é central_HT_cal. `calib` = saída de carregar_calibracao (None = a checagem de calibração não é
    feita; o CLI e a tabela sempre a passam). Exploratórios (sem veredito): expl_ht_msp, expl_ht_k1, lookahead."""
    out = dict(tarefas={}, lookahead={}, expl_ht_msp={}, expl_ht_k1={}, calibracao=calib, veredito=None,
               motivo=None, motivos=[], calib_checada=calib is not None, calib_falha=[])
    itens = [(c, t) for c in (HTCAL, "media_sem_particao", "sem_media_k1") for t in TIPOS_AB]
    falta = texto_falta(df, itens)
    for t in TIPOS_AB:
        ht, msp, k1 = (resumo(df, c, t) for c in (HTCAL, "media_sem_particao", "sem_media_k1"))
        lh = {c: float(valores_semente(df, c, t, "loss_nova").mean()) if len(valores_semente(
            df, c, t, "loss_nova")) else np.nan for c in (HTCAL, "media_sem_particao")}
        rel = abs(lh[HTCAL] - lh["media_sem_particao"]) / lh["media_sem_particao"] \
            if lh["media_sem_particao"] and not np.isnan(lh["media_sem_particao"]) else np.nan
        ah = valores_semente(df, HTCAL, t, "acc_nova")
        am = valores_semente(df, "media_sem_particao", t, "acc_nova")
        dacc = float(ah.mean() - am.mean()) if len(ah) and len(am) else np.nan
        d = dif_pareada(df, HTCAL, "media_sem_particao", t)
        sec_a = bool(msp["inf"] <= ht["media"] <= msp["sup"]) if _ok_ic(msp) and not np.isnan(ht["media"]) else None
        sec_b = bool(msp["inf"] > ht["sup"]) if _ok_ic(msp) and _ok_ic(ht) else None
        out["tarefas"][t] = dict(
            ht=ht, msp=msp, k1=k1, d=d, classe=classe_d(d),
            p_vant=bool(msp["inf"] > k1["sup"]) if _ok_ic(msp) and _ok_ic(k1) else None,
            loss_ht=lh[HTCAL], loss_msp=lh["media_sem_particao"], rel_ce=rel,
            ce_ok=bool(rel <= TOL_CE) if not np.isnan(rel) else None,
            dacc=dacc, acc_ok=bool(abs(dacc) <= TOL_ACC) if not np.isnan(dacc) else None,
            sec_dentro=sec_a, sec_acima=sec_b)
        out["lookahead"][t] = _expl_ht(df, "central_lookahead", "media_sem_particao", t)
        out["expl_ht_msp"][t] = _expl_ht(df, HT, "media_sem_particao", t)
        out["expl_ht_k1"][t] = _expl_ht(df, HT, "sem_media_k1", t)
    # motivos de "não avaliável", na ordem: falta de dado > calibração não atingida > P_vant / elegibilidade
    partes_falta = [falta[len("falta "):]] if falta else []
    if calib is not None:
        ausentes, falhas = _checar_calibracao(calib)
        partes_falta += ausentes
        out["calib_falha"] = falhas
    motivos = ["falta " + "; ".join(partes_falta)] if partes_falta else []
    if out["calib_falha"]:
        motivos.append("calibração não atingida")
    tar = out["tarefas"]
    if not falta:   # P_vant e elegibilidade só são calculáveis com os dados completos
        pv = [t for t in TIPOS_AB if tar[t]["p_vant"] is not True]
        if pv:
            motivos.append("P_vant falhou em " + ",".join(pv))
        ce = [t for t in TIPOS_AB if tar[t]["ce_ok"] is not True]
        if ce:
            motivos.append("nível de aprendizado diferente (perda de treino da rede nova do central_HT_cal fora de "
                           "10% de media_sem_particao) em " + ",".join(ce))
    if motivos:
        out["motivos"] = motivos
        out["motivo"] = "; ".join(motivos)
        out["veredito"] = f"não avaliável ({out['motivo']})"
        return out
    cl = [tar[t]["classe"] for t in TIPOS_AB]
    if "indefinido" in cl:
        out["motivo"] = "IC da diferença indefinido (n<2)"
        out["veredito"] = f"não avaliável ({out['motivo']})"
    elif cl[0] == cl[1]:
        out["veredito"] = {"dentro": "H_tempo", "acima": "HT acima", "abaixo": "H_dist"}.get(cl[0], "intermediário")
    else:
        out["veredito"] = f"misto (A: {cl[0]}, B: {cl[1]})"
    return out


# ---------------------------------------------------------------- H_het

def h_het(df):
    """Por tarefa A e B (ordem estrita) e veredito geral atendida / parcial / não atendida."""
    out = dict(tarefas={}, geral=None)
    for t in TIPOS_AB:
        s = {a: serie_nome("fedavg_dirichlet", a) for a in ALFAS}
        r = dict(gap={a: resumo(df, s[a], t) for a in ALFAS},
                 acc_cont={a: resumo(df, s[a], t, "acc_cont") for a in ALFAS},
                 acc_nova={a: resumo(df, s[a], t, "acc_nova") for a in ALFAS})
        falta = texto_falta(df, [(s[a], t) for a in ALFAS])
        r["d"] = dif_pareada(df, s[100.0], s[0.1], t)
        if falta:
            r["veredito"] = f"não avaliável ({falta})"
            r["ok"] = None
        else:
            m = {a: r["gap"][a]["media"] for a in ALFAS}
            d = r["d"]
            if all(abs(m[a]) < LIM_HET for a in ALFAS):
                r["veredito"], r["ok"] = "inconclusivo (sem faixa dinâmica)", False
            elif m[100.0] > m[1.0] > m[0.1] and _ok_ic(d) and d["inf"] > 0:
                r["veredito"], r["ok"] = "atendida", True
            elif m[100.0] < m[1.0] < m[0.1] and _ok_ic(d) and d["sup"] < 0:
                r["veredito"], r["ok"] = "H_het reversa", False
            else:
                r["veredito"], r["ok"] = "não atendida", False
        out["tarefas"][t] = r
    oks = [out["tarefas"][t]["ok"] for t in TIPOS_AB]
    if None in oks:
        motivos = "; ".join(f"{t}: " + out["tarefas"][t]["veredito"].removeprefix("não avaliável (").rstrip(")")
                            for t in TIPOS_AB if out["tarefas"][t]["ok"] is None)
        out["geral"] = f"não avaliável ({motivos})"
    else:
        n = sum(oks)
        out["geral"] = {2: "atendida", 1: "parcial", 0: "não atendida"}[n]
    return out


# ---------------------------------------------------------------- rank (exploratório, sem veredito)

def rank_tabela(df):
    """Linhas: (nome, tipo, resumo rank_ef, resumo srank_feat), por série e por grupo ('com média'/'sem média').
    Média das 5 últimas tarefas (todas). Sempre juntos."""
    linhas = []
    if df.empty or not {"rank_ef", "srank_feat"} <= set(df.columns):
        return linhas
    for t in TIPOS_AB:
        por_cond = {}
        for serie in sorted(df[(df["braco"] == "T30") & (df["tipo"] == t)]["serie"].unique()):
            re_, sr = (valores_semente(df, serie, t, c, so_ref=False) for c in ("rank_ef", "srank_feat"))
            fora = " (fora do grupo)" if serie in FORA_GRUPO else ""
            linhas.append((rotulo(serie) + fora, t, resumir(re_.values), resumir(sr.values)))
            cond = "fedavg_dirichlet" if serie.startswith("fedavg_dirichlet") else serie
            por_cond.setdefault(cond, []).append((re_, sr))
        for nome, grupo in (("grupo: com média", GRUPO_COM), ("grupo: sem média", GRUPO_SEM)):
            res = []
            for k in range(2):
                cols = [pd.concat([p[k] for p in por_cond[c]], axis=1).mean(axis=1)
                        for c in grupo if c in por_cond]
                res.append(resumir(pd.concat(cols, axis=1).mean(axis=1).values) if cols else resumir([]))
            linhas.append((nome, t, res[0], res[1]))
    return linhas


# ---------------------------------------------------------------- sonda v2

def dlog_norm_linhas(d):
    """mediana_k[log10(Lc20/Lc0) - log10(Ln20/Ln0)] por linha (k=0,1,2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.stack([np.log10(d[f"sonda2_Lc20_{k}"] / d[f"sonda2_Lc0_{k}"])
                      - np.log10(d[f"sonda2_Ln20_{k}"] / d[f"sonda2_Ln0_{k}"]) for k in range(3)], axis=1)
    v = np.where(np.isfinite(v), v, np.nan)
    med = np.full(len(v), np.nan)
    ok = ~np.isnan(v).any(axis=1)
    med[ok] = np.median(v[ok], axis=1)
    return med


def sonda_tabela(df):
    """Linhas (serie, tipo, resumo dlog_norm, resumo dlog_raw): média por semente nas tarefas de sonda."""
    linhas = []
    if df.empty or not set(SONDA_COLS) <= set(df.columns):
        return linhas
    for t in TIPOS_AB:
        for serie in sorted(df[(df["braco"] == "T30") & (df["tipo"] == t)]["serie"].unique()):
            d = _sub(df, serie, t).dropna(subset=SONDA_COLS)
            if d.empty:
                continue
            d = d.assign(dlog_norm=dlog_norm_linhas(d))
            raw = d.dropna(subset=["sonda2_dlog_raw"]).groupby("semente")["sonda2_dlog_raw"].mean() \
                if "sonda2_dlog_raw" in d.columns else pd.Series(dtype=float)
            nrm = d.dropna(subset=["dlog_norm"]).groupby("semente")["dlog_norm"].mean()
            linhas.append((rotulo(serie), t, resumir(nrm.values), resumir(raw.values)))
    return linhas


# ---------------------------------------------------------------- B_fixa

def bfixa_tabela(df):
    """Por condição: gap (métrica principal), esquecimento e BWT na última tarefa (T-1), por semente."""
    linhas = []
    for c in COND_BFIXA:
        d = _sub(df, c, "B_fixa")
        if d.empty:
            continue
        ult = d[d["tarefa"] == T_CURTO - 1]
        esq = ult.dropna(subset=["esquecimento"]).groupby("semente")["esquecimento"].mean() \
            if "esquecimento" in ult.columns else pd.Series(dtype=float)
        bwt = ult.dropna(subset=["bwt"]).groupby("semente")["bwt"].mean() if "bwt" in ult.columns \
            else pd.Series(dtype=float)
        linhas.append((c, resumo(df, c, "B_fixa"), resumir(esq.values), resumir(bwt.values)))
    return linhas


def _med_texto(x, cens):
    return f"> {TETO}" if cens > 0.5 else f"{x:g}"


def reaprendizado_tabela(rea):
    """Por condição: medianas pooled, fração censurada, economia e fração no limiar no passo 0."""
    linhas = []
    if rea.empty:
        return linhas
    for c in [c for c in COND_BFIXA if c in set(rea["cond"])] + sorted(set(rea["cond"]) - set(COND_BFIXA)):
        g = rea[rea["cond"] == c]
        mc, mn = float(g["passos_copia"].median()), float(g["passos_nova"].median())
        cc, cn = float(g["censurado_copia"].astype(float).mean()), float(g["censurado_nova"].astype(float).mean())
        if cc > 0.5:
            eco = "n/d (cópia censurada)"
        elif cn > 0.5:
            eco = f"≥ {1 - mc / TETO:.2f} (nova censurada)"
        elif mn > 0:
            eco = f"{1 - mc / mn:.2f}"
        else:
            eco = "n/d (mediana da nova = 0)"
        por_sem = g.groupby("semente")[["passos_copia", "passos_nova"]].median()
        linhas.append(dict(cond=c, n=len(g), med_copia=mc, med_nova=mn, cens_copia=cc, cens_nova=cn,
                           txt_copia=_med_texto(mc, cc), txt_nova=_med_texto(mn, cn), economia=eco,
                           limiar0_copia=float((g["passos_copia"] == 0).mean()),
                           limiar0_nova=float((g["passos_nova"] == 0).mean()), por_semente=por_sem))
    return linhas


# ---------------------------------------------------------------- braço longo

def indice_colapso(ts, gaps):
    """Primeiro ponto (entre referências) cuja média móvel de 3 consecutivas <= -0,10; índice da última do
    trio; None se nunca."""
    for i in range(2, len(gaps)):
        if np.mean(gaps[i - 2:i + 1]) <= LIM_COLAPSO + 1e-12:
            return int(ts[i])
    return None


def colapso_tabela(df):
    """{(cond, tipo): dict(falta, por_semente {s: idx|None}, mediana, minimo, maximo, frac_sem)}."""
    out = {}
    n_ref = len(range(2, T_LONGO, 3))
    for t in TIPOS_AB:
        for c in COND_LONGO:
            d = _sub(df, c, t, "T100")
            d = d[(d["tarefa"] % 3 == 2)].dropna(subset=["gap"]) if len(d) else d
            falta, por = [], {}
            for s in SEMENTES_LONGO:
                g = d[d["semente"] == s].sort_values("tarefa") if len(d) else d
                if len(g) < n_ref:
                    falta.append(s)
                else:
                    por[s] = indice_colapso(g["tarefa"].values, g["gap"].values)
            out[(c, t)] = stats_colapso(por, falta)
    return out


def stats_colapso(por, falta=()):
    """por {semente: índice|None} -> dict. `mediana`/`minimo`/`maximo`: só entre as sementes que colapsaram
    (descritivas). `mediana_cens`: mediana nas sementes com 'sem colapso' = +inf (censura à direita, > T), usada
    só no julgamento PROPOSTO da previsão (None se faltar semente)."""
    col = [v for v in por.values() if v is not None]
    cens = [np.inf if v is None else float(v) for v in por.values()]
    return dict(falta=list(falta), por_semente=por,
                mediana=float(np.median(col)) if col else None, minimo=min(col) if col else None,
                maximo=max(col) if col else None,
                frac_sem=(len(por) - len(col)) / len(por) if por else None,
                mediana_cens=float(np.median(cens)) if cens and not falta else None)


def julgar_horizonte(col):
    """Julgamento da previsão do autor na tarefa A (ambiguidade 12)."""
    f, k, ca = (col[(c, "A")] for c in ("fedavg_iid", "sem_media_k1", "central_aprendizado"))
    falt = [f"{c}/A sementes {','.join(map(str, x['falta']))}" for c, x in
            (("fedavg_iid", f), ("sem_media_k1", k), ("central_aprendizado", ca)) if x["falta"]]
    if falt:
        return "não avaliável (falta " + "; ".join(falt) + ")"
    k_col, ca_col, f_col = k["mediana"] is not None, ca["mediana"] is not None, f["mediana"] is not None
    if not f_col and k_col and ca_col:
        return "impede a perda"
    if f_col and k_col and ca_col and f["mediana"] > k["mediana"] and f["mediana"] > ca["mediana"]:
        return "só atrasa a perda"
    desc = (f"fedavg_iid {'colapsa' if f_col else 'não colapsa'} (mediana {f['mediana']}); "
            f"sem_media_k1 {'colapsa' if k_col else 'não colapsa'} (mediana {k['mediana']}); "
            f"central_aprendizado {'colapsa' if ca_col else 'não colapsa'} (mediana {ca['mediana']})")
    return f"outro ({desc})"


def julgar_horizonte_proposta(col):
    """PROPOSTA (aguarda confirmação do autor), tarefa A: igual a `julgar_horizonte`, mas a mediana de cada condição
    é calculada nas 3 sementes com 'sem colapso' = censura à direita (+inf, ordenado como o último, > T); 'mais tarde'
    é estrito (empate não é mais tarde). 'só atrasa': o fedavg_iid colapsa em pelo menos uma semente e a mediana dele
    é maior que as de sem_media_k1 e central_aprendizado; 'impede': nenhuma semente do fedavg_iid colapsa e as
    duas outras colapsam em pelo menos uma. A mediana só entre as que colapsaram continua descritiva."""
    f, k, ca = (col[(c, "A")] for c in ("fedavg_iid", "sem_media_k1", "central_aprendizado"))
    falt = [f"{c}/A sementes {','.join(map(str, x['falta']))}" for c, x in
            (("fedavg_iid", f), ("sem_media_k1", k), ("central_aprendizado", ca)) if x["falta"]]
    if falt:
        return "não avaliável (falta " + "; ".join(falt) + ")"
    f_col, k_col, ca_col = (x["mediana"] is not None for x in (f, k, ca))
    if not f_col and k_col and ca_col:
        return "impede a perda"
    if f_col and f["mediana_cens"] > k["mediana_cens"] and f["mediana_cens"] > ca["mediana_cens"]:
        return "só atrasa a perda"
    txt = lambda x: "sem colapso (> T)" if np.isinf(x["mediana_cens"]) else f"{x['mediana_cens']:g}"  # noqa: E731
    return (f"outro (mediana com censura: fedavg_iid {txt(f)}; sem_media_k1 {txt(k)}; "
            f"central_aprendizado {txt(ca)})")


# ---------------------------------------------------------------- julgamento da previsão (PROPOSTA)

ROTULO_PROPOSTA = "PROPOSTA — aguarda confirmação do autor"


def _julg(item, previsao, calculado, julgamento):
    return dict(item=item, previsao=previsao, calculado=calculado, julgamento=julgamento)


def julgar_previsao(ht, hh, col):
    """Julga a previsão do autor, item a item, pelas 4 regras PROPOSTAS (rotulo ROTULO_PROPOSTA). Devolve uma lista
    de dicts (item, previsao, calculado, julgamento); julgamento é um de: 'acerto', 'acerto parcial', 'erro',
    'não conta (não avaliável)'. Itens: (1) H_tempo e H_dist, um só (desfecho previsto: 'HT acima');
    (2) H_het (previsto: 'atendida'); (3) horizonte longo (previsto: 'só atrasa a perda')."""
    itens = []
    v = ht["veredito"]
    cls = [ht["tarefas"][t]["classe"] for t in TIPOS_AB]
    if v.startswith("não avaliável"):
        j = "não conta (não avaliável)"
    elif v == "HT acima":
        j = "acerto"
    elif v.startswith("misto") and "acima" in cls:
        j = "acerto parcial"
    else:
        j = "erro"
    itens.append(_julg("H_tempo e H_dist (um só item)", 'H_tempo não e H_dist não (desfecho "HT acima")', v, j))
    g = hh["geral"]
    j = {"atendida": "acerto", "parcial": "acerto parcial"}.get(g, "não conta (não avaliável)"
                                                                if g.startswith("não avaliável") else "erro")
    itens.append(_julg("H_het", "sim (atendida nas duas tarefas)", g, j))
    h = julgar_horizonte_proposta(col)
    j = "não conta (não avaliável)" if h.startswith("não avaliável") else \
        ("acerto" if h == "só atrasa a perda" else "erro")
    itens.append(_julg("horizonte longo (tarefa A)", "o FedAvg só atrasa a perda", h, j))
    return itens


# ---------------------------------------------------------------- CLI

def resumo_vereditos(df, calib=None):
    ht, hh, col = h_tempo(df, calib), h_het(df), colapso_tabela(df)
    return dict(h_tempo=ht["veredito"], h_dist=ht["veredito"], h_het=hh["geral"], horizonte=julgar_horizonte(col))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Vereditos da rodada 2 (texto)")
    p.add_argument("--results", default="results_rodada2")
    a = p.parse_args(argv)
    df = carregar(a.results) if os.path.isdir(a.results) else pd.DataFrame()
    if df.empty:
        print(f"sem resultados em {a.results}/ (nenhum CSV válido das sementes 6 a 10); nada foi calculado",
              file=sys.stderr)
        return 1
    calib = carregar_calibracao(a.results)
    for k, v in resumo_vereditos(df, calib).items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
