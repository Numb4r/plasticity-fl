"""Biblioteca de análise (esquema v3): carrega results/, calcula a métrica principal e o esquecimento por
NÍVEL de orçamento m e aplica as regras de decisão de PREREGISTRO.md ("Hipóteses e critérios", "Adendo 1"
e "Desvios v3"). Decisões e figuras antigas usam o nível m* de results/calibracao/regime.json.

AMBIGUIDADES do PREREGISTRO.md e como foram implementadas (texto literal), 27 itens:
 1. H_reg: "média ... fica dentro do IC 95% do fedavg_iid". Implementado: a média da métrica
    principal da condição central cai em [inf, sup] do IC de fedavg_iid, limites inclusivos.
 2. H_reg: o mesmo candidato (L2 ou SP) precisa passar em A e em B (Desvios, auditoria nº 2, item 1);
    janela padrão (últimas 5 tarefas). L2 e SP só existem no nível m*; é onde H_reg é avaliada.
 3. "sobrepõe": dois ICs se sobrepõem se inf1 <= sup2 e inf2 <= sup1 (fechados).
 4. "ICs sem sobreposição, fedavg acima" (pré-condição de H_part): inf(fedavg) > sup(central_aprendizado).
 5. H_real: "limite inferior do IC de fedavg_iid acima do limite superior" -> desigualdade estrita (>).
 6. Janela "últimas 5 tarefas": índices T-5..T-1 com T da própria condição (30 ou 12), T = maior
    índice de tarefa da condição no CSV (todos os níveis e tarefas juntos) + 1. Só entram tarefas com gap
    definido (t%3==2, as 5 últimas e os índices 7..11: 5 referências por janela; decisão (ii)). H_part (i)/(ii)
    usa a janela fixa 7..11 para todas as condições.
 7. Sementes cuja janela não tem nenhum gap definido são descartadas (n menor).
 8. Cada IC é um bootstrap percentil próprio (10 000 reamostragens, default_rng(0)); as reamostragens
    usam os mesmos índices de semente em todas as condições com o mesmo n (Desvios, item 5).
 9. Com n<2 sementes o IC é indefinido: toda regra que depende dele devolve "não avaliável (...)";
    isso não conta como "atendida" nem como "não atendida".
10. "Nenhuma atendida" é reportado como "nenhum". A linha final separa "sem vantagem a explicar"
    (pré-condição de H_part falhou) de "dados insuficientes" (falta dado ou IC indefinido).
11. ADENDO, texto contraditório: "marcada 'ganho com violação de esquecimento' se melhora plasticidade
    e passa o limite" diz "passa" onde o sentido pede "ultrapassa". Implementado: melhora e
    média_esq <= limite -> "ganho válido"; melhora e média_esq > limite -> "ganho com violação de
    esquecimento"; não melhora -> "sem ganho".
12. Esquecimento por semente = valor da coluna `esquecimento` na linha da tarefa T-1 (11 em
    media_sem_particao), no nível analisado. Semente sem linha ou com NaN nessa tarefa é descartada.
13. Limite: comparação literal entre MÉDIAS, média_esq(cond) <= média_esq(central_aprendizado) + 0,05,
    sem tolerância numérica. Os ICs do esquecimento só informam.
14. "Melhora plasticidade" usa a métrica principal (janela padrão) em B_disjunta no nível m*;
    inf(cond) > sup(central_aprendizado), estrita. central_aprendizado fica "sem ganho" (referência).
15. Rótulo "não avaliável" quando a métrica principal não tem IC (n<2 ou sem dados) ou quando a
    condição melhora mas falta esquecimento dela ou de central_aprendizado.
16. H_real com o Adendo: sem esquecimento de fedavg_iid ou de central_aprendizado em B_disjunta no nível m*,
    H_real é "não avaliável (falta B_disjunta)" MESMO QUE o critério original já falhe.
17. H_real com critério original válido e fedavg_iid acima do limite: veredito falso e a linha final traz
    "H_real: ganho de plasticidade com esquecimento acima do limite".
18. `carregar` lê só results/*.csv (um nível, sem subpastas): ignora *_rodadas.csv, *_matriz.csv,
    *.params.json e results/calibracao/; pula (com aviso) CSV sem tarefa/tipo/cond/semente/nivel;
    cond, tipo, semente e nivel vêm das colunas, nunca do nome do arquivo; MANTÉM só semente em
    SEMENTES_AVALIACAO (1 a 5) e avisa (warnings.warn) quais sementes foram descartadas (0, 6, 1000 etc.),
    mesmo que apareçam na raiz.
19. Nível de decisão = m_star de regime.json. m_star null: todas as decisões (H_reg, H_part, H_real,
    H_orc e rótulos de ganho) saem "não avaliável (regime sem perda; H_real, H_reg, H_part e H_orc não
    testáveis neste toy)"; as figuras antigas usam o MAIOR nível presente nos CSVs, com aviso no título.
    regime.json ausente: mesmo tratamento, com o motivo "regime.json ausente".
20. H_orc (Desvios v3): no nível m*, média da métrica principal do fedavg_iid dentro do IC 95% fechado
    da métrica principal de central_aprendizado, em A E em B. Só o IC de central_aprendizado precisa existir
    (n>=2); de fedavg_iid basta a média. Janela: últimas 5 tarefas da condição. Sem pré-condição, então
    H_orc nunca sai "sem vantagem a explicar".
21. H_part continua com a letra original no nível m*; (i) agora compara media_sem_particao com o mesmo
    orçamento por cliente (Desvios v3 a). A linha final cita o nível de cada hipótese.
22. Eixo x da dose-resposta: acc_nova média da PRÓPRIA condição nas tarefas da janela principal que têm
    gap definido, média por semente, depois bootstrap entre as mesmas sementes da métrica principal.
23. Sanidade (Desvios v3 a): sem_media_k1 e central_passos "coincidem" se os ICs da métrica principal se
    sobrepõem (fechados), por nível e tarefa; com IC indefinido o resultado é "n/d".
24. Rede nova do fedavg_iid não passa de 0,90: a linha de registro só aparece quando
    `alvo_acima_de_0.90` for falso em calibracao_A_m10.json (literal); ausente do JSON -> "n/d".
25. Marcação de "dados insuficientes" vs "sem vantagem" não muda com o nível.
26. O gap da decisão de m* (regime.json) vem do pipeline (semente 1000); aqui só é reproduzido na tabela.
27. JANELA 7..11 em H_part: o texto manda usar essa janela "para todas as condições", mas a rede nova só
    existia só em t%3==2 e nas 5 últimas tarefas (2 referências na janela em T=30). RESOLVIDA pela decisão (ii)
    da Errata (2026-10-20): `tarefas_com_nova(T)` inclui também os índices 7 a 11, então a janela tem 5
    referências pareadas em TODAS as condições (T=30: 16 tarefas; T=12: 7). Implementação inalterada
    (só entram tarefas com gap definido); a tabela só avisa se n_ref divergir entre condições.
"""
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

TIPOS_AB = ["A", "B"]                 # tarefas das hipóteses originais
TIPO_BD = "B_disjunta"                # tarefa do Adendo 1 (esquecimento)
TIPOS = ["A", "B", "B_disjunta"]      # todos os painéis das figuras
NIVEIS = [1, 3, 10]
CONDICOES = ["fedavg_iid", "sem_media_k1", "media_sem_particao",
             "central_passos", "central_aprendizado", "central_L2", "central_SP"]
CONDICOES_DOSE = ["fedavg_iid", "sem_media_k1", "media_sem_particao",
                  "central_passos", "central_aprendizado"]   # as que existem em todos os níveis
JANELA_PART = (7, 11)  # janela fixa de H_part (índices, inclusiva)
LIMITE_ESQ = 0.05      # Adendo 1: 5 pontos percentuais (fração de acurácia)
# Réplica de src.config.SEMENTES_AVALIACAO (plots/ não importa src/). Só estas sementes entram na análise;
# a 1000 é calibração e a 0 é exploratória do desenho antigo.
SEMENTES_AVALIACAO = (1, 2, 3, 4, 5)
SUFIXOS_IGNORADOS = ("_rodadas.csv", "_matriz.csv")
COLUNAS_CHAVE = {"tarefa", "tipo", "cond", "semente", "nivel"}
MSG_SEM_REGIME = ("regime sem perda; H_real, H_reg, H_part e H_orc não testáveis neste toy")

# Estilo por condição: cor + estilo de linha + marcador (legível em preto e branco)
ESTILOS = {
    "fedavg_iid":          dict(color="#000000", linestyle="-",  marker="o"),
    "sem_media_k1":        dict(color="#d55e00", linestyle="--", marker="s"),
    "media_sem_particao":  dict(color="#0072b2", linestyle="-.", marker="^"),
    "central_passos":      dict(color="#009e73", linestyle=":",  marker="D"),
    "central_aprendizado": dict(color="#cc79a7", linestyle=(0, (5, 1)), marker="v"),
    "central_L2":          dict(color="#e69f00", linestyle=(0, (3, 1, 1, 1)), marker="P"),
    "central_SP":          dict(color="#56b4e9", linestyle=(0, (1, 1)), marker="X"),
}


# ---------------------------------------------------------------- carga

def carregar(results="results"):
    """Lê só os CSVs principais de results/ (um nível, sem subpastas); mantém só semente em
    SEMENTES_AVALIACAO e emite aviso listando as descartadas."""
    arqs = sorted(glob.glob(os.path.join(results, "*.csv")))
    arqs = [a for a in arqs if not a.endswith(SUFIXOS_IGNORADOS)]
    dfs = []
    for a in arqs:
        d = pd.read_csv(a)
        if not COLUNAS_CHAVE <= set(d.columns):
            print(f"aviso: {a} não tem as colunas {sorted(COLUNAS_CHAVE)}; ignorado", file=sys.stderr)
            continue
        dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    fora = sorted(set(df["semente"].unique()) - set(SEMENTES_AVALIACAO))
    if fora:
        warnings.warn("sementes descartadas (fora de SEMENTES_AVALIACAO=1..5): "
                      + ", ".join(str(x) for x in fora), stacklevel=2)
    df = df[df["semente"].isin(SEMENTES_AVALIACAO)].reset_index(drop=True)
    df["nivel"] = df["nivel"].astype(int)
    return df


def carregar_regime(results="results"):
    """results/calibracao/regime.json -> dict(presente, m_star, gaps {nivel: gap}, nota)."""
    arq = os.path.join(results, "calibracao", "regime.json")
    if not os.path.exists(arq):
        return dict(presente=False, m_star=None, gaps={}, nota="regime.json ausente")
    with open(arq, encoding="utf-8") as f:
        j = json.load(f)
    gaps = {int(k): v for k, v in (j.get("gap_central_aprendizado_A") or {}).items()}
    m = j.get("m_star")
    return dict(presente=True, m_star=None if m is None else int(m), gaps=gaps, nota=j.get("nota"))


def carregar_calibracao(results="results"):
    """Lê calibracao_<tarefa>_m<m>.json e selecao_<L2|SP>_<tarefa>_m<m>.json de results/calibracao/.
    Devolve (calib {(tarefa, m): dict}, selecao {(metodo, tarefa, m): dict})."""
    pasta = os.path.join(results, "calibracao")
    calib, sel = {}, {}
    for a in sorted(glob.glob(os.path.join(pasta, "*.json"))):
        nome = os.path.basename(a)
        m1 = re.match(r"^calibracao_(.+)_m(\d+)\.json$", nome)
        m2 = re.match(r"^selecao_(L2|SP)_(.+)_m(\d+)\.json$", nome)
        if not (m1 or m2):
            continue
        with open(a, encoding="utf-8") as f:
            j = json.load(f)
        if m1:
            calib[(m1.group(1), int(m1.group(2)))] = j
        else:
            sel[(m2.group(1), m2.group(2), int(m2.group(3)))] = j
    return calib, sel


def nivel_decisao(regime):
    """m* (int) ou None se não há regime de teste."""
    return regime["m_star"]


def nivel_exibicao(df, regime):
    """(nível, aviso) das figuras antigas e das tabelas numéricas: m*, ou o maior nível presente."""
    if regime["m_star"] is not None:
        return regime["m_star"], ""
    if df.empty:
        return None, "sem dados"
    m = int(df["nivel"].max())
    motivo = "regime sem perda (m* nulo)" if regime["presente"] else "regime.json ausente"
    return m, f"AVISO: {motivo}; maior nível disponível (m={m}), sem critério de decisão"


def texto_sem_regime(regime):
    return ("não avaliável (" + (MSG_SEM_REGIME if regime["presente"] else "regime.json ausente") + ")")


# ---------------------------------------------------------------- métrica principal

def _tamanho(df, cond):
    """T da condição = maior índice de tarefa + 1."""
    d = df[df["cond"] == cond]
    return int(d["tarefa"].max()) + 1 if len(d) else 0


def _janela_dados(df, cond, tipo, nivel, janela=None):
    d = df[(df["cond"] == cond) & (df["tipo"] == tipo) & (df["nivel"] == nivel)]
    if d.empty:
        return d
    if janela is None:
        T = _tamanho(df, cond)
        janela = (T - 5, T - 1)
    d = d[(d["tarefa"] >= janela[0]) & (d["tarefa"] <= janela[1])]
    return d.dropna(subset=["gap"])


def metrica_por_semente(df, cond, tipo, nivel, janela=None):
    """Por semente: média do gap pareado nas tarefas da janela com gap definido.
    janela=None -> últimas 5 da própria condição. pd.Series indexada por semente."""
    d = _janela_dados(df, cond, tipo, nivel, janela)
    if d.empty:
        return pd.Series(dtype=float)
    return d.groupby("semente")["gap"].mean()


def acc_nova_por_semente(df, cond, tipo, nivel, janela=None):
    """Por semente: média de acc_nova nas MESMAS tarefas da métrica principal (eixo x da dose)."""
    d = _janela_dados(df, cond, tipo, nivel, janela).dropna(subset=["acc_nova"])
    if d.empty:
        return pd.Series(dtype=float)
    return d.groupby("semente")["acc_nova"].mean()


def n_referencias(df, cond, tipo, nivel, janela=None):
    """Número de tarefas com gap definido na janela, por semente (5 em todas as condições, decisão (ii))."""
    d = _janela_dados(df, cond, tipo, nivel, janela)
    return d.groupby("semente").size() if len(d) else pd.Series(dtype=int)


def n_ref_texto(df, cond, tipo, nivel, janela=None):
    """'n_ref=5' (ou 'n_ref=2-5' se variar entre sementes): referências pareadas por semente na janela."""
    n = n_referencias(df, cond, tipo, nivel, janela)
    if n.empty:
        return "n_ref=0"
    lo, hi = int(n.min()), int(n.max())
    return f"n_ref={lo}" if lo == hi else f"n_ref={lo}-{hi}"


def bootstrap_ic(valores, n=10000, semente=0):
    """(média, inf, sup): IC 95% percentil reamostrando as sementes. n<2 -> IC = (nan, nan)."""
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    media = float(v.mean())
    if len(v) < 2:
        return media, np.nan, np.nan
    rng = np.random.default_rng(semente)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    medias = v[idx].mean(axis=1)
    return media, float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def _resumir(s):
    m, lo, hi = bootstrap_ic(s.values)
    return dict(media=m, inf=lo, sup=hi, n=len(s))


def resumo(df, cond, tipo, nivel, janela=None):
    """Dicionário com média, inf, sup e n sementes da métrica principal."""
    return _resumir(metrica_por_semente(df, cond, tipo, nivel, janela))


def resumo_acc_nova(df, cond, tipo, nivel, janela=None):
    """Média e IC 95% da acurácia da rede nova (eixo x da dose-resposta)."""
    return _resumir(acc_nova_por_semente(df, cond, tipo, nivel, janela))


def curva(df, cond, tipo, coluna, nivel):
    """Por tarefa: média e IC 95% bootstrap entre sementes (só pontos com valor definido)."""
    if df.empty or coluna not in df.columns:
        return pd.DataFrame(columns=["tarefa", "media", "inf", "sup", "n"])
    d = df[(df["cond"] == cond) & (df["tipo"] == tipo) & (df["nivel"] == nivel)].dropna(subset=[coluna])
    linhas = []
    for t, g in d.groupby("tarefa"):
        m, lo, hi = bootstrap_ic(g[coluna].values)
        linhas.append(dict(tarefa=t, media=m, inf=lo, sup=hi, n=len(g)))
    return pd.DataFrame(linhas, columns=["tarefa", "media", "inf", "sup", "n"])


# ---------------------------------------------------------------- esquecimento (Adendo 1)

def esquecimento_por_semente(df, cond, nivel):
    """Coluna `esquecimento` na última tarefa (T-1) de cond em B_disjunta, por semente."""
    if df.empty or "esquecimento" not in df.columns:
        return pd.Series(dtype=float)
    d = df[(df["cond"] == cond) & (df["tipo"] == TIPO_BD) & (df["nivel"] == nivel)]
    d = d[d["tarefa"] == _tamanho(df, cond) - 1].dropna(subset=["esquecimento"])
    return d.groupby("semente")["esquecimento"].mean()


def resumo_esq(df, cond, nivel):
    """Média, IC 95% bootstrap e n do esquecimento final."""
    return _resumir(esquecimento_por_semente(df, cond, nivel))


def limite_esquecimento(df, nivel):
    """média_esq(central_aprendizado) + 0,05; NaN se central_aprendizado não tem esquecimento."""
    r = resumo_esq(df, "central_aprendizado", nivel)
    return r["media"] + LIMITE_ESQ if r["n"] > 0 else np.nan


def respeita_limite(df, cond, nivel):
    """True/False se média_esq(cond) <= limite; None se faltar esquecimento de cond ou do limite."""
    r, lim = resumo_esq(df, cond, nivel), limite_esquecimento(df, nivel)
    if r["n"] == 0 or np.isnan(lim):
        return None
    return bool(r["media"] <= lim)


def rotulo_ganho(df, cond, regime):
    """Marcação de B_disjunta no nível m*: 'ganho válido', 'ganho com violação de esquecimento',
    'sem ganho' (ou 'não avaliável (...)')."""
    m = nivel_decisao(regime)
    if m is None:
        return texto_sem_regime(regime)
    r = resumo(df, cond, TIPO_BD, m)
    c = resumo(df, "central_aprendizado", TIPO_BD, m)
    if r["n"] == 0 or c["n"] == 0:
        return "não avaliável (sem dados)"
    if _sem_ic(r) or _sem_ic(c):
        return "não avaliável (IC indefinido)"
    if not r["inf"] > c["sup"]:
        return "sem ganho"
    ok = respeita_limite(df, cond, m)
    if ok is None:
        return "não avaliável (falta esquecimento)"
    return "ganho válido" if ok else "ganho com violação de esquecimento"


# ---------------------------------------------------------------- regras de decisão (no nível m)

def _sem_ic(r):
    return np.isnan(r["inf"]) or np.isnan(r["sup"])


def _sobrepoe(a, b):
    return a["inf"] <= b["sup"] and b["inf"] <= a["sup"]


def _checar(df, itens, m, janela=None):
    """Devolve (resumos, motivo_nao_avaliavel). itens = [(cond, tipo), ...]."""
    res, faltam, sem_ic = {}, [], []
    for cond, tipo in itens:
        r = resumo(df, cond, tipo, m, janela)
        if r["n"] == 0:
            faltam.append(f"{cond}/{tipo}")
        elif _sem_ic(r):
            sem_ic.append(f"{cond}/{tipo} (n={r['n']})")
        res[(cond, tipo)] = r
    if faltam:
        return res, "falta " + ", ".join(faltam)
    if sem_ic:
        return res, "IC indefinido (n<2) em " + ", ".join(sem_ic)
    return res, None


def _regra(nome, ok, texto, causa=None, motivo="", nivel=None):
    """ok: True/False/None. causa (se ok não é True): 'sem_vantagem', 'insuficiente', 'limite', 'regime'."""
    return dict(nome=nome, ok=ok, texto=texto, causa=causa, motivo=motivo, nivel=nivel)


def h_reg(df, m):
    """H_reg: existe central_L2 ou central_SP cuja média fica dentro do IC de fedavg_iid em A e B."""
    res, motivo = _checar(df, [("fedavg_iid", t) for t in TIPOS_AB], m)
    if motivo:
        return _regra("H_reg", None, f"H_reg: não avaliável ({motivo})", "insuficiente", motivo, m)
    candidatos, ok = [], []
    for c in ("central_L2", "central_SP"):
        rs = [resumo(df, c, t, m) for t in TIPOS_AB]
        if any(r["n"] == 0 for r in rs):
            continue
        candidatos.append(c)
        if all(res[("fedavg_iid", t)]["inf"] <= r["media"] <= res[("fedavg_iid", t)]["sup"]
               for t, r in zip(TIPOS_AB, rs)):
            ok.append(c)
    if not candidatos:
        mt = "falta central_L2 e central_SP em A e B"
        return _regra("H_reg", None, f"H_reg: não avaliável ({mt})", "insuficiente", mt, m)
    if ok:
        return _regra("H_reg", True, "H_reg: atendida por " + ", ".join(ok), nivel=m)
    if len(candidatos) < 2:
        mt = f"falta a outra entre central_L2/central_SP; {candidatos[0]} não atende"
        return _regra("H_reg", None, f"H_reg: não avaliável ({mt})", "insuficiente", mt, m)
    return _regra("H_reg", False, "H_reg: não atendida", nivel=m)


def h_part(df, m):
    """H_part conforme PREREGISTRO.md, incluindo a pré-condição (tarefa B, janela principal)."""
    res, motivo = _checar(df, [("fedavg_iid", "B"), ("central_aprendizado", "B")], m)
    if motivo:
        mt = f"{motivo}; pré-condição"
        return _regra("H_part", None, f"H_part: não avaliável ({mt})", "insuficiente", mt, m)
    f, c = res[("fedavg_iid", "B")], res[("central_aprendizado", "B")]
    if not (f["inf"] > c["sup"]):
        return _regra("H_part", None, "H_part: não avaliável (sem vantagem a explicar)", "sem_vantagem",
                      nivel=m)
    itens = [(c_, t) for c_ in ("media_sem_particao", "central_aprendizado") for t in TIPOS_AB]
    itens += [("fedavg_iid", "B")]
    res, motivo = _checar(df, itens, m, JANELA_PART)
    if motivo:
        mt = f"{motivo}; janela 7..11"
        return _regra("H_part", None, f"H_part: não avaliável ({mt})", "insuficiente", mt, m)
    i_ok = all(_sobrepoe(res[("media_sem_particao", t)], res[("central_aprendizado", t)]) for t in TIPOS_AB)
    ii_ok = _sobrepoe(res[("fedavg_iid", "B")], res[("central_aprendizado", "B")])
    if i_ok and ii_ok:
        return _regra("H_part", True, "H_part: atendida", nivel=m)
    return _regra("H_part", False, f"H_part: não atendida ((i)={i_ok}, (ii)={ii_ok})", nivel=m)


def h_real(df, m):
    """H_real: inf(fedavg_iid) > sup(central_L2) e > sup(central_SP), em A e B + cláusula do Adendo 1."""
    itens = [(c, t) for c in ("fedavg_iid", "central_L2", "central_SP") for t in TIPOS_AB]
    res, motivo = _checar(df, itens, m)
    falta_bd = any(resumo_esq(df, c, m)["n"] == 0 for c in ("fedavg_iid", "central_aprendizado"))
    if falta_bd:
        mt = "falta B_disjunta" + (f"; {motivo}" if motivo else "")
        txt = "H_real: não avaliável (falta B_disjunta)" if not motivo else f"H_real: não avaliável ({mt})"
        return _regra("H_real", None, txt, "insuficiente", mt, m)
    if motivo:
        return _regra("H_real", None, f"H_real: não avaliável ({motivo})", "insuficiente", motivo, m)
    ok = all(res[("fedavg_iid", t)]["inf"] > res[(c, t)]["sup"]
             for t in TIPOS_AB for c in ("central_L2", "central_SP"))
    if not ok:
        return _regra("H_real", False, "H_real: não atendida", nivel=m)
    if not respeita_limite(df, "fedavg_iid", m):
        return _regra("H_real", False, "H_real: ganho de plasticidade com esquecimento acima do limite",
                      "limite", nivel=m)
    return _regra("H_real", True, "H_real: atendida", nivel=m)


def h_orc(df, m):
    """H_orc (Desvios v3): média de fedavg_iid dentro do IC 95% fechado de central_aprendizado, em A e B."""
    motivos = []
    for t in TIPOS_AB:
        f, c = resumo(df, "fedavg_iid", t, m), resumo(df, "central_aprendizado", t, m)
        if f["n"] == 0:
            motivos.append(f"falta fedavg_iid/{t}")
        if c["n"] == 0:
            motivos.append(f"falta central_aprendizado/{t}")
        elif _sem_ic(c):
            motivos.append(f"IC indefinido (n<2) em central_aprendizado/{t} (n={c['n']})")
    if motivos:
        mt = "; ".join(motivos)
        return _regra("H_orc", None, f"H_orc: não avaliável ({mt})", "insuficiente", mt, m)
    dentro = {t: (resumo(df, "central_aprendizado", t, m)["inf"] <= resumo(df, "fedavg_iid", t, m)["media"]
                  <= resumo(df, "central_aprendizado", t, m)["sup"]) for t in TIPOS_AB}
    if all(dentro.values()):
        return _regra("H_orc", True, "H_orc: atendida", nivel=m)
    return _regra("H_orc", False, f"H_orc: não atendida (dentro do IC: A={dentro['A']}, B={dentro['B']})",
                  nivel=m)


def regras(df, regime):
    """Lista de regras (H_reg, H_part, H_real, H_orc) no nível m*; sem m*, todas 'não avaliável'."""
    m = nivel_decisao(regime)
    if m is None:
        txt = texto_sem_regime(regime)
        return [_regra(n, None, f"{n}: {txt}", "regime", txt) for n in ("H_reg", "H_part", "H_real", "H_orc")]
    return [h_reg(df, m), h_part(df, m), h_real(df, m), h_orc(df, m)]


def criterios_atendidos(df, regime):
    """Aplica as 4 regras. Devolve (texto da linha final, lista de textos das regras)."""
    rs = regras(df, regime)
    m = nivel_decisao(regime)
    if m is None:
        return "Critérios atendidos: " + texto_sem_regime(regime), [r["texto"] for r in rs]
    atendidos = [f"{r['nome']} (m={m})" for r in rs if r["ok"] is True]
    linha = "Critérios atendidos: " + (", ".join(atendidos) if atendidos else "nenhum")
    sem_vant = [r["nome"] for r in rs if r["causa"] == "sem_vantagem"]
    if sem_vant:
        linha += f". Sem vantagem a explicar: {', '.join(sem_vant)}"
    insuf = [f"{r['nome']} ({r['motivo']})" for r in rs if r["causa"] == "insuficiente"]
    if insuf:
        linha += f". Dados insuficientes: {'; '.join(insuf)}"
    for r in rs:
        if r["causa"] == "limite":
            linha += f". {r['texto']}"
    linha += f". Todas as hipóteses avaliadas no nível m*={m} (regime.json)"
    return linha, [r["texto"] for r in rs]
