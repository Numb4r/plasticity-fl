"""Biblioteca de análise: carrega results/, calcula a métrica principal e o esquecimento e aplica
as regras de decisão de PREREGISTRO.md ("Hipóteses e critérios" + "Adendo 1").

AMBIGUIDADES do PREREGISTRO.md e como foram implementadas (texto literal), 18 itens:
 1. H_reg: "média ... fica dentro do IC 95% do fedavg_iid". Implementado: a média da métrica
    principal da condição central cai em [inf, sup] do IC de fedavg_iid, limites inclusivos.
 2. H_reg: o mesmo candidato (L2 ou SP) precisa passar em A e em B (Desvios, auditoria nº 2, item 1);
    janela padrão (últimas 5 tarefas).
 3. "sobrepõe": dois ICs se sobrepõem se inf1 <= sup2 e inf2 <= sup1 (fechados).
 4. "ICs sem sobreposição, fedavg acima" (pré-condição de H_part): inf(fedavg) > sup(central_aprendizado).
 5. H_real: "limite inferior do IC de fedavg_iid acima do limite superior" -> desigualdade estrita (>).
 6. Janela "últimas 5 tarefas": índices T-5..T-1 com T da própria condição (30 ou 12), T = maior
    índice de tarefa da condição no CSV (todas as tarefas juntas) + 1. Só entram tarefas com gap
    definido (t%3==2). H_part (i)/(ii) usa a janela fixa 7..11 para todas as condições.
 7. Sementes cuja janela não tem nenhum gap definido são descartadas (n menor).
 8. Cada IC é um bootstrap percentil próprio (10 000 reamostragens, default_rng(0)); as reamostragens
    usam os mesmos índices de semente em todas as condições com o mesmo n (Desvios, item 5).
 9. Com n<2 sementes o IC é indefinido: toda regra que depende dele devolve "não avaliável (...)";
    isso não conta como "atendida" nem como "não atendida".
10. "Nenhuma atendida" é reportado como "nenhum". A linha final separa "sem vantagem a explicar"
    (pré-condição de H_part falhou) de "dados insuficientes" (falta dado ou IC indefinido).
11. ADENDO, texto contraditório: a linha "marcada 'ganho com violação de esquecimento' se melhora
    plasticidade e passa o limite" diz "passa" onde o sentido (e o efeito sobre H_real) pede
    "ultrapassa". Implementado: melhora e média_esq <= limite -> "ganho válido"; melhora e
    média_esq > limite -> "ganho com violação de esquecimento"; não melhora -> "sem ganho".
12. Esquecimento por semente = valor da coluna `esquecimento` na linha da tarefa T-1 (T como no item 6;
    11 em media_sem_particao). Semente sem linha ou com NaN nessa tarefa é descartada.
13. Limite: comparação literal entre MÉDIAS, média_esq(cond) <= média_esq(central_aprendizado) + 0,05,
    sem tolerância numérica (empate exato conta como "respeita"). Os ICs do esquecimento só informam.
14. "Melhora plasticidade" usa a métrica principal (janela padrão, item 6) em B_disjunta;
    inf(cond) > sup(central_aprendizado), desigualdade estrita. central_aprendizado nunca melhora
    sobre si mesma: fica "sem ganho" (é a referência).
15. Rótulo "não avaliável" quando a métrica principal não tem IC (n<2 ou sem dados) ou quando a
    condição melhora mas falta esquecimento dela ou de central_aprendizado.
16. H_real com o Adendo: sem esquecimento de fedavg_iid ou de central_aprendizado em B_disjunta,
    H_real é "não avaliável (falta B_disjunta)" MESMO QUE o critério original já falhe (literal:
    o texto não diz que a falta de B_disjunta só importa se o critério original valer).
17. H_real com critério original válido e fedavg_iid acima do limite: veredito falso (não conta como
    atendida) e a linha final traz "H_real: ganho de plasticidade com esquecimento acima do limite".
18. `carregar` lê só os CSVs principais: ignora *_rodadas.csv, *_matriz.csv, subpastas e .params.json
    e pula (com aviso) CSV sem as colunas tarefa/tipo/cond/semente. cond, tipo e semente vêm das
    colunas, nunca do nome do arquivo. Em A e B as colunas do esquecimento não existem (NaN após concat).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

TIPOS_AB = ["A", "B"]                 # tarefas das hipóteses originais
TIPO_BD = "B_disjunta"                # tarefa do Adendo 1 (esquecimento)
TIPOS = ["A", "B", "B_disjunta"]      # todos os painéis das figuras
CONDICOES = ["fedavg_iid", "sem_media_k1", "media_sem_particao",
             "central_passos", "central_aprendizado", "central_L2", "central_SP"]
JANELA_PART = (7, 11)  # janela fixa de H_part (índices, inclusiva)
LIMITE_ESQ = 0.05      # Adendo 1: 5 pontos percentuais (fração de acurácia)
SUFIXOS_IGNORADOS = ("_rodadas.csv", "_matriz.csv")
COLUNAS_CHAVE = {"tarefa", "tipo", "cond", "semente"}

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


def carregar(results="results"):
    """Lê só os CSVs principais de results/ (um nível, sem subpastas).

    Ignora *_rodadas.csv e *_matriz.csv (e *.params.json, que nem é CSV). cond/tipo/semente vêm
    das colunas, não do nome do arquivo. Em A e B as colunas acc_media_vistas, esquecimento e bwt
    não existem: viram NaN no concat."""
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
    return pd.concat(dfs, ignore_index=True)


def _tamanho(df, cond):
    """T da condição = maior índice de tarefa + 1."""
    d = df[df["cond"] == cond]
    return int(d["tarefa"].max()) + 1 if len(d) else 0


def metrica_por_semente(df, cond, tipo, janela=None):
    """Para cada semente: média do gap nas tarefas da janela com gap não-NaN.

    janela=None -> últimas 5 tarefas da própria condição (T-5..T-1).
    janela=(ini, fim) -> índices ini..fim inclusive. Devolve pd.Series indexada por semente
    (sementes sem nenhum gap na janela são omitidas)."""
    d = df[(df["cond"] == cond) & (df["tipo"] == tipo)]
    if d.empty:
        return pd.Series(dtype=float)
    if janela is None:
        T = _tamanho(df, cond)
        janela = (T - 5, T - 1)
    d = d[(d["tarefa"] >= janela[0]) & (d["tarefa"] <= janela[1])]
    d = d.dropna(subset=["gap"])
    return d.groupby("semente")["gap"].mean()


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


def resumo(df, cond, tipo, janela=None):
    """Dicionário com média, inf, sup e n sementes da métrica principal."""
    s = metrica_por_semente(df, cond, tipo, janela)
    m, lo, hi = bootstrap_ic(s.values)
    return dict(media=m, inf=lo, sup=hi, n=len(s))


def curva(df, cond, tipo, coluna):
    """Por tarefa: média e IC 95% bootstrap entre sementes (só pontos com valor definido).
    Devolve DataFrame com tarefa, media, inf, sup, n."""
    d = df[(df["cond"] == cond) & (df["tipo"] == tipo)].dropna(subset=[coluna])
    linhas = []
    for t, g in d.groupby("tarefa"):
        m, lo, hi = bootstrap_ic(g[coluna].values)
        linhas.append(dict(tarefa=t, media=m, inf=lo, sup=hi, n=len(g)))
    return pd.DataFrame(linhas, columns=["tarefa", "media", "inf", "sup", "n"])




# ---------------------------------------------------------------- esquecimento (Adendo 1)

def esquecimento_por_semente(df, cond):
    """Valor da coluna `esquecimento` na última tarefa (T-1) de cond em B_disjunta, por semente."""
    if df.empty or "esquecimento" not in df.columns:
        return pd.Series(dtype=float)
    d = df[(df["cond"] == cond) & (df["tipo"] == TIPO_BD)]
    d = d[d["tarefa"] == _tamanho(df, cond) - 1].dropna(subset=["esquecimento"])
    return d.groupby("semente")["esquecimento"].mean()


def resumo_esq(df, cond):
    """Média, IC 95% bootstrap (mesmo método da métrica principal) e n do esquecimento final."""
    s = esquecimento_por_semente(df, cond)
    m, lo, hi = bootstrap_ic(s.values)
    return dict(media=m, inf=lo, sup=hi, n=len(s))


def limite_esquecimento(df):
    """média_esq(central_aprendizado) + 0,05; NaN se central_aprendizado não tem esquecimento."""
    r = resumo_esq(df, "central_aprendizado")
    return r["media"] + LIMITE_ESQ if r["n"] > 0 else np.nan


def respeita_limite(df, cond):
    """True/False se média_esq(cond) <= limite; None se faltar esquecimento de cond ou do limite."""
    r, lim = resumo_esq(df, cond), limite_esquecimento(df)
    if r["n"] == 0 or np.isnan(lim):
        return None
    return bool(r["media"] <= lim)


def rotulo_ganho(df, cond):
    """Marcação de B_disjunta: 'ganho válido', 'ganho com violação de esquecimento', 'sem ganho'
    (ou 'não avaliável (...)')."""
    r = resumo(df, cond, TIPO_BD)
    c = resumo(df, "central_aprendizado", TIPO_BD)
    if r["n"] == 0 or c["n"] == 0:
        return "não avaliável (sem dados)"
    if _sem_ic(r) or _sem_ic(c):
        return "não avaliável (IC indefinido)"
    if not r["inf"] > c["sup"]:
        return "sem ganho"
    ok = respeita_limite(df, cond)
    if ok is None:
        return "não avaliável (falta esquecimento)"
    return "ganho válido" if ok else "ganho com violação de esquecimento"


# ---------------------------------------------------------------- regras de decisão
# Cada regra devolve um dict (ver _regra); ok: True, False ou None (não avaliável).

def _res(df, cond, tipo, janela=None):
    return resumo(df, cond, tipo, janela)


def _sem_ic(r):
    return np.isnan(r["inf"]) or np.isnan(r["sup"])


def _sobrepoe(a, b):
    return a["inf"] <= b["sup"] and b["inf"] <= a["sup"]


def _checar(df, itens, janela=None):
    """Devolve (resumos, motivo_nao_avaliavel). itens = [(cond, tipo), ...]."""
    res = {}
    faltam = []
    sem_ic = []
    for cond, tipo in itens:
        r = _res(df, cond, tipo, janela)
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


def _regra(nome, ok, texto, causa=None, motivo=""):
    """ok: True/False/None. causa (se ok não é True): 'sem_vantagem', 'insuficiente' ou 'limite'."""
    return dict(nome=nome, ok=ok, texto=texto, causa=causa, motivo=motivo)


def h_reg(df):
    """H_reg: existe central_L2 ou central_SP cuja média fica dentro do IC de fedavg_iid em A e B."""
    # a média de L2/SP existe mesmo com n=1; só o IC do fedavg é necessário
    res, motivo = _checar(df, [("fedavg_iid", t) for t in TIPOS_AB])
    if motivo:
        return _regra("H_reg", None, f"H_reg: não avaliável ({motivo})", "insuficiente", motivo)
    candidatos, ok = [], []
    for c in ("central_L2", "central_SP"):
        rs = [_res(df, c, t) for t in TIPOS_AB]
        if any(r["n"] == 0 for r in rs):
            continue
        candidatos.append(c)
        dentro = all(res[("fedavg_iid", t)]["inf"] <= r["media"] <= res[("fedavg_iid", t)]["sup"]
                     for t, r in zip(TIPOS_AB, rs))
        if dentro:
            ok.append(c)
    if not candidatos:
        m = "falta central_L2 e central_SP em A e B"
        return _regra("H_reg", None, f"H_reg: não avaliável ({m})", "insuficiente", m)
    if ok:
        return _regra("H_reg", True, "H_reg: atendida por " + ", ".join(ok))
    if len(candidatos) < 2:
        m = f"falta a outra entre central_L2/central_SP; {candidatos[0]} não atende"
        return _regra("H_reg", None, f"H_reg: não avaliável ({m})", "insuficiente", m)
    return _regra("H_reg", False, "H_reg: não atendida")


def h_part(df):
    """H_part conforme PREREGISTRO.md, incluindo a pré-condição."""
    # pré-condição: janela padrão, tarefa B, fedavg acima de central_aprendizado sem sobreposição
    res, motivo = _checar(df, [("fedavg_iid", "B"), ("central_aprendizado", "B")])
    if motivo:
        m = f"{motivo}; pré-condição"
        return _regra("H_part", None, f"H_part: não avaliável ({m})", "insuficiente", m)
    f, c = res[("fedavg_iid", "B")], res[("central_aprendizado", "B")]
    if not (f["inf"] > c["sup"]):
        return _regra("H_part", None, "H_part: não avaliável (sem vantagem a explicar)", "sem_vantagem")
    # (i) A e B: media_sem_particao sobrepõe central_aprendizado, janela 7..11
    itens = [(c_, t) for c_ in ("media_sem_particao", "central_aprendizado") for t in TIPOS_AB]
    itens += [("fedavg_iid", "B")]
    res, motivo = _checar(df, itens, JANELA_PART)
    if motivo:
        m = f"{motivo}; janela 7..11"
        return _regra("H_part", None, f"H_part: não avaliável ({m})", "insuficiente", m)
    i_ok = all(_sobrepoe(res[("media_sem_particao", t)], res[("central_aprendizado", t)]) for t in TIPOS_AB)
    ii_ok = _sobrepoe(res[("fedavg_iid", "B")], res[("central_aprendizado", "B")])
    if i_ok and ii_ok:
        return _regra("H_part", True, "H_part: atendida")
    return _regra("H_part", False, f"H_part: não atendida ((i)={i_ok}, (ii)={ii_ok})")


def h_real(df):
    """H_real: inf(fedavg_iid) > sup(central_L2) e > sup(central_SP), em A e B (critério original)
    + cláusula do Adendo 1: fedavg_iid precisa respeitar o limite de esquecimento em B_disjunta."""
    itens = [(c, t) for c in ("fedavg_iid", "central_L2", "central_SP") for t in TIPOS_AB]
    res, motivo = _checar(df, itens)
    falta_bd = any(resumo_esq(df, c)["n"] == 0 for c in ("fedavg_iid", "central_aprendizado"))
    if falta_bd:
        # literal: sem B_disjunta o Adendo não permite concluir, mesmo que o critério original falhe
        m = "falta B_disjunta" + (f"; {motivo}" if motivo else "")
        txt = "H_real: não avaliável (falta B_disjunta)" if not motivo else f"H_real: não avaliável ({m})"
        return _regra("H_real", None, txt, "insuficiente", m)
    if motivo:
        return _regra("H_real", None, f"H_real: não avaliável ({motivo})", "insuficiente", motivo)
    ok = all(res[("fedavg_iid", t)]["inf"] > res[(c, t)]["sup"]
             for t in TIPOS_AB for c in ("central_L2", "central_SP"))
    if not ok:
        return _regra("H_real", False, "H_real: não atendida")
    if not respeita_limite(df, "fedavg_iid"):
        return _regra("H_real", False, "H_real: ganho de plasticidade com esquecimento acima do limite",
                      "limite")
    return _regra("H_real", True, "H_real: atendida")


def criterios_atendidos(df):
    """Aplica as 3 regras. Devolve (texto da linha final, lista de explicações)."""
    regras = [h_reg(df), h_part(df), h_real(df)]
    atendidos = [r["nome"] for r in regras if r["ok"] is True]
    linha = "Critérios atendidos: " + (", ".join(atendidos) if atendidos else "nenhum")
    sem_vant = [r["nome"] for r in regras if r["causa"] == "sem_vantagem"]
    if sem_vant:
        linha += f". Sem vantagem a explicar: {', '.join(sem_vant)}"
    insuf = [f"{r['nome']} ({r['motivo']})" for r in regras if r["causa"] == "insuficiente"]
    if insuf:
        linha += f". Dados insuficientes: {'; '.join(insuf)}"
    for r in regras:
        if r["causa"] == "limite":
            linha += f". {r['texto']}"
    return linha, [r["texto"] for r in regras]
