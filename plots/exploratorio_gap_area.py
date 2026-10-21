"""EXPLORATÓRIO (PREREGISTRO.md, "Análises exploratórias pós-resultados", item B). Não altera nenhum veredito.

Gera figures/exploratorio_fig_gap_curvas.png (curva do gap por tarefa, por nível e tarefa) e
figures/exploratorio_gap_area.md (área do gap = integral trapezoidal do gap sobre o índice da tarefa, do primeiro
ao último ponto observado, dividida pelo comprimento do intervalo; média e IC 95% bootstrap entre sementes).
Só lê results/ (via analise.carregar) e só escreve em figures/ (ou na pasta --saida)."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise  # noqa: E402

CONDS_TODAS = analise.CONDICOES_DOSE
CONDS_M3 = analise.CONDICOES_DOSE + ["central_L2", "central_SP"]
NIVEL_EXTRAS = 3


def _trapz(y, x):
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def condicoes_do_nivel(m):
    return CONDS_M3 if m == NIVEL_EXTRAS else CONDS_TODAS


def area_por_semente(df, cond, tipo, nivel):
    """Série semente -> gap médio ponderado pelo espaçamento (trapézios / comprimento do intervalo).
    Semente com menos de 2 pontos de gap observados é descartada."""
    d = df[(df["cond"] == cond) & (df["tipo"] == tipo) & (df["nivel"] == nivel)].dropna(subset=["gap"])
    out = {}
    for s, g in d.groupby("semente"):
        g = g.sort_values("tarefa")
        x, y = g["tarefa"].to_numpy(float), g["gap"].to_numpy(float)
        if len(x) < 2 or x[-1] == x[0]:
            continue
        out[s] = float(_trapz(y, x) / (x[-1] - x[0]))
    return pd.Series(out, dtype=float)


def figura(df, saida):
    fig, axs = plt.subplots(3, 3, figsize=(17, 14), sharex=True)
    for i, m in enumerate(analise.NIVEIS):
        for j, tipo in enumerate(analise.TIPOS):
            ax = axs[i, j]
            for cond in condicoes_do_nivel(m):
                c = analise.curva(df, cond, tipo, "gap", m)
                if c.empty:
                    continue
                est = analise.ESTILOS[cond]
                ax.plot(c["tarefa"], c["media"], label=cond, linewidth=1.3, markersize=4.5, **est)
                ok = c.dropna(subset=["inf"])
                if len(ok):
                    ax.fill_between(ok["tarefa"], ok["inf"], ok["sup"], color=est["color"], alpha=0.10, linewidth=0)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_title(f"nível m={m} · tarefa {tipo}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == 2:
                ax.set_xlabel("tarefa (índice)")
            if j == 0:
                ax.set_ylabel("gap pareado (diferença de acurácia, fração)")
    h, l = [], []
    for ax in axs.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=7)
    fig.suptitle("EXPLORATÓRIO · curva completa do gap por tarefa (média e IC 95% bootstrap entre sementes 1 a 5)\n"
                 "central_L2 e central_SP só existem em m=3; sem efeito sobre nenhum veredito confirmatório",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(saida, dpi=200)
    plt.close(fig)


def _fmt(r):
    if r["n"] == 0:
        return "n/d", "n/d"
    ic = "n/d" if np.isnan(r["inf"]) else f"[{r['inf']:+.4f}; {r['sup']:+.4f}]"
    return f"{r['media']:+.4f}", ic


def tabela(df, saida):
    L = ["# EXPLORATÓRIO · área do gap (média ao longo da sequência)", "",
         "**Análise exploratória pós-resultados (PREREGISTRO.md, item B). Não altera nenhum veredito, não entra na "
         "`tabela_decisao.md` nem em regra de decisão.**", "",
         "- Área por (condição, nível, tarefa, semente) = integral trapezoidal do gap sobre o índice da tarefa, do "
         "primeiro ao último ponto observado (2 a 29; 2 a 11 em `media_sem_particao`), dividida pelo comprimento do "
         "intervalo (27; 9) = gap médio ao longo da sequência, ponderado pelo espaçamento.",
         "- Média e IC 95% por bootstrap percentil (10 000 reamostragens das sementes, `default_rng(0)`), sementes 1 a 5.",
         "- **Limitação:** os pontos de gap são mais densos nas tarefas 7 a 11 e 25 a 29, então a ponderação por "
         "espaçamento usa trapézios sobre pontos desigualmente espaçados.",
         "- A coluna \"confirmatória, não alterada\" é a métrica principal (gap médio das 5 últimas tarefas, "
         "`analise.metrica_por_semente`), mostrada só para comparação visual; não é recalculada nem reinterpretada.",
         "- `media_sem_particao` tem T=12: o intervalo cobre só as tarefas 2 a 11, e a janela \"5 últimas\" é 7 a 11.",
         "- Gap = acc_cont − acc_nova (diferença de acurácia, fração).", ""]
    for tipo in analise.TIPOS:
        for m in analise.NIVEIS:
            L += [f"## Tarefa {tipo} · nível m={m}", "",
                  "| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | "
                  "confirmatória, não alterada: IC 95% | n |",
                  "|---|---|---|---|---|---|---|"]
            for cond in condicoes_do_nivel(m):
                ra = analise._resumir(area_por_semente(df, cond, tipo, m))
                rc = analise.resumo(df, cond, tipo, m)
                a1, a2 = _fmt(ra)
                c1, c2 = _fmt(rc)
                L.append(f"| {cond} | {a1} | {a2} | {ra['n']} | {c1} | {c2} | {rc['n']} |")
            L.append("")
    with open(saida, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--saida", default="figures")
    a = p.parse_args(argv)
    df = analise.carregar(a.results)
    if df.empty or "gap" not in df.columns:
        print(f"sem resultados válidos em {a.results}/; nada foi gerado", file=sys.stderr)
        sys.exit(1)
    os.makedirs(a.saida, exist_ok=True)
    figura(df, os.path.join(a.saida, "exploratorio_fig_gap_curvas.png"))
    tabela(df, os.path.join(a.saida, "exploratorio_gap_area.md"))


if __name__ == "__main__":
    main()
