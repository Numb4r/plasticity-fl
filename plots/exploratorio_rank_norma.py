"""EXPLORATÓRIO (PREREGISTRO.md, "Exploratório da rodada 1: rank efetivo e norma por tarefa"). Não altera veredito.

Tarefa A, nível m=3, sementes 1 a 5, 7 condições: por tarefa t (fim da tarefa), média entre sementes e IC 95%
bootstrap percentil (10 000, default_rng(0)) das colunas `rank_ef` e `norma`. Fonte: results/<cond>_m3_A_<semente>.csv.
Gera figures/exploratorio_rank_norma.png e figures/exploratorio_rank_norma.md. Só lê results/; só escreve em
figures/ (ou na pasta --saida). Usa de analise.py apenas carregar, ESTILOS e bootstrap_ic."""
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

NIVEL, TIPO = 3, "A"
CONDS = ["fedavg_iid", "sem_media_k1", "media_sem_particao", "central_passos",
         "central_aprendizado", "central_L2", "central_SP"]
COM_MEDIA = {"fedavg_iid", "media_sem_particao"}
METRICAS = [("rank_ef", "rank efetivo (entropia das ativações da última camada oculta, adimensional)", "rank efetivo"),
            ("norma", "norma L2 global dos parâmetros (adimensional)", "norma dos parâmetros")]
TAREFAS_RESUMO = [0, 4, 9, 14, 19, 24, 29]
DEF = ("`rank_ef`: rank efetivo por entropia das ativações da última camada oculta, medido no fim de cada tarefa. "
       "`norma`: norma L2 global dos parâmetros, no fim de cada tarefa.")


def tabela_curvas(df, coluna):
    """DataFrame cond, tarefa, media, inf, sup, n (IC por bootstrap percentil entre sementes)."""
    linhas = []
    for cond in CONDS:
        d = df[(df["cond"] == cond) & (df["tipo"] == TIPO) & (df["nivel"] == NIVEL)].dropna(subset=[coluna])
        for t, g in d.groupby("tarefa"):
            m, lo, hi = analise.bootstrap_ic(g[coluna].values, n=10000, semente=0)
            linhas.append(dict(cond=cond, tarefa=int(t), media=m, inf=lo, sup=hi, n=len(g)))
    return pd.DataFrame(linhas, columns=["cond", "tarefa", "media", "inf", "sup", "n"])


def figura(curvas, saida):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (col, rotulo, titulo) in zip(axs, METRICAS):
        c = curvas[col]
        for cond in CONDS:
            g = c[c["cond"] == cond].sort_values("tarefa")
            if g.empty:
                continue
            est = analise.ESTILOS[cond]
            grupo = "com média" if cond in COM_MEDIA else "sem média"
            ax.plot(g["tarefa"], g["media"], label=f"{cond} ({grupo})", linewidth=1.4, markersize=4.5,
                    markevery=1, **est)
            ok = g.dropna(subset=["inf"])
            if len(ok):
                ax.fill_between(ok["tarefa"], ok["inf"], ok["sup"], color=est["color"], alpha=0.12, linewidth=0)
        ax.set_title(f"EXPLORATÓRIO · {titulo} · tarefa A, m=3", fontsize=11)
        ax.set_xlabel("tarefa (índice)")
        ax.set_ylabel(rotulo, fontsize=9)
        ax.grid(alpha=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("EXPLORATÓRIO (descritivo, sem critério de decisão; média entre 5 sementes e IC 95% bootstrap). "
                 "Legenda: 'com média' = fedavg_iid e media_sem_particao; 'sem média' = sem_media_k1 e central_*.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(saida, dpi=200)
    plt.close(fig)


def _cel(r, casas):
    if r is None:
        return "—"
    if np.isnan(r["inf"]):
        return f"{r['media']:.{casas}f} [—]"
    return f"{r['media']:.{casas}f} [{r['inf']:.{casas}f}; {r['sup']:.{casas}f}]"


def _tabela(c, casas, tarefas):
    idx = {(r.cond, r.tarefa): r._asdict() for r in c.itertuples(index=False)}
    out = ["| condição | " + " | ".join(f"t={t}" for t in tarefas) + " |",
           "|---|" + "---|" * len(tarefas)]
    for cond in CONDS:
        out.append(f"| {cond} | " + " | ".join(_cel(idx.get((cond, t)), casas) for t in tarefas) + " |")
    return out


def markdown(curvas, saida, nsem):
    L = ["# EXPLORATÓRIO: rank efetivo e norma por tarefa",
         "",
         "**Exploratório, descritivo, sem critério de decisão, calculado depois de ver os resultados. "
         "Não altera nenhum veredito.** Definição escrita antes de calcular em PREREGISTRO.md, bloco "
         "\"Exploratório da rodada 1: rank efetivo e norma por tarefa\".",
         "",
         f"- Recorte: tarefa A, nível m=3, sementes 1 a 5, 7 condições. Fonte: `results/<cond>_m3_A_<semente>.csv`.",
         f"- {DEF}",
         "- Valor por célula: média entre sementes [IC 95% bootstrap percentil, 10 000 reamostragens das sementes, "
         "`default_rng(0)`]. `media_sem_particao` tem T=12 (tarefas 0 a 11); as demais, T=30 (0 a 29). "
         "'—' = sem dado naquela tarefa.",
         "- Grupos no gráfico (só rótulo de agrupamento, sem interpretação): com média = fedavg_iid, "
         "media_sem_particao; sem média = sem_media_k1, central_passos, central_aprendizado, central_L2, central_SP.",
         "- Figura: `figures/exploratorio_rank_norma.png`. Gerador: `plots/exploratorio_rank_norma.py`.",
         "",
         "## n de sementes por condição",
         "",
         "| condição | n sementes (mín–máx entre tarefas) |", "|---|---|"]
    n = curvas["rank_ef"].groupby("cond")["n"].agg(["min", "max"])
    for cond in CONDS:
        if cond in n.index:
            lo, hi = int(n.loc[cond, "min"]), int(n.loc[cond, "max"])
            L.append(f"| {cond} | {lo if lo == hi else f'{lo}–{hi}'} |")
        else:
            L.append(f"| {cond} | 0 |")
    tmax = int(max(c["tarefa"].max() for c in curvas.values()))
    todas = list(range(tmax + 1))
    for col, _, titulo in METRICAS:
        casas = 2
        L += ["", f"## Resumo: {col} nas tarefas {', '.join(map(str, TAREFAS_RESUMO))}", ""]
        L += _tabela(curvas[col], casas, TAREFAS_RESUMO)
    for col, _, titulo in METRICAS:
        L += ["", f"## Tabela completa: {col} por condição e por tarefa (média [IC 95%])", ""]
        c = curvas[col]
        idx = {(r.cond, r.tarefa): r._asdict() for r in c.itertuples(index=False)}
        L += ["| tarefa | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
        for t in todas:
            L.append(f"| {t} | " + " | ".join(_cel(idx.get((cond, t)), 2) for cond in CONDS) + " |")
    with open(saida, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(results="results", saida="figures"):
    df = analise.carregar(results)
    curvas = {col: tabela_curvas(df, col) for col, _, _ in METRICAS}
    if curvas["rank_ef"].empty:
        sys.exit("sem dados da tarefa A, m=3 em " + results)
    os.makedirs(saida, exist_ok=True)
    figura(curvas, os.path.join(saida, "exploratorio_rank_norma.png"))
    markdown(curvas, os.path.join(saida, "exploratorio_rank_norma.md"), None)
    return curvas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--saida", default="figures")
    a = ap.parse_args()
    main(a.results, a.saida)
