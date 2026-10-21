"""EXPLORATÓRIO (PREREGISTRO.md, "Análises exploratórias pós-resultados", item C). Não altera nenhum veredito.

Δlog = log10(sonda_perda_copia + 1e-6) − log10(sonda_perda_nova + 1e-6), por semente, nas tarefas de sonda
(4, 9, ..., 29; 4 e 9 em media_sem_particao). LIMITE: as colunas dos CSVs são as MÉDIAS das 3 perdas por alvo, então
isto é o log da média, não a mediana entre alvos do log (esta NÃO é computável com estes CSVs).
Gera figures/exploratorio_fig_sonda_log.png e figures/exploratorio_sonda_log.md. Só lê results/."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise  # noqa: E402
from exploratorio_gap_area import condicoes_do_nivel, _fmt  # noqa: E402

EPS = 1e-6
COL_NORMA = "fim_norma_global"


def preparar(df):
    """Cópia só com linhas de sonda (perdas definidas) e a coluna dlog."""
    d = df.dropna(subset=["sonda_perda_copia", "sonda_perda_nova"]).copy()
    d["dlog"] = np.log10(d["sonda_perda_copia"] + EPS) - np.log10(d["sonda_perda_nova"] + EPS)
    return d


def media_sonda_por_semente(d, cond, tipo, nivel, coluna):
    x = d[(d["cond"] == cond) & (d["tipo"] == tipo) & (d["nivel"] == nivel)].dropna(subset=[coluna])
    return x.groupby("semente")[coluna].mean()


def figura(d, saida):
    fig, axs = plt.subplots(6, 3, figsize=(17, 24), sharex="col")
    for i, m in enumerate(analise.NIVEIS):
        for j, tipo in enumerate(analise.TIPOS):
            for k, (col, rot) in enumerate([("dlog", "Δlog (log10 de perda, adimensional)"),
                                            (COL_NORMA, "norma L2 global dos parâmetros")]):
                ax = axs[2 * i + k, j]
                for cond in condicoes_do_nivel(m):
                    c = analise.curva(d, cond, tipo, col, m)
                    if c.empty:
                        continue
                    est = analise.ESTILOS[cond]
                    ax.plot(c["tarefa"], c["media"], label=cond, linewidth=1.3, markersize=5, **est)
                    ok = c.dropna(subset=["inf"])
                    if len(ok):
                        ax.fill_between(ok["tarefa"], ok["inf"], ok["sup"], color=est["color"], alpha=0.10,
                                        linewidth=0)
                if col == "dlog":
                    ax.axhline(0, color="gray", linewidth=0.8)
                ax.set_title(f"m={m} · tarefa {tipo} · " + ("Δlog da sonda" if col == "dlog" else "norma global"),
                             fontsize=9)
                ax.grid(alpha=0.3)
                ax.set_ylabel(rot, fontsize=8)
                if 2 * i + k == 5:
                    ax.set_xlabel("tarefa de sonda (índice)")
    h, l = [], []
    for ax in axs.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=7)
    fig.suptitle("EXPLORATÓRIO · sonda em log: Δlog = log10(perda cópia + 1e-6) − log10(perda rede nova + 1e-6) "
                 "(média sobre os 3 alvos) e norma global\nmédia e IC 95% bootstrap entre sementes 1 a 5; "
                 "central_L2/SP só em m=3; sem efeito sobre nenhum veredito", fontsize=12)
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    fig.savefig(saida, dpi=200)
    plt.close(fig)


def tabela(d, saida):
    L = ["# EXPLORATÓRIO · sonda em escala logarítmica (Δlog)", "",
         "**Análise exploratória pós-resultados (PREREGISTRO.md, item C). Não altera nenhum veredito, não entra na "
         "`tabela_decisao.md` nem em regra de decisão. A sonda original (razão) fica como está.**", "",
         "- Δlog = log10(média das perdas da cópia + 1e-6) − log10(média das perdas da rede nova + 1e-6), por semente, "
         "nas tarefas de sonda (índices 4, 9, 14, 19, 24, 29; 4 e 9 em `media_sem_particao`).",
         "- **Limite:** os CSVs guardam só a **média das 3 perdas por alvo** (`sonda_perda_copia`, `sonda_perda_nova`); "
         "a **mediana entre alvos não é computável** com estes dados. Log da média não é média do log.",
         f"- Por semente: média do Δlog (e da norma global, coluna `{COL_NORMA}`) ao longo das tarefas de sonda; depois "
         "média e IC 95% por bootstrap percentil (10 000, `default_rng(0)`) entre as sementes 1 a 5.",
         "- A norma global é mostrada ao lado só para comparação; nenhuma relação é testada aqui.", ""]
    for tipo in analise.TIPOS:
        for m in analise.NIVEIS:
            L += [f"## Tarefa {tipo} · nível m={m}", "",
                  "| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |",
                  "|---|---|---|---|---|---|---|"]
            for cond in condicoes_do_nivel(m):
                rd = analise._resumir(media_sonda_por_semente(d, cond, tipo, m, "dlog"))
                rn = analise._resumir(media_sonda_por_semente(d, cond, tipo, m, COL_NORMA))
                d1, d2 = _fmt(rd)
                n1, n2 = _fmt(rn)
                L.append(f"| {cond} | {d1} | {d2} | {rd['n']} | {n1} | {n2} | {rn['n']} |")
            L.append("")
    with open(saida, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--saida", default="figures")
    a = p.parse_args(argv)
    df = analise.carregar(a.results)
    need = {"sonda_perda_copia", "sonda_perda_nova", COL_NORMA}
    if df.empty or not need <= set(df.columns):
        print(f"sem resultados válidos com {sorted(need)} em {a.results}/; nada foi gerado", file=sys.stderr)
        sys.exit(1)
    d = preparar(df)
    os.makedirs(a.saida, exist_ok=True)
    figura(d, os.path.join(a.saida, "exploratorio_fig_sonda_log.png"))
    tabela(d, os.path.join(a.saida, "exploratorio_sonda_log.md"))


if __name__ == "__main__":
    main()
