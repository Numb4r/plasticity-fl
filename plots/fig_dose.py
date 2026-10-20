"""figures/fig_dose.png (FIGURA PRINCIPAL): dose-resposta.

Eixo x = acurácia da rede nova (média nas tarefas da janela principal com rede nova, e entre sementes) da
PRÓPRIA condição em cada nível m; eixo y = métrica principal (gap pareado contínua − nova). Uma linha por
condição presente em todos os níveis, um painel por tarefa. Média ± IC 95% bootstrap em x e y.
Anel no nível m* (se existir). Linha cheia em 0 e tracejada em −0,10 (limiar do regime, definido na tarefa A). Nota de rodapé: janela
da métrica (7..11 em media_sem_particao, 25..29 nas demais)."""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _comum
import analise

ROTULO_LIMIAR = "limiar do regime (definido na tarefa A)"
NOTA_JANELA = ("Janela da métrica principal (x e y): media_sem_particao usa as tarefas 7..11 (T=12); "
               "as demais condições usam 25..29 (T=30). Só tarefas com rede nova entram.")

DESLOCAMENTO = {c: (6, 6 + 9 * (i % 3) - 9) for i, c in enumerate(analise.CONDICOES_DOSE)}


def _barra(m, lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return 0.0, 0.0
    return max(m - lo, 0.0), max(hi - m, 0.0)


def _painel(ax, df, tipo, m_star):
    desenhou = False
    for cond in analise.CONDICOES_DOSE:
        pts = []
        for m in analise.NIVEIS:
            y = analise.resumo(df, cond, tipo, m)
            x = analise.resumo_acc_nova(df, cond, tipo, m)
            if y["n"] > 0 and x["n"] > 0:
                pts.append((m, x, y))
        if not pts:
            continue
        est = analise.ESTILOS[cond]
        xs = [p[1]["media"] for p in pts]
        ys = [p[2]["media"] for p in pts]
        ax.plot(xs, ys, color=est["color"], linestyle=est["linestyle"], marker=est["marker"],
                markersize=8, markeredgecolor="black", linewidth=1.4, zorder=3)
        for m, x, y in pts:
            xl, xh = _barra(x["media"], x["inf"], x["sup"])
            yl, yh = _barra(y["media"], y["inf"], y["sup"])
            ax.errorbar(x["media"], y["media"], xerr=[[xl], [xh]], yerr=[[yl], [yh]], fmt="none",
                        ecolor=est["color"], elinewidth=1.0, capsize=2, alpha=0.8, zorder=2)
            ax.annotate(f"m={m}", (x["media"], y["media"]), textcoords="offset points",
                        xytext=DESLOCAMENTO[cond], fontsize=7)
            if m_star is not None and m == m_star:
                ax.scatter([x["media"]], [y["media"]], s=330, facecolors="none", edgecolors="red",
                           linewidths=1.8, zorder=4)
        desenhou = True
    if not desenhou:
        ax.text(0.5, 0.5, "sem dados", transform=ax.transAxes, ha="center")
    ax.axhline(0, color="gray", linewidth=0.9)
    ax.axhline(-0.10, color="black", linestyle="--", linewidth=0.9, label=ROTULO_LIMIAR)
    ax.annotate("−0,10: " + ROTULO_LIMIAR, xy=(0.01, -0.10), xycoords=("axes fraction", "data"),
                xytext=(0, 2), textcoords="offset points", fontsize=7, va="bottom")
    ax.set_title(f"Tarefa {tipo}")
    ax.set_xlabel("acurácia da rede nova (fração)")
    ax.grid(alpha=0.3)


def main():
    a = _comum.args_padrao(__doc__)
    df, regime, nivel, aviso = _comum.contexto(a)
    os.makedirs(a.saida, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
    for ax, tipo in zip(axs, analise.TIPOS):
        _painel(ax, df, tipo, regime["m_star"])
    axs[0].set_ylabel("gap pareado: acurácia contínua − acurácia nova (fração)")
    handles = [Line2D([], [], color=analise.ESTILOS[c]["color"], linestyle=analise.ESTILOS[c]["linestyle"],
                      marker=analise.ESTILOS[c]["marker"], markeredgecolor="black", markersize=8, label=c)
               for c in analise.CONDICOES_DOSE]
    handles.append(Line2D([], [], color="gray", linestyle="-", label="gap = 0"))
    handles.append(Line2D([], [], color="black", linestyle="--", label="−0,10: " + ROTULO_LIMIAR))
    if regime["m_star"] is not None:
        handles.append(Line2D([], [], color="red", marker="o", markerfacecolor="none", markersize=12,
                              linestyle="none", label=f"nível m*={regime['m_star']} (regime de teste)"))
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.012, NOTA_JANELA, ha="center", va="bottom", fontsize=8, style="italic")
    sub = "" if regime["m_star"] is not None else (
        "\n" + ("regime sem perda (m* nulo): só dose-resposta, sem critério de decisão" if regime["presente"]
                else "regime.json ausente: sem anel de m*"))
    fig.suptitle("Dose-resposta: plasticidade (gap pareado) × acurácia da rede nova, por nível de orçamento m"
                 " (média e IC 95% bootstrap entre sementes)" + sub)
    fig.tight_layout(rect=(0, 0.17, 1, 0.93))
    fig.savefig(os.path.join(a.saida, "fig_dose.png"), dpi=200)


if __name__ == "__main__":
    main()
