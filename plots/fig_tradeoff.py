"""figures/fig_tradeoff.png: B_disjunta, plasticidade (x) contra esquecimento médio final (y).

Um ponto por condição com IC 95% bootstrap nos dois eixos. A faixa hachurada marca a região
acima do limite do Adendo 1: média_esq(central_aprendizado) + 0,05."""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _comum
import analise


def _barra(m, lo, hi):
    """Meia-largura inferior e superior da barra de erro; zeros se IC indefinido."""
    if np.isnan(lo) or np.isnan(hi):
        return 0.0, 0.0
    return max(m - lo, 0.0), max(hi - m, 0.0)


def main():
    a = _comum.args_padrao(__doc__)
    df = analise.carregar(a.results)
    os.makedirs(a.saida, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ys, handles = [], []
    for cond in analise.CONDICOES:
        x = analise.resumo(df, cond, analise.TIPO_BD)
        y = analise.resumo_esq(df, cond)
        if x["n"] == 0 or y["n"] == 0:
            continue
        est = analise.ESTILOS[cond]
        xl, xh = _barra(x["media"], x["inf"], x["sup"])
        yl, yh = _barra(y["media"], y["inf"], y["sup"])
        _, caps, barras = ax.errorbar(
            x["media"], y["media"], xerr=[[xl], [xh]], yerr=[[yl], [yh]], fmt=est["marker"],
            color=est["color"], markersize=10, markeredgecolor="black", capsize=4,
            elinewidth=1.6, zorder=3)
        for lc in barras:                       # estilo de linha da barra distingue em preto e branco
            lc.set_linestyle(est["linestyle"])
        ys += [y["media"] - yl, y["media"] + yh]
        handles.append(Line2D([], [], color=est["color"], linestyle=est["linestyle"], marker=est["marker"],
                              markersize=9, markeredgecolor="black", label=cond))
    lim = analise.limite_esquecimento(df)
    if not ys:
        ax.text(0.5, 0.5, "sem dados", transform=ax.transAxes, ha="center")
    else:
        topo = max(ys + ([lim] if not np.isnan(lim) else []))
        base = min(ys + ([lim] if not np.isnan(lim) else []))
        folga = 0.1 * max(topo - base, 0.05)
        ax.set_ylim(base - folga, topo + folga)
        if not np.isnan(lim):
            ax.axhspan(lim, topo + folga, facecolor="none", edgecolor="gray", hatch="//", alpha=0.6,
                       linewidth=0, zorder=1)
            ax.axhline(lim, color="black", linestyle="--", linewidth=1.3, zorder=2)
            ax.text(0.99, lim, f"limite de esquecimento = {lim:.3f}\n(central_aprendizado + 0,05)",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85), zorder=4)
        ax.legend(handles=handles, fontsize=9, loc="upper left", title="condição")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("plasticidade: gap contínua − nova, últimas 5 tarefas (fração de acurácia)")
    ax.set_ylabel("esquecimento médio final (fração de acurácia)")
    ax.set_title("B_disjunta: plasticidade × esquecimento (média e IC 95% bootstrap entre sementes)\n"
                 "região hachurada: esquecimento acima do limite do Adendo 1")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.saida, "fig_tradeoff.png"), dpi=200)


if __name__ == "__main__":
    main()
