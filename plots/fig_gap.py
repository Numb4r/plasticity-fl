"""figures/fig_gap.png: gap (contínua - nova) por tarefa, média e IC 95% entre sementes."""
import os

import matplotlib.pyplot as plt

import _comum
import analise


def main():
    a = _comum.args_padrao(__doc__)
    df = analise.carregar(a.results)
    os.makedirs(a.saida, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, tipo in zip(axs, analise.TIPOS):
        _comum.desenhar(ax, df, "gap", tipo)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"Tarefa {tipo}")
        ax.set_xlabel("tarefa (índice)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("gap = acurácia contínua − acurácia nova (fração)")
    _comum.legenda(axs)
    fig.suptitle("Gap de plasticidade por tarefa (média e IC 95% bootstrap entre sementes)")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(a.saida, "fig_gap.png"), dpi=200)


if __name__ == "__main__":
    main()
