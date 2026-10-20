"""figures/fig_sonda.png: razão da sonda de alvos aleatórios (perda cópia / perda rede nova)."""
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
        _comum.desenhar(ax, df, "sonda_razao", tipo)
        ax.axhline(1, color="gray", linewidth=0.8)
        ax.set_title(f"Tarefa {tipo}")
        ax.set_xlabel("tarefa (índice)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("razão perda_cópia / perda_rede_nova (adimensional)")
    _comum.legenda(axs)
    fig.suptitle("Sonda de alvos aleatórios (razão > 1: cópia aprende pior que rede nova)")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(a.saida, "fig_sonda.png"), dpi=200)


if __name__ == "__main__":
    main()
