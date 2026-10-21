"""figures/fig_sonda.png: razão da sonda de alvos aleatórios (perda cópia / perda rede nova)."""
import os

import matplotlib.pyplot as plt

import _comum
import analise


def main():
    a = _comum.args_padrao(__doc__)
    df, regime, nivel, aviso = _comum.contexto(a)
    os.makedirs(a.saida, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, tipo in zip(axs, analise.TIPOS):
        _comum.desenhar(ax, df, "sonda_razao", tipo, nivel)
        ax.axhline(1, color="gray", linewidth=0.8)
        ax.set_yscale("symlog", linthresh=1)  # razão vai de ~0 a 1e6: escala linear achata as demais condições
        ax.set_title(f"Tarefa {tipo}")
        ax.set_xlabel("tarefa (índice)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("razão perda cópia / rede nova\n(adimensional, escala symlog)")
    _comum.legenda(axs)
    fig.suptitle(_comum.titulo("Sonda de alvos aleatórios (razão > 1: cópia aprende pior que rede nova)", nivel, aviso))
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(os.path.join(a.saida, "fig_sonda.png"), dpi=200)


if __name__ == "__main__":
    main()
