"""figures/fig_norma.png: norma dos parâmetros e ‖Δθ‖ na tarefa (medida após o passo SP), painéis A, B e B_disjunta."""
import os

import matplotlib.pyplot as plt

import _comum
import analise


def main():
    a = _comum.args_padrao(__doc__)
    df = analise.carregar(a.results)
    os.makedirs(a.saida, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(17, 8), sharex="col")
    linhas = [("norma", "norma L2 dos parâmetros"),
              ("distancia", "‖Δθ‖ na tarefa (após SP) (norma L2 dos parâmetros)")]
    for i, (col, rotulo) in enumerate(linhas):
        for j, tipo in enumerate(analise.TIPOS):
            ax = axs[i, j]
            _comum.desenhar(ax, df, col, tipo)
            ax.set_title(f"Tarefa {tipo}: {col}")  # painel sem dados mostra "sem dados"
            if j == 0:
                ax.set_ylabel(rotulo)
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("tarefa (índice)")
    _comum.legenda(axs.ravel())
    fig.suptitle("Norma dos parâmetros e ‖Δθ‖ na tarefa (após SP) (média e IC 95% bootstrap entre sementes)")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(a.saida, "fig_norma.png"), dpi=200)


if __name__ == "__main__":
    main()
