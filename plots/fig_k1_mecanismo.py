"""figures/fig_k1_mecanismo.png: no nível m=10, sem_media_k1 vs fedavg_iid, por tarefa (A, B, B_disjunta).

Linhas: dormentes da última camada oculta (tau=0 e tau=0,025), norma dos parâmetros, rank efetivo.
Objetivo: ver se o colapso do K=1 é o mecanismo clássico (neurônios mortos, perda de rank) ou outro
(explosão de norma). Média ± IC 95% bootstrap entre sementes."""
import os

import matplotlib.pyplot as plt

import _comum
import analise

NIVEL = 10
CONDS = ("fedavg_iid", "sem_media_k1")
LINHAS = [("dorm_tau0_l2", "dormentes, tau=0 (fração da camada)"),
          ("dorm_tau0025_l2", "dormentes, tau=0,025 (fração da camada)"),
          ("norma", "norma L2 dos parâmetros"),
          ("rank_ef", "rank efetivo (adimensional)")]


def main():
    a = _comum.args_padrao(__doc__)
    df = analise.carregar(a.results)
    os.makedirs(a.saida, exist_ok=True)
    fig, axs = plt.subplots(len(LINHAS), 3, figsize=(15, 12), sharex="col")
    for i, (col, rotulo) in enumerate(LINHAS):
        for j, tipo in enumerate(analise.TIPOS):
            ax = axs[i, j]
            tem = False
            for cond in CONDS:
                c = analise.curva(df, cond, tipo, col, NIVEL) if len(df) and col in df.columns else None
                if c is None or c.empty:
                    continue
                est = analise.ESTILOS[cond]
                ax.plot(c["tarefa"], c["media"], label=cond, linewidth=1.5, markersize=5, **est)
                ok = c.dropna(subset=["inf"])
                if len(ok):
                    ax.fill_between(ok["tarefa"], ok["inf"], ok["sup"], color=est["color"], alpha=0.15,
                                    linewidth=0)
                tem = True
            if not tem:
                ax.text(0.5, 0.5, "sem dados", transform=ax.transAxes, ha="center")
            if i == 0:
                ax.set_title(f"Tarefa {tipo}")
            if j == 0:
                ax.set_ylabel(rotulo, fontsize=9)
            if i == len(LINHAS) - 1:
                ax.set_xlabel("tarefa (índice)")
            ax.grid(alpha=0.3)
    _comum.legenda(axs.ravel())
    fig.suptitle("Mecanismo do colapso do K=1 no nível m=10: neurônios mortos e perda de rank (mecanismo clássico)\n"
                 "ou outro (explosão de norma)? sem_media_k1 vs fedavg_iid (média e IC 95% bootstrap entre sementes)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(a.saida, "fig_k1_mecanismo.png"), dpi=200)


if __name__ == "__main__":
    main()
