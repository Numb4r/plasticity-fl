"""figures/fig_k1_mecanismo.png (m=10) e fig_k1_mecanismo_mstar.png (m*, se m* existir e for != 10).

Condições: sem_media_k1, fedavg_iid, central_passos, central_aprendizado. Colunas: tarefas A, B, B_disjunta.
Linhas: (1) dormentes tau=0 e tau=0,025 da última camada oculta, norma, rank_ef (painéis originais);
(2) DIAGNÓSTICOS (Desvios v3.1, protocolo de Klein et al. 2026): norma dos pesos por camada (fim da tarefa),
norma L2 do gradiente por camada (início), srank do gradiente (início), maior autovalor da Hessiana (início e fim),
e acc_cont das mesmas condições para a comparação temporal. A rede nova é a média das 4 condições (cinza, tracejada).
Média ± IC 95% bootstrap entre sementes. Métricas diagnósticas: não entram em regra de decisão nem na tabela."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import _comum
import analise

NIVEL = 10
CONDS = ("sem_media_k1", "fedavg_iid", "central_passos", "central_aprendizado")
CINZA = "#7f7f7f"
# (coluna do contínuo, coluna da rede nova ou None, rótulo do eixo y, é Hessiana?)
ANTIGAS = [("dorm_tau0_l2", None, "dormentes, tau=0\n(fração da camada)"),
           ("dorm_tau0025_l2", None, "dormentes, tau=0,025\n(fração da camada)"),
           ("norma", None, "norma L2 dos parâmetros"),
           ("rank_ef", None, "rank efetivo (adimensional)")]
DIAG = ([(f"fim_norma_l{k}_w", f"nova_fim_norma_l{k}_w", f"norma L2 dos pesos l{k}\n(fim da tarefa)", False)
         for k in (1, 2, 3)]
        + [(f"ini_grad_l2_l{k}_w", f"nova_ini_grad_l2_l{k}_w", f"norma L2 do gradiente, pesos l{k}\n(início da tarefa)",
            False) for k in (1, 2, 3)]
        + [("ini_grad_srank", "nova_ini_grad_srank", "srank do gradiente\n(início da tarefa, adimensional)", False),
           ("ini_hess_max", "nova_ini_hess_max", "maior autovalor da Hessiana\n(início da tarefa)", True),
           ("fim_hess_max", "nova_fim_hess_max", "maior autovalor da Hessiana\n(fim da tarefa)", True),
           ("acc_cont", None, "acc_cont (fração)\n(referência temporal)", False)])
SEM_DIAG = "sem diagnósticos"


def _tem(df, col, nivel):
    if df.empty or col not in df.columns:
        return False
    d = df[(df["nivel"] == nivel) & df["cond"].isin(CONDS)]
    return bool(d[col].notna().any())


def curva_nova(df, col, tipo, nivel):
    """Rede nova: por semente, média entre as condições presentes; depois bootstrap entre sementes."""
    d = df[(df["tipo"] == tipo) & (df["nivel"] == nivel) & df["cond"].isin(CONDS)].dropna(subset=[col])
    linhas = []
    por_sem = d.groupby(["tarefa", "semente"])[col].mean().reset_index()
    for t, g in por_sem.groupby("tarefa"):
        m, lo, hi = analise.bootstrap_ic(g[col].values)
        linhas.append(dict(tarefa=t, media=m, inf=lo, sup=hi))
    return pd.DataFrame(linhas, columns=["tarefa", "media", "inf", "sup"])


def nao_convergiu(df, cond, tipo, nivel, col):
    """Tarefas em que alguma semente tem hess_conv == 0 (com o autovalor medido)."""
    cc = col.replace("hess_max", "hess_conv")
    if cc not in df.columns:
        return set()
    d = df[(df["cond"] == cond) & (df["tipo"] == tipo) & (df["nivel"] == nivel)]
    d = d[d[col].notna() & (d[cc] == 0)]
    return set(d["tarefa"])


def _faixa(ax, c, cor, alpha):
    ok = c.dropna(subset=["inf"])
    if len(ok):
        ax.fill_between(ok["tarefa"], ok["inf"], ok["sup"], color=cor, alpha=alpha, linewidth=0)


def painel(ax, df, tipo, nivel, col, col_nova, hess, diag):
    if diag and not _tem(df, col, nivel):
        ax.text(0.5, 0.5, SEM_DIAG, transform=ax.transAxes, ha="center", color=CINZA)
        return
    tem = False
    for cond in CONDS:
        c = analise.curva(df, cond, tipo, col, nivel)
        if c.empty:
            continue
        est = analise.ESTILOS[cond]
        ax.plot(c["tarefa"], c["media"], linewidth=1.4, markersize=4.5, **est)
        _faixa(ax, c, est["color"], 0.15)
        if hess:
            nc = nao_convergiu(df, cond, tipo, nivel, col)
            pts = c[c["tarefa"].isin(nc)]
            if len(pts):   # marcador vazado por cima do ponto: power iteration não convergiu
                ax.plot(pts["tarefa"], pts["media"], linestyle="none", marker=est["marker"], markersize=8,
                        markerfacecolor="white", markeredgecolor=est["color"], markeredgewidth=1.2, zorder=5)
        tem = True
    if col_nova and col_nova in df.columns:
        cn = curva_nova(df, col_nova, tipo, nivel)
        if len(cn):
            ax.plot(cn["tarefa"], cn["media"], color=CINZA, linestyle="--", linewidth=1.3, marker="x",
                    markersize=4, zorder=1)
            _faixa(ax, cn, CINZA, 0.15)
    if not tem:
        ax.text(0.5, 0.5, "sem dados", transform=ax.transAxes, ha="center")


def legenda_fig(fig, com_nova, com_hess):
    hs = [Line2D([], [], linewidth=1.4, markersize=5, label=c, **analise.ESTILOS[c]) for c in CONDS]
    if com_nova:
        hs.append(Line2D([], [], color=CINZA, linestyle="--", marker="x", markersize=4,
                         label="rede nova (média das 4 condições)"))
    if com_hess:
        hs.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor="black",
                         markersize=8, label="marcador vazado: >=1 semente com hess_conv=0 (não convergiu)"))
    fig.legend(handles=hs, fontsize=8, loc="lower center", ncol=3)


def figura(df, nivel, aviso, saida, nome):
    linhas = [(c, n, r, False, False) for c, n, r in ANTIGAS] + [(c, n, r, h, True) for c, n, r, h in DIAG]
    fig, axs = plt.subplots(len(linhas), 3, figsize=(15, 2.3 * len(linhas) + 1.5), sharex="col")
    for i, (col, col_nova, rot, hess, diag) in enumerate(linhas):
        for j, tipo in enumerate(analise.TIPOS):
            ax = axs[i, j]
            painel(ax, df, tipo, nivel, col, col_nova, hess, diag)
            if i == 0:
                ax.set_title(f"Tarefa {tipo}")
            if j == 0:
                ax.set_ylabel(rot, fontsize=8)
            if i == len(linhas) - 1:
                ax.set_xlabel("tarefa (índice)")
            ax.grid(alpha=0.3)
    legenda_fig(fig, any(_tem(df, n, nivel) for _, n, _, _, d in linhas if n), any(h for *_, h, _ in linhas))
    t = (f"Mecanismo do colapso do K=1 no nível m={nivel}: perda de rank/neurônios mortos ou crescimento de norma e curvatura?\n"
         "Linhas inferiores: diagnósticos exploratórios (Klein et al. 2026), sem regra de decisão. "
         "Média e IC 95% bootstrap entre sementes")
    fig.suptitle(t + ("\n" + aviso if aviso else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97 - (0.01 if aviso else 0)))
    fig.savefig(os.path.join(saida, nome), dpi=200)
    plt.close(fig)


def main():
    a = _comum.args_padrao(__doc__)
    df = analise.carregar(a.results)
    _comum.exigir_resultados(df, a.results)
    regime = analise.carregar_regime(a.results)
    os.makedirs(a.saida, exist_ok=True)
    m_star = regime["m_star"]
    aviso = ""
    if m_star is None:
        motivo = "regime sem perda (m* nulo)" if regime["presente"] else "regime.json ausente"
        aviso = f"AVISO: {motivo}; figura só em m=10, sem nível m*"
    figura(df, NIVEL, aviso, a.saida, "fig_k1_mecanismo.png")
    if m_star is not None and m_star != NIVEL:
        figura(df, m_star, "", a.saida, "fig_k1_mecanismo_mstar.png")


if __name__ == "__main__":
    main()
