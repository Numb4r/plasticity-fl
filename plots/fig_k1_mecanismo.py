"""figures/fig_k1_mecanismo.png (m=10) e fig_k1_mecanismo_mstar.png (m*, se m* existir e for != 10).

Condições: sem_media_k1, fedavg_iid, central_passos, central_aprendizado. Colunas: tarefas A, B, B_disjunta.
Linhas: (1) dormentes tau=0 e tau=0,025 da última camada oculta, norma, rank_ef (painéis originais);
(2) DIAGNÓSTICOS (Desvios v3.1, protocolo de Klein et al. 2026): norma dos pesos por camada (fim da tarefa),
norma L2 do gradiente por camada (início), srank do gradiente (início), maior autovalor da Hessiana (início e fim),
e acc_cont das mesmas condições para a comparação temporal.
Rede nova: nas linhas de INÍCIO (nova_ini_*) é idêntica nas 4 condições (mesma semente de init_nova, sem SP em
nenhuma): uma linha cinza (média). Nas linhas de FIM (nova_fim_*) e em acc_cont (acc_nova) a rede nova é treinada
com o protocolo da própria condição: uma linha fina tracejada POR CONDIÇÃO (cor/marcador da condição).
Média ± IC 95% bootstrap entre sementes. Métricas diagnósticas: não entram em regra de decisão nem na tabela.
Segunda figura (fig_k1_por_semente*.png): uma linha fina por semente, sem média, para fim_hess_max, fim_norma_l2_w
e acc_cont (ordem temporal semente a semente)."""
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
           ("acc_cont", "acc_nova", "acc_cont (fração)\n(tracejada: acc_nova da condição)", False)])
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


def nao_convergiu_media(df, tipo, nivel, col):
    """Tarefas em que alguma condição/semente da rede nova tem hess_conv == 0 (linha cinza, só nova_ini_*)."""
    nc = set()
    for cond in CONDS:
        nc |= nao_convergiu(df, cond, tipo, nivel, col)
    return nc


def _vazado(ax, pts, marker, cor, tam=8):
    ax.plot(pts["tarefa"], pts["media"], linestyle="none", marker=marker, markersize=tam,
            markerfacecolor="white", markeredgecolor=cor, markeredgewidth=1.2, zorder=5)


def painel(ax, df, tipo, nivel, col, col_nova, hess, diag):
    """Devolve True se desenhou algum marcador vazado (para a entrada de legenda)."""
    if diag and not _tem(df, col, nivel):
        ax.text(0.5, 0.5, SEM_DIAG, transform=ax.transAxes, ha="center", color=CINZA)
        return False
    tem, vazado = False, False
    por_cond = bool(col_nova) and not col_nova.startswith("nova_ini_")
    for cond in CONDS:
        c = analise.curva(df, cond, tipo, col, nivel)
        if c.empty:
            continue
        est = analise.ESTILOS[cond]
        ax.plot(c["tarefa"], c["media"], linewidth=1.4, markersize=4.5, **est)
        _faixa(ax, c, est["color"], 0.15)
        if hess:
            pts = c[c["tarefa"].isin(nao_convergiu(df, cond, tipo, nivel, col))]
            if len(pts):   # marcador vazado por cima do ponto: power iteration não convergiu
                _vazado(ax, pts, est["marker"], est["color"])
                vazado = True
        if por_cond and col_nova in df.columns:   # rede nova treinada com o protocolo desta condição
            cn = analise.curva(df, cond, tipo, col_nova, nivel)
            if len(cn):
                ax.plot(cn["tarefa"], cn["media"], color=est["color"], linestyle=(0, (4, 2)), linewidth=0.9,
                        marker=est["marker"], markersize=3, alpha=0.85, zorder=2)
                if hess:
                    pn = cn[cn["tarefa"].isin(nao_convergiu(df, cond, tipo, nivel, col_nova))]
                    if len(pn):
                        _vazado(ax, pn, est["marker"], est["color"], 6)
                        vazado = True
        tem = True
    if col_nova and not por_cond and col_nova in df.columns:   # início: idêntica nas 4 condições -> média única
        cn = curva_nova(df, col_nova, tipo, nivel)
        if len(cn):
            ax.plot(cn["tarefa"], cn["media"], color=CINZA, linestyle="--", linewidth=1.3, marker="x",
                    markersize=4, zorder=1)
            _faixa(ax, cn, CINZA, 0.15)
            if hess:
                pn = cn[cn["tarefa"].isin(nao_convergiu_media(df, tipo, nivel, col_nova))]
                if len(pn):
                    _vazado(ax, pn, "o", CINZA, 6)
                    vazado = True
    if not tem:
        ax.text(0.5, 0.5, "sem dados", transform=ax.transAxes, ha="center")
    return vazado


NOTAS = ("Norma dos parâmetros por camada: só pesos (os vieses estão nos CSVs: norma_l*_b). "
         "hess_max é o autovalor de maior módulo COM SINAL: a média entre sementes pode ficar perto de 0 se o sinal muda.\n"
         "ini_* na tarefa t é medido ANTES de treinar t (estado após t-1 e após o shrink-and-perturb); "
         "fim_* e acc_cont, DEPOIS de treinar t.")
NOTA_NOVA = ("Rede nova no INÍCIO (nova_ini_*): idêntica nas 4 condições (mesma semente de init_nova, sem SP em nenhuma) -> "
             "uma linha cinza (média). No FIM (nova_fim_*) e em acc_cont (acc_nova): treinada com o protocolo de cada "
             "condição -> uma linha fina tracejada por condição.")


def legenda_fig(fig, com_media, com_cond, com_vazado, notas=True):
    hs = [Line2D([], [], linewidth=1.4, markersize=5, label=c, **analise.ESTILOS[c]) for c in CONDS]
    if com_cond:
        hs.append(Line2D([], [], color="black", linestyle=(0, (4, 2)), linewidth=0.9, marker="o", markersize=3,
                         label="rede nova, por condição (fim e acc_nova): tracejada fina, cor/marcador da condição"))
    if com_media:
        hs.append(Line2D([], [], color=CINZA, linestyle="--", marker="x", markersize=4,
                         label="rede nova no início (média das 4: idêntica entre condições)"))
    if com_vazado:
        hs.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor="black",
                         markersize=8, label="marcador vazado: >=1 semente com hess_conv=0 (não convergiu)"))
    fig.legend(handles=hs, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.045 if notas else 0.0))
    if notas:
        fig.text(0.5, 0.005, NOTA_NOVA + "\n" + NOTAS if com_media or com_cond else NOTAS, ha="center",
                 va="bottom", fontsize=7)


def figura(df, nivel, aviso, saida, nome):
    linhas = [(c, n, r, False, False) for c, n, r in ANTIGAS] + [(c, n, r, h, True) for c, n, r, h in DIAG]
    fig, axs = plt.subplots(len(linhas), 3, figsize=(15, 2.3 * len(linhas) + 2.6), sharex="col")
    vaz = False
    for i, (col, col_nova, rot, hess, diag) in enumerate(linhas):
        for j, tipo in enumerate(analise.TIPOS):
            ax = axs[i, j]
            vaz = painel(ax, df, tipo, nivel, col, col_nova, hess, diag) or vaz
            if i == 0:
                ax.set_title(f"Tarefa {tipo}")
            if j == 0:
                ax.set_ylabel(rot, fontsize=8)
            if i == len(linhas) - 1:
                ax.set_xlabel("tarefa (índice)")
            ax.grid(alpha=0.3)
    legenda_fig(fig, any(_tem(df, n, nivel) for _, n, *_ in linhas if n and n.startswith("nova_ini_")),
                any(_tem(df, n, nivel) for _, n, *_ in linhas if n and not n.startswith("nova_ini_")), vaz)
    t = (f"Mecanismo do colapso do K=1 no nível m={nivel}: perda de rank/neurônios mortos ou crescimento de norma e curvatura?\n"
         "Linhas inferiores: diagnósticos exploratórios (Klein et al. 2026), sem regra de decisão. "
         "Média e IC 95% bootstrap entre sementes")
    fig.suptitle(t + ("\n" + aviso if aviso else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97 - (0.01 if aviso else 0)))
    fig.savefig(os.path.join(saida, nome), dpi=200)
    plt.close(fig)


SEMENTE_LINHAS = (("fim_hess_max", "maior autovalor da Hessiana\n(fim da tarefa)", True, True),
                  ("fim_norma_l2_w", "norma L2 dos pesos l2\n(fim da tarefa)", False, True),
                  ("acc_cont", "acc_cont (fração)", False, False))


def painel_semente(ax, df, cond, tipo, nivel, col, hess, diag):
    """Uma linha fina por semente (sem média). Devolve True se desenhou marcador vazado."""
    if diag and not _tem(df, col, nivel):
        ax.text(0.5, 0.5, SEM_DIAG, transform=ax.transAxes, ha="center", color=CINZA)
        return False
    est = analise.ESTILOS[cond]
    d = df[(df["cond"] == cond) & (df["tipo"] == tipo) & (df["nivel"] == nivel)].dropna(subset=[col]) \
        if col in df.columns else df.iloc[0:0]
    if d.empty:
        ax.text(0.5, 0.5, "sem dados", transform=ax.transAxes, ha="center", color=CINZA)
        return False
    vaz = False
    cc = col.replace("hess_max", "hess_conv")
    for _, g in d.groupby("semente"):
        g = g.sort_values("tarefa")
        ax.plot(g["tarefa"], g[col], color=est["color"], linestyle=est["linestyle"], marker=est["marker"],
                linewidth=0.7, markersize=2.5, alpha=0.75)
        if hess and cc in g.columns:
            nc = g[g[cc] == 0]
            if len(nc):
                ax.plot(nc["tarefa"], nc[col], linestyle="none", marker=est["marker"], markersize=5,
                        markerfacecolor="white", markeredgecolor=est["color"], markeredgewidth=0.9, zorder=5)
                vaz = True
    if hess:
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    return vaz


def figura_semente(df, nivel, aviso, saida, nome):
    """Linhas agrupadas por condição (hess, norma, acc_cont) x colunas A, B, B_disjunta; uma linha por semente."""
    linhas = [(c, ln) for c in CONDS for ln in SEMENTE_LINHAS]
    fig, axs = plt.subplots(len(linhas), 3, figsize=(15, 1.9 * len(linhas) + 2.4), sharex="col")
    vaz = False
    for i, (cond, (col, rot, hess, diag)) in enumerate(linhas):
        for j, tipo in enumerate(analise.TIPOS):
            ax = axs[i, j]
            vaz = painel_semente(ax, df, cond, tipo, nivel, col, hess, diag) or vaz
            if i == 0:
                ax.set_title(f"Tarefa {tipo}")
            if j == 0:
                ax.set_ylabel(f"{cond}\n{rot}", fontsize=7)
            if i == len(linhas) - 1:
                ax.set_xlabel("tarefa (índice)")
            ax.grid(alpha=0.3)
    hs = [Line2D([], [], linewidth=0.8, markersize=3, label=f"{c} (uma linha por semente)", **analise.ESTILOS[c])
          for c in CONDS]
    if vaz:
        hs.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor="black",
                         markersize=6, label="marcador vazado: hess_conv=0 (não convergiu)"))
    fig.legend(handles=hs, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.0, NOTAS, ha="center", va="bottom", fontsize=7)
    t = (f"Ordem temporal por semente no nível m={nivel}: norma e Hessiana sobem ANTES da queda de acc_cont?\n"
         "Uma linha fina por semente (sem média entre sementes); diagnósticos exploratórios, sem regra de decisão")
    fig.suptitle(t + ("\n" + aviso if aviso else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97 - (0.01 if aviso else 0)))
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
    figura_semente(df, NIVEL, aviso, a.saida, "fig_k1_por_semente.png")
    if m_star is not None and m_star != NIVEL:
        figura(df, m_star, "", a.saida, "fig_k1_mecanismo_mstar.png")
        figura_semente(df, m_star, "", a.saida, "fig_k1_por_semente_mstar.png")


if __name__ == "__main__":
    main()
