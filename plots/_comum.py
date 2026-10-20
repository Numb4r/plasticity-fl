"""Utilidades compartilhadas pelos scripts de figuras."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise  # noqa: E402


def args_padrao(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--results", default="results")
    p.add_argument("--saida", default="figures")
    return p.parse_args()


COLUNAS_METRICA = {"gap", "acc_nova"}   # sem elas não há métrica principal


def exigir_resultados(df, results):
    """Sem CSVs válidos (ou sem as colunas da métrica): mensagem clara e saída com código 1,
    antes de criar qualquer figura ou pasta de saída (ver README_plots.md)."""
    if df.empty or not COLUNAS_METRICA <= set(df.columns):
        print(f"sem resultados em {results}/ (nenhum CSV válido das sementes 1 a 5 com as colunas "
              f"{sorted(analise.COLUNAS_CHAVE | COLUNAS_METRICA)}); nada foi gerado", file=sys.stderr)
        sys.exit(1)


def contexto(a):
    """Carrega CSVs e regime; devolve (df, regime, nivel de exibição, aviso). Sai com código 1 se vazio."""
    df = analise.carregar(a.results)
    exigir_resultados(df, a.results)
    regime = analise.carregar_regime(a.results)
    nivel, aviso = analise.nivel_exibicao(df, regime)
    return df, regime, nivel, aviso


def titulo(base, nivel, aviso):
    """Título com o nível usado e, se houver, o aviso (m* nulo ou ausente)."""
    t = f"{base} (nível m={nivel})"
    return t + ("\n" + aviso if aviso else "")


def desenhar(ax, df, coluna, tipo, nivel):
    """Uma linha por condição: média por tarefa + faixa de IC 95% (se n>=2). Devolve True se desenhou algo."""
    desenhou = False
    for cond in analise.CONDICOES:
        c = analise.curva(df, cond, tipo, coluna, nivel)
        if c.empty:
            continue
        est = analise.ESTILOS[cond]
        ax.plot(c["tarefa"], c["media"], label=cond, linewidth=1.5, markersize=5, **est)
        ok = c.dropna(subset=["inf"])
        if len(ok):
            ax.fill_between(ok["tarefa"], ok["inf"], ok["sup"], color=est["color"], alpha=0.15, linewidth=0)
        desenhou = True
    if not desenhou:
        ax.text(0.5, 0.5, "sem dados", transform=ax.transAxes, ha="center")
    return desenhou


def legenda(eixos):
    """Legenda única embaixo da figura, com os handles do primeiro eixo que tenha curvas
    (o painel B pode estar vazio no modo quick)."""
    for ax in eixos:
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.figure.legend(h, l, fontsize=8, loc="lower center", ncol=4)
            return
