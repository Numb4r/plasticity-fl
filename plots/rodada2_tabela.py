"""Tabela de decisão e figuras da RODADA 2. Uso (da raiz do projeto):
    .venv/bin/python plots/rodada2_tabela.py [--results results_rodada2] [--saida figures]
Gera <saida>/rodada2_tabela_decisao.md e <saida>/rodada2_*.png (dpi 200). Sem resultados: mensagem e código 1,
sem criar pasta nem figura. Regras e ambiguidades: ver plots/rodada2_analise.py."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rodada2_analise as R  # noqa: E402

ESTILOS = {
    "fedavg_iid": dict(color="#000000", linestyle="-", marker="o"),
    "media_sem_particao": dict(color="#0072b2", linestyle="-.", marker="^"),
    "sem_media_k1": dict(color="#d55e00", linestyle="--", marker="s"),
    "central_aprendizado": dict(color="#cc79a7", linestyle=(0, (5, 1)), marker="v"),
    "central_HT": dict(color="#009e73", linestyle=":", marker="D"),
    "central_lookahead": dict(color="#e69f00", linestyle=(0, (3, 1, 1, 1)), marker="P"),
    "fedavg_dirichlet_a0.1": dict(color="#882255", linestyle=(0, (1, 1)), marker="X"),
    "fedavg_dirichlet_a1": dict(color="#56b4e9", linestyle=(0, (4, 2, 1, 2)), marker="*"),
    "fedavg_dirichlet_a100": dict(color="#999933", linestyle=(0, (6, 2)), marker="h"),
}
LABEL_GAP = "gap (acurácia contínua − acurácia da rede nova, fração)"


def f3(x):
    return "n/d" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ic(r):
    if r["n"] == 0:
        return "sem dados"
    return f"{f3(r['media'])} [{f3(r['inf'])}, {f3(r['sup'])}] (n={r['n']})"


# ---------------------------------------------------------------- figuras

def _salvar(fig, saida, nome):
    caminho = os.path.join(saida, nome)
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _linha(ax, c, serie, **kw):
    est = ESTILOS.get(serie, dict(color="gray", linestyle="-", marker="o"))
    ax.plot(c["tarefa"], c["media"], label=R.rotulo(serie), linewidth=1.5, markersize=5, **est, **kw)
    ok = c.dropna(subset=["inf"])
    if len(ok):
        ax.fill_between(ok["tarefa"], ok["inf"], ok["sup"], color=est["color"], alpha=0.12, linewidth=0)


def fig_gap(df, saida):
    series = list(R.COND_T30)
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    desenhou = False
    for ax, t in zip(eixos, R.TIPOS_AB):
        for s in series:
            c = R.curva(df, s, t, "gap")
            if len(c):
                _linha(ax, c, s)
                desenhou = True
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"tarefa {t}")
        ax.set_xlabel("tarefa (índice)")
    if not desenhou:
        plt.close(fig)
        return None
    eixos[0].set_ylabel(LABEL_GAP)
    h, l = eixos[0].get_legend_handles_labels()
    fig.suptitle("Gap por tarefa e condição, rodada 2 (m*=3, T=30, média e IC 95% bootstrap, sementes 6 a 10)")
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    return _salvar(fig, saida, "rodada2_gap_por_tarefa.png")


def fig_het(df, saida):
    fig, eixos = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    desenhou = False
    hh = R.h_het(df)
    for j, t in enumerate(R.TIPOS_AB):
        r = hh["tarefas"][t]
        ax = eixos[0, j]
        pts = [(a, r["gap"][a]) for a in R.ALFAS if r["gap"][a]["n"]]
        if pts:
            desenhou = True
            ax.errorbar([p[0] for p in pts], [p[1]["media"] for p in pts],
                        yerr=[[p[1]["media"] - p[1]["inf"] for p in pts], [p[1]["sup"] - p[1]["media"] for p in pts]],
                        color="black", linestyle="-", marker="o", capsize=4, label="gap")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"tarefa {t}: gap (H_het: {r['veredito']})", fontsize=9)
        ax.set_ylabel(LABEL_GAP, fontsize=8)
        ax2 = eixos[1, j]
        for chave, rot, est in (("acc_cont", "acc_cont (rede contínua)", dict(color="black", linestyle="-", marker="o")),
                                ("acc_nova", "acc_nova (rede nova)", dict(color="#d55e00", linestyle="--", marker="s"))):
            pts = [(a, r[chave][a]) for a in R.ALFAS if r[chave][a]["n"]]
            if pts:
                ax2.errorbar([p[0] for p in pts], [p[1]["media"] for p in pts],
                             yerr=[[p[1]["media"] - p[1]["inf"] for p in pts],
                                   [p[1]["sup"] - p[1]["media"] for p in pts]], capsize=4, label=rot, **est)
        ax2.set_xscale("log")
        ax2.set_xticks(list(R.ALFAS))
        ax2.set_xticklabels(["0,1", "1", "100"])
        ax2.set_xlabel("α de Dirichlet (menor = mais heterogêneo)")
        ax2.set_ylabel("acurácia de treino (fração)")
        ax2.legend(fontsize=8)
    if not desenhou:
        plt.close(fig)
        return None
    fig.suptitle("H_het por α, rodada 2 (fedavg_dirichlet, média e IC 95%, janela das 5 últimas tarefas)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _salvar(fig, saida, "rodada2_h_het.png")


def fig_longo(df, col, saida):
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    desenhou = False
    for ax, t in zip(eixos, R.TIPOS_AB):
        for i, c in enumerate(R.COND_LONGO):
            cu = R.curva(df, c, t, "gap", "T100")
            if cu.empty:
                continue
            desenhou = True
            _linha(ax, cu, c)
            est = ESTILOS[c]
            idx = [v for v in col[(c, t)]["por_semente"].values() if v is not None]
            if idx:  # tarefa de colapso de cada semente, marcada sobre a faixa y = -0,10 (deslocada por condição)
                ax.plot(idx, [R.LIM_COLAPSO - 0.015 * (i + 1)] * len(idx), linestyle="none", marker=est["marker"],
                        markersize=9, markerfacecolor="white", markeredgecolor=est["color"], markeredgewidth=1.8)
        ax.axhline(R.LIM_COLAPSO, color="gray", linestyle=(0, (2, 2)), linewidth=1)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_title(f"tarefa {t}")
        ax.set_xlabel("tarefa (índice)")
    if not desenhou:
        plt.close(fig)
        return None
    eixos[0].set_ylabel(LABEL_GAP)
    h, l = eixos[0].get_legend_handles_labels()
    fig.suptitle("Braço longo (T=100), rodada 2: gap nas referências; marcadores vazados = tarefa de colapso "
                 "por semente (média móvel de 3 referências ≤ −0,10)", fontsize=10)
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return _salvar(fig, saida, "rodada2_longo_colapso.png")


def fig_reaprendizado(rea_tab, saida):
    if not rea_tab:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(rea_tab):
        for dx, chave, tipo, est in ((-0.15, "passos_copia", "cópia", dict(marker="o", markerfacecolor="black")),
                                     (0.15, "passos_nova", "rede nova", dict(marker="s", markerfacecolor="white"))):
            v = r["por_semente"][chave].values
            ax.plot([i + dx] * len(v), v, linestyle="none", marker=est["marker"], markersize=5, alpha=0.5,
                    markerfacecolor=est["markerfacecolor"], markeredgecolor="black")
            ax.plot([i + dx - 0.1, i + dx + 0.1], [r["med_copia" if chave == "passos_copia" else "med_nova"]] * 2,
                    color="black", linewidth=2.5 if tipo == "cópia" else 1.5,
                    linestyle="-" if tipo == "cópia" else "--",
                    label=f"mediana ({tipo})" if i == 0 else None)
    ax.axhline(R.TETO, color="gray", linestyle=(0, (2, 2)), linewidth=1)
    ax.text(len(rea_tab) - 0.5, R.TETO, "teto 1500 (censurado)", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rea_tab)))
    ax.set_xticklabels([r["cond"] for r in rea_tab], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("passos de reaprendizado (nº de passos de treino)")
    ax.set_xlabel("condição (círculo cheio = cópia, quadrado vazado = rede nova; pontos = mediana por semente)",
                  fontsize=8)
    ax.set_title("Reaprendizado em B_fixa, rodada 2 (passos até 90% da acurácia antiga)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _salvar(fig, saida, "rodada2_reaprendizado.png")


def fig_rank(df, saida):
    fig, eixos = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    desenhou = False
    series = list(R.COND_T30) + [R.serie_nome("fedavg_dirichlet", a) for a in R.ALFAS]
    for i, t in enumerate(R.TIPOS_AB):
        for j, (col, rot) in enumerate((("rank_ef", "rank efetivo por entropia (dimensão)"),
                                        ("srank_feat", "srank das features, δ=0,01 (dimensão)"))):
            ax = eixos[i, j]
            for s in series:
                c = R.curva(df, s, t, col)
                if len(c):
                    _linha(ax, c, s)
                    desenhou = True
            ax.set_title(f"tarefa {t}: {col}")
            ax.set_ylabel(rot, fontsize=8)
            if i == 1:
                ax.set_xlabel("tarefa (índice)")
    if not desenhou:
        plt.close(fig)
        return None
    h, l = eixos[0, 0].get_legend_handles_labels()
    fig.suptitle("Rank das ativações da última camada oculta por tarefa, rodada 2 (exploratório; as duas medidas "
                 "juntas)", fontsize=10)
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    return _salvar(fig, saida, "rodada2_rank.png")


# ---------------------------------------------------------------- markdown

def md_metrica(df):
    L = ["## 1. Métrica principal por condição e tarefa (gap médio pareado das 5 últimas tarefas; T=30: 25..29)",
         "", "Média entre sementes [IC 95% bootstrap percentil, 10 000, default_rng(0)] (n sementes).", "",
         "| condição | A | B | B_fixa |", "|---|---|---|---|"]
    series = list(R.COND_T30) + [R.serie_nome("fedavg_dirichlet", a) for a in R.ALFAS]
    for s in series:
        cel = [ic(R.resumo(df, s, t)) if len(R._sub(df, s, t)) else "sem dados" for t in ("A", "B", "B_fixa")]
        L.append(f"| {R.rotulo(s)} | " + " | ".join(cel) + " |")
    return L


def md_h_tempo(df):
    ht = R.h_tempo(df)
    L = ["## 2. H_tempo / H_dist (central_HT contra media_sem_particao)", "",
         f"**Veredito: {ht['veredito']}**", ""]
    L += ["| tarefa | P_vant: inf(msp) > sup(k1) | perda CE nova HT / msp (rel.) | elegível (≤10%) | "
          "acc nova HT − msp (checagem ±0,03) | d = HT − msp, IC | classe de d (δ=0,02) |", "|---|---|---|---|---|---|---|"]
    for t in R.TIPOS_AB:
        x = ht["tarefas"][t]
        L.append(f"| {t} | {x['p_vant']} (msp {ic(x['msp'])}; k1 {ic(x['k1'])}) | "
                 f"{f3(x['loss_ht'])} / {f3(x['loss_msp'])} ({f3(x['rel_ce'])}) | {x['ce_ok']} | "
                 f"{f3(x['dacc'])} ({x['acc_ok']}) | {ic(x['d'])} | {x['classe']} |")
    L += ["", "Critério secundário (literal; sem veredito): média do central_HT dentro do IC de "
          "media_sem_particao; inf(IC msp) acima do sup(IC HT).", "",
          "| tarefa | HT dentro do IC de msp | inf(msp) > sup(HT) |", "|---|---|---|"]
    for t in R.TIPOS_AB:
        x = ht["tarefas"][t]
        L.append(f"| {t} | {x['sec_dentro']} | {x['sec_acima']} |")
    L += ["", "central_lookahead (exploratório, sem veredito): diferença pareada contra media_sem_particao.", "",
          "| tarefa | d = lookahead − msp, IC |", "|---|---|"]
    for t in R.TIPOS_AB:
        x = ht["lookahead"][t]
        L.append(f"| {t} | {ic(x['d']) if not x['falta'] else 'não avaliável (' + x['falta'] + ')'} |")
    return L, ht


def md_h_het(df):
    hh = R.h_het(df)
    L = ["## 3. H_het (fedavg_dirichlet, α = 0,1; 1; 100)", "", f"**Veredito geral: {hh['geral']}**", ""]
    L += ["| tarefa | α | gap | acc_cont | acc_nova |", "|---|---|---|---|---|"]
    for t in R.TIPOS_AB:
        r = hh["tarefas"][t]
        for a in R.ALFAS:
            L.append(f"| {t} | {a:g} | {ic(r['gap'][a])} | {ic(r['acc_cont'][a])} | {ic(r['acc_nova'][a])} |")
    L += ["", "| tarefa | gap(100) − gap(0,1), IC pareado | veredito da tarefa |", "|---|---|---|"]
    for t in R.TIPOS_AB:
        r = hh["tarefas"][t]
        L.append(f"| {t} | {ic(r['d'])} | {r['veredito']} |")
    return L, hh


def md_rank(df):
    L = ["## 4. Rank das ativações (exploratório; SEM critério e SEM veredito)", "",
         "`rank_ef` e `srank_feat` sempre juntos (média das 5 últimas tarefas; média entre sementes e IC 95%). "
         "As duas medidas discordaram na rodada 1 (tarefa A, m=3: rank_ef do fedavg_iid 6,3, abaixo de "
         "central_aprendizado 9,8; srank_feat do fedavg_iid 63, acima de 19,6), e o autor já viu esses números.", "",
         "| tarefa | condição / grupo | rank_ef | srank_feat |", "|---|---|---|---|"]
    for nome, t, a, b in R.rank_tabela(df):
        L.append(f"| {t} | {nome} | {ic(a)} | {ic(b)} |")
    return L


def md_sonda(df):
    L = ["## 5. Sonda v2 (20 passos; média por semente nas tarefas de sonda)", "",
         "dlog_norm = mediana_k[log10(Lc20/Lc0) − log10(Ln20/Ln0)] (recalculada de sonda2_L*_k); "
         "dlog_raw = estatística exploratória sem normalizar (coluna sonda2_dlog_raw).", "",
         "| tarefa | condição | dlog_norm | dlog_raw |", "|---|---|---|---|"]
    for nome, t, a, b in R.sonda_tabela(df):
        L.append(f"| {t} | {nome} | {ic(a)} | {ic(b)} |")
    return L


def md_bfixa(df, rea):
    L = ["## 6. B_fixa (T=30)", "",
         "O gap em B_fixa mistura transferência (mesmo professor) e plasticidade.", "",
         "| condição | gap (5 últimas) | esquecimento final | BWT final |", "|---|---|---|---|"]
    for c, g, e, b in R.bfixa_tabela(df):
        L.append(f"| {c} | {ic(g)} | {ic(e)} | {ic(b)} |")
    L += ["", "Reaprendizado (cópia vs. rede nova; medianas pooled sobre semente × tarefa_atual × tarefa_antiga; "
          "'> 1500' se mais da metade censurada).", "",
          "| condição | n | passos cópia (mediana) | passos nova (mediana) | censurado cópia | censurado nova | "
          "economia = 1 − cópia/nova | no limiar no passo 0 (cópia / nova) |", "|---|---|---|---|---|---|---|---|"]
    for r in R.reaprendizado_tabela(rea):
        L.append(f"| {r['cond']} | {r['n']} | {r['txt_copia']} | {r['txt_nova']} | {r['cens_copia']:.2f} | "
                 f"{r['cens_nova']:.2f} | {r['economia']} | {r['limiar0_copia']:.2f} / {r['limiar0_nova']:.2f} |")
    return L


def md_longo(df):
    col = R.colapso_tabela(df)
    L = ["## 7. Braço longo (T=100, sementes 6 a 8): tarefa de colapso", "",
         "Primeiro ponto, entre as referências t%3==2, em que a média móvel de 3 referências do gap pareado fica "
         "≤ −0,10 (índice da última referência da janela). Mediana [mín, máx] entre as sementes que colapsaram.", "",
         "| tarefa | condição | gap médio (95..99) | colapso por semente | mediana [mín, máx] | fração sem colapso |",
         "|---|---|---|---|---|---|"]
    for t in R.TIPOS_AB:
        for c in R.COND_LONGO:
            x = col[(c, t)]
            gm = ic(R.resumo(df, c, t, braco="T100"))
            if x["falta"]:
                L.append(f"| {t} | {c} | {gm} | não avaliável (falta sementes {','.join(map(str, x['falta']))}) "
                         "| n/d | n/d |")
                continue
            por = ", ".join(f"{s}: {v if v is not None else 'sem colapso'}" for s, v in x["por_semente"].items())
            med = "sem colapso" if x["mediana"] is None else \
                f"{x['mediana']:g} [{x['minimo']}, {x['maximo']}]"
            L.append(f"| {t} | {c} | {gm} | {por} | {med} | {x['frac_sem']:.2f} |")
    j = R.julgar_horizonte(col)
    L += ["", f"**Julgamento da previsão do autor (tarefa A): {j}**"]
    return L, col, j


def md_previsao(ht, hh, j):
    L = ["## 8. Previsão do autor × resultado (calculado, sem interpretar)", "",
         "| hipótese | previsão do autor | resultado calculado |", "|---|---|---|",
         f"| H_tempo | não (HT protege mais que a média entre aprendizes) | {ht['veredito']} |",
         f"| H_dist | não | {ht['veredito']} |",
         f"| H_het | sim (gap piora quando α diminui) | {hh['geral']} |",
         f"| horizonte longo | o FedAvg só atrasa a perda | {j} |"]
    return L


def gerar(results, saida):
    import pandas as pd
    df = R.carregar(results) if os.path.isdir(results) else pd.DataFrame()
    if df.empty:
        print(f"sem resultados em {results}/ (nenhum CSV válido das sementes 6 a 10); nada foi gerado",
              file=sys.stderr)
        return 1
    rea = R.carregar_reaprendizado(results)
    os.makedirs(saida, exist_ok=True)
    corpo_t, ht = md_h_tempo(df)
    corpo_h, hh = md_h_het(df)
    corpo_l, col, j = md_longo(df)
    L = ["# Tabela de decisão da rodada 2", "",
         f"Fonte: `{results}/` (sementes 6 a 10; semente 1000 nunca entra). Regras literais de PREREGISTRO.md "
         "(rodada 2 e blocos de 2026-10-21); ambiguidades em `plots/rodada2_analise.py`. "
         "Vereditos: texto calculado, sem interpretação.", ""]
    L += md_metrica(df) + [""] + corpo_t + [""] + corpo_h + [""] + md_rank(df) + [""] + md_sonda(df) + [""]
    L += md_bfixa(df, rea) + [""] + corpo_l + [""] + md_previsao(ht, hh, j) + [""]
    with open(os.path.join(saida, "rodada2_tabela_decisao.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    gerados = [fig_gap(df, saida), fig_het(df, saida), fig_longo(df, col, saida),
               fig_reaprendizado(R.reaprendizado_tabela(rea), saida), fig_rank(df, saida)]
    print("gerado:", os.path.join(saida, "rodada2_tabela_decisao.md"))
    for g in gerados:
        print("gerado:" if g else "sem dados para a figura (não gerada)", g or "")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Tabela de decisão e figuras da rodada 2")
    p.add_argument("--results", default="results_rodada2")
    p.add_argument("--saida", default="figures")
    a = p.parse_args(argv)
    return gerar(a.results, a.saida)


if __name__ == "__main__":
    sys.exit(main())
