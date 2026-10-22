"""Tabela de decisão e figura da RODADA 2b. Uso (da raiz do projeto):
    .venv/bin/python plots/rodada2b_tabela.py --results results_rodada2b --saida <arquivo.md> [--figura <arquivo.png>]
Sem --figura, o PNG (dpi 200) vai ao lado do .md com o sufixo _gap.png. Sem resultados: mensagem e código 1, sem
criar nada. Regras e decisões de leitura: plots/rodada2b_analise.py (que reaproveita a rodada 2)."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rodada2_analise as R  # noqa: E402
import rodada2_tabela as T2  # noqa: E402
import rodada2b_analise as B  # noqa: E402

ESTILOS = {c: T2.ESTILOS[c] for c in ("media_sem_particao", "sem_media_k1")}
ESTILOS[B.HTCAL2] = dict(color="#117733", linestyle=(0, (1, 2, 5, 2)), marker="d")
LABEL_GAP = T2.LABEL_GAP
f3, ic = T2.f3, T2.ic


def _mu(sel):
    if not sel:
        return "n/d"
    return f"{sel['mu']:g}" + (" [borda]" if sel.get("borda") else "")


def _notas(sel):
    """Notas da seleção de μ como tabela (μ: nota) numa célula; aceita dict, lista ou texto."""
    n = (sel or {}).get("notas")
    if isinstance(n, dict):
        return "; ".join(f"μ={m}: {f3(v)}" for m, v in n.items())
    if isinstance(n, list):
        return ", ".join(map(str, n))
    return str(n) if n else ""


def md_calibracao(calib):
    L = ["## 1. Calibração de R (semente 1000) e μ", "",
         "R = rodadas por tarefa (grade 75 a 150); erro_rel = |loss_ht − alvo| / alvo; ok = erro_rel ≤ 10%; "
         "alvo = `loss_nova` do media_sem_particao; [borda] = μ na borda da grade (mantido).", "",
         "| tarefa | R | erro_rel | ok | alvo (loss_nova msp) | loss_ht | μ | notas |", "|---|---|---|---|---|---|---|---|"]
    for t in B.TIPOS:
        c = (calib or {}).get(t, {})
        k, sel = c.get("calibracao"), c.get("sel")
        notas = _notas(sel)
        if k:
            cel = [str(k.get("rodadas", "n/d")), f3(k.get("erro_rel")), str(k.get("ok")),
                   f3(k.get("alvo_loss_nova_msp")), f3(k.get("loss_ht"))]
        else:
            cel = [f"n/d (falta calibracao_HT2_{t}_m3.json)", "n/d", "n/d", "n/d", "n/d"]
        L.append(f"| {t} | " + " | ".join(cel) + f" | {_mu(sel)} | {notas} |")
    return L


def md_metrica(df):
    L = ["## 2. Métrica principal por condição e tarefa (gap pareado médio das 5 últimas tarefas: 25 a 29)", "",
         "Média entre sementes [IC 95% bootstrap percentil, 10 000, default_rng(0)] (n sementes).", "",
         "| condição | A | B |", "|---|---|---|"]
    for c in B.CONDICOES_2B:
        cel = [ic(B.resumo_serie(df, c, t)) if len(R._sub(df, c, t)) else "sem dados" for t in B.TIPOS]
        L.append(f"| {c} | " + " | ".join(cel) + " |")
    return L


def md_diferenca(ht):
    nao_av = str(ht["veredito"]).startswith("não avaliável")
    L = [f"## 3. Diferença pareada {B.HTCAL2} − media_sem_particao (δ = 0,02)", "",
         f"**Veredito geral (H_tempo e H_dist, o mesmo): {ht['veredito']}**", "",
         "Classe: dentro = IC todo em [−δ, +δ]; acima (\"HT acima\") = inf > +δ; abaixo (H_dist) = sup < −δ; "
         "senão intermediário." + (" Veredito \"não avaliável\": a classe por tarefa é só descritiva (não decisória)."
                                   if nao_av else ""), "",
         "Nota: a perda CE da rede nova está igualada; a acurácia pode não estar; gap = acc_cont − acc_nova.", "",
         f"| tarefa | d (gap), IC 95% | classe | diferença média de acc_nova ({B.HTCAL2} − msp, 5 últimas "
         "tarefas, média entre sementes) | dentro de ±0,03 |", "|---|---|---|---|---|"]
    for t in B.TIPOS:
        x = ht["tarefas"][t]
        cl = f"{x['classe']} (não decisória)" if nao_av else x["classe"]
        dentro = "n/d" if x["acc_ok"] is None else ("sim" if x["acc_ok"] else "não")
        L.append(f"| {t} | {ic(x['d'])} | {cl} | {f3(x['dacc'])} | {dentro} |")
    L += ["", "Critério secundário (literal; só reportado, sem veredito): média de " + B.HTCAL2 +
          " dentro do IC de media_sem_particao; inf(IC msp) acima do sup(IC " + B.HTCAL2 + ").", "",
          f"| tarefa | {B.HTCAL2} dentro do IC de msp (sec_dentro) | inf(msp) > sup({B.HTCAL2}) (sec_acima) |",
          "|---|---|---|"]
    for t in B.TIPOS:
        x = ht["tarefas"][t]
        L.append(f"| {t} | {x['sec_dentro']} | {x['sec_acima']} |")
    return L


def md_elegibilidade(ht):
    L = ["## 4. P_vant e elegibilidade", "",
         "P_vant: inf(IC msp) > sup(IC sem_media_k1). Elegibilidade: |perda CE da rede nova HT − msp| / msp ≤ 10% "
         "(sementes 11 a 15); acurácia ±0,03 só reportada.", "",
         "| tarefa | P_vant | perda CE nova HT | perda CE nova msp | razão relativa | elegível | "
         "Δ acc nova (±0,03, só reportada) |", "|---|---|---|---|---|---|---|"]
    for t in B.TIPOS:
        x = ht["tarefas"][t]
        L.append(f"| {t} | {x['p_vant']} (msp {ic(x['msp'])}; k1 {ic(x['k1'])}) | {f3(x['loss_ht'])} | "
                 f"{f3(x['loss_msp'])} | {f3(x['rel_ce'])} | {x['ce_ok']} | {f3(x['dacc'])} ({x['acc_ok']}) |")
    if ht["motivos"]:
        L += ["", "Motivos de \"não avaliável\": " + "; ".join(ht["motivos"])]
    return L


def md_previsao(ht):
    j = B.julgar_previsao_item(ht)
    return ["## 5. Previsão do autor × resultado (só o item H_tempo e H_dist)", "",
            f"Previsão literal da rodada 2 (valendo sem alteração): \"{B.PREVISAO_LITERAL}\". Na 2b só se julga o "
            "item H_tempo e H_dist (H_het e horizonte não entram). Regras confirmadas em 2026-10-21/22: acerto só "
            "com \"HT acima\" no veredito geral; \"misto\" com \"HT acima\" numa tarefa = acerto parcial; "
            "\"não avaliável\" não conta.", "",
            "| item | previsão do autor | veredito calculado | julgamento |", "|---|---|---|---|",
            f"| {j['item']} | {j['previsao']} | {j['calculado']} | {j['julgamento']} |"]


def fig_gap(df, caminho):
    """Gap médio (5 últimas tarefas) por condição e tarefa, com IC 95%. None se não há dado."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    desenhou = False
    for i, c in enumerate(B.CONDICOES_2B):
        pts = [(k, B.resumo_serie(df, c, t)) for k, t in enumerate(B.TIPOS)]
        pts = [(k, r) for k, r in pts if r["n"] and r["inf"] == r["inf"]]
        if not pts:
            continue
        desenhou = True
        est = ESTILOS[c]
        x = [k + 0.12 * (i - 1) for k, _ in pts]
        ax.errorbar(x, [r["media"] for _, r in pts],
                    yerr=[[r["media"] - r["inf"] for _, r in pts], [r["sup"] - r["media"] for _, r in pts]],
                    capsize=4, markersize=7, linewidth=1.5, label=c, **est)
    if not desenhou:
        plt.close(fig)
        return None
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(range(len(B.TIPOS)))
    ax.set_xticklabels([f"tarefa {t}" for t in B.TIPOS])
    ax.set_xlabel("tipo de tarefa")
    ax.set_ylabel(LABEL_GAP)
    ax.set_title("Gap por condição, rodada 2b (m*=3, T=30, 5 últimas tarefas, média e IC 95% bootstrap, sementes 11 a 15)",
                 fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return caminho


def texto(results):
    """Markdown completo da 2b e o DataFrame carregado (vazio se não há CSV)."""
    df = B.carregar(results) if os.path.isdir(results) else pd.DataFrame()
    if df.empty:
        return None, df
    calib = B.carregar_calibracao(results)
    ht = B.h_tempo(df, calib)
    L = ["# Tabela de decisão da rodada 2b (central_HT_cal2)", "",
         f"Fonte: `{results}/` (sementes 11 a 15; a semente 1000 nunca entra). Regras literais de PREREGISTRO.md "
         "(rodada 2b e blocos de 2026-10-21/22); decisões de leitura em `plots/rodada2b_analise.py`. "
         "Vereditos: texto calculado, sem interpretação.", ""]
    L += md_calibracao(calib) + [""] + md_metrica(df) + [""] + md_diferenca(ht) + [""] + md_elegibilidade(ht) + [""]
    L += md_previsao(ht) + [""]
    return "\n".join(L), df


def gerar(results, saida, figura=None):
    md, df = texto(results)
    if md is None:
        print(f"sem resultados em {results}/ (nenhum CSV válido das sementes 11 a 15); nada foi gerado",
              file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        f.write(md)
    figura = figura or os.path.splitext(saida)[0] + "_gap.png"
    print("gerado:", saida)
    g = fig_gap(df, figura)
    print("gerado:" if g else "sem dados para a figura (não gerada)", g or "")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Tabela de decisão e figura da rodada 2b")
    p.add_argument("--results", default="results_rodada2b")
    p.add_argument("--saida", required=True, help="arquivo .md de saída")
    p.add_argument("--figura", default=None, help="PNG de saída (padrão: ao lado do .md)")
    a = p.parse_args(argv)
    return gerar(a.results, a.saida, a.figura)


if __name__ == "__main__":
    sys.exit(main())
