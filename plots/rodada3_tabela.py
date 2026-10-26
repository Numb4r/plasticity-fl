"""Tabela de decisão e figura da RODADA 3. Uso (da raiz do projeto):
    .venv/bin/python plots/rodada3_tabela.py --results results_rodada3 --saida <arquivo.md> [--figura <arquivo.png>]
Sem --figura, o PNG (dpi 200) vai ao lado do .md com o sufixo _gap.png. Sem resultados: mensagem e código 1, sem
criar nada. Regras e decisões de leitura: plots/rodada3_analise.py."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rodada3_analise as R  # noqa: E402

LABEL_GAP = "gap (acurácia contínua − acurácia da rede nova, fração)"
ESTILOS = {"particao_iid": dict(color="#000000", linestyle="-", marker="o"),
           "copias": dict(color="#0072b2", linestyle="-.", marker="^"),
           "sem_media_k1": dict(color="#d55e00", linestyle="--", marker="s")}


def f3(x):
    return "n/d" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def ic(r):
    if r["n"] == 0:
        return "sem dados"
    return f"{f3(r['media'])} [{f3(r['inf'])}, {f3(r['sup'])}] (n={r['n']})"


def cc(c):
    """Célula de um contraste: IC ou o que falta."""
    return f"não avaliável (falta {c['falta']})" if c["falta"] else ic(c)


def md_prep(dec):
    pr = dec["p_rep"]
    L = ["## 1. Pré-condição P_rep", "",
         f"Regra: inf do IC pareado de gap(fedavg_iid) − gap(sem_media_k1) > +δ = {R.DELTA} em A e em B. "
         "Sem isso, todas as hipóteses saem \"não avaliável (a vantagem do FedAvg não se reproduziu)\".", "",
         f"**P_rep: {'atendida' if pr['ok'] else pr['msg']}**", "", "| tarefa | diferença (IC 95%) | inf > δ |", "|---|---|---|"]
    for t in R.TIPOS:
        x = pr["por"][t]
        L.append(f"| {t} | {cc(x['c'])} | {'n/d' if x['ok'] is None else x['ok']} |")
    return L


def md_calibracao(df, calib):
    L = ["## 2. Calibração (semente 1000) e elegibilidade", "",
         "R = rodadas; erro_rel da calibração; elegível = calibração ok nas duas tarefas e razão relativa da perda CE "
         "da rede nova (média das médias por semente, sementes 16 a 23, janela principal) ao fedavg_iid ≤ 0,10 "
         "(fechado). Δ acc_nova ao fedavg_iid (±0,03) só reportada.", "",
         "| condição | tarefa | R | erro_rel | ok | razão da perda | elegível | Δ acc_nova (±0,03, só reportada) | motivo |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in R.CONDICOES:
        for t in R.TIPOS:
            e = R.elegibilidade(df, calib, c, t)
            k = calib.get((c, t))
            ok = "n/d" if k is None else k.get("ok")
            L.append(f"| {c} | {t} | {e['rodadas'] if e['rodadas'] is not None else 'n/d'} | {f3(e['erro_rel'])} | "
                     f"{ok} | {f3(e['razao'])} | {e['elegivel']} | {f3(e['dacc'])} ({e['acc_ok']}) | {e['motivo']} |")
    return L


def md_metrica(df):
    L = ["## 3. Métrica principal por condição e tarefa (gap pareado médio das tarefas 25 a 29)", "",
         "Média entre sementes [IC 95% bootstrap percentil, 10 000, default_rng(0)] (n sementes).", "",
         "| condição | A | B |", "|---|---|---|"]
    for c in R.CONDICOES:
        L.append(f"| {c} | " + " | ".join(ic(R.resumo(df, c, t)) for t in R.TIPOS) + " |")
    return L


def _linhas_k(nome, h, dec, regra, contraste_txt):
    pr = dec["p_rep"]["ok"] is not True
    L = [f"### {nome}", "", f"**Veredito geral: {dec['hip'][nome]['geral']}** (A: {dec['hip'][nome]['vers']['A']}; "
         f"B: {dec['hip'][nome]['vers']['B']})", "",
         f"Regra: {regra}." + (" P_rep não atendida: os números abaixo são descritivos, não decisórios." if pr else ""),
         "", f"| tarefa | K | {contraste_txt}, IC 95% | resultado |", "|---|---|---|---|"]
    for t in R.TIPOS:
        for k in (2,) + R.KS:
            x = h[t]["porK"][k]
            nota = " (K=2 só reportado)" if k == 2 else ""
            L.append(f"| {t} | {k} | {cc(x['c']) if x['c'] else '—'} | {x['res']}{nota} |")
        L.append(f"| {t} | veredito | | **{h[t]['veredito']}** |")
    return L


def md_hk(dec):
    h = dec["hip"]["H_K"]
    L = ["### H_K", "", f"**Veredito geral: {h['geral']}** (A: {h['vers']['A']}; B: {h['vers']['B']})", "",
         "Regra: na particao_iid, gap(20) > gap(10) > gap(5) e inf do IC pareado de gap(20) − gap(5) > 0; três "
         "médias a menos de 0,01 de 0 = \"inconclusivo (sem faixa dinâmica)\"." +
         (" P_rep não atendida: números descritivos." if dec["p_rep"]["ok"] is not True else ""), "",
         "| tarefa | gap K=2 (só reportado) | gap K=5 | gap K=10 | gap K=20 | 20 > 10 > 5 | gap(20) − gap(5) | inf > 0 | "
         "veredito |", "|---|---|---|---|---|---|---|---|---|"]
    for t in R.TIPOS:
        x = h["por"][t]["t"]
        L.append(f"| {t} | " + " | ".join(ic(x["medias"][k]) for k in (2, 5, 10, 20)) + f" | {x['ordem']} | "
                 f"{cc(x['c'])} | {x['ic_acima0']} | {h['por'][t]['veredito']} |")
    return L


def md_hipoteses(dec):
    L = ["## 4. Hipóteses (δ = 0,05; comparações pareadas por semente; IC fechado)", ""]
    L += _linhas_k("H_dados", dec["hip"]["H_dados"]["por"], dec, "d_K = gap(particao_iid_kK) − gap(copias_kK) "
                   "(K=10: fedavg_iid − copias_k10) com inf do IC > +δ, em K = 5, 10, 20", "d_K")
    L += [""] + _linhas_k("H_cópias", dec["hip"]["H_cópias"]["por"], dec, "IC pareado de gap(copias_kK) − "
                          "gap(sem_media_k1) inteiro em [−δ, +δ], em K = 5, 10, 20", "copias − sem_media_k1")
    L += [""] + md_hk(dec)
    return L


def md_relato_k(dec):
    L = ["## 5. À parte: gap(20) − gap(5) tem IC acima de 0? (para a leitura da previsão \"K=20 melhor que K=5\")", "",
         "| tarefa | família | gap(20) − gap(5), IC 95% | inf > 0 | elegibilidade |", "|---|---|---|---|---|"]
    for t in R.TIPOS:
        x = dec["hip"]["H_K"]["por"][t]["t"]
        L.append(f"| {t} | particao_iid | {cc(x['c'])} | {x['ic_acima0']} | {x['elegivel'] or 'elegível'} |")
    return L


def md_epocas(df, calib):
    L = ["## 6. Braço de épocas locais fixas (exploratório, SEM veredito)", "",
         "Mesmos contrastes: d_K = gap(particao_epocas_kK) − gap(copias_kK) (K=10: fedavg_iid, igual a "
         "particao_epocas_k10 por igualdade); S_local = 75, 30 e 8 passos (K = 2, 5, 20). Só reportado.", "",
         "| tarefa | K | gap particao_epocas | d_K, IC 95% | inf > δ | elegibilidade (informativa) |",
         "|---|---|---|---|---|---|"]
    tend = []
    for t in R.TIPOS:
        b = R.braco_epocas(df, calib, t)
        for k in (2,) + R.KS:
            x = b["dk"][k]
            L.append(f"| {t} | {k} | {ic(b['tend']['medias'][k])} | {cc(x['c'])} | {x['vale']} | "
                     f"{x['eleg'] or 'elegível'} |")
        x = b["tend"]
        tend.append(f"| {t} | {x['ordem']} | {cc(x['c'])} | {x['ic_acima0']} |")
    L += ["", "Tendência em K (gap(20) > gap(10) > gap(5) e gap(20) − gap(5)):", "",
          "| tarefa | 20 > 10 > 5 | gap(20) − gap(5), IC 95% | inf > 0 |", "|---|---|---|---|"] + tend
    return L


def md_mecanismo(df):
    L = ["## 7. Mecanismo (exploratório, sem veredito)", "",
         "Esperado, se \"as atualizações discordam e a média as amortece\": cosseno menor e encolhimento maior com dados "
         "diferentes. Média entre sementes das tarefas 25 a 29 [IC 95%]. rank_ef e srank_feat sempre juntos.", ""]
    for t in R.TIPOS:
        m = R.mecanismo(df, t)
        L += [f"### tarefa {t}", "", "| família | K | cos_medio | encolhimento | rank_ef | srank_feat |",
              "|---|---|---|---|---|---|"]
        for (fam, k), r in m.items():
            L.append(f"| {fam} | {k} | " + " | ".join(ic(r[x]) for x in R.MEDIDAS_MEC) + " |")
        L.append("")
    return L[:-1]


def md_previsao(dec):
    L = ["## 8. Previsão do autor × resultado", "",
         "Regra: \"não\" acerta se \"não atendida\", em parte se \"parcial\", erra se \"atendida\"; \"sim\" acerta se "
         "\"atendida\", em parte se \"parcial\", erra se \"não atendida\"; \"não avaliável\" não conta. Leituras "
         "(2026-10-26): H_K como registrada; H_cópias \"não\" = \"não atendida\".", "",
         "| item | previsão | veredito geral | julgamento |", "|---|---|---|---|"]
    for j in R.julgar_previsao(dec):
        L.append(f"| {j['item']} | {j['previsao']} | {j['calculado']} | {j['julgamento']} |")
    return L


def fig_gap(df, caminho):
    """Gap por K (famílias particao_iid e copias, mais sem_media_k1 em K=1), A e B. None se não há dado."""
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    desenhou = False
    for ax, t in zip(eixos, R.TIPOS):
        for i, (fam, f) in enumerate((("particao_iid", R.cond_part), ("copias", R.cond_cop))):
            pts = [(k, R.resumo(df, f(k), t)) for k in (2, 5, 10, 20)]
            pts = [(k, r) for k, r in pts if r["n"]]
            if not pts:
                continue
            desenhou = True
            ax.errorbar([k * (1 + 0.04 * (2 * i - 1)) for k, _ in pts], [r["media"] for _, r in pts],
                        yerr=[[r["media"] - r["inf"] for _, r in pts], [r["sup"] - r["media"] for _, r in pts]],
                        capsize=3, markersize=7, linewidth=1.5, label=fam, **ESTILOS[fam])
        r = R.resumo(df, "sem_media_k1", t)
        if r["n"]:
            desenhou = True
            ax.errorbar([1], [r["media"]], yerr=[[r["media"] - r["inf"]], [r["sup"] - r["media"]]], capsize=3,
                        markersize=8, linestyle="none", label="sem_media_k1 (K=1)",
                        **{k: v for k, v in ESTILOS["sem_media_k1"].items() if k != "linestyle"})
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_xlabel("K (número de trabalhadores)")
        ax.set_title(f"tarefa {t}", fontsize=9)
    if not desenhou:
        plt.close(fig)
        return None
    eixos[0].set_ylabel(LABEL_GAP, fontsize=8)
    eixos[0].legend(fontsize=8)
    fig.suptitle("Gap por K, rodada 3 (m*=3, T=30, tarefas 25 a 29, média e IC 95% bootstrap, sementes 16 a 23)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return caminho


def texto(results):
    """Markdown completo e o DataFrame (None, vazio se não há CSV)."""
    df = R.carregar(results) if os.path.isdir(results) else pd.DataFrame()
    if df.empty:
        return None, df
    calib = R.carregar_calibracao(results)
    dec = R.decidir(df, calib)
    L = ["# Tabela de decisão da rodada 3 (média entre clientes com dados diferentes)", "",
         f"Fonte: `{results}/` (sementes 16 a 23; a 1000 só calibra). Regras literais de PREREGISTRO.md (rodada 3 e "
         "bloco de 2026-10-26); decisões de leitura em `plots/rodada3_analise.py`. Vereditos: texto calculado, sem "
         "interpretação.", ""]
    for bloco in (md_prep(dec), md_calibracao(df, calib), md_metrica(df), md_hipoteses(dec), md_relato_k(dec),
                  md_epocas(df, calib), md_mecanismo(df), md_previsao(dec)):
        L += bloco + [""]
    return "\n".join(L), df


def gerar(results, saida, figura=None):
    md, df = texto(results)
    if md is None:
        print(f"sem resultados em {results}/ (nenhum CSV válido das sementes 16 a 23); nada foi gerado",
              file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        f.write(md)
    print("gerado:", saida)
    g = fig_gap(df, figura or os.path.splitext(saida)[0] + "_gap.png")
    print("gerado:" if g else "sem dados para a figura (não gerada)", g or "")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Tabela de decisão e figura da rodada 3")
    p.add_argument("--results", default="results_rodada3")
    p.add_argument("--saida", required=True, help="arquivo .md de saída")
    p.add_argument("--figura", default=None, help="PNG de saída (padrão: ao lado do .md)")
    a = p.parse_args(argv)
    return gerar(a.results, a.saida, a.figura)


if __name__ == "__main__":
    sys.exit(main())
