"""figures/tabela_decisao.md (esquema v3): cabeçalho com o nível m*, métrica principal em A e B no nível m*,
B_disjunta (gap + esquecimento + marcação), dose-resposta, calibração, seleção de λ, sanidade K=1 e
a linha final de critérios (H_reg, H_part, H_real, H_orc)."""
import os

import numpy as np

import _comum
import analise


def fmt(r, nref=None):
    """Formata média [IC 95%] (n); sem IC (n<2) escreve a nota. nref (ex.: 'n_ref=5') vai ao lado."""
    if r["n"] == 0:
        return "sem dados"
    extra = f", {nref}" if nref else ""
    if np.isnan(r["inf"]):
        return f"{r['media']:.4f} (IC indefinido (n={r['n']}){extra})"
    return f"{r['media']:.4f} [{r['inf']:.4f}, {r['sup']:.4f}] (n={r['n']}{extra})"


def fmt_m(df, cond, tipo, nivel, janela=None):
    """Métrica principal com n_ref (referências pareadas por semente na janela)."""
    return fmt(analise.resumo(df, cond, tipo, nivel, janela),
               analise.n_ref_texto(df, cond, tipo, nivel, janela))


def _cabecalho(regime, nivel, aviso):
    L = ["# Tabela de decisão", "", "## (a) Nível de avaliação"]
    if not regime["presente"]:
        L.append("- `results/calibracao/regime.json` ausente: nenhuma decisão é avaliável.")
    else:
        gaps = ", ".join(f"m={m}: {g:.3f}" for m, g in sorted(regime["gaps"].items())) or "sem gaps no JSON"
        if regime["m_star"] is None:
            L.append(f"- m* = null (regime.json). {regime['nota'] or analise.MSG_SEM_REGIME}")
        else:
            L.append(f"- m* = {regime['m_star']} (de `results/calibracao/regime.json`: menor nível com gap "
                     "de central_aprendizado ≤ −0,10 na tarefa A, semente 1000).")
        L.append(f"- Gap de central_aprendizado, tarefa A, semente 1000, por nível: {gaps}.")
    if aviso:
        L.append(f"- {aviso}. Os números abaixo são descritivos; as decisões saem não avaliáveis.")
    elif nivel is not None:
        L.append(f"- As tabelas (b), (b'), (c) e as decisões usam o nível m={nivel}; (d) e (g) percorrem todos os níveis.")
    return L


def _tabela_ab(df, nivel, janela=None):
    L = ["| condição | tarefa A | tarefa B |", "|---|---|---|"]
    conds = analise.CONDICOES if janela is None else ("fedavg_iid", "central_aprendizado", "media_sem_particao")
    for c in conds:
        L.append(f"| {c} | " + " | ".join(fmt_m(df, c, t, nivel, janela)
                                         for t in analise.TIPOS_AB) + " |")
    return L


def _tabela_bd(df, nivel, regime):
    lim = analise.limite_esquecimento(df, nivel)
    L = ["Limite: média_esq(condição) ≤ média_esq(central_aprendizado) + 0,05 = "
         + ("indefinido (sem esquecimento de central_aprendizado)" if np.isnan(lim) else f"{lim:.4f}"), "",
         "| condição | gap contínua−nova, últimas 5 tarefas (média [IC 95%], n) "
         "| esquecimento médio final (média [IC 95%], n) | marcação |", "|---|---|---|---|"]
    for c in analise.CONDICOES:
        marca = analise.rotulo_ganho(df, c, regime)
        if c == "central_aprendizado" and analise.nivel_decisao(regime) is not None:
            marca += " (referência do limite)"
        L.append(f"| {c} | {fmt_m(df, c, analise.TIPO_BD, nivel)} "
                 f"| {fmt(analise.resumo_esq(df, c, nivel))} | {marca} |")
    return L


def _tabela_dose(df):
    L = []
    for t in analise.TIPOS:
        L += [f"**Tarefa {t}**", "", "| condição | " + " | ".join(f"m={m}" for m in analise.NIVEIS) + " |",
              "|---|" + "---|" * len(analise.NIVEIS)]
        for c in analise.CONDICOES:
            cel = [fmt_m(df, c, t, m) for m in analise.NIVEIS]
            if all(x == "sem dados" for x in cel):
                continue
            L.append(f"| {c} | " + " | ".join(cel) + " |")
        L.append("")
    return L


def _tabela_calib(calib):
    L = ["| tarefa | nível | alvo (acc nova do fedavg_iid) | passos | acc_central | erro | ok | alvo > 0,90 |",
         "|---|---|---|---|---|---|---|---|"]

    def f(v, casas=3):
        return "n/d" if v is None else (f"{v:.{casas}f}" if isinstance(v, float) else str(v))
    for (t, m), j in sorted(calib.items(), key=lambda kv: (analise.TIPOS.index(kv[0][0])
                                                           if kv[0][0] in analise.TIPOS else 9, kv[0][1])):
        L.append(f"| {t} | {m} | {f(j.get('alvo_acc_nova_fedavg'))} | {f(j.get('passos'))} "
                 f"| {f(j.get('acc_central'))} | {f(j.get('erro'))} | {f(j.get('ok'))} "
                 f"| {f(j.get('alvo_acima_de_0.90'))} |")
    if not calib:
        L.append("| sem calibracao_*.json | | | | | | | |")
    j10 = calib.get(("A", 10))
    if j10 is None:
        L += ["", "Registro: calibracao_A_m10.json ausente; não dá para dizer se o alvo passa de 0,90 em m=10."]
    elif j10.get("alvo_acima_de_0.90") is False:
        L += ["", "**Registro: a rede nova do fedavg_iid não passa de 0,90 nem em m=10: já é um resultado.**"]
    elif j10.get("alvo_acima_de_0.90") is None:
        L += ["", "Registro: calibracao_A_m10.json sem o campo `alvo_acima_de_0.90` (n/d)."]
    else:
        L += ["", "Registro: alvo da rede nova do fedavg_iid passa de 0,90 em m=10 na tarefa A."]
    return L


def _num(v, casas=4):
    if isinstance(v, bool):
        return str(v)
    return "n/d" if v is None else (f"{v:.{casas}f}" if isinstance(v, (int, float)) else str(v))


def _por_lambda(d):
    """{λ (str ou float): valor} -> lista [(λ float, valor)] ordenada por λ; não-dict -> []."""
    if not isinstance(d, dict):
        return []
    return sorted(((float(k), v) for k, v in d.items()), key=lambda kv: kv[0])


def _tabela_selecao(sel):
    """Uma subseção por selecao_<método>_<tarefa>_m<m>.json: tabela por λ (acc, esquecimento, respeita o
    limite esq_ref + 0,05?) e a flag `violacao`. Formato real: notas e esquecimento são dicts {λ: valor},
    esquecimento_ref é float (ou null fora de B_disjunta; esquecimento então é {})."""
    if not sel:
        return ["sem selecao_*.json"]
    L = []
    for (met, t, m), j in sorted(sel.items()):
        ref = j.get("esquecimento_ref")
        esq = dict(_por_lambda(j.get("esquecimento")))
        notas = _por_lambda(j.get("notas"))
        lam = j.get("lambda")
        L += [f"**{met}, tarefa {t}, m={m}**: λ escolhido = {_num(lam, 6)}; "
              f"esquecimento_ref = {_num(ref)}; violação = {_num(j.get('violacao'))}", "",
              "| λ | acc contínua (últimas 5 tarefas) | esquecimento final | respeita limite (≤ ref + 0,05)? |",
              "|---|---|---|---|"]
        for l, acc in notas:
            e = esq.get(l)
            if e is None or ref is None:
                resp = "n/d (limite só medido em B_disjunta)"
            else:
                resp = "sim" if e <= ref + analise.LIMITE_ESQ else "NÃO"
            marca = " (escolhido)" if lam is not None and abs(l - float(lam)) < 1e-12 else ""
            L.append(f"| {l:g}{marca} | {_num(acc)} | {_num(e)} | {resp} |")
        if not notas:
            L.append("| sem notas no JSON | | | |")
        L.append("")
    return L[:-1]


def _tabela_sanidade(df):
    L = ["Pelo Desvios v3 (a), `sem_media_k1` e `central_passos` executam o mesmo algoritmo e deveriam "
         "coincidir dentro do ruído. Coincidem = ICs da métrica principal se sobrepõem.", "",
         "| tarefa | nível | sem_media_k1 | central_passos | ICs sobrepostos? |", "|---|---|---|---|---|"]
    for t in analise.TIPOS:
        for m in analise.NIVEIS:
            a, b = analise.resumo(df, "sem_media_k1", t, m), analise.resumo(df, "central_passos", t, m)
            na = analise.n_ref_texto(df, "sem_media_k1", t, m)
            nb = analise.n_ref_texto(df, "central_passos", t, m)
            if a["n"] == 0 or b["n"] == 0:
                ver = "n/d (sem dados)"
            elif analise._sem_ic(a) or analise._sem_ic(b):
                ver = "n/d (IC indefinido)"
            else:
                ver = "sim" if analise._sobrepoe(a, b) else "NÃO (reportar: deveriam coincidir)"
            L.append(f"| {t} | {m} | {fmt(a, na)} | {fmt(b, nb)} | {ver} |")
    return L


def main():
    a = _comum.args_padrao(__doc__)
    df, regime, nivel, aviso = _comum.contexto(a)
    calib, sel = analise.carregar_calibracao(a.results)
    os.makedirs(a.saida, exist_ok=True)
    L = _cabecalho(regime, nivel, aviso)
    if nivel is not None and not df.empty:
        L += ["", f"## (b) Métrica principal no nível m={nivel} (gap médio nas últimas 5 tarefas; "
              "média [IC 95% bootstrap] entre sementes)", ""] + _tabela_ab(df, nivel)
        L += ["", f"## (b') fedavg_iid vs central_aprendizado, janela 7..11 (índices), m={nivel}", ""] \
            + _tabela_ab(df, nivel, analise.JANELA_PART)
        L += ["", f"## (c) B_disjunta no nível m={nivel}: plasticidade, esquecimento e marcação do Adendo 1", "",
              "Nas tarefas A e B o esquecimento não é medido (o mesmo X recebe rótulos conflitantes); a marcação "
              "só vale para B_disjunta.", ""] + _tabela_bd(df, nivel, regime)
    L += ["", "## (d) Dose-resposta: métrica principal por condição × nível × tarefa "
          "(central_L2 e central_SP só existem em m*)", ""] + _tabela_dose(df)
    L += ["## (e) Calibração de central_aprendizado (semente 1000)", ""] + _tabela_calib(calib)
    L += ["", "## (f) Seleção de λ (semente 1000, nível m*)", ""] + _tabela_selecao(sel)
    L += ["", "## (g) Sanidade: sem_media_k1 vs central_passos", ""] + _tabela_sanidade(df)
    linha, expl = analise.criterios_atendidos(df, regime)
    m = analise.nivel_decisao(regime)
    L += ["", "## Regras" + (f" (nível m*={m})" if m is not None else ""), ""] + [f"- {e}" for e in expl]
    if m is not None:
        ruins = [t for t in analise.TIPOS_AB if calib.get((t, m), {}).get("ok") is False]
        if ruins:
            linha += (f". Aviso (tarefa {', '.join(ruins)}): calibração fora de ±0,03 no nível m*: premissa de H_orc "
                      "(mesmo nível de aprendizado) enfraquecida")
    L += ["", "## (h) Linha final", "", linha, ""]
    with open(os.path.join(a.saida, "tabela_decisao.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
