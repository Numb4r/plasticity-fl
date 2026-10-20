"""figures/tabela_decisao.md: métrica principal (A e B), tabela de B_disjunta (plasticidade, esquecimento,
marcação), janela 7..11 e critérios."""
import os

import numpy as np

import _comum
import analise


def fmt(r):
    """Formata média ± IC e n; sem IC (n<2) escreve a nota pedida."""
    if r["n"] == 0:
        return "sem dados"
    if np.isnan(r["inf"]):
        return f"{r['media']:.4f} (IC indefinido (n={r['n']}))"
    return f"{r['media']:.4f} [{r['inf']:.4f}, {r['sup']:.4f}] (n={r['n']})"


def main():
    a = _comum.args_padrao(__doc__)
    df = analise.carregar(a.results)
    os.makedirs(a.saida, exist_ok=True)
    L = ["# Tabela de decisão", "",
         "## Métrica principal (gap médio nas últimas 5 tarefas da condição; média e IC 95% bootstrap entre sementes)",
         "", "| condição | tarefa A | tarefa B |", "|---|---|---|"]
    for c in analise.CONDICOES:
        L.append(f"| {c} | " + " | ".join(fmt(analise.resumo(df, c, t)) for t in analise.TIPOS_AB) + " |")
    L += ["", "## fedavg_iid vs central_aprendizado, janela 7..11 (índices)", "",
          "| condição | tarefa A | tarefa B |", "|---|---|---|"]
    for c in ("fedavg_iid", "central_aprendizado", "media_sem_particao"):
        L.append(f"| {c} | " + " | ".join(
            fmt(analise.resumo(df, c, t, analise.JANELA_PART)) for t in analise.TIPOS_AB) + " |")
    lim = analise.limite_esquecimento(df)
    L += ["", "## B_disjunta: plasticidade, esquecimento e marcação do Adendo 1", "",
          "Nas tarefas A e B o esquecimento não é medido: o mesmo X recebe rótulos conflitantes a cada "
          "tarefa, por isso a marcação abaixo só vale para B_disjunta.",
          "",
          "Limite: média_esq(condição) ≤ média_esq(central_aprendizado) + 0,05 = "
          + ("indefinido (sem esquecimento de central_aprendizado)" if np.isnan(lim) else f"{lim:.4f}"),
          "", "| condição | plasticidade: gap contínua−nova, últimas 5 tarefas (média [IC 95%], n) "
          "| esquecimento médio final (média [IC 95%], n) | marcação |", "|---|---|---|---|"]
    for c in analise.CONDICOES:
        marca = analise.rotulo_ganho(df, c)
        if c == "central_aprendizado":
            marca += " (referência do limite)"
        L.append(f"| {c} | {fmt(analise.resumo(df, c, analise.TIPO_BD))} "
                 f"| {fmt(analise.resumo_esq(df, c))} | {marca} |")
    linha, expl = analise.criterios_atendidos(df)
    L += ["", "## Regras", ""] + [f"- {e}" for e in expl] + ["", linha, ""]
    with open(os.path.join(a.saida, "tabela_decisao.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
