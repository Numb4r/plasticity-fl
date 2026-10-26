"""Análises EXPLORATÓRIAS posteriores aos resultados da rodada 2 (PREREGISTRO.md, "Desvios da rodada 2", 2026-10-26).

Respondem a uma discrepância (media_sem_particao, tarefa A, m=3: rodada 1 contra rodada 2) e ao pedido sobre H_het.
Não alteram nenhum veredito. Só leem CSVs. Bootstrap percentil 10 000, default_rng(0), como a métrica principal.
Uso: python plots/exploratorio_rodada2_pos.py [--saida figures/rodada2_pos_resultados.md]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analise  # noqa: E402

J_FIXA, J_FIM = (7, 11), (25, 29)       # janela das tarefas 8 a 12 (índices 7 a 11) e das 5 últimas de T=30
SEM_R1, SEM_R2 = (1, 2, 3, 4, 5), (6, 7, 8, 9, 10)


def ler(pasta, nivel=3):
    linhas = []
    for f in sorted(glob.glob(os.path.join(pasta, f"*_m{nivel}_[AB]_*.csv"))):
        n = os.path.basename(f)
        m = re.match(rf"^(.+?)(_a[\d.]+|_T100)?_m{nivel}_([AB])_(\d+)\.csv$", n)
        if not m or "_rodadas" in n or "_matriz" in n or "_reaprend" in n:
            continue
        cond, suf, tipo, sem = m.group(1), m.group(2) or "", m.group(3), int(m.group(4))
        if suf == "_T100":
            continue
        d = pd.read_csv(f)
        d["cond"] = cond + (f" (α={suf[2:]})" if suf.startswith("_a") else "")
        d["tipo"], d["sem"] = tipo, sem
        linhas.append(d)
    return pd.concat(linhas, ignore_index=True)


def por_semente(df, cond, tipo, col, janela, sems):
    d = df[(df["cond"] == cond) & (df["tipo"] == tipo) & (df["sem"].isin(sems))
           & (df["tarefa"] >= janela[0]) & (df["tarefa"] <= janela[1])].dropna(subset=[col, "gap"])
    return d.groupby("sem")[col].mean()


def fmt(s, casas=3):
    if len(s) == 0:
        return "—"
    m, lo, hi = analise.bootstrap_ic(s.values)
    return f"{m:.{casas}f} [{lo:.{casas}f}, {hi:.{casas}f}] (n={len(s)})"


def tabela(df, conds, sems, janelas, col="gap", tipos=("A", "B")):
    L = ["| condição | " + " | ".join(f"{t}, tarefas {j[0] + 1}–{j[1] + 1}" for t in tipos for j in janelas) + " |",
         "|---|" + "---|" * (len(tipos) * len(janelas))]
    for c in conds:
        L.append(f"| {c} | " + " | ".join(fmt(por_semente(df, c, t, col, j, sems)) for t in tipos for j in janelas) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="figures/rodada2_pos_resultados.md")
    a = ap.parse_args()
    r1, r2 = ler("results"), ler("results_rodada2")
    r1 = r1[r1["sem"].isin(SEM_R1)]
    r2 = r2[r2["sem"].isin(SEM_R2)]
    janelas = (J_FIXA, J_FIM)
    L = ["# Análises exploratórias posteriores aos resultados da rodada 2", "",
         "Exploratórias, sem critério de decisão, nenhum veredito muda. Métrica = gap pareado (acc_cont − acc_nova) médio nas tarefas "
         "da janela com rede nova, média entre sementes [IC 95% bootstrap percentil] (n sementes). Índices: tarefas 8–12 = índices 7 a 11; "
         "tarefas 26–30 = índices 25 a 29. Nível m=3.", "",
         "## 1. Rodada 1: o `media_sem_particao` rodou com T=12; a métrica principal dele usa a janela das tarefas 8–12",
         "`src/config.py:26` (`T_SEM_PARTICAO = 12`), `src/condicoes.py:69` (a condição `media_sem_particao` usa `T=T_SEM_PARTICAO`) e "
         "`plots/analise.py:194-201` (janela padrão = `(T-5, T-1)` com o T da própria condição: 7 a 11 para ele, 25 a 29 para as de T=30). "
         "Na rodada 2 ele passou a rodar com T=30 (`src/condicoes.py:condicao_r2`), e a métrica principal passou a ser a das tarefas 26–30.", ""]
    conds1 = ["fedavg_iid", "sem_media_k1", "media_sem_particao", "central_passos", "central_aprendizado", "central_L2", "central_SP"]
    L += ["## 2. Rodada 1 (sementes 1 a 5), todas as condições, nas duas janelas",
          "A janela 8–12 é a única em que o `media_sem_particao` (T=12) tem rede nova; a janela 26–30 não existe para ele nessa rodada.", ""]
    L += tabela(r1, conds1, SEM_R1, janelas)
    conds2 = ["fedavg_iid", "media_sem_particao", "sem_media_k1", "central_aprendizado", "central_HT", "central_lookahead"]
    L += ["", "## 3. Rodada 2 (sementes 6 a 10), as mesmas condições comuns, nas duas janelas",
          "O `media_sem_particao` da rodada 2 roda com T=30: a janela 8–12 dele é comparável à da rodada 1 (mesmas tarefas, T maior).", ""]
    L += tabela(r2, conds2, SEM_R2, janelas)
    L += ["", "## 4. `fedavg_iid` da rodada 2 (sementes 6 a 10) ao lado do da rodada 1 (sementes 1 a 5), m=3",
          "(gap, acc_cont e acc_nova; a rodada 1 não gravou a perda)", ""]
    for col in ("gap", "acc_cont", "acc_nova"):
        L += [f"**{col}**", "", "| rodada | " + " | ".join(f"{t}, tarefas {j[0] + 1}–{j[1] + 1}" for t in "AB" for j in janelas) + " |",
              "|---|---|---|---|---|"]
        for nome, df, sems in (("rodada 1 (sementes 1–5)", r1, SEM_R1), ("rodada 2 (sementes 6–10)", r2, SEM_R2)):
            L.append(f"| {nome} | " + " | ".join(fmt(por_semente(df, "fedavg_iid", t, col, j, sems)) for t in "AB" for j in janelas) + " |")
        L.append("")
    L += ["## 5. H_het exploratório: rede nova e rede contínua do `fedavg_dirichlet` por α (sementes 6 a 10)",
          "Janela das tarefas 26–30 (as 5 últimas, todas com rede nova). `fedavg_iid` e `media_sem_particao` entram só como referência de nível. "
          "A perda é a entropia cruzada de treino ao fim da tarefa.", ""]
    alfas = [f"fedavg_dirichlet (α={a_})" for a_ in ("100", "1", "0.1")]
    for t in ("A", "B"):
        L += [f"**Tarefa {t}**", "", "| condição | acc_nova | loss_nova | acc_cont | loss_cont | gap |", "|---|---|---|---|---|---|"]
        for c in alfas + ["fedavg_iid", "media_sem_particao"]:
            L.append(f"| {c} | " + " | ".join(fmt(por_semente(r2, c, t, col, J_FIM, SEM_R2)) for col in
                                              ("acc_nova", "loss_nova", "acc_cont", "loss_cont", "gap")) + " |")
        L.append("")
    L += ["Mesma tabela na janela das tarefas 8–12:", ""]
    for t in ("A", "B"):
        L += [f"**Tarefa {t}**", "", "| condição | acc_nova | loss_nova | acc_cont | loss_cont | gap |", "|---|---|---|---|---|---|"]
        for c in alfas:
            L.append(f"| {c} | " + " | ".join(fmt(por_semente(r2, c, t, col, J_FIXA, SEM_R2)) for col in
                                              ("acc_nova", "loss_nova", "acc_cont", "loss_cont", "gap")) + " |")
        L.append("")
    txt = "\n".join(L)
    os.makedirs(os.path.dirname(a.saida) or ".", exist_ok=True)
    open(a.saida, "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
