"""Análise EXPLORATÓRIA dos diagnósticos de Klein et al. (Desvios v3.1 e "Análises exploratórias
pós-resultados"). Sem critério de decisão. Escrita depois de ver os resultados.

Pergunta: na condição que perde plasticidade, a norma dos parâmetros e o maior autovalor da
Hessiana sobem ANTES da queda da acurácia do contínuo? (ordem da cadeia causal da Fig. 1 do survey)

Seções do relatório:
 (1) trajetórias médias por blocos de tarefas;
 (2) VERSÃO 1, por limiar: DEGENERADA (o limiar é cruzado já em t = 5, o primeiro instante
     permitido, e o "adiantamento" repete t_acc - 5). Mantida só como registro;
 (3) VERSÃO 2, meia-excursão SEM filtro: rodada uma vez antes da definição da versão 3
     ("prévia"; commit 1e3de74). Mantida como registro;
 (4) VERSÃO 3, definição final (PREREGISTRO.md, "Análises exploratórias pós-resultados", A):
     só (condição, nível, tarefa, semente) com perda real (gap médio das 5 últimas tarefas <= -0,10);
     "indefinido" se o percurso da acurácia (|média das 5 últimas - média das 3 primeiras|) < 0,10;
     t50 = primeiro t em que a média móvel (3 tarefas, para trás) percorre metade do caminho
     entre o nível inicial (3 primeiras) e o final (5 últimas); adiantamento = t50(acc) - t50(sinal)
     por semente; mediana entre sementes com IC 95% bootstrap da mediana (10 000, rng(0));
     n < 3: sem IC. Resolução temporal: uma tarefa, suavização de 3 tarefas.
Uso: python -m experiments.analise_exploratoria_klein [--saida analise_exploratoria_klein.md]
"""
import argparse
import glob

import numpy as np
import pandas as pd

CONDS = ["central_passos", "central_aprendizado", "sem_media_k1", "fedavg_iid", "media_sem_particao",
         "central_L2", "central_SP"]
TIPOS = ["A", "B", "B_disjunta"]
NIVEIS = [1, 3, 10]
SINAIS = {"acc_cont": "acc_cont", "norma": "fim_norma_global", "hess": "fim_hess_max",
          "grad_l2": "ini_grad_l2_global", "grad_srank": "ini_grad_srank", "rank_ef": "rank_ef",
          "srank_feat": "srank_feat", "dorm": "dorm_tau0025_l2"}
SINAIS_LEAD = {"norma": "fim_norma_global", "|hess|": "fim_hess_max", "dormentes": "dorm_tau0025_l2",
               "srank_feat": "srank_feat", "rank_ef": "rank_ef"}
LIMITE_PERDA, LIMITE_PERCURSO = -0.10, 0.10


def carregar(nivel: int, tarefa: str) -> dict:
    out = {}
    for c in CONDS:
        fs = sorted(glob.glob(f"results/{c}_m{nivel}_{tarefa}_[1-5].csv"))
        if fs:
            out[c] = [pd.read_csv(f) for f in fs]
    return out


def gap_principal(d: pd.DataFrame) -> float:
    T = len(d)
    return float(d[d.tarefa >= T - 5]["gap"].dropna().mean())


def cruza(serie, base, delta, rel, queda=False) -> float:
    """VERSÃO 1 (degenerada): primeiro t >= 5 em que a média móvel de 3 passa do limiar."""
    s = pd.Series(serie).rolling(3, min_periods=3).mean().values
    lim = base - delta if queda else (base * (1 + delta) if rel else base + delta)
    for t in range(5, len(s)):
        if not np.isnan(s[t]) and ((s[t] < lim) if queda else (s[t] > lim)):
            return float(t)
    return float("nan")


def t50(serie) -> float:
    """Primeiro t em que a média móvel (3 tarefas, para trás) percorreu metade do caminho
    entre o nível inicial (3 primeiras) e o final (5 últimas). NaN se não houver excursão."""
    serie = np.asarray(serie, dtype=float)
    ini, fim = np.nanmean(serie[:3]), np.nanmean(serie[-5:])
    if not np.isfinite(ini) or not np.isfinite(fim) or abs(fim - ini) < 1e-9:
        return float("nan")
    s = pd.Series(serie).rolling(3, min_periods=1).mean().values
    meio = ini + 0.5 * (fim - ini)
    for t in range(len(s)):
        if (s[t] >= meio) if fim > ini else (s[t] <= meio):
            return float(t)
    return float("nan")


def ic_mediana(v, B=10000, seed=0):
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    m = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return np.percentile(m, [2.5, 97.5])


def fmt_lead(v: list) -> str:
    v = [x for x in v if np.isfinite(x)]
    if not v:
        return "-"
    med = np.median(v)
    if len(v) < 3:
        return f"{med:+.1f} (n={len(v)}, sem IC)"
    lo, hi = ic_mediana(v)
    return f"{med:+.1f} [{lo:+.1f}, {hi:+.1f}] (n={len(v)})"


def secao_trajetorias(dados: dict, titulo: str) -> list[str]:
    L = [f"### {titulo}", ""]
    blocos = {"t0-4": slice(0, 5), "t10-14": slice(10, 15), "t20-24": slice(20, 25), "t25-29": slice(25, 30)}
    for c, ds in dados.items():
        L += [f"**{c}** (T={len(ds[0])})", "", "| sinal | " + " | ".join(blocos) + " |", "|---|" + "---|" * len(blocos)]
        for nome, col in SINAIS.items():
            if col not in ds[0]:
                continue
            vals = []
            for b in blocos.values():
                x = [d[col].iloc[b].mean() for d in ds if len(d.iloc[b]) > 0]
                vals.append(f"{np.mean(x):.3g}" if x else "-")
            L.append(f"| {nome} | " + " | ".join(vals) + " |")
        L.append("")
    return L


def secao_v1(dados: dict) -> list[str]:
    L = ["| condição | sementes com queda de acc | t_acc | adiant. norma +25% (v1) |", "|---|---|---|---|"]
    for c, ds in dados.items():
        ta, ln = [], []
        for d in ds:
            acc, nor = d["acc_cont"].values, d["fim_norma_global"].values
            t = cruza(acc, acc[:5].mean(), 0.10, False, queda=True)
            if np.isnan(t):
                continue
            ta.append(t)
            ln.append(t - cruza(nor, nor[:5].mean(), 0.25, True))
        L.append(f"| {c} | {len(ta)}/{len(ds)} | {np.median(ta) if ta else '-'} | "
                 f"{('med %+.1f [%+.0f,%+.0f]' % (np.median(ln), min(ln), max(ln))) if ln else '-'} |")
    return L


def secao_v2(dados: dict) -> list[str]:
    cab = " | ".join(f"adiant. {n}" for n in SINAIS_LEAD)
    L = [f"| condição | t50 acc (mediana) | {cab} |", "|---|---|" + "---|" * len(SINAIS_LEAD)]
    for c, ds in dados.items():
        if c in ("central_L2", "central_SP", "media_sem_particao"):
            continue
        ta, leads = [], {n: [] for n in SINAIS_LEAD}
        for d in ds:
            t = t50(d["acc_cont"].values)
            if np.isnan(t):
                continue
            ta.append(t)
            for n, col in SINAIS_LEAD.items():
                x = np.abs(d[col].values) if n == "|hess|" else d[col].values
                leads[n].append(t - t50(x))
        f = lambda v: "-" if not len(v) else f"{np.nanmedian(v):+.1f} [{np.nanmin(v):+.0f},{np.nanmax(v):+.0f}]"
        L.append(f"| {c} (n={len(ta)}/{len(ds)}) | {f(ta)} | " + " | ".join(f(leads[n]) for n in SINAIS_LEAD) + " |")
    return L


def versao3() -> list[str]:
    """Definição final: roda UMA vez sobre todas as combinações de nível e tarefa."""
    cab = " | ".join(f"adiant. {n}" for n in SINAIS_LEAD)
    L = [f"| tarefa | m | condição | sementes com perda | acc definida | t50 acc (mediana) | {cab} |",
         "|---|---|---|---|---|---|" + "---|" * len(SINAIS_LEAD)]
    for tipo in TIPOS:
        for nivel in NIVEIS:
            for c, ds in carregar(nivel, tipo).items():
                com_perda = [d for d in ds if gap_principal(d) <= LIMITE_PERDA]
                if not com_perda:
                    continue
                ta, leads, n_def = [], {n: [] for n in SINAIS_LEAD}, 0
                for d in com_perda:
                    acc = d["acc_cont"].values
                    if abs(np.nanmean(acc[-5:]) - np.nanmean(acc[:3])) < LIMITE_PERCURSO:
                        continue                                    # acc indefinida
                    n_def += 1
                    t_acc = t50(acc)
                    if np.isnan(t_acc):
                        continue
                    ta.append(t_acc)
                    for n, col in SINAIS_LEAD.items():
                        x = np.abs(d[col].values) if n == "|hess|" else d[col].values
                        leads[n].append(t_acc - t50(x))
                L.append(f"| {tipo} | {nivel} | {c} | {len(com_perda)}/{len(ds)} | {n_def} | "
                         f"{np.median(ta) if ta else '-'} | " + " | ".join(fmt_lead(leads[n]) for n in SINAIS_LEAD) + " |")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="analise_exploratoria_klein.md")
    a = ap.parse_args()
    L = ["# Análise exploratória dos diagnósticos de Klein et al. (Adendo v3.1)", "",
         "**Exploratória, escrita DEPOIS de ver os resultados, sem critério de decisão e sem auditoria própria.** "
         "Não altera nenhum veredito confirmatório. Gerada por `python -m experiments.analise_exploratoria_klein`.", "",
         "**Pergunta:** na condição que perde plasticidade, a norma dos parâmetros e o maior autovalor da Hessiana "
         "sobem antes da queda da acurácia do contínuo (ordem da cadeia causal da Figura 1 do survey)?", ""]
    L += ["## 1. Trajetórias médias por bloco de tarefas (média entre sementes)", ""]
    for tipo, nivel in (("A", 3), ("A", 10)):
        L += secao_trajetorias(carregar(nivel, tipo), f"Tarefa {tipo}, m={nivel}")
    L += ["## 2. Versão 1, por limiar: DEGENERADA (mantida só como registro)", "",
          "O limiar é cruzado já em t = 5, o primeiro instante permitido; o adiantamento sai sempre t_acc − 5 "
          "e **não informa a ordem entre os sinais**. Tarefa A, m=3:", ""]
    L += secao_v1(carregar(3, "A")) + [""]
    L += ["## 3. Versão 2, meia-excursão sem filtro (prévia; rodada antes da definição da versão 3)", "",
          "Todas as sementes, sem o filtro de perda real. Onde não há queda real de acurácia (por exemplo `fedavg_iid` "
          "na tarefa B) o t50 é ruído. Mediana [mín, máx] entre 5 sementes.", ""]
    for tipo, nivel in (("A", 3), ("A", 10), ("B", 3)):
        L += [f"**Tarefa {tipo}, m={nivel}**", ""] + secao_v2(carregar(nivel, tipo)) + [""]
    L += ["## 4. Versão 3, definição final (filtro de perda real, percurso da acurácia ≥ 0,10, mediana e IC)", "",
          "Combinações (tarefa, nível, condição) com ao menos uma semente com gap médio das 5 últimas tarefas ≤ −0,10. "
          "Adiantamento positivo = o sinal se moveu antes da acurácia. Mediana entre as sementes com perda e acurácia "
          "definida, IC 95% bootstrap da mediana (n < 3: sem IC). Resolução: uma tarefa; suavização de 3 tarefas.", ""]
    L += versao3()
    open(a.saida, "w").write("\n".join(L) + "\n")
    print("\n".join(versao3()))


if __name__ == "__main__":
    main()
