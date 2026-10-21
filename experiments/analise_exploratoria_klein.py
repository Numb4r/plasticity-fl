"""Análise EXPLORATÓRIA dos diagnósticos de Klein et al. (Desvios v3.1). Sem critério de decisão.

Pergunta: na condição que perde plasticidade, a norma dos parâmetros e o maior autovalor da
Hessiana sobem ANTES da queda da acurácia do contínuo? (ordem da cadeia causal da Fig. 1 do survey)

Duas leituras, ambas descritivas:
 (1) trajetórias médias por blocos de tarefas (primeiras 5, 10-14, 20-24, últimas 5);
 (2) por semente, o "tempo de meia-excursão" t50 de cada sinal: o primeiro instante em que a
     série suavizada (média móvel de 3 tarefas) percorreu metade do caminho entre o seu nível
     inicial (média das 3 primeiras tarefas) e o final (média das 5 últimas). Sem limiar
     arbitrário. Adiantamento = t50(acc_cont) - t50(sinal); positivo = o sinal se move ANTES.
     (Uma versão anterior com limiares fixos saiu degenerada: o limiar era cruzado no primeiro
     instante permitido e o adiantamento só repetia t_acc - 5. Foi descartada.)
Uso: python -m experiments.analise_exploratoria_klein --nivel 3 --tarefa A
"""
import argparse
import glob

import numpy as np
import pandas as pd

CONDS = ["central_passos", "central_aprendizado", "sem_media_k1", "fedavg_iid", "media_sem_particao",
         "central_L2", "central_SP"]
SINAIS = {"acc_cont": "acc_cont", "norma": "fim_norma_global", "hess": "fim_hess_max",
          "grad_l2": "ini_grad_l2_global", "grad_srank": "ini_grad_srank", "rank_ef": "rank_ef",
          "srank_feat": "srank_feat", "dorm": "dorm_tau0025_l2"}


def carregar(nivel: int, tarefa: str) -> dict:
    out = {}
    for c in CONDS:
        fs = sorted(glob.glob(f"results/{c}_m{nivel}_{tarefa}_[1-5].csv"))
        if fs:
            out[c] = [pd.read_csv(f) for f in fs]
    return out


def cruza(serie: np.ndarray, base: float, delta: float, rel: bool, queda: bool = False) -> float:
    """Primeiro t >= 5 em que a série (média móvel de 3) passa do limiar; NaN se nunca."""
    s = pd.Series(serie).rolling(3, min_periods=3).mean().values
    lim = base - delta if queda else (base * (1 + delta) if rel else base + delta)
    for t in range(5, len(s)):
        if not np.isnan(s[t]) and ((s[t] < lim) if queda else (s[t] > lim)):
            return float(t)
    return float("nan")


def t50(serie: np.ndarray) -> float:
    """Primeiro t em que a série suavizada (média móvel de 3) percorreu metade do caminho entre o
    nível inicial (3 primeiras tarefas) e o final (5 últimas). NaN se a excursão for desprezível."""
    ini, fim = np.nanmean(serie[:3]), np.nanmean(serie[-5:])
    if not np.isfinite(ini) or not np.isfinite(fim) or abs(fim - ini) < 1e-9:
        return float("nan")
    s = pd.Series(serie).rolling(3, min_periods=1).mean().values
    meio = ini + 0.5 * (fim - ini)
    for t in range(len(s)):
        if (s[t] >= meio) if fim > ini else (s[t] <= meio):
            return float(t)
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nivel", type=int, default=3)
    ap.add_argument("--tarefa", default="A")
    a = ap.parse_args()
    dados = carregar(a.nivel, a.tarefa)
    print(f"# Exploratório (sem decisão): tarefa {a.tarefa}, nível m={a.nivel}, sementes 1 a 5\n")
    print("## (1) Trajetórias médias por bloco de tarefas (média entre sementes)\n")
    blocos = {"t0-4": slice(0, 5), "t10-14": slice(10, 15), "t20-24": slice(20, 25), "t25-29": slice(25, 30)}
    for c, ds in dados.items():
        T = len(ds[0])
        print(f"**{c}**  (T={T})")
        print("| sinal | " + " | ".join(blocos) + " |\n|---|" + "---|" * len(blocos))
        for nome, col in SINAIS.items():
            if col not in ds[0]:
                continue
            vals = []
            for b in blocos.values():
                x = [d[col].iloc[b].mean() for d in ds if len(d.iloc[b]) > 0]
                vals.append(f"{np.mean(x):.3g}" if x else "-")
            print(f"| {nome} | " + " | ".join(vals) + " |")
        print()
    print("## (2) Tempo de meia-excursão t50 por semente; adiantamento = t50(acc) - t50(sinal)\n")
    sinais = {"norma": "fim_norma_global", "|hess|": "fim_hess_max", "dormentes": "dorm_tau0025_l2",
              "srank_feat": "srank_feat", "rank_ef": "rank_ef"}
    print("| condição | t50 acc (mediana) | " + " | ".join(f"adiant. {n}" for n in sinais) + " |")
    print("|---|---|" + "---|" * len(sinais))
    for c, ds in dados.items():
        if c in ("central_L2", "central_SP", "media_sem_particao"):
            continue
        t_acc, leads = [], {n: [] for n in sinais}
        for d in ds:
            ta = t50(d["acc_cont"].values)
            if np.isnan(ta):
                continue
            t_acc.append(ta)
            for n, col in sinais.items():
                x = np.abs(d[col].values) if n == "|hess|" else d[col].values
                leads[n].append(ta - t50(x))
        f = lambda v: ("-" if not len(v) else f"{np.nanmedian(v):+.1f} [{np.nanmin(v):+.0f},{np.nanmax(v):+.0f}]")
        print(f"| {c} (n={len(t_acc)}/{len(ds)}) | {f(t_acc)} | " + " | ".join(f(leads[n]) for n in sinais) + " |")

if __name__ == "__main__":
    main()
