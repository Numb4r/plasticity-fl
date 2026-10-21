"""Mede o custo das métricas de Klein et al. e projeta o acréscimo no tempo total do completo.

Uso: python -m experiments.medir_custo [--log logs/pipeline_teste_curto.log]
O tempo de treino de cada tipo de execução vem das linhas finais do log do teste curto
("... tarefa 30/30 ... (XXXs)"); o custo por medição é medido aqui, em regime (após aquecimento).
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import torch

from src import config, diagnosticos as dg
from src.config import tarefas_com_nova
from src.dados import Sequencia
from src.modelo import criar_mlp


def tempo_medicao(completo: bool = True, n: int = 12) -> float:
    seq = Sequencia("A", 1)
    modelo = criar_mlp(5)
    dg.medir(modelo, *seq.dados(0), 1, 0, 0, 0, "x_", None, completo)            # aquecimento
    ts = []
    for t in range(n):
        X, y = seq.dados(t)
        t0 = time.perf_counter()
        dg.medir(modelo, X, y, 1, t, t % 2, 0, "x_", None, completo)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def tempos_de_treino(log: Path) -> dict:
    """(condição, nível) -> segundos do treino de uma execução de T=30, lidos do log do teste curto."""
    out = {}
    for linha in log.read_text().splitlines():
        m = re.search(r"(\w+) m(\d+) A s1000 tarefa 30/30 .*\((\d+)s\)", linha)
        if m:
            out[(m.group(1), int(m.group(2)))] = float(m.group(3))
    return out


def projetar(c_cheio: float, c_leve: float, treino: dict, m_star: int = 10) -> dict:
    """Soma, em segundos de CPU, o treino do completo (aprox.) e as medições, por configuração."""
    def t_fl(cond, m):          # linear em m entre os dois níveis medidos (1 e 10)
        a, b = treino[(cond, 1)], treino[(cond, 10)]
        return a + (b - a) * (m - 1) / 9
    niveis = config.NIVEIS
    n30, n12 = len(tarefas_com_nova(30)), len(tarefas_com_nova(12))
    # por (semente, tarefa), execuções de T=30: fedavg_iid, sem_media_k1, central_passos e
    # central_aprendizado (esta aproximada pelo custo de central_passos) em cada nível,
    # mais L2 e SP (idem) só no nível m*
    treino_30 = sum(t_fl("fedavg_iid", m) + t_fl("sem_media_k1", m) + 2 * t_fl("central_passos", m)
                    for m in niveis) + 2 * t_fl("central_passos", m_star)
    n_exec_30 = 4 * len(niveis) + 2
    # media_sem_particao (T=12): mesmo custo por tarefa do fedavg_iid, 19 redes em vez de 46
    treino_12 = sum(t_fl("fedavg_iid", m) for m in niveis) * (12 + n12) / (30 + n30)
    n_exec_12 = len(niveis)
    treino_total = 15 * (treino_30 + treino_12)                         # 3 tarefas x 5 sementes
    medicoes_30 = 2 * (30 + n30)                                        # contínuo (T) + redes novas, início e fim
    medicoes_12 = 2 * (12 + n12)
    # com o corte (c): Hessiana+espectro só nas tarefas com referência
    cheias_30 = 2 * (n30 + n30)
    cheias_12 = 2 * (n12 + n12)
    diag_cheio = 15 * (n_exec_30 * medicoes_30 + n_exec_12 * medicoes_12) * c_cheio
    diag_c = 15 * (n_exec_30 * (cheias_30 * c_cheio + (medicoes_30 - cheias_30) * c_leve)
                   + n_exec_12 * (cheias_12 * c_cheio + (medicoes_12 - cheias_12) * c_leve))
    fixo = 3 * 1800                                                     # calibração + regime + seleção (estimativa grosseira)
    base = treino_total + fixo
    return {"treino_total_cpu_s": treino_total, "fixo_cpu_s": fixo, "base_cpu_s": base,
            "diag_cheio_cpu_s": diag_cheio, "acrescimo_cheio": diag_cheio / base,
            "diag_corte_c_cpu_s": diag_c, "acrescimo_corte_c": diag_c / base,
            "medicoes_por_execucao": {"T30": medicoes_30, "T12": medicoes_12},
            "execucoes": {"T30": 15 * n_exec_30, "T12": 15 * n_exec_12}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/pipeline_teste_curto.log")
    ap.add_argument("--saida", default="custo_diagnosticos.json")
    ap.add_argument("--sem-s", type=float, default=None, help="segundos de fedavg_iid m=1 T=30 SEM diagnósticos")
    ap.add_argument("--com-s", type=float, default=None, help="idem COM diagnósticos (mesma máquina, em paralelo)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    cheio, leve = tempo_medicao(True), tempo_medicao(False)
    treino = tempos_de_treino(Path(a.log))
    res = {"isolado": {"c_cheio_s": cheio, "c_leve_s": leve, **projetar(cheio, leve, treino)}}
    if a.sem_s and a.com_s:      # custo por medição DENTRO de uma execução real (inclui disputa de CPU)
        n_med = 2 * (30 + len(tarefas_com_nova(30)))
        c_situ = (a.com_s - a.sem_s) / n_med
        res["fedavg_m1_medido"] = {"sem_s": a.sem_s, "com_s": a.com_s, "acrescimo": a.com_s / a.sem_s - 1,
                                   "medicoes": n_med, "c_situ_s": c_situ}
        res["em_situacao"] = projetar(c_situ, c_situ * leve / cheio, treino)
    Path(a.saida).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
