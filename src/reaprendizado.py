"""Reaprendizado (B_fixa, rodada 2): quantos passos para voltar a 90% da acurácia antiga.

Uma CÓPIA do modelo (nunca o original) treina centralizado na tarefa antiga j (batch 20, Adam novo a
cada 15 passos, 1000 pontos), igual para todas as condições (régua comum). Conta o primeiro múltiplo
de 15 (contando 0) com acurácia >= REAPRENDE_FRACAO * a[j][j], até o teto. Rede nova na mesma tarefa
serve de linha de base (economia = 1 - passos_cópia / passos_nova).
"""
import copy

import torch

from src.config import BATCH, N, REAPRENDE_PASSO, REAPRENDE_TETO
from src.metricas import acuracia
from src.modelo import MLP, criar_mlp
from src.sementes import gerador, semente
from src.treino import treinar_passos


def passos_para_limiar(modelo: MLP, X, y, limiar: float, g: torch.Generator,
                       teto: int = REAPRENDE_TETO, passo: int = REAPRENDE_PASSO) -> tuple[int, bool, float]:
    """(passos, censurado, acurácia inicial). Passos = 0 se já está no limiar; censurado se o teto
    foi atingido sem chegar ao limiar (passos = teto)."""
    copia = copy.deepcopy(modelo)
    inicial = acuracia(copia, X, y)
    if inicial >= limiar:
        return 0, False, inicial
    todos = torch.arange(N, device=X.device)
    feitos = 0
    while feitos < teto:
        treinar_passos(copia, X, y, todos, passo, BATCH, passo, g)
        feitos += passo
        if acuracia(copia, X, y) >= limiar:
            return feitos, False, inicial
    return teto, True, inicial


def reaprender(modelo: MLP, seq, sem: int, t: int, j: int, a_jj: float, fracao: float) -> dict:
    """Uma tentativa (tarefa atual t, tarefa antiga j): cópia do modelo e rede nova na tarefa j."""
    X, y = seq.dados(j)
    limiar = fracao * a_jj
    ps_c, cen_c, ini_c = passos_para_limiar(modelo, X, y, limiar, gerador(sem, "reaprende", t, 2 * j))
    nova = criar_mlp(semente(sem, "reaprende_nova", t, j))
    ps_n, cen_n, _ = passos_para_limiar(nova, X, y, limiar, gerador(sem, "reaprende", t, 2 * j + 1))
    return {"tarefa_atual": t, "tarefa_antiga": j, "a_jj": a_jj, "limiar": limiar, "acc_atual_j": ini_c,
            "passos_copia": ps_c, "censurado_copia": cen_c, "passos_nova": ps_n, "censurado_nova": cen_n}
