"""Esquecimento em B_disjunta: matriz de acurácias e métricas derivadas.

a[i][j] = acurácia na tarefa j (no X_j dela) depois de treinar até a tarefa i.
Só existe para j <= i; o resto fica NaN.
"""
import numpy as np

from src.dados import Sequencia
from src.metricas import acuracia
from src.modelo import MLP


def linha_da_matriz(modelo: MLP, seq: Sequencia, i: int, T: int) -> np.ndarray:
    """Acurácia do modelo em cada tarefa j <= i (NaN para j > i)."""
    linha = np.full(T, np.nan)
    for j in range(i + 1):
        X, y = seq.dados(j)
        linha[j] = acuracia(modelo, X, y)
    return linha


def metricas_esquecimento(a: np.ndarray, i: int) -> dict:
    """Métricas depois da tarefa i (a: matriz T x T com NaN acima da diagonal).

    acc_media_vistas: média de a[i][j], j <= i.
    esquecimento: Chaudhry et al. (2018), média sobre j < i de
        max_{j <= l < i} a[l][j] - a[i][j].
    bwt: Lopez-Paz e Ranzato (2017), média sobre j < i de a[i][j] - a[j][j].
    Em i = 0 não há tarefa anterior: esquecimento e bwt são NaN.
    """
    out = {"acc_media_vistas": float(np.mean(a[i, :i + 1]))}
    if i == 0:
        out["esquecimento"] = out["bwt"] = float("nan")
        return out
    esq = [np.max(a[j:i, j]) - a[i, j] for j in range(i)]
    bwt = [a[i, j] - a[j, j] for j in range(i)]
    out["esquecimento"] = float(np.mean(esq))
    out["bwt"] = float(np.mean(bwt))
    return out
