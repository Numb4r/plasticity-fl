"""Métricas de plasticidade: acurácia, normas, dormentes, rank efetivo, cosseno."""
import math

import torch

from src.modelo import MLP


@torch.no_grad()
def acuracia(modelo: MLP, X: torch.Tensor, y: torch.Tensor) -> float:
    return (modelo(X).argmax(dim=1) == y).float().mean().item()


def vetor_parametros(modelo: MLP) -> torch.Tensor:
    """Todos os parâmetros achatados num vetor (cópia)."""
    return torch.cat([p.detach().flatten() for p in modelo.parameters()]).clone()


def norma_parametros(modelo: MLP) -> float:
    return vetor_parametros(modelo).norm().item()


def fracao_dormentes(h: torch.Tensor, tau: float) -> float:
    """Fração de neurônios dormentes numa camada (Sokar et al., 2023).

    h: ativações pós-ReLU, forma (pontos, neurônios).
    score_i = E|h_i| / média_j E|h_j|; o neurônio é dormente se score_i <= tau.
    Camada com todas as ativações zero (média 0) conta 100% dormente.
    """
    score = h.abs().mean(dim=0)
    media = score.mean()
    if media.item() == 0.0:
        return 1.0
    return ((score / media) <= tau).float().mean().item()


@torch.no_grad()
def rank_efetivo(h: torch.Tensor) -> float:
    """Rank efetivo (Roy e Vetterli, 2007): exp da entropia dos valores singulares
    normalizados da matriz de ativações."""
    s = torch.linalg.svdvals(h.float())
    total = s.sum()
    if total.item() == 0.0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return math.exp(-(p * p.log()).sum().item())


def cosseno_medio(atualizacoes: list[torch.Tensor]) -> float:
    """Cosseno médio entre pares de atualizações de clientes. NaN se há 1 cliente."""
    k = len(atualizacoes)
    if k < 2:
        return float("nan")
    U = torch.stack(atualizacoes)
    U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
    G = U @ U.T
    return ((G.sum() - G.diagonal().sum()) / (k * (k - 1))).item()


@torch.no_grad()
def metricas_modelo(modelo: MLP, X: torch.Tensor, taus: tuple[float, ...]) -> dict:
    """Dormentes por camada e tau, e rank efetivo da última camada oculta."""
    h1, h2, _ = modelo.ativacoes(X)
    out = {}
    for tau in taus:
        nome = f"{tau:g}".replace(".", "")
        out[f"dorm_tau{nome}_l1"] = fracao_dormentes(h1, tau)
        out[f"dorm_tau{nome}_l2"] = fracao_dormentes(h2, tau)
    out["rank_ef"] = rank_efetivo(h2)
    return out
