"""Laço de treino com passos contados e Adam recriado em ritmo fixo."""
import torch
from torch import nn

from src.config import LR
from src.modelo import MLP


def treinar_passos(
    modelo: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    n_passos: int,
    batch: int,
    reset_cada: int,
    gerador: torch.Generator,
    wd: float = 0.0,
) -> int:
    """Treina `modelo` por exatamente `n_passos` passos de otimizador.

    - Os lotes vêm de permutações de `indices` (uma nova a cada época, o lote
      nunca atravessa a fronteira de época).
    - O Adam é RECRIADO a cada `reset_cada` passos (estado do otimizador zerado).
      No FL local, reset_cada = n_passos: um Adam novo por rodada.
    - wd é L2 clássico: soma wd*theta ao gradiente (weight_decay do Adam).
    Devolve o número de passos executados (usado nos testes).
    """
    n = len(indices)
    assert n % batch == 0, "n deve ser múltiplo do batch"
    lotes_por_epoca = n // batch
    perda = nn.CrossEntropyLoss()
    opt = None
    perm = indices
    for passo in range(n_passos):
        if passo % reset_cada == 0:
            opt = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=wd)
        if passo % lotes_por_epoca == 0:
            ordem = torch.randperm(n, generator=gerador).to(indices.device)
            perm = indices[ordem]
        k = passo % lotes_por_epoca
        idx = perm[k * batch:(k + 1) * batch]
        opt.zero_grad(set_to_none=True)
        perda(modelo(X[idx]), y[idx]).backward()
        opt.step()
    return n_passos


@torch.no_grad()
def shrink_and_perturb(modelo: MLP, lam: float, escala: float, gerador: torch.Generator) -> None:
    """Ash e Adams (2020): theta <- lam*theta + N(0, escala^2), in-place."""
    for p in modelo.parameters():
        ruido = torch.randn(p.shape, generator=gerador).to(p.device) * escala
        p.mul_(lam).add_(ruido)
