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
    ao_fim_de_bloco=None,
) -> int:
    """Treina `modelo` por exatamente `n_passos` passos de otimizador.

    - Os lotes vêm de permutações de `indices` (uma nova a cada época, o lote
      nunca atravessa a fronteira de época).
    - O Adam é RECRIADO a cada `reset_cada` passos (estado do otimizador zerado).
      No FL local, reset_cada = n_passos: um Adam novo por rodada.
    - wd é L2 clássico: soma wd*theta ao gradiente (weight_decay do Adam).
    - ao_fim_de_bloco(passos_feitos), se dado, é chamado cada vez que o Adam
      completa um bloco de `reset_cada` passos (usado na calibração incremental).
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
        if ao_fim_de_bloco is not None and (passo + 1) % reset_cada == 0:
            ao_fim_de_bloco(passo + 1)
    return n_passos


@torch.no_grad()
def shrink_and_perturb(modelo: MLP, lam: float, escala: float, gerador: torch.Generator) -> None:
    """Ash e Adams (2020): theta <- lam*theta + N(0, escala^2), in-place."""
    for p in modelo.parameters():
        ruido = torch.randn(p.shape, generator=gerador).to(p.device) * escala
        p.mul_(lam).add_(ruido)


def treinar_passos_geral(
    modelo: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    n_passos: int,
    batch: int,
    gerador: torch.Generator,
) -> int:
    """Treino local para clientes de qualquer tamanho (Dirichlet, rodada 2).

    Caminho SEPARADO do `treinar_passos` (que exige n múltiplo do batch e é o que gerou os
    resultados da rodada 1). Um Adam novo por chamada (uma rodada local). Lotes:
      n >= batch: de permutações encadeadas (o lote pode atravessar a fronteira de época);
      n <  batch: `batch` índices sorteados com reposição.
    Devolve o número de passos executados. `indices` deve ter n >= 1.
    """
    n = len(indices)
    assert n >= 1, "cliente vazio não treina"
    perda = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    fila = indices[:0]
    for _ in range(n_passos):
        if n < batch:
            idx = indices[torch.randint(0, n, (batch,), generator=gerador).to(indices.device)]
        else:
            while len(fila) < batch:
                ordem = torch.randperm(n, generator=gerador).to(indices.device)
                fila = torch.cat([fila, indices[ordem]])
            idx, fila = fila[:batch], fila[batch:]
        opt.zero_grad(set_to_none=True)
        perda(modelo(X[idx]), y[idx]).backward()
        opt.step()
    return n_passos
