"""MLP 32-128-128-10 com ReLU e inicialização controlada por semente."""
import math

import torch
from torch import nn

from src.config import CLASSES, D, OCULTA, dispositivo


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(D, OCULTA)
        self.l2 = nn.Linear(OCULTA, OCULTA)
        self.l3 = nn.Linear(OCULTA, CLASSES)

    def ativacoes(self, x: torch.Tensor):
        """Devolve (h1, h2, logits); h1 e h2 são pós-ReLU."""
        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        return h1, h2, self.l3(h2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ativacoes(x)[2]


def criar_mlp(semente_int: int) -> MLP:
    """MLP nova. Mesma regra do nn.Linear (uniforme em ±1/sqrt(fan_in)),
    mas sorteada com um gerador próprio, sem tocar no estado global do torch."""
    modelo = MLP()
    g = torch.Generator()
    g.manual_seed(semente_int)
    with torch.no_grad():
        for camada in (modelo.l1, modelo.l2, modelo.l3):
            limite = 1.0 / math.sqrt(camada.in_features)
            camada.weight.copy_((torch.rand(camada.weight.shape, generator=g) * 2 - 1) * limite)
            camada.bias.copy_((torch.rand(camada.bias.shape, generator=g) * 2 - 1) * limite)
    return modelo.to(dispositivo())
