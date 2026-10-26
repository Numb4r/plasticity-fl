"""Peças locais das técnicas da rodada 4: treino local com ajustes de gradiente e o ReDo.

`treinar_passos_ext` repete EXATAMENTE o laço de `treinar_passos` (mesmos sorteios, mesmas contas) e só
acrescenta dois ganchos opcionais: um `ajuste` do gradiente (FedProx, SCAFFOLD, L2Init) e a escolha dos
parâmetros que o otimizador atualiza (FedRep). Sem ganchos, o resultado é idêntico bit a bit.
"""
import math

import torch
from torch import nn

from src.config import LR
from src.modelo import MLP


def treinar_passos_ext(modelo: MLP, X, y, indices, n_passos: int, batch: int, reset_cada: int, gerador, *,
                       wd: float = 0.0, ajuste=None, so_params=None, ao_fim_de_bloco=None) -> int:
    """Como `treinar_passos`, com `ajuste()` (muda `p.grad` depois do backward, antes do passo) e
    `so_params` (lista dos parâmetros que o Adam atualiza; padrão: todos)."""
    n = len(indices)
    assert n % batch == 0, "n deve ser múltiplo do batch"
    lotes_por_epoca = n // batch
    perda = nn.CrossEntropyLoss()
    params = list(modelo.parameters()) if so_params is None else list(so_params)
    opt, perm = None, indices
    for passo in range(n_passos):
        if passo % reset_cada == 0:
            opt = torch.optim.Adam(params, lr=LR, weight_decay=wd)
        if passo % lotes_por_epoca == 0:
            ordem = torch.randperm(n, generator=gerador).to(indices.device)
            perm = indices[ordem]
        k = passo % lotes_por_epoca
        idx = perm[k * batch:(k + 1) * batch]
        opt.zero_grad(set_to_none=True)
        perda(modelo(X[idx]), y[idx]).backward()
        if ajuste is not None:
            ajuste()
        opt.step()
        if ao_fim_de_bloco is not None and (passo + 1) % reset_cada == 0:
            ao_fim_de_bloco(passo + 1)
    return n_passos


def montar_ajuste(modelo: MLP, *, prox=None, corr=None, l2init=None):
    """Monta o `ajuste` de gradiente. Cada termo é opcional:
    prox = (mu, ref): FedProx, grad += mu * (p - ref[nome]);
    corr = {nome: c - c_k}: SCAFFOLD, grad += corr[nome];
    l2init = (lam, theta0): L2Init (Kumar et al., 2023), grad += lam * (p - theta0[nome])."""
    nomes = dict(modelo.named_parameters())

    @torch.no_grad()
    def ajuste() -> None:
        for n, p in nomes.items():
            if p.grad is None:
                continue
            if prox is not None:
                p.grad.add_(p - prox[1][n], alpha=prox[0])
            if corr is not None:
                p.grad.add_(corr[n])
            if l2init is not None:
                p.grad.add_(p - l2init[1][n], alpha=l2init[0])

    return ajuste


# ---------------------------------------------------------------- ReDo (Sokar et al., 2023)
@torch.no_grad()
def scores_neuronios(modelo: MLP, X) -> list[torch.Tensor]:
    """Score de cada neurônio das duas camadas ocultas: E|h_i| / média_j E|h_j| (o mesmo da fração de
    dormentes). Camada com todas as ativações zero: score 0 em todos (todos dormentes)."""
    h1, h2, _ = modelo.ativacoes(X)
    saida = []
    for h in (h1, h2):
        s = h.abs().mean(dim=0)
        m = s.mean()
        saida.append(torch.zeros_like(s) if m.item() == 0.0 else s / m)
    return saida


def mascara_media(scores_clientes: list[list[torch.Tensor]], pesos: list[int], tau: float) -> list[torch.Tensor]:
    """ReDo federado por média: score agregado = média ponderada dos scores dos clientes; dormente se <= tau."""
    total = float(sum(pesos))
    saida = []
    for camada in range(2):
        agg = sum(s[camada].double() * (w / total) for s, w in zip(scores_clientes, pesos))
        saida.append(agg <= tau)
    return saida


def mascara_voto(scores_clientes: list[list[torch.Tensor]], pesos: list[int], tau: float, q: float) -> list[torch.Tensor]:
    """ReDo federado por voto: o neurônio é reinicializado se, em pelo menos a fração q dos clientes
    (ponderada), o score local é <= tau."""
    total = float(sum(pesos))
    saida = []
    for camada in range(2):
        voto = sum((s[camada] <= tau).double() * (w / total) for s, w in zip(scores_clientes, pesos))
        saida.append(voto >= q - 1e-12)
    return saida


@torch.no_grad()
def redo_reiniciar(modelo: MLP, mascaras: list[torch.Tensor], gerador: torch.Generator) -> int:
    """Reinicializa os neurônios marcados (ReDo): pesos de entrada ~ U(±1/sqrt(fan_in)) como a init do
    projeto, viés 0, pesos de saída 0. Devolve quantos neurônios foram reiniciados."""
    total = 0
    for ent, sai, mascara in ((modelo.l1, modelo.l2, mascaras[0]), (modelo.l2, modelo.l3, mascaras[1])):
        idx = mascara.nonzero().flatten()
        if len(idx) == 0:
            continue
        lim = 1.0 / math.sqrt(ent.in_features)
        novo = (torch.rand((len(idx), ent.in_features), generator=gerador) * 2 - 1) * lim
        ent.weight[idx] = novo.to(ent.weight.device)
        ent.bias[idx] = 0.0
        sai.weight[:, idx] = 0.0
        total += len(idx)
    return total
