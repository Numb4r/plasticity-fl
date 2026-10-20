"""Sondas de plasticidade. Trabalham SEMPRE numa cópia; a rede original não muda."""
import copy

import torch
import torch.nn.functional as F

from src.config import LR, SONDA_ALVOS, SONDA_BATCH, SONDA_PASSOS
from src.modelo import MLP, criar_mlp
from src.sementes import gerador, semente


def _treinar_imitando(modelo: MLP, X, alvo, g: torch.Generator) -> float:
    """SONDA_PASSOS passos de Adam com MSE para imitar `alvo`; devolve a perda final em X."""
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    n = len(X)
    for _ in range(SONDA_PASSOS):
        idx = torch.randint(0, n, (SONDA_BATCH,), generator=g).to(X.device)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(modelo(X[idx]), alvo[idx]).backward()
        opt.step()
    with torch.no_grad():
        return F.mse_loss(modelo(X), alvo).item()


def sonda_alvos_aleatorios(modelo: MLP, X, sem: int, t: int) -> dict:
    """Sonda de alvos aleatórios (estilo Lyle et al., 2022).

    Para cada uma de 3 redes-alvo recém-inicializadas (semente nova): treina uma
    CÓPIA do modelo e uma rede nova, com os mesmos alvos, os mesmos lotes e os mesmos
    passos, e compara a perda final. razao = perda_cópia / perda_rede_nova
    (média das 3 razões). Razão > 1 = a rede perdeu capacidade de aprender alvos novos.
    """
    perdas_copia, perdas_nova, razoes = [], [], []
    for k in range(SONDA_ALVOS):
        alvo_net = criar_mlp(semente(sem, "sonda_alvo", t, k))
        with torch.no_grad():
            alvo = alvo_net(X)
        copia = copy.deepcopy(modelo)                      # a original não é tocada
        nova = criar_mlp(semente(sem, "sonda_nova", t, k))
        # mesmos lotes para a cópia e para a rede nova: só a inicialização difere
        pc = _treinar_imitando(copia, X, alvo, gerador(sem, "sonda_embaralha", t, k))
        pn = _treinar_imitando(nova, X, alvo, gerador(sem, "sonda_embaralha", t, k))
        perdas_copia.append(pc)
        perdas_nova.append(pn)
        razoes.append(pc / pn)
    n = len(razoes)
    return {
        "sonda_perda_copia": sum(perdas_copia) / n,
        "sonda_perda_nova": sum(perdas_nova) / n,
        "sonda_razao": sum(razoes) / n,
    }
