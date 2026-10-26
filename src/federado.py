"""FedAvg simulado em PyTorch puro (sem Flower)."""
import torch

from src.config import BATCH, S_LOCAL
from src.metricas import cosseno_medio
from src.modelo import MLP
from src.treino import treinar_passos, treinar_passos_geral


def media_ponderada(estados: list[dict], pesos: list[int]) -> dict:
    """Média ponderada pelo número de amostras de TODAS as entradas do state_dict.

    Entradas de ponto flutuante são ponderadas; entradas inteiras (se houver,
    como contadores) são promediadas e arredondadas para manter o dtype.
    """
    total = float(sum(pesos))
    saida = {}
    for chave in estados[0]:
        acc = sum(e[chave].double() * (w / total) for e, w in zip(estados, pesos))
        if estados[0][chave].is_floating_point():
            saida[chave] = acc.to(estados[0][chave].dtype)
        else:
            saida[chave] = acc.round().to(estados[0][chave].dtype)
    return saida


def _vetor_estado(estado: dict) -> torch.Tensor:
    return torch.cat([v.flatten().float() for v in estado.values() if v.is_floating_point()])


def treino_local(
    modelo_local: MLP, estado_global: dict, X, y, indices, gerador, geral: bool = False, passos: int | None = None
) -> dict:
    """Um cliente numa rodada: parte do modelo global e faz S_LOCAL=15 passos, batch 20.

    O número de passos é FIXO, seja qual for o tamanho dos dados do cliente
    (100 pontos = 3 épocas; 1000 pontos = 0,3 época). O Adam é RECRIADO a cada
    rodada (reset_cada = S_LOCAL): o estado do otimizador local nunca sobrevive
    de uma rodada para a outra.
    """
    modelo_local.load_state_dict(estado_global)
    if geral:        # clientes de tamanho qualquer (Dirichlet, rodada 2); caminho separado
        treinar_passos_geral(modelo_local, X, y, indices, S_LOCAL, BATCH, gerador)
    else:
        # `passos` (só o central_HT_cal): rodada mais longa, com o Adam recriado a cada S_LOCAL passos;
        # com o padrão (S_LOCAL passos) a chamada é idêntica à da rodada 1
        treinar_passos(modelo_local, X, y, indices, passos or S_LOCAL, BATCH, S_LOCAL, gerador)
    return {k: v.detach().clone() for k, v in modelo_local.state_dict().items()}


def treinar_fedavg(
    modelo_global: MLP, X, y, particoes: list[torch.Tensor], gerador: torch.Generator,
    rodadas: int, geral: bool = False, ao_fim_de_rodada=None,
) -> list[float]:
    """`rodadas` rodadas de FedAvg (25·m) sobre o modelo global (in-place).

    Devolve o cosseno médio entre as atualizações dos clientes em cada rodada
    (NaN com 1 cliente). `ao_fim_de_rodada(modelo_global, r)`, se dado, só lê (calibração de R, rodada 4).
    """
    cosenos = []
    local = MLP().to(X.device)
    particoes = [p for p in particoes if len(p) > 0]       # cliente vazio fica fora da rodada e do peso
    pesos = [len(p) for p in particoes]
    for r in range(rodadas):
        global_antes = {k: v.detach().clone() for k, v in modelo_global.state_dict().items()}
        vetor_global = _vetor_estado(global_antes)
        estados = [treino_local(local, global_antes, X, y, p, gerador, geral) for p in particoes]
        cosenos.append(cosseno_medio([_vetor_estado(e) - vetor_global for e in estados]))
        modelo_global.load_state_dict(media_ponderada(estados, pesos))
        if ao_fim_de_rodada is not None:
            ao_fim_de_rodada(modelo_global, r + 1)
    return cosenos
