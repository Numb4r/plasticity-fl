"""Médias no tempo para um aprendiz único (rodada 2): Hare & Tortoise e Lookahead.

Os dois reaproveitam o `treino_local` do FedAvg (15 passos, Adam novo a cada rodada), de modo que
com mu = 0 ambos reproduzem EXATAMENTE o `sem_media_k1` (mesmos sorteios, mesmas contas).
"""
import torch

from src.federado import _vetor_estado, treino_local
from src.metricas import cosseno_medio
from src.modelo import MLP


def clonar_estado(modelo: MLP) -> dict:
    return {k: v.detach().clone() for k, v in modelo.state_dict().items()}


def misturar(tartaruga: dict, lebre: dict, mu: float) -> dict:
    """tartaruga <- mu * tartaruga + (1 - mu) * lebre, em TODAS as entradas do state_dict
    (conta em dupla precisão; com mu = 0 o resultado é exatamente a lebre)."""
    saida = {}
    for k, t in tartaruga.items():
        if t.is_floating_point():
            saida[k] = (t.double() * mu + lebre[k].double() * (1.0 - mu)).to(t.dtype)
        else:
            saida[k] = lebre[k].clone()
    return saida


def treinar_lookahead(modelo: MLP, X, y, particoes: list[torch.Tensor], gerador, rodadas: int, mu: float) -> list[float]:
    """central_lookahead (a especificação literal): a cada rodada a lebre parte da tartaruga,
    treina 15 passos, e tartaruga <- mu*tartaruga + (1-mu)*lebre. `modelo` é a tartaruga
    (o modelo avaliado). É o Lookahead (Zhang et al., 2019) com alfa = 1 - mu: o deslocamento
    por rodada encolhe por (1 - mu)."""
    local = MLP().to(X.device)
    cosenos = []
    for _ in range(rodadas):
        tartaruga = clonar_estado(modelo)
        lebre = treino_local(local, tartaruga, X, y, particoes[0], gerador)
        cosenos.append(cosseno_medio([_vetor_estado(lebre) - _vetor_estado(tartaruga)]))
        modelo.load_state_dict(misturar(tartaruga, lebre, mu))
    return cosenos


def treinar_ht(modelo: MLP, extra: dict, X, y, particoes: list[torch.Tensor], gerador, rodadas: int, mu: float,
                passos: int | None = None, ao_fim_de_rodada=None) -> list[float]:
    """central_HT (Hare & Tortoise de verdade). `modelo` é a LEBRE (a rede que aprende e é medida);
    `extra["tartaruga"]` é a tartaruga (state_dict), que persiste entre tarefas. No início da tarefa
    a lebre volta para a tartaruga (e só aí); em cada rodada a lebre treina 15 passos a partir de
    onde está e a tartaruga vira a média móvel exponencial da trajetória da lebre. `passos` = passos locais
    por rodada (None = 15; o central_HT_cal usa um múltiplo de 15 calibrado na semente 1000)."""
    local = MLP().to(X.device)
    modelo.load_state_dict(extra["tartaruga"])
    cosenos = []
    for _ in range(rodadas):
        antes = clonar_estado(modelo)
        lebre = treino_local(local, antes, X, y, particoes[0], gerador, passos=passos)
        cosenos.append(cosseno_medio([_vetor_estado(lebre) - _vetor_estado(antes)]))
        modelo.load_state_dict(lebre)
        extra["tartaruga"] = misturar(extra["tartaruga"], lebre, mu)
        if ao_fim_de_rodada is not None:        # só leitura (calibração do R): recebe a lebre e o nº de rodadas feitas
            ao_fim_de_rodada(modelo, len(cosenos))
    return cosenos
