"""Toda a aleatoriedade do projeto sai daqui.

Cada fonte de aleatoriedade (dados, rótulos, init da contínua, init da rede nova...)
recebe um "papel" próprio. Papéis diferentes nunca compartilham semente, mesmo com
a mesma semente-base. Assim as sementes são independentes entre si e a mesma
semente-base dá os mesmos X, rótulos e professores em todas as condições (pareado).
"""
import numpy as np
import torch

# ATENÇÃO: só acrescente papéis NO FIM; inserir no meio muda todas as sementes.
PAPEIS = (
    "dados", "rotulos", "professor", "particao",
    "init_continua", "init_nova", "embaralha_continua", "embaralha_nova",
    "sp_ruido", "sonda_alvo", "sonda_nova", "sonda_embaralha",
    "calib_nova", "calib_embaralha",
    "media_tarefa", "dados_tarefa",   # B_disjunta (adendo 1), acrescentados no fim
    "diag_indices", "diag_hess_v0",   # diagnósticos de Klein et al. (Desvios v3.1)
    "particao_dirichlet", "ref_centro", "reaprende", "reaprende_nova",   # rodada 2, acrescentados no fim
    "redo_reinicio",   # rodada 4: sorteio dos pesos de entrada dos neurônios reinicializados (ReDo)
    "sp_rodada",       # rodada 4: ruído do shrink-and-perturb por rodada (não reusa o "sp_ruido" por tarefa)
)


def semente(base: int, papel: str, i: int = 0, j: int = 0) -> int:
    """Semente inteira determinística para (base, papel, i, j)."""
    entropia = [base, PAPEIS.index(papel), i, j]
    return int(np.random.SeedSequence(entropia).generate_state(1)[0])


def gerador(base: int, papel: str, i: int = 0, j: int = 0) -> torch.Generator:
    """torch.Generator (CPU) semeado por semente(base, papel, i, j)."""
    g = torch.Generator()
    g.manual_seed(semente(base, papel, i, j))
    return g
