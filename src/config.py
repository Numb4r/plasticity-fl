"""Constantes do experimento (valores fixados no PREREGISTRO.md)."""
import os

import torch

# --- dados e modelo ---
N = 1000          # número de pontos
D = 32            # dimensão de entrada
CLASSES = 10
OCULTA = 128      # largura das duas camadas ocultas
LR = 1e-3         # taxa de aprendizado do Adam

# --- protocolo local (igual em todas as condições do tipo FL) ---
BATCH = 20
EPOCAS_LOCAIS = 3
RODADAS = 25
CLIENTES = 10

# --- sequência de tarefas ---
T_PADRAO = 30
T_SEM_PARTICAO = 12
FRESCA_CADA = 3   # rede nova de referência nas tarefas t com t % 3 == 2
SONDA_CADA = 5    # sondas nas tarefas t com t % 5 == 4

# --- passos do protocolo ---
# Um cliente do fedavg_iid tem N/CLIENTES pontos: 100/20 = 5 lotes por época.
PASSOS_RODADA = EPOCAS_LOCAIS * (N // CLIENTES // BATCH)   # 15 passos por rodada
PASSOS_CLIENTE = RODADAS * PASSOS_RODADA                    # 375 passos por tarefa
RESET_CENTRAL = PASSOS_RODADA  # nos centralizados, o Adam é recriado no mesmo ritmo

# --- sonda de alvos aleatórios (Lyle et al., 2022) ---
SONDA_PASSOS = 300
SONDA_ALVOS = 3
SONDA_BATCH = 100

# --- demais ---
SP_RUIDO = 0.01   # escala do ruído do shrink-and-perturb
TAUS = (0.0, 0.025)


def dispositivo() -> torch.device:
    """CPU por padrão; CUDA se houver. PLAST_DISPOSITIVO=cpu|cuda força a escolha."""
    escolha = os.environ.get("PLAST_DISPOSITIVO", "auto")
    if escolha == "auto":
        escolha = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(escolha)
