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
S_LOCAL = 15          # passos de otimizador por cliente por rodada, QUALQUER que seja o tamanho dos dados
RODADAS_BASE = 25     # rodadas por tarefa no nível m=1; no nível m são RODADAS_BASE * m
CLIENTES = 10
NIVEIS = (1, 3, 10)   # eixo de orçamento m (Desvios v3)

# --- sementes (Desvios v3) ---
SEMENTE_CALIB = 1000                 # só calibração, seleção de lambda e regime de teste
SEMENTES_AVALIACAO = (1, 2, 3, 4, 5)  # confirmatórias; a 0 do desenho antigo é exploratória

# --- sequência de tarefas ---
T_PADRAO = 30
T_SEM_PARTICAO = 12
FRESCA_CADA = 3       # rede nova nas tarefas t com t % 3 == 2 ...
FRESCA_ULTIMAS = 5    # ... e em todas as 5 últimas tarefas da condição
SONDA_CADA = 5        # sondas nas tarefas t com t % 5 == 4
VERSAO_DADOS = 2      # 2 = professor com logits centralizados (balanceado)

# --- passos do protocolo ---
PASSOS_RODADA = S_LOCAL                       # 15 passos por rodada
PASSOS_CLIENTE = RODADAS_BASE * S_LOCAL       # 375 passos por tarefa no nível m=1
RESET_CENTRAL = S_LOCAL                       # nos centralizados, o Adam é recriado no mesmo ritmo
PASSOS_MAX_CALIB = 15000                      # teto da varredura de calibração (300 épocas)
TOLERANCIA_CALIB = 0.03
LIMITE_ESQUECIMENTO = 0.05                    # Adendo 1: 5 pontos percentuais

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


def tarefas_com_nova(T: int) -> list[int]:
    """Tarefas (0-based) em que há rede nova de referência: t % 3 == 2 e as 5 últimas."""
    return [t for t in range(T) if t % FRESCA_CADA == FRESCA_CADA - 1 or t >= T - FRESCA_ULTIMAS]
