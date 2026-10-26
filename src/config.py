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
JANELA_H_PART = range(7, 12)   # ... e nos índices 7 a 11 (janela de H_part: tarefas 8 a 12), Errata (ii)
SONDA_CADA = 5        # sondas nas tarefas t com t % 5 == 4
VERSAO_DADOS = 2      # 2 = professor com logits centralizados (balanceado)

# --- passos do protocolo ---
PASSOS_RODADA = S_LOCAL                       # 15 passos por rodada
PASSOS_CLIENTE = RODADAS_BASE * S_LOCAL       # 375 passos por tarefa no nível m=1
RESET_CENTRAL = S_LOCAL                       # nos centralizados, o Adam é recriado no mesmo ritmo
PASSOS_MAX_CALIB = 15000                      # teto da varredura de calibração (300 épocas)
TOLERANCIA_CALIB = 0.03
TOLERANCIA_HT_CAL = 0.10                      # central_HT_cal: perda CE da rede nova a até 10% (relativo) da do media_sem_particao
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


def tarefas_com_nova(T: int, janela_hpart: bool = True) -> list[int]:
    """Tarefas (0-based) com rede nova de referência (decisão (ii) da Errata, 2026-10-20):
    t mod 3 = 2, as 5 últimas da condição e, se `janela_hpart`, a janela de H_part (índices 7 a 11).
    Com T=30 são 16 tarefas; com T=12 as janelas coincidem e são 7. No braço longo (T=100) a
    janela de H_part não serve (não há H_part) e fica de fora (rodada 2)."""
    return [t for t in range(T)
            if t % FRESCA_CADA == FRESCA_CADA - 1 or t >= T - FRESCA_ULTIMAS
            or (janela_hpart and t in JANELA_H_PART)]


# --- diagnósticos de Klein et al. (Desvios v3.1) ---
DIAG_PONTOS_ESPECTRO = 64     # pontos fixos da tarefa para o espectro do gradiente
SRANK_DELTA = 0.01            # Definição 4.1 do survey
HESS_ITERS = 30               # iterações da power iteration
HESS_TOL = 0.01               # convergiu se a variação relativa < 1% nas últimas 5 iterações
# Cortes de custo (Desvios v3.1, aplicados em ordem só se o completo crescer > 25%):
HESS_PONTOS = None            # (a) None = todos os pontos da tarefa; 256 = subconjunto fixo
HESS_PARTIDA_QUENTE = False   # (b) começar do autovetor da medição anterior da mesma rede
DIAG_SO_REFERENCIAS = False   # (c) Hessiana e espectro só nas tarefas de tarefas_com_nova


# --- Rodada 2 (PREREGISTRO.md, "Rodada 2 · Qual média protege a plasticidade, e por quê?") ---
NIVEL_R2 = 3                          # m* = 3
SEMENTES_R2 = (6, 7, 8, 9, 10)        # sementes de avaliação da rodada 2
SEMENTES_LONGO = (6, 7, 8)            # braço de horizonte longo
T_LONGO = 100
SEMENTES_R2B = (11, 12, 13, 14, 15)    # sementes de avaliação da rodada 2b (central_HT_cal2)
R_CAL2 = tuple(range(75, 151))          # grade de R (rodadas por tarefa) da calibração do central_HT_cal2: de R a 2R, passo 1
MUS_2B = (0.9, 0.99, 0.995, 0.999)    # grade de μ do central_HT_cal2, escolhido depois de R
MUS = (0.5, 0.9, 0.99)                # grade de μ (central_HT e central_lookahead), escolhido na semente 1000
ALFAS = (0.1, 1.0, 100.0)             # fedavg_dirichlet
SONDA_PASSOS_R2 = 20                  # sonda da rodada 2 (a de 300 passos satura)
REF_CENTRO_PONTOS = 10000             # B_fixa: amostra de referência da centralização dos logits
REAPRENDE_FRACAO = 0.9                # reaprendizado: voltar a 90% da acurácia com que a tarefa foi aprendida
REAPRENDE_PASSO = 15
REAPRENDE_TETO = 1500
REAPRENDE_EM = (9, 19, 29)            # ao fim destas tarefas...
REAPRENDE_ANTIGAS = (0, 5)            # ...reaprende estas tarefas antigas


# --- Rodada 4 (4a: técnicas distribuídas; 4b: remédios de plasticidade). Implementada ANTES do pré-registro
# (decisão do autor, 2026-10-26): métodos genéricos, com testes, sem nenhum dado visto. Estes valores são
# padrões do código, não critérios; o pré-registro da rodada 4 vem depois dos resultados da rodada 3. ---
SEMENTES_R4 = (24, 25, 26, 27, 28, 29, 30, 31)
R_GRADE_R4 = tuple(range(50, 201))            # grade de R (rodadas por tarefa) da calibração, passo 1
TOLERANCIA_R4 = 0.10                          # perda CE da rede nova a até 10% (relativo) do alvo (fedavg_iid K=10)
R_BASE_R4 = 75                                # R do fedavg_iid (o alvo da calibração)
FEDREP_PASSOS_CABECA = 5                      # FedRep: 5 passos só na cabeça, depois 10 só no corpo (total 15)
FEDREP_PASSOS_CORPO = 10
DILOCO_LR_EXTERNA = 0.7                       # DiLoCo (Douillard et al., 2023): SGD externo com momento de Nesterov
DILOCO_MOMENTO = 0.9
DILOCO_HS = (15, 50, 150)
REDO_CADA = 5                                 # ReDo (local e federado): a cada 5 rodadas, começando na rodada 0
GRADE_FEDPROX_MU = (0.001, 0.01, 0.1, 1.0)
GRADE_L2_FED = (1e-5, 1e-4, 1e-3)
GRADE_L2INIT = (1e-4, 1e-3, 1e-2)
GRADE_REDO_TAU = (0.025, 0.1)
GRADE_SP_TAREFA = (0.6, 0.8, 0.9, 0.95)
GRADE_SP_RODADA = (0.99, 0.995, 0.999)        # exploratório
GRADE_REDO_VOTO_Q = (0.5, 0.8)                # exploratório
GRADE_L2_CENTRAL = (1e-6, 1e-5, 1e-4, 1e-3)   # referência central_L2 (grade estendida pelo autor)


# --- Rodada 3 (PREREGISTRO.md, "Rodada 3 · Por que a média entre clientes com dados diferentes protege...") ---
SEMENTES_R3 = (16, 17, 18, 19, 20, 21, 22, 23)
KS_R3 = (2, 5, 10, 20)
EPOCAS_LOCAIS_FIXAS = 3                       # braço secundário: S_local = ceil(3 * n / batch) passos por rodada
