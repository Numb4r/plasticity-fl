# Análises exploratórias posteriores aos resultados da rodada 2

Exploratórias, sem critério de decisão, nenhum veredito muda. Métrica = gap pareado (acc_cont − acc_nova) médio nas tarefas da janela com rede nova, média entre sementes [IC 95% bootstrap percentil] (n sementes). Índices: tarefas 8–12 = índices 7 a 11; tarefas 26–30 = índices 25 a 29. Nível m=3.

## 1. Rodada 1: o `media_sem_particao` rodou com T=12; a métrica principal dele usa a janela das tarefas 8–12
`src/config.py:26` (`T_SEM_PARTICAO = 12`), `src/condicoes.py:69` (a condição `media_sem_particao` usa `T=T_SEM_PARTICAO`) e `plots/analise.py:194-201` (janela padrão = `(T-5, T-1)` com o T da própria condição: 7 a 11 para ele, 25 a 29 para as de T=30). Na rodada 2 ele passou a rodar com T=30 (`src/condicoes.py:condicao_r2`), e a métrica principal passou a ser a das tarefas 26–30.

## 2. Rodada 1 (sementes 1 a 5), todas as condições, nas duas janelas
A janela 8–12 é a única em que o `media_sem_particao` (T=12) tem rede nova; a janela 26–30 não existe para ele nessa rodada.

| condição | A, tarefas 8–12 | A, tarefas 26–30 | B, tarefas 8–12 | B, tarefas 26–30 |
|---|---|---|---|---|
| fedavg_iid | 0.088 [0.070, 0.103] (n=5) | -0.062 [-0.090, -0.033] (n=5) | 0.028 [0.026, 0.030] (n=5) | -0.001 [-0.004, 0.003] (n=5) |
| sem_media_k1 | 0.035 [0.011, 0.060] (n=5) | -0.332 [-0.475, -0.238] (n=5) | 0.024 [0.020, 0.029] (n=5) | -0.066 [-0.076, -0.054] (n=5) |
| media_sem_particao | -0.006 [-0.014, 0.006] (n=5) | — | 0.009 [0.006, 0.011] (n=5) | — |
| central_passos | -0.049 [-0.069, -0.029] (n=5) | -0.463 [-0.583, -0.339] (n=5) | 0.009 [0.004, 0.013] (n=5) | -0.067 [-0.072, -0.062] (n=5) |
| central_aprendizado | -0.097 [-0.129, -0.064] (n=5) | -0.452 [-0.585, -0.318] (n=5) | 0.009 [0.003, 0.016] (n=5) | -0.082 [-0.097, -0.069] (n=5) |
| central_L2 | 0.146 [0.141, 0.151] (n=5) | 0.100 [0.089, 0.113] (n=5) | 0.035 [0.033, 0.038] (n=5) | 0.028 [0.026, 0.031] (n=5) |
| central_SP | 0.158 [0.153, 0.163] (n=5) | 0.148 [0.142, 0.154] (n=5) | 0.045 [0.041, 0.048] (n=5) | 0.041 [0.037, 0.044] (n=5) |

## 3. Rodada 2 (sementes 6 a 10), as mesmas condições comuns, nas duas janelas
O `media_sem_particao` da rodada 2 roda com T=30: a janela 8–12 dele é comparável à da rodada 1 (mesmas tarefas, T maior).

| condição | A, tarefas 8–12 | A, tarefas 26–30 | B, tarefas 8–12 | B, tarefas 26–30 |
|---|---|---|---|---|
| fedavg_iid | 0.101 [0.093, 0.107] (n=5) | -0.044 [-0.055, -0.033] (n=5) | 0.031 [0.028, 0.035] (n=5) | 0.003 [-0.001, 0.008] (n=5) |
| media_sem_particao | -0.018 [-0.032, -0.000] (n=5) | -0.326 [-0.430, -0.259] (n=5) | 0.008 [0.006, 0.010] (n=5) | -0.035 [-0.050, -0.023] (n=5) |
| sem_media_k1 | 0.033 [0.013, 0.052] (n=5) | -0.366 [-0.491, -0.262] (n=5) | 0.022 [0.017, 0.026] (n=5) | -0.080 [-0.103, -0.058] (n=5) |
| central_aprendizado | -0.122 [-0.130, -0.113] (n=5) | -0.592 [-0.660, -0.479] (n=5) | 0.003 [-0.001, 0.008] (n=5) | -0.095 [-0.104, -0.088] (n=5) |
| central_HT | 0.145 [0.136, 0.154] (n=5) | 0.188 [0.183, 0.194] (n=5) | 0.042 [0.039, 0.045] (n=5) | 0.051 [0.048, 0.055] (n=5) |
| central_lookahead | 0.163 [0.158, 0.170] (n=5) | 0.102 [0.082, 0.125] (n=5) | 0.067 [0.061, 0.073] (n=5) | 0.030 [0.024, 0.037] (n=5) |

## 4. `fedavg_iid` da rodada 2 (sementes 6 a 10) ao lado do da rodada 1 (sementes 1 a 5), m=3
(gap, acc_cont e acc_nova; a rodada 1 não gravou a perda)

**gap**

| rodada | A, tarefas 8–12 | A, tarefas 26–30 | B, tarefas 8–12 | B, tarefas 26–30 |
|---|---|---|---|---|
| rodada 1 (sementes 1–5) | 0.088 [0.070, 0.103] (n=5) | -0.062 [-0.090, -0.033] (n=5) | 0.028 [0.026, 0.030] (n=5) | -0.001 [-0.004, 0.003] (n=5) |
| rodada 2 (sementes 6–10) | 0.101 [0.093, 0.107] (n=5) | -0.044 [-0.055, -0.033] (n=5) | 0.031 [0.028, 0.035] (n=5) | 0.003 [-0.001, 0.008] (n=5) |

**acc_cont**

| rodada | A, tarefas 8–12 | A, tarefas 26–30 | B, tarefas 8–12 | B, tarefas 26–30 |
|---|---|---|---|---|
| rodada 1 (sementes 1–5) | 0.919 [0.907, 0.931] (n=5) | 0.767 [0.741, 0.793] (n=5) | 0.990 [0.988, 0.991] (n=5) | 0.963 [0.960, 0.967] (n=5) |
| rodada 2 (sementes 6–10) | 0.928 [0.924, 0.932] (n=5) | 0.785 [0.774, 0.797] (n=5) | 0.991 [0.989, 0.993] (n=5) | 0.963 [0.960, 0.966] (n=5) |

**acc_nova**

| rodada | A, tarefas 8–12 | A, tarefas 26–30 | B, tarefas 8–12 | B, tarefas 26–30 |
|---|---|---|---|---|
| rodada 1 (sementes 1–5) | 0.831 [0.824, 0.838] (n=5) | 0.829 [0.824, 0.834] (n=5) | 0.962 [0.960, 0.964] (n=5) | 0.964 [0.962, 0.965] (n=5) |
| rodada 2 (sementes 6–10) | 0.828 [0.825, 0.831] (n=5) | 0.829 [0.825, 0.832] (n=5) | 0.960 [0.957, 0.962] (n=5) | 0.960 [0.957, 0.963] (n=5) |

## 5. H_het exploratório: rede nova e rede contínua do `fedavg_dirichlet` por α (sementes 6 a 10)
Janela das tarefas 26–30 (as 5 últimas, todas com rede nova). `fedavg_iid` e `media_sem_particao` entram só como referência de nível. A perda é a entropia cruzada de treino ao fim da tarefa.

**Tarefa A**

| condição | acc_nova | loss_nova | acc_cont | loss_cont | gap |
|---|---|---|---|---|---|
| fedavg_dirichlet (α=100) | 0.857 [0.853, 0.862] (n=5) | 0.682 [0.675, 0.690] (n=5) | 0.777 [0.755, 0.796] (n=5) | 0.789 [0.744, 0.842] (n=5) | -0.080 [-0.104, -0.060] (n=5) |
| fedavg_dirichlet (α=1) | 0.660 [0.654, 0.666] (n=5) | 1.128 [1.110, 1.147] (n=5) | 0.758 [0.729, 0.787] (n=5) | 0.744 [0.675, 0.809] (n=5) | 0.098 [0.071, 0.127] (n=5) |
| fedavg_dirichlet (α=0.1) | 0.401 [0.386, 0.414] (n=5) | 2.269 [2.067, 2.442] (n=5) | 0.531 [0.511, 0.552] (n=5) | 1.675 [1.559, 1.791] (n=5) | 0.130 [0.110, 0.153] (n=5) |
| fedavg_iid | 0.829 [0.825, 0.832] (n=5) | 0.740 [0.735, 0.746] (n=5) | 0.785 [0.774, 0.797] (n=5) | 0.748 [0.719, 0.777] (n=5) | -0.044 [-0.055, -0.033] (n=5) |
| media_sem_particao | 0.875 [0.872, 0.879] (n=5) | 0.690 [0.682, 0.697] (n=5) | 0.550 [0.450, 0.615] (n=5) | 1.309 [1.194, 1.496] (n=5) | -0.326 [-0.430, -0.259] (n=5) |

**Tarefa B**

| condição | acc_nova | loss_nova | acc_cont | loss_cont | gap |
|---|---|---|---|---|---|
| fedavg_dirichlet (α=100) | 0.968 [0.964, 0.970] (n=5) | 0.178 [0.172, 0.186] (n=5) | 0.958 [0.950, 0.967] (n=5) | 0.194 [0.165, 0.223] (n=5) | -0.010 [-0.019, -0.000] (n=5) |
| fedavg_dirichlet (α=1) | 0.897 [0.889, 0.904] (n=5) | 0.356 [0.342, 0.369] (n=5) | 0.929 [0.921, 0.938] (n=5) | 0.233 [0.209, 0.257] (n=5) | 0.032 [0.024, 0.040] (n=5) |
| fedavg_dirichlet (α=0.1) | 0.673 [0.657, 0.690] (n=5) | 1.331 [1.158, 1.508] (n=5) | 0.725 [0.694, 0.753] (n=5) | 1.020 [0.908, 1.151] (n=5) | 0.052 [0.033, 0.074] (n=5) |
| fedavg_iid | 0.960 [0.957, 0.963] (n=5) | 0.195 [0.191, 0.201] (n=5) | 0.963 [0.960, 0.966] (n=5) | 0.170 [0.157, 0.185] (n=5) | 0.003 [-0.001, 0.008] (n=5) |
| media_sem_particao | 0.985 [0.984, 0.986] (n=5) | 0.149 [0.145, 0.154] (n=5) | 0.950 [0.934, 0.962] (n=5) | 0.262 [0.224, 0.306] (n=5) | -0.035 [-0.050, -0.023] (n=5) |

Mesma tabela na janela das tarefas 8–12:

**Tarefa A**

| condição | acc_nova | loss_nova | acc_cont | loss_cont | gap |
|---|---|---|---|---|---|
| fedavg_dirichlet (α=100) | 0.853 [0.848, 0.858] (n=5) | 0.691 [0.682, 0.698] (n=5) | 0.939 [0.930, 0.949] (n=5) | 0.347 [0.323, 0.373] (n=5) | 0.086 [0.077, 0.095] (n=5) |
| fedavg_dirichlet (α=1) | 0.667 [0.661, 0.674] (n=5) | 1.123 [1.104, 1.143] (n=5) | 0.774 [0.762, 0.789] (n=5) | 0.717 [0.675, 0.753] (n=5) | 0.107 [0.094, 0.120] (n=5) |
| fedavg_dirichlet (α=0.1) | 0.409 [0.387, 0.434] (n=5) | 2.066 [1.955, 2.176] (n=5) | 0.482 [0.456, 0.516] (n=5) | 1.674 [1.587, 1.774] (n=5) | 0.073 [0.058, 0.087] (n=5) |

**Tarefa B**

| condição | acc_nova | loss_nova | acc_cont | loss_cont | gap |
|---|---|---|---|---|---|
| fedavg_dirichlet (α=100) | 0.967 [0.964, 0.970] (n=5) | 0.177 [0.168, 0.184] (n=5) | 0.993 [0.991, 0.995] (n=5) | 0.063 [0.056, 0.074] (n=5) | 0.026 [0.025, 0.026] (n=5) |
| fedavg_dirichlet (α=1) | 0.900 [0.893, 0.908] (n=5) | 0.347 [0.327, 0.363] (n=5) | 0.946 [0.938, 0.954] (n=5) | 0.196 [0.177, 0.216] (n=5) | 0.046 [0.040, 0.053] (n=5) |
| fedavg_dirichlet (α=0.1) | 0.677 [0.655, 0.704] (n=5) | 1.151 [1.070, 1.229] (n=5) | 0.730 [0.712, 0.744] (n=5) | 0.859 [0.798, 0.947] (n=5) | 0.053 [0.034, 0.074] (n=5) |

