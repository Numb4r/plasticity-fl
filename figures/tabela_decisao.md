# Tabela de decisão

## (a) Nível de avaliação
- m* = 3 (de `results/calibracao/regime.json`: menor nível com gap de central_aprendizado ≤ −0,10 na tarefa A, semente 1000).
- Gap de central_aprendizado, tarefa A, semente 1000, por nível: m=1: 0.102, m=3: -0.351, m=10: -0.785.
- As tabelas (b), (b'), (c) e as decisões usam o nível m=3; (d) e (g) percorrem todos os níveis.

## (b) Métrica principal no nível m=3 (gap médio nas últimas 5 tarefas; média [IC 95% bootstrap] entre sementes)

| condição | tarefa A | tarefa B |
|---|---|---|
| fedavg_iid | -0.0617 [-0.0903, -0.0328] (n=5, n_ref=5) | -0.0006 [-0.0039, 0.0029] (n=5, n_ref=5) |
| sem_media_k1 | -0.3324 [-0.4746, -0.2375] (n=5, n_ref=5) | -0.0660 [-0.0763, -0.0542] (n=5, n_ref=5) |
| media_sem_particao | -0.0058 [-0.0140, 0.0059] (n=5, n_ref=5) | 0.0090 [0.0065, 0.0114] (n=5, n_ref=5) |
| central_passos | -0.4626 [-0.5826, -0.3386] (n=5, n_ref=5) | -0.0674 [-0.0725, -0.0622] (n=5, n_ref=5) |
| central_aprendizado | -0.4518 [-0.5854, -0.3181] (n=5, n_ref=5) | -0.0825 [-0.0974, -0.0692] (n=5, n_ref=5) |
| central_L2 | 0.0997 [0.0893, 0.1130] (n=5, n_ref=5) | 0.0284 [0.0259, 0.0310] (n=5, n_ref=5) |
| central_SP | 0.1481 [0.1418, 0.1540] (n=5, n_ref=5) | 0.0409 [0.0374, 0.0444] (n=5, n_ref=5) |

## (b') fedavg_iid vs central_aprendizado, janela 7..11 (índices), m=3

| condição | tarefa A | tarefa B |
|---|---|---|
| fedavg_iid | 0.0878 [0.0696, 0.1030] (n=5, n_ref=5) | 0.0280 [0.0256, 0.0304] (n=5, n_ref=5) |
| central_aprendizado | -0.0970 [-0.1295, -0.0644] (n=5, n_ref=5) | 0.0093 [0.0033, 0.0158] (n=5, n_ref=5) |
| media_sem_particao | -0.0058 [-0.0140, 0.0059] (n=5, n_ref=5) | 0.0090 [0.0065, 0.0114] (n=5, n_ref=5) |

## (c) B_disjunta no nível m=3: plasticidade, esquecimento e marcação do Adendo 1

Nas tarefas A e B o esquecimento não é medido (o mesmo X recebe rótulos conflitantes); a marcação só vale para B_disjunta.

Limite: média_esq(condição) ≤ média_esq(central_aprendizado) + 0,05 = 0.8924

| condição | gap contínua−nova, últimas 5 tarefas (média [IC 95%], n) | esquecimento médio final (média [IC 95%], n) | marcação |
|---|---|---|---|
| fedavg_iid | 0.0631 [0.0572, 0.0709] (n=5, n_ref=5) | 0.8688 [0.8668, 0.8708] (n=5) | ganho válido |
| sem_media_k1 | 0.0169 [0.0102, 0.0254] (n=5, n_ref=5) | 0.8445 [0.8416, 0.8474] (n=5) | ganho válido |
| media_sem_particao | 0.0512 [0.0470, 0.0556] (n=5, n_ref=5) | 0.8690 [0.8677, 0.8702] (n=5) | ganho válido |
| central_passos | 0.0036 [-0.0128, 0.0174] (n=5, n_ref=5) | 0.8472 [0.8425, 0.8527] (n=5) | sem ganho |
| central_aprendizado | -0.0157 [-0.0274, -0.0019] (n=5, n_ref=5) | 0.8424 [0.8363, 0.8482] (n=5) | sem ganho (referência do limite) |
| central_L2 | 0.0761 [0.0720, 0.0802] (n=5, n_ref=5) | 0.8774 [0.8747, 0.8801] (n=5) | ganho válido |
| central_SP | 0.1024 [0.0960, 0.1087] (n=5, n_ref=5) | 0.8827 [0.8815, 0.8839] (n=5) | ganho válido |

## (d) Dose-resposta: métrica principal por condição × nível × tarefa (central_L2 e central_SP só existem em m*)

**Tarefa A**

| condição | m=1 | m=3 | m=10 |
|---|---|---|---|
| fedavg_iid | 0.1402 [0.1354, 0.1451] (n=5, n_ref=5) | -0.0617 [-0.0903, -0.0328] (n=5, n_ref=5) | -0.0598 [-0.0703, -0.0494] (n=5, n_ref=5) |
| sem_media_k1 | 0.0987 [0.0889, 0.1079] (n=5, n_ref=5) | -0.3324 [-0.4746, -0.2375] (n=5, n_ref=5) | -0.4525 [-0.6709, -0.2342] (n=5, n_ref=5) |
| media_sem_particao | 0.1116 [0.1087, 0.1151] (n=5, n_ref=5) | -0.0058 [-0.0140, 0.0059] (n=5, n_ref=5) | 0.0000 [0.0000, 0.0000] (n=5, n_ref=5) |
| central_passos | 0.0862 [0.0757, 0.0961] (n=5, n_ref=5) | -0.4626 [-0.5826, -0.3386] (n=5, n_ref=5) | -0.6449 [-0.7826, -0.4051] (n=5, n_ref=5) |
| central_aprendizado | 0.0877 [0.0780, 0.0993] (n=5, n_ref=5) | -0.4518 [-0.5854, -0.3181] (n=5, n_ref=5) | -0.6268 [-0.7809, -0.4046] (n=5, n_ref=5) |
| central_L2 | sem dados | 0.0997 [0.0893, 0.1130] (n=5, n_ref=5) | sem dados |
| central_SP | sem dados | 0.1481 [0.1418, 0.1540] (n=5, n_ref=5) | sem dados |

**Tarefa B**

| condição | m=1 | m=3 | m=10 |
|---|---|---|---|
| fedavg_iid | 0.0794 [0.0686, 0.0910] (n=5, n_ref=5) | -0.0006 [-0.0039, 0.0029] (n=5, n_ref=5) | -0.0001 [-0.0002, -0.0000] (n=5, n_ref=5) |
| sem_media_k1 | 0.0102 [-0.0017, 0.0224] (n=5, n_ref=5) | -0.0660 [-0.0763, -0.0542] (n=5, n_ref=5) | -0.0088 [-0.0113, -0.0061] (n=5, n_ref=5) |
| media_sem_particao | 0.0240 [0.0191, 0.0288] (n=5, n_ref=5) | 0.0090 [0.0065, 0.0114] (n=5, n_ref=5) | 0.0000 [0.0000, 0.0000] (n=5, n_ref=5) |
| central_passos | 0.0018 [-0.0072, 0.0109] (n=5, n_ref=5) | -0.0674 [-0.0725, -0.0622] (n=5, n_ref=5) | -0.0026 [-0.0037, -0.0018] (n=5, n_ref=5) |
| central_aprendizado | 0.0179 [0.0114, 0.0267] (n=5, n_ref=5) | -0.0825 [-0.0974, -0.0692] (n=5, n_ref=5) | -0.0318 [-0.0363, -0.0281] (n=5, n_ref=5) |
| central_L2 | sem dados | 0.0284 [0.0259, 0.0310] (n=5, n_ref=5) | sem dados |
| central_SP | sem dados | 0.0409 [0.0374, 0.0444] (n=5, n_ref=5) | sem dados |

**Tarefa B_disjunta**

| condição | m=1 | m=3 | m=10 |
|---|---|---|---|
| fedavg_iid | 0.0951 [0.0858, 0.1063] (n=5, n_ref=5) | 0.0631 [0.0572, 0.0709] (n=5, n_ref=5) | -0.0002 [-0.0005, 0.0000] (n=5, n_ref=5) |
| sem_media_k1 | 0.0518 [0.0428, 0.0626] (n=5, n_ref=5) | 0.0169 [0.0102, 0.0254] (n=5, n_ref=5) | -0.0054 [-0.0096, -0.0027] (n=5, n_ref=5) |
| media_sem_particao | 0.0690 [0.0642, 0.0736] (n=5, n_ref=5) | 0.0512 [0.0470, 0.0556] (n=5, n_ref=5) | 0.0000 [0.0000, 0.0000] (n=5, n_ref=5) |
| central_passos | 0.0418 [0.0289, 0.0530] (n=5, n_ref=5) | 0.0036 [-0.0128, 0.0174] (n=5, n_ref=5) | -0.0028 [-0.0036, -0.0019] (n=5, n_ref=5) |
| central_aprendizado | 0.0602 [0.0504, 0.0699] (n=5, n_ref=5) | -0.0157 [-0.0274, -0.0019] (n=5, n_ref=5) | -0.0152 [-0.0168, -0.0132] (n=5, n_ref=5) |
| central_L2 | sem dados | 0.0761 [0.0720, 0.0802] (n=5, n_ref=5) | sem dados |
| central_SP | sem dados | 0.1024 [0.0960, 0.1087] (n=5, n_ref=5) | sem dados |

## (e) Calibração de central_aprendizado (semente 1000)

| tarefa | nível | alvo (acc nova do fedavg_iid) | passos | acc_central | erro | ok | alvo > 0,90 |
|---|---|---|---|---|---|---|---|
| A | 1 | 0.395 | 420 | 0.392 | -0.003 | True | False |
| A | 3 | 0.823 | 1200 | 0.825 | 0.002 | True | False |
| A | 10 | 1.000 | 2490 | 1.000 | 0.000 | True | True |
| B | 1 | 0.704 | 435 | 0.707 | 0.003 | True | False |
| B | 3 | 0.963 | 1185 | 0.964 | 0.001 | True | True |
| B | 10 | 1.000 | 2025 | 1.000 | 0.000 | True | True |
| B_disjunta | 1 | 0.697 | 435 | 0.696 | -0.001 | True | False |
| B_disjunta | 3 | 0.918 | 1140 | 0.915 | -0.003 | True | True |
| B_disjunta | 10 | 1.000 | 2400 | 1.000 | 0.000 | True | True |

Registro: alvo da rede nova do fedavg_iid passa de 0,90 em m=10 na tarefa A.

## (f) Seleção de λ (semente 1000, nível m*)

**L2, tarefa A, m=3**: λ escolhido = 0.000100; esquecimento_ref = n/d; violação = False

| λ | acc contínua (últimas 5 tarefas) | esquecimento final | respeita limite (≤ ref + 0,05)? |
|---|---|---|---|
| 0.0001 (escolhido) | 0.9144 | n/d | n/d (limite só medido em B_disjunta) |
| 0.001 | 0.8610 | n/d | n/d (limite só medido em B_disjunta) |
| 0.01 | 0.6296 | n/d | n/d (limite só medido em B_disjunta) |

**L2, tarefa B, m=3**: λ escolhido = 0.000100; esquecimento_ref = n/d; violação = False

| λ | acc contínua (últimas 5 tarefas) | esquecimento final | respeita limite (≤ ref + 0,05)? |
|---|---|---|---|
| 0.0001 (escolhido) | 0.9904 | n/d | n/d (limite só medido em B_disjunta) |
| 0.001 | 0.9860 | n/d | n/d (limite só medido em B_disjunta) |
| 0.01 | 0.9008 | n/d | n/d (limite só medido em B_disjunta) |

**L2, tarefa B_disjunta, m=3**: λ escolhido = 0.000100; esquecimento_ref = 0.8523; violação = False

| λ | acc contínua (últimas 5 tarefas) | esquecimento final | respeita limite (≤ ref + 0,05)? |
|---|---|---|---|
| 0.0001 (escolhido) | 0.9886 | 0.8715 | sim |
| 0.001 | 0.9830 | 0.8632 | sim |
| 0.01 | 0.8636 | 0.7434 | sim |

**SP, tarefa A, m=3**: λ escolhido = 0.800000; esquecimento_ref = n/d; violação = False

| λ | acc contínua (últimas 5 tarefas) | esquecimento final | respeita limite (≤ ref + 0,05)? |
|---|---|---|---|
| 0.4 | 0.7726 | n/d | n/d (limite só medido em B_disjunta) |
| 0.6 | 0.8420 | n/d | n/d (limite só medido em B_disjunta) |
| 0.8 (escolhido) | 0.9356 | n/d | n/d (limite só medido em B_disjunta) |

**SP, tarefa B, m=3**: λ escolhido = 0.800000; esquecimento_ref = n/d; violação = False

| λ | acc contínua (últimas 5 tarefas) | esquecimento final | respeita limite (≤ ref + 0,05)? |
|---|---|---|---|
| 0.4 | 0.9554 | n/d | n/d (limite só medido em B_disjunta) |
| 0.6 | 0.9846 | n/d | n/d (limite só medido em B_disjunta) |
| 0.8 (escolhido) | 0.9950 | n/d | n/d (limite só medido em B_disjunta) |

**SP, tarefa B_disjunta, m=3**: λ escolhido = 0.800000; esquecimento_ref = 0.8523; violação = False

| λ | acc contínua (últimas 5 tarefas) | esquecimento final | respeita limite (≤ ref + 0,05)? |
|---|---|---|---|
| 0.4 | 0.8930 | 0.7922 | sim |
| 0.6 | 0.9692 | 0.8640 | sim |
| 0.8 (escolhido) | 0.9902 | 0.8809 | sim |

## (g) Sanidade: sem_media_k1 vs central_passos

Pelo Desvios v3 (a), `sem_media_k1` e `central_passos` executam o mesmo algoritmo e deveriam coincidir dentro do ruído. Coincidem = ICs da métrica principal se sobrepõem.

| tarefa | nível | sem_media_k1 | central_passos | ICs sobrepostos? |
|---|---|---|---|---|
| A | 1 | 0.0987 [0.0889, 0.1079] (n=5, n_ref=5) | 0.0862 [0.0757, 0.0961] (n=5, n_ref=5) | sim |
| A | 3 | -0.3324 [-0.4746, -0.2375] (n=5, n_ref=5) | -0.4626 [-0.5826, -0.3386] (n=5, n_ref=5) | sim |
| A | 10 | -0.4525 [-0.6709, -0.2342] (n=5, n_ref=5) | -0.6449 [-0.7826, -0.4051] (n=5, n_ref=5) | sim |
| B | 1 | 0.0102 [-0.0017, 0.0224] (n=5, n_ref=5) | 0.0018 [-0.0072, 0.0109] (n=5, n_ref=5) | sim |
| B | 3 | -0.0660 [-0.0763, -0.0542] (n=5, n_ref=5) | -0.0674 [-0.0725, -0.0622] (n=5, n_ref=5) | sim |
| B | 10 | -0.0088 [-0.0113, -0.0061] (n=5, n_ref=5) | -0.0026 [-0.0037, -0.0018] (n=5, n_ref=5) | NÃO (reportar: deveriam coincidir) |
| B_disjunta | 1 | 0.0518 [0.0428, 0.0626] (n=5, n_ref=5) | 0.0418 [0.0289, 0.0530] (n=5, n_ref=5) | sim |
| B_disjunta | 3 | 0.0169 [0.0102, 0.0254] (n=5, n_ref=5) | 0.0036 [-0.0128, 0.0174] (n=5, n_ref=5) | sim |
| B_disjunta | 10 | -0.0054 [-0.0096, -0.0027] (n=5, n_ref=5) | -0.0028 [-0.0036, -0.0019] (n=5, n_ref=5) | sim |

## Regras (nível m*=3)

- H_reg: não atendida
- H_part: não atendida ((i)=False, (ii)=False)
- H_real: não atendida
- H_orc: não atendida (dentro do IC: A=False, B=False)

## (h) Linha final

Critérios atendidos: nenhum. Todas as hipóteses avaliadas no nível m*=3 (regime.json)
