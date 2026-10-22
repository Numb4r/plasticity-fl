# Tabela de decisão da rodada 2

Fonte: `results_rodada2/` (sementes 6 a 10; semente 1000 nunca entra). Regras literais de PREREGISTRO.md (rodada 2 e blocos de 2026-10-21); ambiguidades em `plots/rodada2_analise.py`. Vereditos: texto calculado, sem interpretação.

## 1. Métrica principal por condição e tarefa (gap médio pareado das 5 últimas tarefas; T=30: 25..29)

Média entre sementes [IC 95% bootstrap percentil, 10 000, default_rng(0)] (n sementes).

| condição | A | B | B_fixa |
|---|---|---|---|
| fedavg_iid | -0.044 [-0.055, -0.033] (n=5) | 0.003 [-0.001, 0.008] (n=5) | 0.067 [0.057, 0.076] (n=5) |
| media_sem_particao | -0.326 [-0.430, -0.259] (n=5) | -0.035 [-0.050, -0.023] (n=5) | 0.032 [0.027, 0.037] (n=5) |
| sem_media_k1 | -0.366 [-0.491, -0.262] (n=5) | -0.080 [-0.103, -0.058] (n=5) | 0.078 [0.068, 0.087] (n=5) |
| central_aprendizado | -0.592 [-0.660, -0.479] (n=5) | -0.095 [-0.104, -0.088] (n=5) | 0.065 [0.058, 0.073] (n=5) |
| central_HT | 0.188 [0.183, 0.194] (n=5) | 0.051 [0.048, 0.055] (n=5) | 0.080 [0.070, 0.088] (n=5) |
| central_HT_cal | 0.188 [0.183, 0.194] (n=5) | 0.051 [0.048, 0.055] (n=5) | sem dados |
| central_lookahead | 0.102 [0.082, 0.125] (n=5) | 0.030 [0.024, 0.037] (n=5) | sem dados |
| fedavg_dirichlet (α=0.1) | 0.130 [0.110, 0.153] (n=5) | 0.052 [0.033, 0.074] (n=5) | sem dados |
| fedavg_dirichlet (α=1) | 0.098 [0.071, 0.127] (n=5) | 0.032 [0.024, 0.040] (n=5) | sem dados |
| fedavg_dirichlet (α=100) | -0.080 [-0.104, -0.060] (n=5) | -0.010 [-0.019, -0.000] (n=5) | sem dados |

## 2. H_tempo / H_dist (central_HT_cal contra media_sem_particao)

**Veredito (H_tempo e H_dist, o mesmo): não avaliável (calibração não atingida; P_vant falhou em A; nível de aprendizado diferente (perda de treino da rede nova do central_HT_cal fora de 10% de media_sem_particao) em A,B)**

### Calibração do central_HT_cal (semente 1000) e μ escolhidos

S = passos locais por rodada (múltiplo de 15); erro_rel = |loss_ht − alvo| / alvo, com alvo = `loss_nova` do media_sem_particao; ok = erro_rel ≤ 10%. μ do central_HT_cal (com esse S), do central_HT (15 passos) e do central_lookahead; [borda] = escolhido na borda da grade (mantido).

| tarefa | S (passos locais/rodada) | erro_rel | ok | alvo (loss_nova msp) | loss_ht | μ central_HT_cal | μ central_HT | μ central_lookahead |
|---|---|---|---|---|---|---|---|---|
| A | 15 | 0.240 | False | 0.708 | 0.878 | 0.99 [borda] | 0.99 [borda] | 0.5 [borda] |
| B | 15 | 0.670 | False | 0.141 | 0.236 | 0.99 [borda] | 0.99 [borda] | 0.5 [borda] |

| tarefa | P_vant: inf(msp) > sup(k1) | perda CE nova central_HT_cal / msp (rel.) | elegível (≤10%) | acc nova HT_cal − msp (checagem ±0,03) | d = central_HT_cal − msp, IC | classe de d (δ=0,02) |
|---|---|---|---|---|---|---|
| A | False (msp -0.326 [-0.430, -0.259] (n=5); k1 -0.366 [-0.491, -0.262] (n=5)) | 0.890 / 0.690 (0.291) | False | -0.104 (False) | 0.514 [0.447, 0.617] (n=5) | acima |
| B | True (msp -0.035 [-0.050, -0.023] (n=5); k1 -0.080 [-0.103, -0.058] (n=5)) | 0.258 / 0.149 (0.736) | False | -0.045 (False) | 0.086 [0.073, 0.100] (n=5) | acima |

Critério secundário (literal; sem veredito): média do central_HT_cal dentro do IC de media_sem_particao; inf(IC msp) acima do sup(IC central_HT_cal).

| tarefa | central_HT_cal dentro do IC de msp | inf(msp) > sup(central_HT_cal) |
|---|---|---|
| A | False | False |
| B | False | False |

### Exploratórios (sem veredito, não entram em H_tempo nem em H_dist)

**(a) central_HT (15 passos, SEM calibração) contra media_sem_particao** (mesma diferença pareada d e IC; a rede nova do central_HT aprende em outro nível que a do msp).

| tarefa | d = central_HT − msp, IC (gap) | mesma d em acc_cont | d por semente (gap) |
|---|---|---|---|
| A | 0.514 [0.447, 0.617] (n=5) | 0.410 [0.342, 0.512] (n=5) | 6: 0.457, 7: 0.431, 8: 0.509, 9: 0.459, 10: 0.715 |
| B | 0.086 [0.073, 0.100] (n=5) | 0.042 [0.029, 0.057] (n=5) | 6: 0.101, 7: 0.071, 8: 0.109, 9: 0.076, 10: 0.073 |

**(b) central_HT − sem_media_k1** por tarefa e semente (a rede nova é a mesma; gap = acc_cont − acc_nova, então a diferença de gap é a de acc_cont).

| tarefa | d = central_HT − k1, IC (gap) | mesma d em acc_cont | d por semente (gap) |
|---|---|---|---|
| A | 0.554 [0.450, 0.677] (n=5) | 0.554 [0.450, 0.677] (n=5) | 6: 0.659, 7: 0.452, 8: 0.449, 9: 0.450, 10: 0.759 |
| B | 0.131 [0.110, 0.151] (n=5) | 0.131 [0.110, 0.151] (n=5) | 6: 0.128, 7: 0.090, 8: 0.156, 9: 0.123, 10: 0.158 |

**(c) central_lookahead** contra media_sem_particao.

| tarefa | d = lookahead − msp, IC (gap) | mesma d em acc_cont | d por semente (gap) |
|---|---|---|---|
| A | 0.428 [0.359, 0.519] (n=5) | 0.053 [-0.015, 0.143] (n=5) | 6: 0.404, 7: 0.339, 8: 0.434, 9: 0.357, 10: 0.608 |
| B | 0.065 [0.050, 0.081] (n=5) | -0.137 [-0.149, -0.122] (n=5) | 6: 0.084, 7: 0.054, 8: 0.090, 9: 0.049, 10: 0.049 |

## 3. H_het (fedavg_dirichlet, α = 0,1; 1; 100)

**Veredito geral: não atendida**

| tarefa | α | gap | acc_cont | acc_nova |
|---|---|---|---|---|
| A | 0.1 | 0.130 [0.110, 0.153] (n=5) | 0.531 [0.511, 0.552] (n=5) | 0.401 [0.386, 0.414] (n=5) |
| A | 1 | 0.098 [0.071, 0.127] (n=5) | 0.758 [0.729, 0.787] (n=5) | 0.660 [0.654, 0.666] (n=5) |
| A | 100 | -0.080 [-0.104, -0.060] (n=5) | 0.777 [0.755, 0.796] (n=5) | 0.857 [0.853, 0.862] (n=5) |
| B | 0.1 | 0.052 [0.033, 0.074] (n=5) | 0.725 [0.694, 0.753] (n=5) | 0.673 [0.657, 0.690] (n=5) |
| B | 1 | 0.032 [0.024, 0.040] (n=5) | 0.929 [0.921, 0.938] (n=5) | 0.897 [0.889, 0.904] (n=5) |
| B | 100 | -0.010 [-0.019, -0.000] (n=5) | 0.958 [0.950, 0.967] (n=5) | 0.968 [0.964, 0.970] (n=5) |

| tarefa | gap(100) − gap(0,1), IC pareado | veredito da tarefa |
|---|---|---|
| A | -0.210 [-0.231, -0.183] (n=5) | H_het reversa |
| B | -0.062 [-0.074, -0.050] (n=5) | H_het reversa |

## 4. Rank das ativações (exploratório; SEM critério e SEM veredito)

`rank_ef` e `srank_feat` sempre juntos (média das 5 últimas tarefas; média entre sementes e IC 95%). Grupo "com média": fedavg_iid, media_sem_particao, central_HT_cal, fedavg_dirichlet; "sem média": sem_media_k1, central_aprendizado; central_HT e central_lookahead aparecem à parte, fora dos grupos. As duas medidas discordaram na rodada 1 (tarefa A, m=3: rank_ef do fedavg_iid 6,3, abaixo de central_aprendizado 9,8; srank_feat do fedavg_iid 63, acima de 19,6), e o autor já viu esses números.

| tarefa | condição / grupo | rank_ef | srank_feat |
|---|---|---|---|
| A | central_HT (fora do grupo) | 44.712 [40.930, 48.493] (n=5) | 116.840 [114.880, 118.880] (n=5) |
| A | central_HT_cal | 44.712 [40.930, 48.493] (n=5) | 116.840 [114.880, 118.880] (n=5) |
| A | central_aprendizado | 13.472 [9.035, 17.096] (n=5) | 21.160 [19.120, 24.520] (n=5) |
| A | central_lookahead (fora do grupo) | 17.493 [15.751, 19.234] (n=5) | 75.840 [72.080, 79.600] (n=5) |
| A | fedavg_dirichlet (α=0.1) | 5.031 [4.418, 5.434] (n=5) | 77.640 [69.600, 84.680] (n=5) |
| A | fedavg_dirichlet (α=1) | 7.554 [6.610, 8.498] (n=5) | 88.160 [82.720, 93.320] (n=5) |
| A | fedavg_dirichlet (α=100) | 6.764 [6.107, 7.493] (n=5) | 65.040 [62.080, 67.360] (n=5) |
| A | fedavg_iid | 7.126 [6.706, 7.470] (n=5) | 69.320 [66.320, 72.320] (n=5) |
| A | media_sem_particao | 10.455 [5.384, 18.777] (n=5) | 34.960 [31.600, 38.400] (n=5) |
| A | sem_media_k1 | 10.006 [4.722, 15.589] (n=5) | 25.760 [23.720, 28.400] (n=5) |
| A | grupo: com média | 17.186 [14.978, 20.141] (n=5) | 74.517 [72.850, 76.313] (n=5) |
| A | grupo: sem média | 11.739 [7.504, 15.465] (n=5) | 23.460 [21.500, 26.520] (n=5) |
| B | central_HT (fora do grupo) | 28.545 [26.097, 30.770] (n=5) | 113.920 [113.040, 114.800] (n=5) |
| B | central_HT_cal | 28.545 [26.097, 30.770] (n=5) | 113.920 [113.040, 114.800] (n=5) |
| B | central_aprendizado | 4.102 [3.575, 4.658] (n=5) | 18.040 [14.800, 20.120] (n=5) |
| B | central_lookahead (fora do grupo) | 8.571 [7.989, 9.012] (n=5) | 57.240 [55.000, 59.960] (n=5) |
| B | fedavg_dirichlet (α=0.1) | 5.235 [5.027, 5.431] (n=5) | 76.560 [72.240, 81.560] (n=5) |
| B | fedavg_dirichlet (α=1) | 6.736 [6.322, 7.095] (n=5) | 83.480 [80.320, 86.200] (n=5) |
| B | fedavg_dirichlet (α=100) | 6.039 [5.595, 6.599] (n=5) | 55.960 [49.520, 62.400] (n=5) |
| B | fedavg_iid | 6.465 [6.219, 6.711] (n=5) | 63.280 [59.400, 65.760] (n=5) |
| B | media_sem_particao | 4.587 [4.135, 5.040] (n=5) | 33.480 [29.800, 36.720] (n=5) |
| B | sem_media_k1 | 4.929 [4.247, 5.716] (n=5) | 26.240 [21.760, 31.600] (n=5) |
| B | grupo: com média | 11.400 [10.665, 12.143] (n=5) | 70.670 [68.083, 73.007] (n=5) |
| B | grupo: sem média | 4.515 [4.063, 4.967] (n=5) | 22.140 [18.280, 25.700] (n=5) |

## 5. Sonda v2 (20 passos; média por semente nas tarefas de sonda)

dlog_norm = mediana_k[log10(Lc20/Lc0) − log10(Ln20/Ln0)] (recalculada de sonda2_L*_k); dlog_raw = estatística exploratória sem normalizar (coluna sonda2_dlog_raw).

| tarefa | condição | dlog_norm | dlog_raw |
|---|---|---|---|
| A | central_HT | -0.087 [-0.116, -0.056] (n=5) | 4.557 [4.519, 4.594] (n=5) |
| A | central_HT_cal | -0.087 [-0.116, -0.056] (n=5) | 4.557 [4.519, 4.594] (n=5) |
| A | central_aprendizado | -0.151 [-0.246, -0.007] (n=5) | 4.242 [4.093, 4.456] (n=5) |
| A | central_lookahead | 0.080 [0.046, 0.119] (n=5) | 4.814 [4.707, 4.930] (n=5) |
| A | fedavg_dirichlet (α=0.1) | 0.625 [0.609, 0.641] (n=5) | 9.058 [9.026, 9.090] (n=5) |
| A | fedavg_dirichlet (α=1) | 0.548 [0.534, 0.558] (n=5) | 7.725 [7.691, 7.766] (n=5) |
| A | fedavg_dirichlet (α=100) | 0.287 [0.256, 0.312] (n=5) | 5.948 [5.877, 6.012] (n=5) |
| A | fedavg_iid | 0.363 [0.342, 0.385] (n=5) | 6.276 [6.213, 6.349] (n=5) |
| A | media_sem_particao | -0.045 [-0.262, 0.142] (n=5) | 4.830 [4.473, 5.172] (n=5) |
| A | sem_media_k1 | 0.037 [-0.148, 0.222] (n=5) | 4.819 [4.393, 5.210] (n=5) |
| B | central_HT | 0.084 [0.073, 0.094] (n=5) | 4.950 [4.916, 4.978] (n=5) |
| B | central_HT_cal | 0.084 [0.073, 0.094] (n=5) | 4.950 [4.916, 4.978] (n=5) |
| B | central_aprendizado | 0.441 [0.425, 0.455] (n=5) | 5.971 [5.834, 6.108] (n=5) |
| B | central_lookahead | 0.274 [0.262, 0.286] (n=5) | 5.227 [5.188, 5.275] (n=5) |
| B | fedavg_dirichlet (α=0.1) | 0.625 [0.605, 0.643] (n=5) | 9.153 [9.110, 9.191] (n=5) |
| B | fedavg_dirichlet (α=1) | 0.567 [0.548, 0.582] (n=5) | 7.964 [7.939, 7.987] (n=5) |
| B | fedavg_dirichlet (α=100) | 0.423 [0.404, 0.442] (n=5) | 6.431 [6.370, 6.486] (n=5) |
| B | fedavg_iid | 0.460 [0.445, 0.474] (n=5) | 6.752 [6.713, 6.793] (n=5) |
| B | media_sem_particao | 0.462 [0.450, 0.473] (n=5) | 6.327 [6.263, 6.391] (n=5) |
| B | sem_media_k1 | 0.430 [0.425, 0.437] (n=5) | 6.062 [5.949, 6.179] (n=5) |

## 6. B_fixa (T=30)

O gap em B_fixa mistura transferência (mesmo professor) e plasticidade.

| condição | gap (5 últimas) | esquecimento final | BWT final |
|---|---|---|---|
| fedavg_iid | 0.067 [0.057, 0.076] (n=5) | 0.486 [0.482, 0.489] (n=5) | -0.486 [-0.489, -0.482] (n=5) |
| media_sem_particao | 0.032 [0.027, 0.037] (n=5) | 0.478 [0.471, 0.486] (n=5) | -0.478 [-0.486, -0.471] (n=5) |
| sem_media_k1 | 0.078 [0.068, 0.087] (n=5) | 0.496 [0.487, 0.504] (n=5) | -0.496 [-0.504, -0.487] (n=5) |
| central_aprendizado | 0.065 [0.058, 0.073] (n=5) | 0.496 [0.484, 0.508] (n=5) | -0.496 [-0.508, -0.484] (n=5) |
| central_HT | 0.080 [0.070, 0.088] (n=5) | 0.458 [0.448, 0.468] (n=5) | -0.458 [-0.468, -0.448] (n=5) |

Reaprendizado (cópia vs. rede nova; medianas pooled sobre semente × tarefa_atual × tarefa_antiga; '> 1500' se mais da metade censurada).

| condição | n | passos cópia (mediana) | passos nova (mediana) | censurado cópia | censurado nova | economia = 1 − cópia/nova | no limiar no passo 0 (cópia / nova) |
|---|---|---|---|---|---|---|---|
| fedavg_iid | 30 | 382.5 | 907.5 | 0.00 | 0.00 | 0.58 | 0.00 / 0.00 |
| media_sem_particao | 30 | 397.5 | 960 | 0.00 | 0.00 | 0.59 | 0.00 / 0.00 |
| sem_media_k1 | 30 | 330 | 855 | 0.00 | 0.00 | 0.61 | 0.00 / 0.00 |
| central_aprendizado | 30 | 367.5 | 900 | 0.00 | 0.00 | 0.59 | 0.00 / 0.00 |
| central_HT | 30 | 285 | 840 | 0.00 | 0.00 | 0.66 | 0.00 / 0.00 |

## 7. Braço longo (T=100, sementes 6 a 8): tarefa de colapso

Primeiro ponto, entre as referências t%3==2, em que a média móvel de 3 referências do gap pareado fica ≤ −0,10 (índice da última referência da janela). Mediana [mín, máx] entre as sementes que colapsaram.

| tarefa | condição | gap médio (95..99) | colapso por semente | mediana [mín, máx] | fração sem colapso |
|---|---|---|---|---|---|
| A | fedavg_iid | -0.390 [-0.433, -0.352] (n=3) | 6: 44, 7: 44, 8: 44 | 44 [44, 44] | 0.00 |
| A | media_sem_particao | -0.671 [-0.704, -0.638] (n=3) | 6: 17, 7: 17, 8: 17 | 17 [17, 17] | 0.00 |
| A | sem_media_k1 | -0.623 [-0.632, -0.609] (n=3) | 6: 17, 7: 17, 8: 17 | 17 [17, 17] | 0.00 |
| A | central_aprendizado | -0.706 [-0.718, -0.693] (n=3) | 6: 14, 7: 14, 8: 14 | 14 [14, 14] | 0.00 |
| A | central_HT | -0.015 [-0.026, -0.001] (n=3) | 6: sem colapso, 7: sem colapso, 8: sem colapso | sem colapso | 1.00 |
| B | fedavg_iid | -0.260 [-0.267, -0.248] (n=3) | 6: 74, 7: 74, 8: 68 | 74 [68, 74] | 0.00 |
| B | media_sem_particao | -0.339 [-0.343, -0.337] (n=3) | 6: 53, 7: 50, 8: 50 | 50 [50, 53] | 0.00 |
| B | sem_media_k1 | -0.413 [-0.453, -0.386] (n=3) | 6: 38, 7: 41, 8: 32 | 38 [32, 41] | 0.00 |
| B | central_aprendizado | -0.443 [-0.463, -0.424] (n=3) | 6: 32, 7: 35, 8: 29 | 32 [29, 35] | 0.00 |
| B | central_HT | 0.053 [0.051, 0.054] (n=3) | 6: sem colapso, 7: sem colapso, 8: sem colapso | sem colapso | 1.00 |

**Julgamento do horizonte (tarefa A), regra já confirmada (mediana só entre as sementes que colapsaram): só atrasa a perda**

regra de julgamento confirmada pelo autor em 2026-10-21: com "sem colapso" como censura à direita e mediana nas 3 sementes: **só atrasa a perda**

## 8. Previsão do autor × resultado (calculado)

Previsão literal do autor (commit `ab3c8ab`): "H_tempo não, H_dist não, H_het sim, FedAvg só atrasa". Pelo esclarecimento de 2026-10-21, o "central_HT" da previsão é o central_HT_cal.

**Julgamento: regra de julgamento confirmada pelo autor em 2026-10-21.** Regras: (1) H_tempo/H_dist acerta só se o veredito geral for "HT acima"; "misto" com "HT acima" numa tarefa = acerto parcial; "não avaliável" não conta; (2) H_tempo e H_dist são um só item; (3) horizonte: "sem colapso" = censura à direita (> T), mediana nas 3 sementes, empate não é "mais tarde"; (4) H_het: "atendida" = acerto (confirmado), "parcial" = acerto parcial, ordem certa com IC passando por 0 = erro.

| item | previsão do autor (literal) | veredito calculado | julgamento |
|---|---|---|---|
| H_tempo e H_dist (um só item) | H_tempo não e H_dist não (desfecho "HT acima") | não avaliável (calibração não atingida; P_vant falhou em A; nível de aprendizado diferente (perda de treino da rede nova do central_HT_cal fora de 10% de media_sem_particao) em A,B) | não conta (não avaliável) |
| H_het | sim (atendida nas duas tarefas) | não atendida | erro |
| horizonte longo (tarefa A) | o FedAvg só atrasa a perda | só atrasa a perda | acerto |
