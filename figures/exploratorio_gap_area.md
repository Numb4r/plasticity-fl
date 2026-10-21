# EXPLORATÓRIO · área do gap (média ao longo da sequência)

**Análise exploratória pós-resultados (PREREGISTRO.md, item B). Não altera nenhum veredito, não entra na `tabela_decisao.md` nem em regra de decisão.**

- Área por (condição, nível, tarefa, semente) = integral trapezoidal do gap sobre o índice da tarefa, do primeiro ao último ponto observado (2 a 29; 2 a 11 em `media_sem_particao`), dividida pelo comprimento do intervalo (27; 9) = gap médio ao longo da sequência, ponderado pelo espaçamento.
- Média e IC 95% por bootstrap percentil (10 000 reamostragens das sementes, `default_rng(0)`), sementes 1 a 5.
- **Limitação:** os pontos de gap são mais densos nas tarefas 7 a 11 e 25 a 29, então a ponderação por espaçamento usa trapézios sobre pontos desigualmente espaçados.
- A coluna "confirmatória, não alterada" é a métrica principal (gap médio das 5 últimas tarefas, `analise.metrica_por_semente`), mostrada só para comparação visual; não é recalculada nem reinterpretada.
- `media_sem_particao` tem T=12: o intervalo cobre só as tarefas 2 a 11, e a janela "5 últimas" é 7 a 11.
- Gap = acc_cont − acc_nova (diferença de acurácia, fração).

## Tarefa A · nível m=1

| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | confirmatória, não alterada: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +0.1118 | [+0.1043; +0.1177] | 5 | +0.1402 | [+0.1354; +0.1451] | 5 |
| sem_media_k1 | +0.0962 | [+0.0914; +0.1005] | 5 | +0.0987 | [+0.0889; +0.1079] | 5 |
| media_sem_particao | +0.0883 | [+0.0843; +0.0920] | 5 | +0.1116 | [+0.1087; +0.1151] | 5 |
| central_passos | +0.0774 | [+0.0667; +0.0888] | 5 | +0.0862 | [+0.0757; +0.0961] | 5 |
| central_aprendizado | +0.0948 | [+0.0866; +0.1030] | 5 | +0.0877 | [+0.0780; +0.0993] | 5 |

## Tarefa A · nível m=3

| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | confirmatória, não alterada: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +0.0296 | [+0.0146; +0.0474] | 5 | -0.0617 | [-0.0903; -0.0328] | 5 |
| sem_media_k1 | -0.1377 | [-0.2011; -0.0963] | 5 | -0.3324 | [-0.4746; -0.2375] | 5 |
| media_sem_particao | +0.0480 | [+0.0430; +0.0557] | 5 | -0.0058 | [-0.0140; +0.0059] | 5 |
| central_passos | -0.2091 | [-0.2444; -0.1697] | 5 | -0.4626 | [-0.5826; -0.3386] | 5 |
| central_aprendizado | -0.2508 | [-0.3176; -0.1841] | 5 | -0.4518 | [-0.5854; -0.3181] | 5 |
| central_L2 | +0.1272 | [+0.1258; +0.1287] | 5 | +0.0997 | [+0.0893; +0.1130] | 5 |
| central_SP | +0.1521 | [+0.1494; +0.1547] | 5 | +0.1481 | [+0.1418; +0.1540] | 5 |

## Tarefa A · nível m=10

| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | confirmatória, não alterada: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | -0.0252 | [-0.0290; -0.0215] | 5 | -0.0598 | [-0.0703; -0.0494] | 5 |
| sem_media_k1 | -0.2672 | [-0.4365; -0.0980] | 5 | -0.4525 | [-0.6709; -0.2342] | 5 |
| media_sem_particao | +0.0000 | [+0.0000; +0.0000] | 5 | +0.0000 | [+0.0000; +0.0000] | 5 |
| central_passos | -0.3912 | [-0.5148; -0.2285] | 5 | -0.6449 | [-0.7826; -0.4051] | 5 |
| central_aprendizado | -0.3938 | [-0.5468; -0.2350] | 5 | -0.6268 | [-0.7809; -0.4046] | 5 |

## Tarefa B · nível m=1

| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | confirmatória, não alterada: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +0.0603 | [+0.0540; +0.0667] | 5 | +0.0794 | [+0.0686; +0.0910] | 5 |
| sem_media_k1 | +0.0069 | [-0.0021; +0.0183] | 5 | +0.0102 | [-0.0017; +0.0224] | 5 |
| media_sem_particao | +0.0206 | [+0.0176; +0.0241] | 5 | +0.0240 | [+0.0191; +0.0288] | 5 |
| central_passos | -0.0028 | [-0.0082; +0.0026] | 5 | +0.0018 | [-0.0072; +0.0109] | 5 |
| central_aprendizado | +0.0146 | [+0.0115; +0.0177] | 5 | +0.0179 | [+0.0114; +0.0267] | 5 |

## Tarefa B · nível m=3

| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | confirmatória, não alterada: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +0.0167 | [+0.0123; +0.0190] | 5 | -0.0006 | [-0.0039; +0.0029] | 5 |
| sem_media_k1 | -0.0144 | [-0.0210; -0.0073] | 5 | -0.0660 | [-0.0763; -0.0542] | 5 |
| media_sem_particao | +0.0122 | [+0.0102; +0.0141] | 5 | +0.0090 | [+0.0065; +0.0114] | 5 |
| central_passos | -0.0229 | [-0.0260; -0.0199] | 5 | -0.0674 | [-0.0725; -0.0622] | 5 |
| central_aprendizado | -0.0295 | [-0.0364; -0.0217] | 5 | -0.0825 | [-0.0974; -0.0692] | 5 |
| central_L2 | +0.0328 | [+0.0316; +0.0341] | 5 | +0.0284 | [+0.0259; +0.0310] | 5 |
| central_SP | +0.0440 | [+0.0421; +0.0458] | 5 | +0.0409 | [+0.0374; +0.0444] | 5 |

## Tarefa B · nível m=10

| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | confirmatória, não alterada: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | -0.0000 | [-0.0000; -0.0000] | 5 | -0.0001 | [-0.0002; -0.0000] | 5 |
| sem_media_k1 | -0.0019 | [-0.0025; -0.0014] | 5 | -0.0088 | [-0.0113; -0.0061] | 5 |
| media_sem_particao | +0.0000 | [+0.0000; +0.0000] | 5 | +0.0000 | [+0.0000; +0.0000] | 5 |
| central_passos | -0.0004 | [-0.0007; -0.0003] | 5 | -0.0026 | [-0.0037; -0.0018] | 5 |
| central_aprendizado | -0.0128 | [-0.0140; -0.0115] | 5 | -0.0318 | [-0.0363; -0.0281] | 5 |

## Tarefa B_disjunta · nível m=1

| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | confirmatória, não alterada: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +0.0803 | [+0.0756; +0.0853] | 5 | +0.0951 | [+0.0858; +0.1063] | 5 |
| sem_media_k1 | +0.0458 | [+0.0395; +0.0505] | 5 | +0.0518 | [+0.0428; +0.0626] | 5 |
| media_sem_particao | +0.0563 | [+0.0531; +0.0595] | 5 | +0.0690 | [+0.0642; +0.0736] | 5 |
| central_passos | +0.0416 | [+0.0371; +0.0448] | 5 | +0.0418 | [+0.0289; +0.0530] | 5 |
| central_aprendizado | +0.0615 | [+0.0549; +0.0681] | 5 | +0.0602 | [+0.0504; +0.0699] | 5 |

## Tarefa B_disjunta · nível m=3

| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | confirmatória, não alterada: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +0.0721 | [+0.0713; +0.0729] | 5 | +0.0631 | [+0.0572; +0.0709] | 5 |
| sem_media_k1 | +0.0572 | [+0.0523; +0.0622] | 5 | +0.0169 | [+0.0102; +0.0254] | 5 |
| media_sem_particao | +0.0484 | [+0.0451; +0.0514] | 5 | +0.0512 | [+0.0470; +0.0556] | 5 |
| central_passos | +0.0486 | [+0.0429; +0.0538] | 5 | +0.0036 | [-0.0128; +0.0174] | 5 |
| central_aprendizado | +0.0389 | [+0.0317; +0.0468] | 5 | -0.0157 | [-0.0274; -0.0019] | 5 |
| central_L2 | +0.0762 | [+0.0738; +0.0799] | 5 | +0.0761 | [+0.0720; +0.0802] | 5 |
| central_SP | +0.1023 | [+0.0983; +0.1053] | 5 | +0.1024 | [+0.0960; +0.1087] | 5 |

## Tarefa B_disjunta · nível m=10

| condição | área: média | área: IC 95% | n | confirmatória, não alterada: média (5 últimas) | confirmatória, não alterada: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | -0.0001 | [-0.0003; +0.0001] | 5 | -0.0002 | [-0.0005; +0.0000] | 5 |
| sem_media_k1 | -0.0016 | [-0.0026; -0.0008] | 5 | -0.0054 | [-0.0096; -0.0027] | 5 |
| media_sem_particao | +0.0000 | [+0.0000; +0.0000] | 5 | +0.0000 | [+0.0000; +0.0000] | 5 |
| central_passos | -0.0009 | [-0.0011; -0.0008] | 5 | -0.0028 | [-0.0036; -0.0019] | 5 |
| central_aprendizado | -0.0041 | [-0.0050; -0.0032] | 5 | -0.0152 | [-0.0168; -0.0132] | 5 |
