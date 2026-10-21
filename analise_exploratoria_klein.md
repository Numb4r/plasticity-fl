# Análise exploratória dos diagnósticos de Klein et al. (Adendo v3.1)

**Exploratória, sem critério de decisão, sem previsão registrada.** Gerada por `python -m experiments.analise_exploratoria_klein --nivel N --tarefa T` a partir dos CSVs das sementes 1 a 5. Não passou por auditoria própria.

**Pergunta:** na condição que perde plasticidade, a norma dos parâmetros e o maior autovalor da Hessiana sobem antes da queda da acurácia do contínuo (ordem da cadeia causal da Figura 1 do survey)?

**Como ler a seção (2):** `t50` é o instante em que o sinal percorreu metade do caminho entre o nível do começo e o do fim (série suavizada em 3 tarefas); "adiantamento" = t50(acurácia) − t50(sinal), mediana [mín, máx] entre 5 sementes; positivo = o sinal se moveu antes. Onde não há queda real de acurácia (por exemplo `fedavg_iid` na tarefa B) o t50 é ruído e a linha não deve ser lida.

## O que se pode e não se pode dizer (leitura do autor desta análise, a confirmar)
- Descritivo, nas condições que perdem plasticidade (`central_passos`, `central_aprendizado`, `sem_media_k1`) em A: a norma começa a subir desde as primeiras tarefas e percorre metade do caminho **poucas tarefas antes** da metade da queda da acurácia (adiantamento mediano de +1 a +3 em m=3 e de +2 a +10 em m=10). O rank efetivo das ativações (`rank_ef`) é o sinal que mais consistentemente se move antes (+4 a +7 em m=3; +2 a +14 em m=10), seguido por dormentes e `srank_feat`.
- O `|hess_max|` **não** se move de forma consistente antes da acurácia: as medianas mudam de sinal entre condições e as sementes se espalham (por exemplo −6 em `central_aprendizado`, +2 em `central_passos`, m=3, tarefa A).
- **Entre condições, a magnitude da norma e da curvatura não acompanha a perda de plasticidade:** o `fedavg_iid` tem a maior norma (178 contra 143-148) e a maior curvatura (367 contra 42-67) nas 5 últimas tarefas em m=3, e mesmo assim perde bem menos acurácia. As condições que perdem têm muitos neurônios dormentes (79-83%) e `srank_feat` baixo (20-25); o `fedavg_iid` fica em 62% e 63; L2 e SP ficam em 7% e 2% de dormentes com `srank_feat` ≈ 110.
- **Cuidados:** (a) "antes" aqui é sobre a forma das trajetórias (norma e rank mudam cedo e devagar, a acurácia cai depois), não prova causalidade; (b) são 5 sementes e uma única tarefa-tipo (A) com perda clara; (c) `hess_conv` médio 0,99, mas a convergência da power iteration é declarada com tolerância de 1%; (d) a norma e a curvatura crescem monotonicamente também em condições que não perdem plasticidade (L2: norma 26 → 57), então subir sozinho não discrimina.


---
# Exploratório (sem decisão): tarefa A, nível m=3, sementes 1 a 5

## (1) Trajetórias médias por bloco de tarefas (média entre sementes)

**central_passos**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.882 | 0.632 | 0.419 | 0.325 |
| norma | 28.5 | 91.5 | 132 | 143 |
| hess | 7.1 | 22.7 | 40.2 | 41.7 |
| grad_l2 | 1.11 | 2.93 | 3.63 | 3.07 |
| grad_srank | 61.6 | 59.6 | 50.9 | 41.1 |
| rank_ef | 76.4 | 25 | 14 | 12.1 |
| srank_feat | 118 | 68.6 | 28.9 | 22.1 |
| dorm | 0.0544 | 0.474 | 0.757 | 0.811 |

**central_aprendizado**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.913 | 0.609 | 0.418 | 0.379 |
| norma | 30.1 | 97.4 | 137 | 148 |
| hess | 7.12 | 22.5 | 48.4 | 60.7 |
| grad_l2 | 1.19 | 3.22 | 3.79 | 3.58 |
| grad_srank | 61.3 | 57.8 | 45.8 | 42.8 |
| rank_ef | 76.7 | 30.1 | 12.7 | 9.76 |
| srank_feat | 118 | 65 | 27 | 19.6 |
| dorm | 0.0469 | 0.495 | 0.776 | 0.834 |

**sem_media_k1**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.867 | 0.694 | 0.478 | 0.439 |
| norma | 27.7 | 88.6 | 134 | 147 |
| hess | 7.56 | 33.1 | 52.7 | 67.2 |
| grad_l2 | 1.11 | 3.05 | 3.96 | 3.74 |
| grad_srank | 61.2 | 59.4 | 53 | 50.8 |
| rank_ef | 76.8 | 29.1 | 11.7 | 7.61 |
| srank_feat | 118 | 73.3 | 35 | 25.2 |
| dorm | 0.0488 | 0.432 | 0.719 | 0.79 |

**fedavg_iid**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.908 | 0.902 | 0.793 | 0.767 |
| norma | 27.6 | 88.6 | 151 | 178 |
| hess | 26.6 | 232 | 369 | 367 |
| grad_l2 | 1.44 | 4.47 | 6.2 | 6.26 |
| grad_srank | 61.3 | 59.4 | 58.8 | 58.8 |
| rank_ef | 56.2 | 15.9 | 7.68 | 6.31 |
| srank_feat | 120 | 99.7 | 75.8 | 63.4 |
| dorm | 0.0309 | 0.352 | 0.572 | 0.623 |

**media_sem_particao**  (T=12)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.952 | 0.812 | - | - |
| norma | 28.6 | 86.4 | - | - |
| hess | 8.09 | 52 | - | - |
| grad_l2 | 1.23 | 3.57 | - | - |
| grad_srank | 61.1 | 59.4 | - | - |
| rank_ef | 75.3 | 21.7 | - | - |
| srank_feat | 121 | 90.1 | - | - |
| dorm | 0.0291 | 0.337 | - | - |

**central_L2**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.926 | 0.961 | 0.934 | 0.92 |
| norma | 25.9 | 55.5 | 57.6 | 57.4 |
| hess | 7.87 | 56.8 | 130 | 140 |
| grad_l2 | 1.19 | 3.45 | 5 | 4.82 |
| grad_srank | 61.2 | 58.8 | 59.2 | 59.4 |
| rank_ef | 75.8 | 37.4 | 28.6 | 27.3 |
| srank_feat | 121 | 114 | 109 | 109 |
| dorm | 0.00469 | 0.0356 | 0.0625 | 0.0728 |

**central_SP**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.909 | 0.95 | 0.94 | 0.938 |
| norma | 22.6 | 31.7 | 32 | 32.2 |
| hess | 6.52 | 11 | 14.7 | 15.2 |
| grad_l2 | 0.605 | 1.04 | 1.3 | 1.38 |
| grad_srank | 63 | 62.9 | 62.6 | 62.6 |
| rank_ef | 76.2 | 55.3 | 42.3 | 38.2 |
| srank_feat | 121 | 115 | 114 | 113 |
| dorm | 0.0144 | 0.0534 | 0.0241 | 0.0231 |

## (2) Tempo de meia-excursão t50 por semente; adiantamento = t50(acc) - t50(sinal)

| condição | t50 acc (mediana) | adiant. norma | adiant. |hess| | adiant. dormentes | adiant. srank_feat | adiant. rank_ef |
|---|---|---|---|---|---|---|
| central_passos (n=5/5) | +15.0 [+12,+18] | +3.0 [+0,+6] | +2.0 [-6,+12] | +3.0 [-1,+7] | +2.0 [-2,+6] | +7.0 [+3,+10] |
| central_aprendizado (n=5/5) | +13.0 [+11,+14] | +1.0 [+0,+3] | -6.0 [-7,-1] | +0.0 [-1,+3] | +0.0 [-1,+2] | +4.0 [-2,+5] |
| sem_media_k1 (n=5/5) | +15.0 [+13,+16] | +2.0 [+0,+4] | -2.0 [-8,+5] | +2.0 [+1,+3] | +0.0 [+0,+3] | +5.0 [+1,+7] |
| fedavg_iid (n=5/5) | +19.0 [+0,+22] | +4.0 [-16,+7] | +6.0 [-13,+11] | +7.0 [-14,+12] | +1.0 [-17,+7] | +12.0 [-8,+16] |

---
# Exploratório (sem decisão): tarefa A, nível m=10, sementes 1 a 5

## (1) Trajetórias médias por bloco de tarefas (média entre sementes)

**central_passos**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.994 | 0.685 | 0.382 | 0.355 |
| norma | 73.7 | 224 | 277 | 297 |
| hess | 6.84 | 116 | 183 | 189 |
| grad_l2 | 4.23 | 12.4 | 11.6 | 10.8 |
| grad_srank | 58 | 45.3 | 25.1 | 22.7 |
| rank_ef | 66.5 | 19.1 | 18.1 | 16.2 |
| srank_feat | 115 | 37.8 | 28.4 | 28.2 |
| dorm | 0.0772 | 0.697 | 0.749 | 0.748 |

**central_aprendizado**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.993 | 0.635 | 0.464 | 0.373 |
| norma | 54.5 | 171 | 214 | 229 |
| hess | 5.77 | 78.8 | 95.7 | 110 |
| grad_l2 | 2.76 | 8.5 | 7.04 | 6.57 |
| grad_srank | 58 | 46.6 | 33.4 | 28.6 |
| rank_ef | 69.5 | 22.4 | 13.5 | 13.1 |
| srank_feat | 117 | 41.2 | 23 | 22.6 |
| dorm | 0.0584 | 0.665 | 0.784 | 0.781 |

**sem_media_k1**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.999 | 0.76 | 0.607 | 0.547 |
| norma | 71.8 | 227 | 287 | 310 |
| hess | 4.84 | 136 | 214 | 227 |
| grad_l2 | 4.06 | 11.9 | 10.3 | 9.9 |
| grad_srank | 57.9 | 47.1 | 40.8 | 39.2 |
| rank_ef | 62.8 | 17.7 | 12.8 | 11.4 |
| srank_feat | 117 | 42.2 | 30.9 | 27.8 |
| dorm | 0.0559 | 0.676 | 0.753 | 0.764 |

**fedavg_iid**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 1 | 0.983 | 0.964 | 0.94 |
| norma | 68.9 | 248 | 351 | 386 |
| hess | 4.89 | 716 | 983 | 1.27e+03 |
| grad_l2 | 4.86 | 13.4 | 13.2 | 12.8 |
| grad_srank | 58.1 | 55 | 55.4 | 56 |
| rank_ef | 42.6 | 5.03 | 3.93 | 3.85 |
| srank_feat | 119 | 68.7 | 43.2 | 36.1 |
| dorm | 0.0712 | 0.722 | 0.823 | 0.841 |

**media_sem_particao**  (T=12)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 1 | 1 | - | - |
| norma | 63.5 | 175 | - | - |
| hess | 0.000429 | 64.7 | - | - |
| grad_l2 | 5.57 | 11.7 | - | - |
| grad_srank | 58.1 | 56 | - | - |
| rank_ef | 55.3 | 8.68 | - | - |
| srank_feat | 120 | 87.7 | - | - |
| dorm | 0.0512 | 0.641 | - | - |

## (2) Tempo de meia-excursão t50 por semente; adiantamento = t50(acc) - t50(sinal)

| condição | t50 acc (mediana) | adiant. norma | adiant. |hess| | adiant. dormentes | adiant. srank_feat | adiant. rank_ef |
|---|---|---|---|---|---|---|
| central_passos (n=5/5) | +14.0 [+10,+19] | +5.0 [+1,+11] | -1.0 [-3,+13] | +6.0 [+2,+11] | +6.0 [+2,+11] | +9.0 [+1,+15] |
| central_aprendizado (n=5/5) | +11.0 [+7,+27] | +2.0 [-2,+18] | +0.0 [-1,+22] | +3.0 [-2,+19] | +3.0 [-2,+18] | +2.0 [+1,+22] |
| sem_media_k1 (n=5/5) | +19.0 [+11,+22] | +10.0 [+1,+13] | +9.0 [-7,+13] | +10.0 [+4,+13] | +10.0 [+3,+13] | +14.0 [+4,+17] |
| fedavg_iid (n=5/5) | +22.0 [+16,+24] | +10.0 [+5,+12] | +7.0 [+4,+14] | +14.0 [+9,+17] | +10.0 [+4,+13] | +17.0 [+11,+19] |

---
# Exploratório (sem decisão): tarefa B, nível m=3, sementes 1 a 5

## (1) Trajetórias médias por bloco de tarefas (média entre sementes)

**central_passos**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.979 | 0.939 | 0.893 | 0.883 |
| norma | 25.8 | 74.5 | 105 | 114 |
| hess | 7.12 | 80.4 | 177 | 209 |
| grad_l2 | 2.97 | 6.91 | 8.03 | 7.62 |
| grad_srank | 58.8 | 57.2 | 58.1 | 57.8 |
| rank_ef | 50.5 | 12.6 | 5.53 | 4.99 |
| srank_feat | 114 | 63.6 | 29.3 | 25.1 |
| dorm | 0.0813 | 0.562 | 0.787 | 0.816 |

**central_aprendizado**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.987 | 0.95 | 0.899 | 0.879 |
| norma | 26.7 | 78.4 | 111 | 122 |
| hess | 6.77 | 90.7 | 217 | 244 |
| grad_l2 | 3.11 | 7.59 | 9.01 | 8.66 |
| grad_srank | 58.8 | 56.9 | 57.4 | 57.3 |
| rank_ef | 50 | 11.6 | 5.34 | 4.75 |
| srank_feat | 113 | 62.9 | 30.6 | 25.8 |
| dorm | 0.0838 | 0.561 | 0.783 | 0.823 |

**sem_media_k1**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.973 | 0.944 | 0.89 | 0.874 |
| norma | 25.4 | 75.1 | 109 | 121 |
| hess | 7.44 | 72.6 | 162 | 195 |
| grad_l2 | 3 | 7.35 | 8.18 | 7.74 |
| grad_srank | 59 | 57 | 57.4 | 57.8 |
| rank_ef | 50.3 | 13.9 | 6.2 | 5.4 |
| srank_feat | 116 | 70.2 | 36.2 | 29.4 |
| dorm | 0.0569 | 0.498 | 0.743 | 0.788 |

**fedavg_iid**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.985 | 0.987 | 0.966 | 0.963 |
| norma | 26 | 77.5 | 126 | 146 |
| hess | 13.4 | 96.6 | 225 | 256 |
| grad_l2 | 3.68 | 10.3 | 13.4 | 12.8 |
| grad_srank | 58.8 | 56.8 | 57.1 | 56.5 |
| rank_ef | 39.8 | 13.4 | 7.37 | 6.29 |
| srank_feat | 117 | 96.2 | 74.7 | 64.8 |
| dorm | 0.0341 | 0.349 | 0.548 | 0.601 |

**media_sem_particao**  (T=12)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.997 | 0.993 | - | - |
| norma | 26.3 | 70.6 | - | - |
| hess | 5.71 | 68.8 | - | - |
| grad_l2 | 3.39 | 8.02 | - | - |
| grad_srank | 58.6 | 57 | - | - |
| rank_ef | 46 | 13.4 | - | - |
| srank_feat | 116 | 85.7 | - | - |
| dorm | 0.0606 | 0.435 | - | - |

**central_L2**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.987 | 0.996 | 0.991 | 0.989 |
| norma | 22.4 | 37.7 | 43.3 | 46.7 |
| hess | 6.48 | 40.1 | 61.9 | 51.6 |
| grad_l2 | 3.01 | 7.07 | 8.28 | 7.44 |
| grad_srank | 58.8 | 56.6 | 57.4 | 56.6 |
| rank_ef | 50.9 | 22.6 | 15.3 | 14 |
| srank_feat | 119 | 113 | 108 | 107 |
| dorm | 0.0025 | 0.0275 | 0.0522 | 0.0516 |

**central_SP**  (T=30)
| sinal | t0-4 | t10-14 | t20-24 | t25-29 |
|---|---|---|---|---|
| acc_cont | 0.986 | 0.997 | 0.996 | 0.996 |
| norma | 20.9 | 28.8 | 29.7 | 29.9 |
| hess | 5.36 | 6.55 | 7.62 | 8.43 |
| grad_l2 | 1.84 | 3.2 | 3.38 | 3.38 |
| grad_srank | 61.1 | 59.8 | 60 | 59.7 |
| rank_ef | 45.3 | 24.5 | 20.2 | 18.7 |
| srank_feat | 116 | 105 | 103 | 102 |
| dorm | 0.0275 | 0.0553 | 0.0319 | 0.0266 |

## (2) Tempo de meia-excursão t50 por semente; adiantamento = t50(acc) - t50(sinal)

| condição | t50 acc (mediana) | adiant. norma | adiant. |hess| | adiant. dormentes | adiant. srank_feat | adiant. rank_ef |
|---|---|---|---|---|---|---|
| central_passos (n=5/5) | +14.0 [+14,+16] | +3.0 [+1,+4] | -1.0 [-3,+0] | +4.0 [+3,+6] | +3.0 [+1,+5] | +6.0 [+6,+9] |
| central_aprendizado (n=5/5) | +16.0 [+14,+17] | +4.0 [+3,+4] | +1.0 [-2,+1] | +6.0 [+4,+8] | +4.0 [+3,+6] | +9.0 [+6,+10] |
| sem_media_k1 (n=5/5) | +16.0 [+15,+17] | +3.0 [+2,+4] | +0.0 [-2,+1] | +5.0 [+4,+6] | +3.0 [+1,+5] | +7.0 [+7,+9] |
| fedavg_iid (n=5/5) | +0.0 [+0,+0] | -15.0 [-15,-14] | -17.0 [-18,-15] | -11.0 [-13,-11] | -15.0 [-18,-14] | -7.0 [-8,-7] |
