# Tabela de decisão da rodada 2b (central_HT_cal2)

Fonte: `results_rodada2b/` (sementes 11 a 15; a semente 1000 nunca entra). Regras literais de PREREGISTRO.md (rodada 2b e blocos de 2026-10-21/22); decisões de leitura em `plots/rodada2b_analise.py`. Vereditos: texto calculado, sem interpretação.

## 1. Calibração de R (semente 1000) e μ

R = rodadas por tarefa (grade 75 a 150); erro_rel = |loss_ht − alvo| / alvo; ok = erro_rel ≤ 10%; alvo = `loss_nova` do media_sem_particao; [borda] = μ na borda da grade (mantido).

| tarefa | R | erro_rel | ok | alvo (loss_nova msp) | loss_ht | μ | notas |
|---|---|---|---|---|---|---|---|
| A | 86 | -0.006 | True | 0.708 | 0.703 | 0.99 | μ=0.9: 0.633; μ=0.99: 0.983; μ=0.995: 0.976; μ=0.999: 0.907 |
| B | 89 | 0.003 | True | 0.141 | 0.142 | 0.995 | μ=0.9: 0.910; μ=0.99: 0.995; μ=0.995: 0.996; μ=0.999: 0.986 |

## 2. Métrica principal por condição e tarefa (gap pareado médio das 5 últimas tarefas: 25 a 29)

Média entre sementes [IC 95% bootstrap percentil, 10 000, default_rng(0)] (n sementes).

| condição | A | B |
|---|---|---|
| central_HT_cal2 | 0.146 [0.142, 0.150] (n=5) | 0.021 [0.020, 0.023] (n=5) |
| media_sem_particao | -0.264 [-0.289, -0.239] (n=5) | -0.034 [-0.043, -0.026] (n=5) |
| sem_media_k1 | -0.301 [-0.369, -0.254] (n=5) | -0.077 [-0.088, -0.065] (n=5) |

## 3. Diferença pareada central_HT_cal2 − media_sem_particao (δ = 0,02)

**Veredito geral (H_tempo e H_dist, o mesmo): não avaliável (P_vant falhou em A)**

Classe: dentro = IC todo em [−δ, +δ]; acima ("HT acima") = inf > +δ; abaixo (H_dist) = sup < −δ; senão intermediário. Veredito "não avaliável": a classe por tarefa é só descritiva (não decisória).

Nota: a perda CE da rede nova está igualada; a acurácia pode não estar; gap = acc_cont − acc_nova.

| tarefa | d (gap), IC 95% | classe | diferença média de acc_nova (central_HT_cal2 − msp, 5 últimas tarefas, média entre sementes) | dentro de ±0,03 |
|---|---|---|---|---|
| A | 0.410 [0.387, 0.434] (n=5) | acima (não decisória) | -0.037 | não |
| B | 0.055 [0.047, 0.064] (n=5) | acima (não decisória) | -0.013 | sim |

Critério secundário (literal; só reportado, sem veredito): média de central_HT_cal2 dentro do IC de media_sem_particao; inf(IC msp) acima do sup(IC central_HT_cal2).

| tarefa | central_HT_cal2 dentro do IC de msp (sec_dentro) | inf(msp) > sup(central_HT_cal2) (sec_acima) |
|---|---|---|
| A | False | False |
| B | False | False |

## 4. P_vant e elegibilidade

P_vant: inf(IC msp) > sup(IC sem_media_k1). Elegibilidade: |perda CE da rede nova HT − msp| / msp ≤ 10% (sementes 11 a 15); acurácia ±0,03 só reportada.

| tarefa | P_vant | perda CE nova HT | perda CE nova msp | razão relativa | elegível | Δ acc nova (±0,03, só reportada) |
|---|---|---|---|---|---|---|
| A | False (msp -0.264 [-0.289, -0.239] (n=5); k1 -0.301 [-0.369, -0.254] (n=5)) | 0.696 | 0.692 | 0.005 | True | -0.037 (False) |
| B | True (msp -0.034 [-0.043, -0.026] (n=5); k1 -0.077 [-0.088, -0.065] (n=5)) | 0.143 | 0.144 | 0.004 | True | -0.013 (True) |

Motivos de "não avaliável": P_vant falhou em A

## 5. Previsão do autor × resultado (só o item H_tempo e H_dist)

Previsão literal da rodada 2 (valendo sem alteração): "H_tempo não, H_dist não, H_het sim, FedAvg só atrasa". Na 2b só se julga o item H_tempo e H_dist (H_het e horizonte não entram). Regras confirmadas em 2026-10-21/22: acerto só com "HT acima" no veredito geral; "misto" com "HT acima" numa tarefa = acerto parcial; "não avaliável" não conta.

| item | previsão do autor | veredito calculado | julgamento |
|---|---|---|---|
| H_tempo e H_dist (um só item) | H_tempo não e H_dist não (desfecho "HT acima") | não avaliável (P_vant falhou em A) | não conta (não avaliável) |
