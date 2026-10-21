# EXPLORATÓRIO · sonda em escala logarítmica (Δlog)

**Análise exploratória pós-resultados (PREREGISTRO.md, item C). Não altera nenhum veredito, não entra na `tabela_decisao.md` nem em regra de decisão. A sonda original (razão) fica como está.**

- Δlog = log10(média das perdas da cópia + 1e-6) − log10(média das perdas da rede nova + 1e-6), por semente, nas tarefas de sonda (índices 4, 9, 14, 19, 24, 29; 4 e 9 em `media_sem_particao`).
- **Limite:** os CSVs guardam só a **média das 3 perdas por alvo** (`sonda_perda_copia`, `sonda_perda_nova`); a **mediana entre alvos não é computável** com estes dados. Log da média não é média do log.
- Por semente: média do Δlog (e da norma global, coluna `fim_norma_global`) ao longo das tarefas de sonda; depois média e IC 95% por bootstrap percentil (10 000, `default_rng(0)`) entre as sementes 1 a 5.
- A norma global é mostrada ao lado só para comparação; nenhuma relação é testada aqui.

## Tarefa A · nível m=1

| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +3.5887 | [+3.5293; +3.6475] | 5 | +34.6869 | [+33.9480; +35.6096] | 5 |
| sem_media_k1 | +2.9915 | [+2.9614; +3.0227] | 5 | +36.1454 | [+35.5954; +36.6701] | 5 |
| media_sem_particao | +1.8610 | [+1.8251; +1.8925] | 5 | +20.7458 | [+20.3275; +21.1838] | 5 |
| central_passos | +2.8604 | [+2.7839; +2.9088] | 5 | +36.3319 | [+35.8245; +36.7369] | 5 |
| central_aprendizado | +3.0798 | [+2.9715; +3.1904] | 5 | +41.0055 | [+40.6167; +41.4340] | 5 |

## Tarefa A · nível m=3

| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +5.3757 | [+5.3062; +5.4576] | 5 | +115.4029 | [+112.8634; +117.9424] | 5 |
| sem_media_k1 | +3.8089 | [+3.3510; +4.1910] | 5 | +104.0112 | [+101.6915; +106.0850] | 5 |
| media_sem_particao | +3.4795 | [+3.3561; +3.6764] | 5 | +58.4719 | [+57.9597; +58.9841] | 5 |
| central_passos | +3.3479 | [+3.0659; +3.7808] | 5 | +104.0997 | [+102.2675; +106.2529] | 5 |
| central_aprendizado | +3.3676 | [+3.0371; +3.6981] | 5 | +108.8155 | [+107.7795; +110.4165] | 5 |
| central_L2 | +4.1829 | [+3.7245; +4.5428] | 5 | +52.5782 | [+51.9633; +53.2631] | 5 |
| central_SP | +3.0775 | [+3.0451; +3.1082] | 5 | +31.1295 | [+30.8603; +31.3987] | 5 |

## Tarefa A · nível m=10

| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +7.9007 | [+7.7943; +8.0072] | 5 | +279.6821 | [+274.7008; +284.3378] | 5 |
| sem_media_k1 | +5.9854 | [+5.2168; +6.7539] | 5 | +240.1389 | [+231.3543; +248.9234] | 5 |
| media_sem_particao | +7.5237 | [+7.4199; +7.6437] | 5 | +121.1183 | [+117.4761; +125.7026] | 5 |
| central_passos | +5.2216 | [+4.6916; +5.9196] | 5 | +234.6239 | [+225.6110; +242.7916] | 5 |
| central_aprendizado | +4.5095 | [+3.9860; +5.0452] | 5 | +179.3343 | [+175.1101; +183.5585] | 5 |

## Tarefa B · nível m=1

| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +3.9011 | [+3.8363; +3.9738] | 5 | +35.3714 | [+34.7203; +35.8941] | 5 |
| sem_media_k1 | +3.4716 | [+3.4240; +3.5190] | 5 | +30.7275 | [+30.1300; +31.2470] | 5 |
| media_sem_particao | +2.4162 | [+2.3598; +2.4664] | 5 | +19.7779 | [+19.3418; +20.2059] | 5 |
| central_passos | +3.5647 | [+3.5270; +3.6027] | 5 | +29.8400 | [+29.3795; +30.2400] | 5 |
| central_aprendizado | +3.7625 | [+3.6903; +3.8361] | 5 | +34.2787 | [+33.4437; +35.1137] | 5 |

## Tarefa B · nível m=3

| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +5.9731 | [+5.9319; +6.0142] | 5 | +97.6261 | [+95.8735; +99.1874] | 5 |
| sem_media_k1 | +5.2626 | [+5.2019; +5.3519] | 5 | +86.8069 | [+84.8045; +88.8092] | 5 |
| media_sem_particao | +4.5077 | [+4.4736; +4.5514] | 5 | +49.5132 | [+48.9508; +50.0757] | 5 |
| central_passos | +5.1705 | [+5.0907; +5.2481] | 5 | +84.0902 | [+81.7255; +86.5399] | 5 |
| central_aprendizado | +5.3470 | [+5.1852; +5.4990] | 5 | +88.9116 | [+85.4866; +91.7685] | 5 |
| central_L2 | +5.3350 | [+5.2340; +5.4213] | 5 | +39.2957 | [+38.2600; +40.2775] | 5 |
| central_SP | +3.6982 | [+3.6600; +3.7364] | 5 | +28.4894 | [+28.1044; +28.9047] | 5 |

## Tarefa B · nível m=10

| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +8.7763 | [+8.7548; +8.8038] | 5 | +133.5792 | [+129.8551; +137.3033] | 5 |
| sem_media_k1 | +8.2237 | [+8.1585; +8.2985] | 5 | +180.2772 | [+177.1829; +183.3715] | 5 |
| media_sem_particao | +7.5591 | [+7.4877; +7.6413] | 5 | +53.5051 | [+52.6051; +54.4432] | 5 |
| central_passos | +8.1256 | [+8.1005; +8.1499] | 5 | +178.1779 | [+172.5024; +184.1355] | 5 |
| central_aprendizado | +6.5789 | [+6.5321; +6.6257] | 5 | +127.5815 | [+124.5468; +130.6795] | 5 |

## Tarefa B_disjunta · nível m=1

| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +3.7642 | [+3.7316; +3.7963] | 5 | +36.6209 | [+35.3756; +37.8662] | 5 |
| sem_media_k1 | +3.3982 | [+3.3758; +3.4180] | 5 | +32.2617 | [+31.7176; +32.7605] | 5 |
| media_sem_particao | +2.3656 | [+2.3093; +2.4144] | 5 | +20.2720 | [+20.0204; +20.4404] | 5 |
| central_passos | +3.4198 | [+3.3702; +3.4729] | 5 | +32.5844 | [+31.9222; +33.0919] | 5 |
| central_aprendizado | +3.6710 | [+3.6368; +3.7168] | 5 | +37.2394 | [+36.1299; +38.2266] | 5 |

## Tarefa B_disjunta · nível m=3

| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +5.7626 | [+5.7341; +5.7943] | 5 | +95.0805 | [+92.8460; +97.8365] | 5 |
| sem_media_k1 | +5.2853 | [+5.2071; +5.3605] | 5 | +89.8769 | [+87.5958; +92.2184] | 5 |
| media_sem_particao | +4.4496 | [+4.4061; +4.4998] | 5 | +47.1089 | [+46.1314; +48.0317] | 5 |
| central_passos | +5.2189 | [+5.1542; +5.2801] | 5 | +93.1675 | [+91.6098; +94.7108] | 5 |
| central_aprendizado | +5.2027 | [+5.1220; +5.2833] | 5 | +95.7112 | [+94.6804; +96.7538] | 5 |
| central_L2 | +5.0387 | [+5.0121; +5.0672] | 5 | +37.7571 | [+37.5297; +37.9844] | 5 |
| central_SP | +3.4889 | [+3.4555; +3.5230] | 5 | +26.8376 | [+26.4652; +27.1221] | 5 |

## Tarefa B_disjunta · nível m=10

| condição | Δlog: média | Δlog: IC 95% | n | norma global: média | norma global: IC 95% | n |
|---|---|---|---|---|---|---|
| fedavg_iid | +8.7470 | [+8.6764; +8.8044] | 5 | +131.2759 | [+128.4984; +134.5226] | 5 |
| sem_media_k1 | +8.1957 | [+8.1603; +8.2312] | 5 | +163.1044 | [+162.6966; +163.3317] | 5 |
| media_sem_particao | +7.8735 | [+7.7434; +8.0037] | 5 | +69.9200 | [+69.1189; +70.9619] | 5 |
| central_passos | +8.1047 | [+8.0017; +8.1914] | 5 | +162.9162 | [+159.2275; +166.6049] | 5 |
| central_aprendizado | +7.1689 | [+7.1095; +7.2393] | 5 | +153.6733 | [+150.8202; +155.9846] | 5 |
