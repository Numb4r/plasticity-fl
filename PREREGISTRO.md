# Pré-registro

Congelado antes de qualquer experimento. Não editar o texto acima da seção "Desvios". Mudanças vão para "Desvios", com data e motivo.

## Pergunta
A vantagem do FedAvg sobre o treino centralizado em preservar plasticidade sobrevive a controles de passos, de partição dos dados e de regularização?

Plasticidade = capacidade da rede de continuar aprendendo, medida contra uma rede nova (Dohare et al., 2024; Lyle et al., 2022).

## Métricas
- **Principal**: média, nas últimas 5 tarefas, de (acurácia de treino da rede contínua − acurácia de treino da rede nova). Só entram as tarefas dessa janela que têm rede nova de referência (ver "Escolhas operacionais", item 1). Agregação entre sementes: média das médias por semente, com IC 95% por bootstrap percentil (10 000 reamostragens das sementes).
- **Secundária**: razão da sonda de alvos aleatórios, perda_cópia / perda_rede_nova.

## Hipóteses e critérios
- **H_reg** (a vantagem é regularização conhecida). Atendida se existir ao menos um entre `central_L2` e `central_SP` cuja média da métrica principal fica dentro do IC 95% do `fedavg_iid` nas duas tarefas (A e B).
- **H_part** (a vantagem vem de dividir os dados entre clientes que discordam, não da média). Atendida se: (i) nas tarefas A e B, o IC 95% de `media_sem_particao` sobrepõe o IC 95% de `central_aprendizado`; e (ii) na tarefa B, o IC 95% de `fedavg_iid` sobrepõe o IC 95% de `central_aprendizado`. Como `media_sem_particao` tem T=12, a comparação de H_part usa, para todas as condições, a janela das tarefas 8 a 12 (índices 7 a 11), com a mesma regra da métrica principal. Pré-condição para H_part ser avaliável: na métrica principal (janela 26 a 30), `fedavg_iid` tem vantagem sobre `central_aprendizado` na tarefa B (ICs sem sobreposição, fedavg acima). Sem a pré-condição, a tabela reporta "H_part: não avaliável (sem vantagem a explicar)".
- **H_real** (há efeito além disso). Atendida se, nas tarefas A e B, o limite inferior do IC 95% de `fedavg_iid` fica acima do limite superior do IC 95% de `central_L2` e de `central_SP`.
- Mais de uma hipótese pode ser atendida; a tabela lista todas. Nenhuma atendida também é um resultado e é reportado como "nenhum".

## Condições (T=30 tarefas, 5 sementes, tarefas A e B)
1. `fedavg_iid`: 10 clientes × 100 pontos, partição aleatória, média ponderada de todo o `state_dict`.
2. `sem_media_k1`: 1 cliente com os 1000 pontos, mesmo protocolo local.
3. `media_sem_particao`: 10 clientes com os mesmos 1000 pontos; T=12.
4. `central_passos`: mesmo número de passos sequenciais por tarefa que um cliente do `fedavg_iid` (25 rodadas × 3 épocas × 5 lotes = 375 passos), batch 20, Adam recriado a cada 15 passos.
5. `central_aprendizado`: batch 20, Adam recriado a cada 15 passos, número de passos calibrado na semente 0 para a rede nova atingir a acurácia da rede nova do `fedavg_iid` (±0,03); congelado para as outras sementes.
6. `central_L2`: como 5, com L2; λ ∈ {1e-4, 1e-3, 1e-2} escolhido na semente 0 e congelado.
7. `central_SP`: como 5, com shrink-and-perturb (Ash e Adams, 2020) no início de cada tarefa, θ ← λθ + N(0, 0,01²); λ ∈ {0,4; 0,6; 0,8} escolhido na semente 0 e congelado.

Protocolo FL local: batch 20, 3 épocas locais, Adam (lr 1e-3) recriado a cada rodada, R=25 rodadas por tarefa. Modelo MLP 32-128-128-10 com ReLU. N=1000, D=32, X ~ N(0,1).

## Escolhas operacionais (fixadas aqui para fechar ambiguidades do enunciado)
1. **Rede nova de referência**: treinada nas tarefas de índice (a partir de 0) t com t mod 3 = 2, isto é, a 3ª, 6ª, …, 30ª. Na janela das últimas 5 tarefas (índices 25 a 29) isso dá 2 referências (26 e 29); com T=12 e janela 8 a 12 (índices 7 a 11), 2 referências (8 e 11). A diferença contínua − nova de uma tarefa usa a rede nova da mesma tarefa (mesmos rótulos).
2. **Acurácia de treino** = acurácia do modelo (global, no FL) nos 1000 pontos ao fim da tarefa.
3. **Rede nova de cada condição** usa o mesmo protocolo da condição (FL, passos, L2, SP). Em `central_SP`, a rede nova também recebe o passo de shrink-and-perturb antes de treinar. O passo SP da rede contínua é aplicado no início das tarefas de índice ≥ 1.
4. **Calibração (condição 5)**: varre o número de passos em múltiplos de 15 na semente 0 e escolhe o que deixa a acurácia média da rede nova mais perto da do `fedavg_iid` (média das redes novas do `fedavg_iid` na semente 0, por tipo de tarefa). A calibração usa sementes de rede nova próprias, distintas das da avaliação. Se nenhum valor ficar a ±0,03, o fato é registrado em "Desvios" e a execução para.
5. **Escolha de λ** (condições 6 e 7): maior média, nas últimas 5 tarefas, da acurácia de treino da rede contínua na semente 0, por tipo de tarefa. A semente 0 também entra na avaliação final; esse reuso é um limite conhecido.
6. **L2** = termo λ·θ somado ao gradiente (`weight_decay` do Adam), aplicado em todos os parâmetros.
7. **Dormentes** (Sokar et al., 2023): ativações pós-ReLU das duas camadas ocultas nos 1000 pontos; score = E|h_i| / média_j E|h_j| na camada; dormente se score ≤ τ, com τ ∈ {0; 0,025}. Camada com todas as ativações zero conta 100%.
8. **Rank efetivo**: exp da entropia dos valores singulares normalizados da matriz de ativações (1000 × 128) da última camada oculta (Roy e Vetterli, 2007).
9. **Sonda de alvos aleatórios**: a cada 5 tarefas (índices 4, 9, …), na cópia do modelo; 300 passos de Adam (lr 1e-3, lotes de 100) com MSE contra as saídas de uma rede-alvo nova; perda final medida nos 1000 pontos; rede nova treinada nos mesmos alvos e passos; razão por alvo, média de 3 alvos.
10. **Cosseno entre clientes**: média dos cossenos entre pares de (θ_cliente − θ_global) achatados, por rodada; o CSV por tarefa guarda a média das 25 rodadas.
11. **Sementes**: a semente s define dados X, rótulos, professores, partição e inicializações por fontes independentes (`src/sementes.py`). A mesma semente dá os mesmos X, rótulos e professores entre condições (desenho pareado).
12. **IC 95%**: bootstrap percentil sobre as 5 sementes, 10 000 reamostragens, semente fixa.

## Desvios
(nenhum)
