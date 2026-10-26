# RASCUNHO · Rodada 4 · Alavancas do mecanismo e remédios no servidor

**Status: rascunho para revisão do autor (2026-10-26). Nada implementado, nada em `PREREGISTRO.md`.** A rodada 4 só deve ser pré-registrada **depois dos resultados da rodada 3**, porque as previsões das alavancas dependem do que a rodada 3 disser sobre o mecanismo (desacordo entre clientes amortecido pela média). Este rascunho fixa o desenho para que o autor possa revisar já.

## Pergunta e lógica
A leitura exploratória atual é que o efeito protetor vem da média entre clientes com dados diferentes. A rodada 3 testa isso variando K e a origem dos dados. A rodada 4 usa o mecanismo como **alavanca** e como **alvo de remédio**:
- **4A, alavancas.** Se a proteção vem do desacordo entre as atualizações dos clientes amortecido pela média, métodos que mudam o desacordo ou o modo de promediar devem mover a proteção na direção prevista pelo mecanismo.
- **4B, remédios no servidor.** Onde a média sozinha não protege (a média de cópias, K = 10, `copias`), remédios aplicados ao modelo global pelo servidor devem restaurar a proteção; onde protege (`particao_iid`, K = 10), não devem estragá-la.
- **Referência.** L2 (central e federado), a regularização que já protegeu na rodada 1.

## Cenário comum (herdado da rodada 3)
m\* = 3, T = 30, tarefas A e B, K = 10 clientes, S_local = 15, batch 20, Adam recriado a cada rodada, métrica principal e bootstrap como na rodada 2 (gap pareado médio das tarefas 26 a 30). Bases: `particao_iid` K = 10 (protegida, = `fedavg_iid`) e `copias` K = 10 (desprotegida, = `media_sem_particao`), mais `sem_media_k1`. Calibração do nível da rede nova pela **perda** (alvo = `fedavg_iid` K = 10, semente 1000), calibrando o número de rodadas R por condição numa grade de passo 1 (faixa a decidir; a rodada 3 usa 50 a 200); fora de 10% = "não avaliável", sem forçar. δ = 0,05. Sementes novas (24 a 31, 8 sementes; semente 1000 só para calibrar e escolher parâmetros). Medidas por rodada: cosseno entre atualizações dos clientes, fator de encolhimento da média, norma dos parâmetros; por tarefa: os diagnósticos de sempre (norma, `rank_ef` e `srank_feat` juntos, dormentes, Hessiana).

## 4A · Alavancas do mecanismo
Cada alavanca age num ponto diferente da cadeia "clientes com dados diferentes → atualizações que discordam → média que as amortece". Aplicadas à base `particao_iid` K = 10 e, quando fizer sentido, à `copias`.

1. **SCAFFOLD** (Karimireddy et al., 2020). Variáveis de controle corrigem o desvio local: cada cliente corrige o gradiente local por (c − c_k). Alavanca: **remove o desacordo**. Previsão do mecanismo: o cosseno entre atualizações sobe e a proteção cai em direção à da `copias`. Critério (proposta): gap(SCAFFOLD) − gap(`particao_iid`) com IC pareado inteiro abaixo de −δ, nas duas tarefas.
2. **FedRep** (Collins et al., 2021). Só o corpo (as duas camadas ocultas) é promediado; a cabeça (camada de saída) fica local, treinada primeiro por alguns passos, depois o corpo. Alavanca: **onde mora a média**. Se a proteção vem de promediar a representação, ela fica (dentro de ±δ da `particao_iid`); se vem de promediar a cabeça, cai. **Decisão a tomar:** qual é o modelo avaliado e medido (corpo global com a média das cabeças locais, ou corpo global com a cabeça refeita em poucos passos nos dados de cada cliente), porque muda a acurácia da rede contínua e da nova. Critério: equivalência de dois lados com δ.
3. **Gossip** (Boyd et al., 2006; Koloskova et al., 2019). Sem servidor: cada cliente promedia com os vizinhos por uma matriz de mistura W. Topologias: anel, grafo aleatório regular de grau 4 e completo (= FedAvg). Alavanca: **a dose de média**, medida pelo gap espectral 1 − λ₂(W) e pela distância ao consenso. Previsão: a proteção cresce com o gap espectral (completo > aleatório > anel). Modelo avaliado: a média dos pesos dos clientes (e a distância de cada cliente a ela reportada). Critério: gap(completo) − gap(anel) com IC pareado inteiro acima de +δ, e a ordem monotônica nas médias.
4. **DiLoCo** (Douillard et al., 2023). H passos locais com otimizador interno e otimizador externo (SGD com momento de Nesterov) sobre o pseudo-gradiente (diferença média dos pesos). Alavanca: **tamanho do passo local e do passo externo**. Grade: H ∈ {15, 50, 150} com taxa externa e momento do artigo (0,7 e 0,9) fixos, R recalibrado pelo nível da rede nova. **Exploratória**, sem critério, porque há muitos botões e a previsão do mecanismo é ambígua (mais passos locais aumentam o desacordo, mas o momento externo também muda o passo efetivo).

## 4B · Remédios no servidor (sem mexer nos clientes)
Aplicados ao modelo global antes de difundi-lo. Bases: `copias` K = 10 (para restaurar) e `particao_iid` K = 10 (para não estragar). A rede nova recebe o mesmo protocolo (como o shrink-and-perturb da rodada 1).

5. **Shrink-and-perturb no modelo global** (Ash e Adams, 2020). θ ← λθ + N(0, σ²), com σ = 0,01 como na rodada 1. Variantes: (a) no início de cada tarefa; (b) a cada rodada, com λ mais perto de 1. Grade de λ escolhida na semente 1000 (a rodada 1 tinha o λ na borda de 0,8; incluir valores além), por variante.
6. **ReDo federado** (Sokar et al., 2023, adaptado). Cada cliente mede a atividade média normalizada de cada neurônio das camadas ocultas nos seus dados; o servidor agrega (média ponderada por n) e reinicializa no modelo global os neurônios com atividade agregada ≤ τ (pesos de entrada reinicializados, pesos de saída zerados). Variantes: (a) agregação por média; (b) por voto (dormente em pelo menos q% dos clientes). Disparo: no início de cada tarefa (e, como variante, a cada N rodadas). τ = 0,025 como o dos diagnósticos do projeto; grade de q ∈ {50%, 80%}. Na `copias`, todos os clientes têm os mesmos dados, então as duas agregações coincidem; na `particao_iid`, não.
7. **L2 de referência.** `central_L2` (rodada 1; λ na borda de 1e-4, grade a estender abaixo: {1e-5, 1e-4, 1e-3}) e `fed_L2` (weight decay no Adam local dos clientes, mesma grade), escolhidos na semente 1000 pela mesma regra da rodada 1.

Critérios (proposta): **H_restaura** = em `copias` K = 10, gap(remédio) − gap(`copias`) com IC pareado inteiro acima de +δ e gap(remédio) ≥ gap(`particao_iid`) − δ; **H_não_estraga** = em `particao_iid` K = 10, gap(remédio) − gap(`particao_iid`) com IC inteiro dentro de ou acima de −δ. Comparação direta com o L2: diferença pareada reportada, sem critério. Veredito geral por tarefa: "atendida" só se vale nas duas; "parcial" em uma; "não avaliável" por calibração fora de 10% ou falta de dado.

## Custo e ordem
Condições: 3 bases; 4A: SCAFFOLD, FedRep, gossip × 3 topologias, DiLoCo × 3 valores de H = 8; 4B: shrink-and-perturb × 2 variantes e ReDo × 2 variantes, cada um em 2 bases = 8; L2: 2. Total 21 condições × 2 tarefas × 8 sementes ≈ 336 execuções (as bases repetem as da rodada 3, mas rodam nas sementes novas), mais a calibração de R e a escolha de parâmetros por condição na semente 1000. Custo não medido, da ordem de 40 a 50 CPU-h no total, ou de 2 a 3 h no cluster com 20 processos. **Sugestão:** dividir em 4a e 4b, pré-registrando 4b só depois de ver 4a e a rodada 3. Código novo: um treino federado com variáveis de controle (SCAFFOLD), com cabeça local (FedRep), com matriz de mistura (gossip), com otimizador externo (DiLoCo), e dois passos de servidor (perturbação dos pesos e reinicialização de neurônios); o laço de calibração por rodadas, que a rodada 3 já exige, serve para todos.

## Ameaças à validade (a declarar)
- Cada alavanca muda o treino também da rede nova; a calibração por perda iguala o nível dela, mas não o da rede contínua, e R diferente entre condições significa passos totais diferentes.
- O modelo avaliado em FedRep e em gossip não é um só modelo global; a escolha altera a acurácia e deve ser fixada antes de qualquer dado.
- O mecanismo "desacordo amortecido" ainda é uma leitura exploratória; se a rodada 3 não a sustentar, as previsões de 4A perdem o sentido.
- ReDo e shrink-and-perturb têm hiperparâmetros (τ, q, λ, σ, frequência); escolhidos numa só semente, podem se ajustar ao ruído.
- Muitos testes: a decisão usa critérios pré-registrados por hipótese, mas o número de comparações é grande; reportar todas, sem correção, com essa nota.

## Previsão do autor (rodada 4)
(a preencher pelo autor, antes de qualquer código da rodada 4)

## Decisões a confirmar pelo autor antes de pré-registrar
1. Dividir em 4a (alavancas) e 4b (remédios), pré-registrando a 4b depois da 4a e da rodada 3.
2. Modelo avaliado em FedRep (média das cabeças ou cabeça refeita) e em gossip (média dos pesos).
3. Topologias de gossip (anel, grau 4, completo) e valores de H do DiLoCo (15, 50, 150).
4. Variantes dos remédios (por tarefa e por rodada; ReDo por média e por voto) e as grades de λ, τ e q.
5. Grade do L2 estendida abaixo de 1e-4, escolhida na semente 1000.
6. Critérios: o limiar δ = 0,05, e se DiLoCo fica exploratório.
7. Sementes (24 a 31) e a faixa de R da calibração (a da rodada 3, 50 a 200, ou outra).
