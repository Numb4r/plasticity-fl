# RASCUNHO · Rodada 5 · Benchmarks reais (depois das rodadas 3 e 4)

**Status: ideia registrada em 2026-10-26; decisões do autor incorporadas no fim (2026-10-26). Nada implementado, nada pré-registrado, nenhum dado baixado.** Vem depois das rodadas 3 e 4; as hipóteses dependem do que elas disserem. Os números de custo são estimativas por contagem de operações, não medidas (incerteza de um fator 3 ou mais).

## Pergunta
O que as rodadas 1 a 4 acharam no problema sintético (a média entre trabalhadores com dados diferentes protege a plasticidade, e a média de cópias não) vale em benchmarks reais de aprendizado contínuo, com redes e dados reais, e no mesmo nível de aprendizado da rede nova?

## Benchmarks e o que está confirmado nas fontes
1. **Permuted MNIST com MLP.** Dohare et al. (versão arXiv 2306.13812v1; a versão da Nature pode diferir): "Online Permuted MNIST", MLP com 3 camadas ocultas de 2000 unidades, 800 tarefas, cada uma de 60 000 imagens vistas uma vez, em ordem aleatória. A rodada 5 **não** copia o regime online (48 milhões de exemplos com lote 1 não cabem no nosso protocolo de passos locais); usa MNIST inteiro por tarefa, MLP menor (proposta: 784-1000-1000-10, ≈ 1,8 M parâmetros), T = 100 tarefas (a reduzir ou ampliar na semente 1000).
2. **CIFAR-10 com rótulos reembaralhados a cada tarefa.** Lyle et al. (2023, arXiv 2303.01486): o estudo supervisionado do artigo usa MNIST com rótulos permutados de 5000 imagens, MLP de 2 camadas de 1024, Adam 1e-3; **o protocolo de CIFAR-10 com rótulos aleatórios não está nesse artigo** (a conferir na fonte original, que pode ser Lyle et al. 2021 ou um trabalho posterior). Proposta: subconjunto fixo de 10 000 imagens, rótulos sorteados de novo a cada tarefa, T = 50, com CNN pequena (a da Dohare, abaixo) e, como braço pesado, ResNet-18. Decisão: BatchNorm entre trabalhadores (média das estatísticas, ou trocar por GroupNorm).
3. **Continual ImageNet 32×32 com CNN.** Dohare et al. (arXiv 2306.13812v1): 1000 classes, 700 imagens por classe (600 de treino e 100 de teste), 32×32; cada tarefa é uma classificação binária entre duas classes sorteadas sem reposição (1200 imagens de treino e 200 de teste por tarefa); 2000 tarefas no experimento principal e 5000 no de backpropagation contínuo; CNN com 3 blocos convolucionais (32 filtros 5×5, 64 e 128 de 3×3, com max-pooling) e 3 camadas totalmente conectadas (128, 128 e 2 unidades), SGD com momento 0,9, **250 passagens** por tarefa com lote 100, cabeça zerada a cada tarefa. A rodada 5 propõe T = 200 tarefas (a ajustar na semente 1000) e o Adam do projeto como otimizador primário (ver decisões). O ImageNet reduzido exige licença e download; o projeto tem a regra "nenhum download de dados": **decisão do autor**.

## Condições (nomes neutros, sem "federado")
`localsgd_part` (K trabalhadores, cada um com um pedaço IID dos dados da tarefa; antes `fedavg_iid`), `localsgd_copias` (K trabalhadores com os mesmos dados; antes `media_sem_particao`), `sem_media_k1` (um trabalhador) e `central_aprendizado` (centralizado com passos calibrados). Mesmo protocolo local do toy (15 passos por rodada, Adam recriado a cada rodada, K = 10) e **calibração do número de rodadas R pela perda da rede nova** (alvo = `localsgd_part`, semente 1000, a até 10%), como nas rodadas 2b, 3 e 4. Fica para o pré-registro a faixa de R (50 a 200 foi escolhida para o toy). Remédios e alavancas da rodada 4 não entram, salvo uma escolha do autor depois da 4.

## Regime de teste (análogo ao m\*)
Antes de comparar, cada benchmark precisa mostrar perda de plasticidade no `sem_media_k1` dentro do horizonte (gap das 5 últimas tarefas ≤ −0,10, na semente 1000), senão o benchmark sai como "sem regime" e não decide nada. T, a taxa de aprendizado e o número de passos por tarefa seriam escolhidos na semente 1000 antes de qualquer semente de avaliação.

## Custo estimado (CPU e GPU no cluster)
Modelo: FLOPs por exemplo (ida e volta) × exemplos por tarefa (R = 100 rodadas × K = 10 × 15 passos × lote 20 = 300 000 amostras) × (1 + 0,36 de redes novas) × T tarefas. Eficiência suposta: CPU 15 (MLP) a 30 (CNN) GFLOP/s por núcleo; GPU L40S 3 TFLOP/s efetivos com um trabalhador por vez e lote 20, 20 TFLOP/s se os 10 trabalhadores forem vetorizados num só lote (exige código novo); piso por sobrecarga de núcleos da GPU ≈ 0,5 ms por passo. Totais para 4 condições × 8 sementes, mais 30% de calibração, sem os diagnósticos de Hessiana.

| Benchmark | PFLOP por execução `localsgd` | CPU, uma execução | GPU sequencial, uma execução | GPU vetorizada, uma execução | CPU-h total | GPU-h total (sequencial / vetorizada) |
|---|---|---|---|---|---|---|
| Permuted MNIST, MLP 1,8 M, T = 100 | 0,44 | ≈ 8 h | ≈ 0,3 h (piso por passo) | ≈ 0,04 h | ≈ 190 | ≈ 7 / ≈ 1 |
| CIFAR-10 relabel, CNN pequena, T = 50 | 8 | ≈ 76 h | ≈ 0,8 h | ≈ 0,1 h | ≈ 1 700 | ≈ 17 / ≈ 3 |
| CIFAR-10 relabel, ResNet-18, T = 50 | 67 | ≈ 620 h | ≈ 6 h | ≈ 1 h | ≈ 14 000 | ≈ 140 / ≈ 21 |
| Continual ImageNet 32×32, CNN, T = 200 | 33 | ≈ 300 h | ≈ 3 h | ≈ 0,45 h | ≈ 6 900 | ≈ 70 / ≈ 10 |

Leitura: só o Permuted MNIST cabe numa CPU do cluster; os outros três precisam de GPU. Em GPU, treinar os 10 trabalhadores vetorizados (uma implementação nova, com a mesma matemática) corta o custo de 6 a 7 vezes e vale a pena para a CNN e a ResNet. Sem GPU determinística, a reprodutibilidade deixa de ser bit a bit e passa a ser estatística (pareamento por semente de dados e de inicialização, não por contas idênticas); isso muda a regra do projeto e precisa ser registrado.

## O que continua valendo das métricas da rodada 1 e o que muda
- **Gap pareado (acurácia da contínua menos a da rede nova, na mesma tarefa)**: continua, com a rede nova de referência nas mesmas tarefas (t mod 3 = 2, as 5 últimas e uma janela intermediária). Muda o que é "acurácia": nas tarefas de rótulos aleatórios continua sendo acurácia de treino (capacidade de ajuste); no Continual ImageNet e no Permuted MNIST a literatura usa acurácia de teste. O pré-registro fixa a métrica principal (proposta: treino, com teste como secundária).
- **Perda de treino da rede nova** (a calibração de R): continua e é o centro do método.
- **Rank efetivo e `srank_feat`**: continuam na última camada oculta (nas CNNs, a 128 unidades). Sempre os dois juntos.
- **Dormentes**: continuam; nas convoluções, a média por canal sobre as posições.
- **Norma dos parâmetros**: perde significado com BatchNorm (invariância de escala); com BatchNorm usar norma por camada ou outra medida; sem BatchNorm continua.
- **Cosseno entre atualizações e fator de encolhimento**: continuam como estão (é a medida do mecanismo).
- **Maior autovalor da Hessiana e espectro do gradiente**: caros em redes grandes; usar subconjunto fixo de pontos (256) ou só as últimas camadas; opcional na ResNet-18.
- **Sonda de alvos aleatórios**: não validada nas rodadas 2 e 2b; só reaparece se for refeita e validada antes (por exemplo, com saídas normalizadas).
- **Esquecimento e reaprendizado**: sem sentido nas tarefas de rótulos aleatórios; no Permuted MNIST seria medível (acurácia nas permutações antigas) e entra como secundária.

## Ameaças e decisões para o pré-registro
Dados: baixar MNIST e CIFAR-10 (sem licença) e ImageNet reduzido (com licença) contraria a regra "nenhum download"; R diferente entre condições = passos totais diferentes; K = 10 e IID fixos; BatchNorm; otimizador (Adam do toy contra SGD com momento da Dohare); horizonte T e escala dos tarefas bem menores que os dos artigos (2000 a 5000 tarefas); a CNN de Dohare tem pooling de passo 1 (custo maior que o de uma CNN comum); reprodutibilidade em GPU; hiperparâmetros escolhidos numa só semente; previsão do autor.

## Decisões do autor (2026-10-26)
1. **Escopo:** começar por **Permuted MNIST e CIFAR-10**; o Continual ImageNet fica de fora até resolver a licença (e a regra de não baixar dados para ele).
2. **Otimizador:** Adam como principal; um braço com **SGD com momento só no Permuted MNIST**.
3. **Normalização:** **GroupNorm** no lugar de BatchNorm (resolve a média das estatísticas entre trabalhadores e a invariância de escala da norma).
4. **Métrica:** **acurácia de treino como principal**, de teste como secundária.
5. **Horizonte:** T menor que o dos artigos, desde que a regra do regime (perda de plasticidade do `sem_media_k1` na semente 1000) seja cumprida.
6. **GPU:** a reprodutibilidade **não é bit a bit** (registrar no pré-registro) e se **compensa com mais sementes**.
7. **Origem do protocolo de CIFAR-10 com rótulos aleatórios:** continua "a conferir".
Consequência para o custo: sem o Continual ImageNet, ficam o Permuted MNIST (≈ 190 CPU-h, ou 1 a 7 GPU-h) e o CIFAR-10 (CNN: 3 a 17 GPU-h; ResNet-18: 21 a 140 GPU-h; com GroupNorm o custo não muda de forma relevante), mais as sementes extras.

## Previsão do autor (rodada 5)
(a preencher pelo autor, antes de qualquer código da rodada 5)

Fontes: Dohare et al., "Loss of plasticity in deep continual learning" (Nature, 2024; protocolo conferido na versão arXiv 2306.13812v1); Lyle et al., "Understanding plasticity in neural networks" (ICML 2023, arXiv 2303.01486).
