# RASCUNHO · Rodada 3 · Por que a média entre clientes com dados diferentes protege, e a média de cópias não?

**Status: decisões do autor incorporadas em 2026-10-26 (seção no fim); aguardando a previsão do autor. Nada implementado, nada commitado em `PREREGISTRO.md`.** Vira pré-registro (datado, antes de qualquer código ou dado) quando o autor mandar o "pode registrar"; a previsão entra depois, commitada sozinha, como na rodada 2.

## Pergunta e lógica
A rodada 2 mostrou, em m = 3 e nas tarefas 26 a 30, que o `fedavg_iid` (10 clientes, 100 pontos cada) perde pouco plasticidade (A: −0,044), enquanto a média de 10 cópias com os mesmos 1000 pontos (`media_sem_particao`: −0,326) e o aprendiz único (`sem_media_k1`: −0,366) perdem muito. Leitura exploratória: o que protege é a média entre clientes com **dados diferentes**. A rodada 3 testa isso com um desenho fatorial que separa dois fatores:
1. **quantos clientes** (K), com os dados totais e os passos por cliente fixos;
2. **se os clientes têm dados diferentes** (partição IID de N = 1000 pontos) **ou os mesmos dados** (K cópias dos 1000 pontos, só a ordem dos lotes difere).

## Condições (m\* = 3, T = 30, tarefas A e B)
| família | K = 2 | K = 5 | K = 10 | K = 20 | pontos por cliente |
|---|---|---|---|---|---|
| `particao_iid` (dados diferentes) | sim | sim | sim (= `fedavg_iid`) | sim | 500, 200, 100, 50 |
| `copias` (mesmos dados) | sim | sim | sim (= `media_sem_particao`) | sim | 1000 |
| `sem_media_k1` (K = 1) | referência comum às duas famílias | | | | 1000 |

- Dados totais fixos (N = 1000), passos por cliente fixos (S_local = 15, batch 20, Adam recriado a cada rodada), rodadas por tarefa R calibrado (abaixo). K = 10 reproduz condições já existentes; elas são reexecutadas nas sementes novas.
- Partição IID refeita por semente, como na rodada 1 (`particionar("iid")`), generalizada para K clientes; para n por cliente não múltiplo de 20 (K = 20: 50 pontos) usa-se o caminho `treinar_passos_geral` já validado na rodada 2.

## Calibração do nível de aprendizado (semente 1000)
- Alvo comum: a média da `loss_nova` (entropia cruzada de treino) do `fedavg_iid` K = 10, T = 30, semente 1000, nas 16 tarefas de referência, por tipo de tarefa. (Decisão do autor: alvo = `fedavg_iid` K = 10.)
- Cada condição calibra o número de rodadas R por tarefa, numa grade fina de passo 1, para a perda da rede nova ficar a até 10% do alvo (como na 2b). Grade (decidida): R inteiro de 50 a 200, passo 1. Fora de 10%: `ok = false`, a execução segue e a condição entra como "não avaliável", sem forçar.
- Como na 2b, a rede nova de cada condição é treinada pelo mesmo código da execução real, lendo a perda a cada rodada (precisa de um gancho por rodada no treino de FedAvg, como o `ao_fim_de_rodada` do `treinar_ht`).
- Aviso: R diferente entre condições significa passos totais diferentes por tarefa; isso é uma ameaça (ver abaixo), declarada.

## O que se mede (por execução, em todas as condições com K ≥ 2)
- **Por rodada:** cosseno médio entre as atualizações dos clientes (já gravado como `cos_medio`); **fator de encolhimento** = ‖média das atualizações‖ / média das ‖atualizações‖ (1 = cópias idênticas, perto de 0 = atualizações que se cancelam); norma dos parâmetros ao fim da rodada.
- **Por tarefa:** os diagnósticos de sempre (`norma`, `rank_ef`, `srank_feat`, dormentes, Hessiana), `acc_nova`, `loss_nova`, `acc_cont`, `loss_cont`.
- **Métrica principal:** a mesma da rodada 2: gap pareado médio nas tarefas 26 a 30, média entre sementes, IC 95% por bootstrap percentil (10 000, `default_rng(0)`).

## Hipóteses e critérios (nível de aprendizado igualado; δ = 0,05, decisão do autor; tarefas A e B)
Condição geral: uma comparação só é avaliável se a perda da rede nova das duas condições está a até 10% do alvo (calibração `ok` e checagem nas sementes de avaliação, como em H_tempo). Caso contrário, "não avaliável".
- **P_rep (pré-condição de replicação):** na semente de avaliação, o gap médio da `particao_iid` K = 10 fica acima do do `sem_media_k1` com IC pareado inteiro acima de +δ, em A e em B. Sem isso, "não avaliável (a vantagem do FedAvg não se reproduziu)".
- **H_dados (o efeito é de dados diferentes):** para cada K ≥ 5, d_K = gap(`particao_iid` K) − gap(`copias` K) com IC pareado por semente inteiro acima de +δ. Atendida se vale para K = 5, 10 e 20 (K = 2 só reportado). "Parcial" se vale em K = 10 e em mais um valor de K.
- **H_cópias (copiar não protege):** para cada K, o IC pareado de gap(`copias` K) − gap(`sem_media_k1`) fica inteiro dentro de [−δ, +δ] (equivalência de dois lados). Atendida se vale em K = 5, 10 e 20.
- **H_K (dose de clientes):** na `particao_iid`, gap(K = 20) > gap(10) > gap(5) em média (K = 2 só reportado), e o IC pareado de gap(20) − gap(5) inteiro acima de 0. (Se as três médias ficarem a menos de 0,01 de 0, "inconclusivo (sem faixa dinâmica)", como em H_het; esperado em B.)
- **Mecanismo (exploratório, sem critério):** cosseno e fator de encolhimento por K e por família; o cosseno deve ser menor e o encolhimento maior com dados diferentes se o mecanismo for "as atualizações discordam e a média as amortece". `rank_ef` e `srank_feat` sempre juntos.
- Veredito geral por hipótese: "atendida" só se vale nas duas tarefas; "parcial" se vale em exatamente uma; "não atendida" nos demais casos; "não avaliável" por falta de dado, calibração ou nível.

## Sementes, custo, ordem
- Sementes de avaliação novas, **16 a 23 (8 sementes)**; semente 1000 só para calibração.
- 9 condições × 2 tarefas × 8 sementes = 144 execuções (as K = 10 existem nas rodadas anteriores, mas reexecutam nas sementes novas), mais o braço de épocas locais fixas (K ∈ {2, 5, 10, 20}, `particao_iid`, S_local proporcional a n_k: 4 condições × 2 tarefas × 8 sementes = 64 execuções). Custo estimado, não medido: o custo de uma execução de FL cresce com K (≈ 80, 200, 400 e 800 s em K = 2, 5, 10 e 20, mais ~130 s do `sem_media_k1`), dobrando com R até 2×; da ordem de 25 CPU-h no total (com o braço extra), ou de 1 a 2 h com 20 processos.
- Ordem: pré-registro commitado → implementação, testes, auditoria pré-execução, sem rodar → teste curto e completo no cluster → auditoria dos resultados antes de interpretar.

## Ameaças à validade (a declarar)
- **Épocas locais confundidas com K:** com 15 passos de batch 20 por rodada, cada cliente vê 300 amostras por rodada: 0,6 época em K = 2 (500 pontos), 6 épocas em K = 20 (50 pontos). A `particao_iid` varia as épocas locais e o tamanho do cliente junto com K; as `copias` não (1000 pontos em todos os K). Se H_K valer, não se sabe se é "mais clientes" ou "mais épocas locais sobre pontos menores". **Braço incluído (secundário, reportado à parte):** S_local proporcional a n_k (épocas locais fixas), `particao_iid`, K ∈ {2, 5, 10, 20}.
- **R diferente entre condições** (calibração por perda): mais rodadas por tarefa em algumas condições; o `central_HT_cal2` da 2b teve o mesmo problema.
- **Calibração com uma só semente** e 16 redes por condição; a perda muda de 2% a 5% por rodada, então o R escolhido pode se ajustar ao ruído.
- Igualar a perda da rede nova não iguala a rede contínua.
- A tarefa B tem faixa dinâmica pequena (H_K tende a "inconclusivo").
- O `media_sem_particao` precisa seguir a convenção correta de janela (T = 30, tarefas 26 a 30); nada da rodada 1 com T = 12 entra.

## Previsão do autor (rodada 3)
(a preencher pelo autor, antes de qualquer código da rodada 3)

## Decisões do autor, fechadas em 2026-10-26
1. Alvo de calibração: `fedavg_iid` K = 10.
2. Calibrar R (rodadas), grade de 50 a 200, passo 1.
3. Braço de épocas locais fixas (S_local proporcional a n_k): incluído, como secundário.
4. Oito sementes de avaliação (16 a 23).
5. δ = 0,05 em H_dados e H_cópias (e nos demais critérios desta rodada).
6. K = 2 só reportado, fora dos critérios (H_dados, H_cópias e H_K usam K = 5, 10 e 20; H_K compara gap(20) com gap(5)).
7. A pré-condição de replicação P_rep é regra, não só reporte.
8. A previsão do autor chega depois e é commitada sozinha antes de qualquer código.
