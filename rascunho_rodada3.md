# RASCUNHO · Rodada 3 · Por que a média entre clientes com dados diferentes protege, e a média de cópias não?

**Status: rascunho para revisão do autor. Nada implementado, nada commitado em `PREREGISTRO.md`.** Só vira pré-registro depois do "de acordo" do autor, com data, e antes de qualquer código ou dado da rodada 3.

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
- Alvo comum: a média da `loss_nova` (entropia cruzada de treino) do `fedavg_iid` K = 10, T = 30, semente 1000, nas 16 tarefas de referência, por tipo de tarefa. (**Decisão a confirmar:** alvo = `fedavg_iid` K = 10, ou o `media_sem_particao`, que aprende melhor.)
- Cada condição calibra o número de rodadas R por tarefa, numa grade fina de passo 1, para a perda da rede nova ficar a até 10% do alvo (como na 2b). Grade proposta: R de 50 a 200. Fora de 10%: `ok = false`, a execução segue e a condição entra como "não avaliável", sem forçar.
- Como na 2b, a rede nova de cada condição é treinada pelo mesmo código da execução real, lendo a perda a cada rodada (precisa de um gancho por rodada no treino de FedAvg, como o `ao_fim_de_rodada` do `treinar_ht`).
- Aviso: R diferente entre condições significa passos totais diferentes por tarefa; isso é uma ameaça (ver abaixo), declarada.

## O que se mede (por execução, em todas as condições com K ≥ 2)
- **Por rodada:** cosseno médio entre as atualizações dos clientes (já gravado como `cos_medio`); **fator de encolhimento** = ‖média das atualizações‖ / média das ‖atualizações‖ (1 = cópias idênticas, perto de 0 = atualizações que se cancelam); norma dos parâmetros ao fim da rodada.
- **Por tarefa:** os diagnósticos de sempre (`norma`, `rank_ef`, `srank_feat`, dormentes, Hessiana), `acc_nova`, `loss_nova`, `acc_cont`, `loss_cont`.
- **Métrica principal:** a mesma da rodada 2: gap pareado médio nas tarefas 26 a 30, média entre sementes, IC 95% por bootstrap percentil (10 000, `default_rng(0)`).

## Hipóteses e critérios (nível de aprendizado igualado; δ = 0,02; tarefas A e B)
Condição geral: uma comparação só é avaliável se a perda da rede nova das duas condições está a até 10% do alvo (calibração `ok` e checagem nas sementes de avaliação, como em H_tempo). Caso contrário, "não avaliável".
- **P_rep (pré-condição de replicação):** na semente de avaliação, o gap médio da `particao_iid` K = 10 fica acima do do `sem_media_k1` com IC pareado inteiro acima de +δ, em A e em B. Sem isso, "não avaliável (a vantagem do FedAvg não se reproduziu)".
- **H_dados (o efeito é de dados diferentes):** para cada K ≥ 5, d_K = gap(`particao_iid` K) − gap(`copias` K) com IC pareado por semente inteiro acima de +δ. Atendida se vale para K = 5, 10 e 20 (e K = 2 é reportado). "Parcial" se vale em K = 10 e em mais um valor de K.
- **H_cópias (copiar não protege):** para cada K, o IC pareado de gap(`copias` K) − gap(`sem_media_k1`) fica inteiro dentro de [−δ, +δ] (equivalência de dois lados). Atendida se vale em K = 5, 10 e 20.
- **H_K (dose de clientes):** na `particao_iid`, gap(K = 20) > gap(10) > gap(5) > gap(2) em média, e o IC pareado de gap(20) − gap(2) inteiro acima de 0. (Se as três médias ficarem a menos de 0,01 de 0, "inconclusivo (sem faixa dinâmica)", como em H_het; esperado em B.)
- **Mecanismo (exploratório, sem critério):** cosseno e fator de encolhimento por K e por família; o cosseno deve ser menor e o encolhimento maior com dados diferentes se o mecanismo for "as atualizações discordam e a média as amortece". `rank_ef` e `srank_feat` sempre juntos.
- Veredito geral por hipótese: "atendida" só se vale nas duas tarefas; "parcial" se vale em exatamente uma; "não atendida" nos demais casos; "não avaliável" por falta de dado, calibração ou nível.

## Sementes, custo, ordem
- Sementes de avaliação novas, 16 a 20 (ou 16 a 23, 8 sementes, para IC mais estreito: **decisão**); semente 1000 só para calibração.
- 9 condições × 2 tarefas × 5 sementes = 90 execuções (a K = 10 já existe, mas reexecuta nas sementes novas). Custo estimado, não medido: o custo de uma execução de FL cresce com K (≈ 80, 200, 400 e 800 s em K = 2, 5, 10 e 20, mais ~130 s do `sem_media_k1`), dobrando com R até 2×; da ordem de 10 CPU-h, ou de 30 a 60 min com 20 processos.
- Ordem: pré-registro commitado → implementação, testes, auditoria pré-execução, sem rodar → teste curto e completo no cluster → auditoria dos resultados antes de interpretar.

## Ameaças à validade (a declarar)
- **Épocas locais confundidas com K:** com 15 passos de batch 20 por rodada, cada cliente vê 300 amostras por rodada: 0,6 época em K = 2 (500 pontos), 6 épocas em K = 20 (50 pontos). A `particao_iid` varia as épocas locais e o tamanho do cliente junto com K; as `copias` não (1000 pontos em todos os K). Se H_K valer, não se sabe se é "mais clientes" ou "mais épocas locais sobre pontos menores". **Braço opcional:** S_local proporcional a n_k (épocas locais fixas), reportado à parte.
- **R diferente entre condições** (calibração por perda): mais rodadas por tarefa em algumas condições; o `central_HT_cal2` da 2b teve o mesmo problema.
- **Calibração com uma só semente** e 16 redes por condição; a perda muda de 2% a 5% por rodada, então o R escolhido pode se ajustar ao ruído.
- Igualar a perda da rede nova não iguala a rede contínua.
- A tarefa B tem faixa dinâmica pequena (H_K tende a "inconclusivo").
- O `media_sem_particao` precisa seguir a convenção correta de janela (T = 30, tarefas 26 a 30); nada da rodada 1 com T = 12 entra.

## Previsão do autor (rodada 3)
(a preencher pelo autor, antes de qualquer código da rodada 3)

## Decisões a confirmar pelo autor antes de pré-registrar
1. Alvo de calibração: `fedavg_iid` K = 10 ou `media_sem_particao`.
2. Calibrar R (rodadas) com grade [50, 200] passo 1, ou outra faixa; ou calibrar o S_local por rodada.
3. Incluir o braço de épocas locais fixas (S_local proporcional a n_k) como secundário.
4. 5 ou 8 sementes de avaliação.
5. δ = 0,02 também para H_dados e H_cópias, ou um δ maior (a tarefa A varia de −0,3 a +0,2 entre condições).
6. K = 2 entra nos critérios ou só é reportado.
7. P_rep entra como pré-condição (proposta) ou só como reporte.
