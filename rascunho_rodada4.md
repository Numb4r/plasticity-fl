# RASCUNHO · Rodada 4 (4a e 4b) · estado em 2026-10-26: IMPLEMENTADA, NÃO PRÉ-REGISTRADA, NADA RODADO

Substitui a versão anterior deste arquivo. Por decisão do autor, a implementação veio antes do pré-registro (métodos genéricos, com testes, sem dado visto). O pré-registro (hipóteses, critérios, previsão) entra **depois dos resultados da rodada 3 e antes de rodar a 4**. Este arquivo lista o que foi construído e o que o pré-registro precisa decidir. A auditoria pré-execução (2026-10-26) não achou P0.

## O que está implementado
**Cenário comum.** m\* = 3, T = 30, tarefas A e B, K = 10 clientes com partição IID (100 pontos cada), 15 passos locais, batch 20, Adam recriado a cada rodada. Todas as condições calibram o número de rodadas R (50 a 200, passo 1) para a perda da rede nova ficar a até 10% da do `fedavg_iid` K = 10 (semente 1000), ou ficam `ok = false` ("não avaliável"), sem forçar. Hiperparâmetros: para cada candidato calibra-se R, roda-se a semente 1000 e escolhe-se a maior acurácia contínua nas 5 últimas tarefas entre os candidatos com calibração ok. Sementes 24 a 31.

**4a, técnicas distribuídas** (`fedavg_iid`, `fedprox` mu em {0,001; 0,01; 0,1; 1}, `scaffold`, `fedrep`, `gossip_anel`, `gossip_grau4`, `diloco_h15/h50/h150` exploratórios, referências `sem_media_k1` e `central_aprendizado`). Gossip completo = FedAvg (bit a bit), por isso não é condição.

**4b, remédios no FedAvg** (`fed_l2`, `fed_l2init`, `fed_redo_local`, `fed_sp` por tarefa, `fed_redo_fed` por média, as combinações `fed_l2_sp`, `fed_l2init_sp`, `fed_redo_ambos`, os exploratórios `fed_sp_rodada` e `fed_redo_fed_voto`, e a referência `central_l2` com {1e-6 … 1e-3}). SP no cliente não existe como condição (ver abaixo).

**Medidas gravadas por tarefa:** cosseno entre atualizações dos clientes, fator de encolhimento da média, norma, rank efetivo e `srank_feat`, dormentes, Hessiana; no gossip, acurácia do modelo médio e consenso; nos remédios de ReDo, **quantos neurônios foram reiniciados** (servidor e cliente, contínua e nova). Avaliação: no FedRep e no gossip, a média dos clientes, cada um com o seu modelo, nos 1000 pontos da tarefa; diagnósticos no modelo representativo.

## Decisões que o pré-registro precisa fechar (da auditoria)
1. **SP.** "Cliente e servidor" não é igual a "servidor"; "SP no cliente, aplicado no envio" ≈ SP do servidor com σ/√K (não testado). Decidir se σ/√K vira condição e se o σ do SP por rodada escala com R (com σ fixo = 0,01 e λ = 0,999, o desvio estacionário passa de 0,2 por peso).
2. **DiLoCo.** Com R ≥ 50, H = 15, 50 e 150 ficam muito acima do alvo de passos (o passo externo efetivo do H = 15 é ≈ 7×); esperar `ok = false`. Decidir: baixar o piso de R, tirar o H = 150 ou deixar com R fixo. É ~60% do custo confirmatório da 4a.
3. **FedRep e gossip.** "Corpo global agregado + cabeça local" (o corpo local pós-treino é descartado pelo algoritmo); avaliar nos 1000 pontos ou nos 100 do próprio cliente (90% da medida é generalização); o corpo treina 10 passos, não 15 (confundidor de passos; controle opcional: FedAvg com 5+10 passos).
4. **ReDo.** Federado por média usa a média dos scores normalizados por cliente (promediar E|h| bruto seria o ReDo nos dados juntos); τ em {0,025; 0,1} pode não reiniciar nada: "remédio não acionado" é um resultado próprio (a contagem está gravada). `fed_redo_ambos` pode colapsar em `fed_redo_fed` (neurônio morto está morto em todos os clientes).
5. **Combinações.** Hoje só 3 das 6 combinações cliente × servidor rodam; os hiperparâmetros vêm dos componentes, sem busca conjunta.
6. **Base desprotegida.** O código só tem a base IID; `H_restaura` (restaurar onde a média sozinha não protege) precisa de uma base com cópias ou do `sem_media_k1`; ou a 4b vira "ajuda além do FedAvg".
7. **Checagens de manipulação** por alavanca (por exemplo, SCAFFOLD precisa aumentar o cosseno com IC acima de 0) como condição para avaliar cada hipótese.
8. **Grades e convenções:** L2 federado {1e-5 … 1e-3} contra central {1e-6 … 1e-3}; L2Init com λ(θ − θ0) (metade da convenção 2λ de Kumar et al.); grau 4 circulante (k±1, k±2), não aleatório.
9. **`fedavg_iid` pela calibração** como piso de ruído do procedimento (R e erro), e o `fedavg_iid` duplicado em 4a e 4b como checagem de determinismo.
10. **Critérios, δ, janela, bootstrap e a pré-condição de replicação**, como na rodada 3; a previsão do autor.

## Ameaças a declarar
R diferente entre condições = passos totais diferentes; igualar a rede nova não iguala a contínua; ReDo, SP e L2Init mudam também a rede nova; diluição e escala do ReDo local depois da média; modelo avaliado e diagnósticos no representativo em FedRep e gossip; K = 10 fixo e só IID; sonda v2 não validada; hiperparâmetros escolhidos numa só semente; muitas comparações sem correção.

## Previsão do autor (rodada 4)
(a preencher pelo autor, antes de qualquer execução da rodada 4)
