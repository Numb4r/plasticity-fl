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

## Adendo 1 · Esquecimento como medida secundária (2026-10-20, antes de qualquer resultado)
Acrescentado antes de rodar qualquer experimento. Não altera as hipóteses H_reg, H_part e H_real nem as tarefas A e B.

**Tarefa B_disjunta.** Cada tarefa t tem o seu conjunto X_t de N=1000 pontos, X_t ~ N(mu_t, I), com mu_t sorteado com norma 3 (direção uniforme na esfera). Os rótulos vêm de uma MLP professora nova por tarefa (argmax das 10 saídas, aplicado a X_t). Todas as 7 condições rodam em B_disjunta, com T=30 (T=12 em `media_sem_particao`) e 5 sementes. A partição entre clientes é a mesma de A e B (índices fixos por semente, aplicados a cada X_t). Calibração de `central_aprendizado` e escolha de λ (L2, SP) são feitas separadamente para B_disjunta, na semente 0, com as mesmas regras. Rede nova, sondas, dormentes e rank efetivo usam o X_t da tarefa corrente.

**Esquecimento (só em B_disjunta).** Nas tarefas A e B o mesmo X recebe rótulos conflitantes a cada tarefa, então esquecimento não é medido nelas. Em B_disjunta, depois de cada tarefa i, a rede é avaliada em (X_j, y_j) de todas as tarefas j ≤ i. A matriz a[i][j] (acurácia na tarefa j depois de treinar até a tarefa i; vazia para j > i) vai para `results/<condicao>_B_disjunta_<semente>_matriz.csv`. A diagonal a[i][i] é a acurácia de treino no fim da tarefa i. Medidas, por tarefa i:
- acurácia média sobre as tarefas vistas: média de a[i][j], j ≤ i;
- esquecimento médio (Chaudhry et al., 2018): média, sobre j < i, de max_{j≤l<i} a[l][j] − a[i][j] (o máximo corre de quando a tarefa j foi aprendida até a tarefa anterior a i);
- BWT (Lopez-Paz e Ranzato, 2017): média, sobre j < i, de a[i][j] − a[j][j]. O BWT final é o do último i.
Valor de decisão: o do último i (T−1; 11 em `media_sem_particao`), agregado entre sementes com média e IC 95% por bootstrap, como na métrica principal.

**Limite de esquecimento (fixado agora).** Uma condição só conta como ganho de plasticidade se o seu esquecimento médio final em B_disjunta não passar o do contínuo sem intervenção em mais de 5 pontos percentuais, isto é, média_esq(condição) ≤ média_esq(central_aprendizado) + 0,05. O "contínuo sem intervenção" é `central_aprendizado` (centralizado sem L2 nem SP, calibrado no mesmo nível de aprendizado).
- "Melhora plasticidade" em B_disjunta = limite inferior do IC 95% da métrica principal da condição > limite superior do IC 95% da de `central_aprendizado`.
- A condição é marcada "ganho com violação de esquecimento" se melhora plasticidade e passa o limite; "ganho válido" se melhora e respeita o limite.
- Efeito sobre H_real: além do critério original, `fedavg_iid` precisa respeitar o limite de esquecimento em B_disjunta. Se H_real valer pelo critério original mas `fedavg_iid` violar o limite, a tabela reporta "H_real: ganho de plasticidade com esquecimento acima do limite" e H_real não conta como atendida. H_reg e H_part não mudam.
- A tabela de decisão e a figura `fig_tradeoff.png` (plasticidade no eixo x, esquecimento médio final no eixo y, IC 95% nos dois, faixa do limite desenhada) são só de B_disjunta.

## Desvios

### 2026-10-20 · Esclarecimentos antes de qualquer resultado (auditoria pré-execução)
Nenhum resultado existia. Não mudam hipóteses nem critérios; só fecham lacunas apontadas pelo auditor.
1. **Calibração**: as 10 redes novas centralizadas de cada valor testado usam os rótulos das tarefas de índice 2, 5, …, 29 (as mesmas em que o `fedavg_iid` tem rede nova), em vez de 5 redes nas tarefas 0 a 4. Varredura: 15 a 1200 passos (múltiplos de 15). Empate vai para menos passos.
2. **Seleção de λ**: empate vai para o primeiro λ da grade. Cada candidato roda a sequência completa de 30 tarefas na semente 0.
3. **Pareamento**: dentro de uma semente, além de X, rótulos e professores, também são iguais entre condições a inicialização da rede contínua, a das redes novas, as sondas e o ruído do SP. Não muda o IC marginal de cada condição.
4. **`sem_media_k1`** usa 150 passos por rodada (1000 pontos, batch 20, 3 épocas) com Adam recriado a cada rodada, isto é, 3750 passos por tarefa contra 375 do cliente do `fedavg_iid`. É o "mesmo protocolo local" pedido, e a interpretação deve lembrar dessa diferença de passos.
5. **Determinismo**: o pipeline fixa CPU (`PLAST_DISPOSITIVO=cpu`) para a semente 0 sair idêntica na seleção e na avaliação.
6. **Retomada**: o pipeline só pula um CSV se ele tiver T linhas e um arquivo `.params.json` igual aos parâmetros congelados atuais.
7. O Adendo 1 (esquecimento e B_disjunta) foi acrescentado no mesmo dia, antes de qualquer resultado.

### 2026-10-20 · Regras de análise fixadas antes de qualquer resultado (auditoria nº 2)
Leituras do texto de "Hipóteses e critérios" adotadas no código de análise (`plots/analise.py`):
1. **H_reg**: o mesmo candidato (L2 ou SP) precisa ficar dentro do IC 95% do `fedavg_iid` em A e em B. Candidatos diferentes por tarefa não contam. Intervalos fechados (limites inclusivos).
2. **H_part (ii)** usa a janela das tarefas 8 a 12 (índices 7 a 11) em B; a **pré-condição** usa a janela principal (índices 25 a 29). As duas janelas coexistem na mesma regra.
3. "Sobrepõe" = inf₁ ≤ sup₂ e inf₂ ≤ sup₁. "Sem sobreposição, fedavg acima" = inf(fedavg) > sup(outra). H_real usa desigualdade estrita.
4. A janela "últimas 5" usa o T da própria condição. Só entram tarefas com gap definido (as que têm rede nova de referência: t mod 3 = 2, as 5 últimas da condição e os índices 7 a 11; ver "Decisão da Errata do item (c): opção (ii)"). Sementes sem nenhum gap na janela são descartadas.
5. Cada IC é um bootstrap percentil próprio (10 000 reamostragens das sementes, `default_rng(0)`); como o gerador e n são iguais, as reamostragens usam os mesmos índices de semente em todas as condições. Isso não muda o IC marginal de cada condição.
6. Com menos de 2 sementes o IC é indefinido e o critério sai "não avaliável"; isso não conta como atendido nem como não atendido. A linha final separa "sem vantagem a explicar" (pré-condição de H_part falhou) de "dados insuficientes".
7. Rótulo da figura de distância: ‖Δθ‖ na tarefa, medida depois do passo SP.

### 2026-10-20 · Redação do Adendo 1, antes de qualquer resultado
No Adendo 1, "passa o limite" significa **ultrapassa** o limite (média_esq > média_esq(central_aprendizado) + 0,05). Portanto: melhora plasticidade e ultrapassa o limite = "ganho com violação de esquecimento"; melhora plasticidade e respeita o limite (média_esq ≤ limite) = "ganho válido". O código de análise já implementa esse sentido. A comparação do limite usa as médias entre sementes, sem tolerância numérica; os ICs do esquecimento são só informativos.

### 2026-10-20 · Leituras registradas após a auditoria nº 3 (antes de qualquer resultado)
1. "A tabela de decisão e a figura são só de B_disjunta" (Adendo 1) vale para as colunas de plasticidade-e-esquecimento, a marcação de ganho e `fig_tradeoff.png`. As hipóteses H_reg, H_part e H_real continuam avaliadas em A e B, com a cláusula extra de esquecimento em H_real.
2. `media_sem_particao` tem esquecimento medido na sua última tarefa (índice 11, 11 tarefas anteriores), comparado com o de `central_aprendizado` na tarefa 29. Os horizontes diferem; a marcação dela em B_disjunta deve ser lida com essa ressalva. Nenhuma hipótese depende dessa marcação.
3. Limitação conhecida do código de análise: sem dados de B_disjunta, H_real sai "não avaliável (falta B_disjunta)" mesmo que o critério original já falhe. Só afeta execuções sem B_disjunta (como `--quick`); a leitura literal do Adendo ("além do critério original") tornaria H_real "não atendida" nesse caso. A interpretação final usa a execução completa.
4. Risco de desenho a checar antes de interpretar esquecimento: o professor tem inicialização padrão e X_t é deslocado; se as classes ficarem muito desbalanceadas, a acurácia em tarefas antigas pode ficar alta só pela classe majoritária. O histograma de y_t é reportado junto com os resultados.

## Desvios v3 · Eixo de orçamento e protocolo corrigido (2026-10-20)
Registrado ANTES de qualquer mudança de código e antes de qualquer execução confirmatória.

### Motivo
No modo rápido (semente 0), nenhuma condição de orçamento baixo (375 a 420 passos por tarefa) perdeu plasticidade. O único colapso (`sem_media_k1`, tarefa A: contínuo de 1,00 para 0,11, com acaso em 0,10, enquanto a rede nova ficou em 1,00) veio com 10 vezes mais passos por tarefa (3750 contra 375). Partição e orçamento estavam confundidos, porque o treino local era definido em épocas e cada cliente do `fedavg_iid` tem só 100 pontos. O desenho original não testa se o FedAvg protege, porque o FedAvg nunca chega ao regime em que a perda aparece.

### Auditoria do Passo 1 (fatos que fundamentam as mudanças)
- O gap já é pareado em todo o código (tarefa a tarefa, só onde há rede nova). Nenhuma tabela, figura ou decisão usa a versão não pareada.
- A semente 0 foi usada ao mesmo tempo na calibração (alvo, dados e rótulos), na seleção de λ e na avaliação. O pré-registro só admitia o reuso para λ.
- Passos por tarefa por cliente (T=30): `fedavg_iid` 375; `sem_media_k1` 3750; `media_sem_particao` 3750 por cliente; `central_passos` 375; `central_aprendizado` 420 (A), 405 (B_disjunta), 435 (B).
- Classe majoritária > 0,3 em 83% a 97% das tarefas de B e de B_disjunta, em todas as sementes (0 a 5 e 1000); há tarefas com 1 classe só (B, semente 4, t=0) ou quase (B_disjunta, semente 4, t=0: 0,999). Dados em `notes.md`.

### Semente 0 e sementes
- A semente 0 do modo rápido e todos os CSVs, JSONs e matrizes do desenho antigo (incluindo os 13 CSVs do pipeline completo interrompido) passam a ser **exploratórios** e saem da análise confirmatória. Serão movidos (não apagados) para `results_v1_exploratorio/`.
- **Semente 1000**: só para (i) alvo e calibração de `central_aprendizado`, (ii) seleção de λ de L2 e SP, (iii) definição do regime de teste. Não entra na avaliação. Seus resultados ficam em `results/calibracao/`, fora do alcance da análise.
- **Sementes confirmatórias: 1 a 5.**

### Mudanças de protocolo
a) **Treino local em passos.** Cada cliente faz S_local = 15 passos por rodada (batch 20), seja qual for o tamanho dos seus dados, em `fedavg_iid`, `sem_media_k1` e `media_sem_particao`. O Adam continua recriado a cada rodada. Em m=1, `fedavg_iid` é idêntico ao já rodado (5 lotes × 3 épocas = 15 passos). Isso **substitui** o Desvio 4 (150 passos por rodada em `sem_media_k1`) e a leitura de "mesmo protocolo local" do item 4 de "Escolhas operacionais".
   - Consequência declarada: `sem_media_k1` passa a executar o mesmo algoritmo que `central_passos` (1 cliente, 15 passos por rodada, Adam recriado a cada 15 passos, 25·m rodadas = 375·m passos; a média de um só modelo é exata). A única diferença é a amostragem dos lotes (permutação nova por rodada, com só 300 dos 1000 pontos usados, contra épocas contínuas). Mantida como controle de sanidade: deve coincidir com `central_passos` dentro do ruído. Se não coincidir, é resultado a reportar.
   - `media_sem_particao` passa a diferir do `fedavg_iid` só pela fonte dos minilotes (1000 pontos comuns contra 100 próprios), não mais pelo número de passos. Isso muda a leitura de H_part (ver abaixo).
b) **Eixo de orçamento m ∈ {1, 3, 10}**: rodadas R = 25·m em todas as condições FL. `central_passos` faz 375·m passos. `central_aprendizado` é calibrado em cada nível para a rede nova ficar a ±0,03 da rede nova do `fedavg_iid` naquele nível (calibração na semente 1000; varredura de passos em múltiplos de 15 até 15 000, por treino incremental único com leitura a cada 15 passos). Se a calibração não chegar a ±0,03, registra-se a falha e o pipeline segue com o valor mais próximo, marcado `ok=false`. Se a rede nova do `fedavg_iid` não passar de 0,90 na tarefa A nem em m=10, isso é registrado como resultado.
c) **Rede nova** treinada em t mod 3 = 2 **e** em todas as 5 últimas tarefas da condição (0-based 25 a 29 com T=30; 7 a 11 com T=12). O gap é sempre pareado, tarefa a tarefa. Isso **substitui** o item 1 de "Escolhas operacionais" (2 referências por janela): a janela principal e a de H_part têm agora 5 referências. A calibração usa as mesmas tarefas do alvo.
d) **L2 e SP só no regime de teste** (definido abaixo), com λ e calibração escolhidos na semente 1000, mesmas grades. Critério: maior acurácia contínua média nas tarefas 26 a 30 (0-based 25 a 29), entre os λ que respeitam o limite de esquecimento do Adendo 1. O limite só é medido em B_disjunta (nas tarefas A e B não há esquecimento medido); nesta tarefa ele exige rodar também `central_aprendizado` na semente 1000 como referência. Se nenhum λ respeitar o limite, escolhe-se o de menor esquecimento médio final e a escolha é marcada "violação".
e) **Balanceamento das tarefas B e B_disjunta.** O gatilho (classe majoritária > 0,3 em mais de 20% das tarefas) disparou em todas as 14 combinações. O professor passa a rotular por argmax dos logits **centralizados**: de cada logit subtrai-se a média daquela classe sobre os pontos da tarefa (`logits − logits.mean(0)`). Muda a definição de professor do pré-registro original. Não se aplica à tarefa A. Cada CSV passa a ter a coluna `maj_frac` (fração da classe majoritária da tarefa), em todas as tarefas A, B e B_disjunta; as frações depois do balanceamento serão reportadas em `notes.md`.
f) **Mantidos**: métricas, sonda de Lyle, matriz e métricas de esquecimento, B_disjunta, testes, controles e sementes.
g) **Nomes de arquivo**: `results/<condicao>_m<m>_<tarefa>_<semente>.csv` (e `_matriz`, `_rodadas`, `.params.json` análogos), com coluna `nivel` (=m) no CSV. Os parâmetros gravados incluem m, R, S_local e uma versão dos dados (`versao_dados`), para a retomada do pipeline não aceitar CSVs do desenho antigo.

### Regime de teste e hipóteses
- **Regime de teste (m\*)**: o menor nível m ∈ {1, 3, 10} em que `central_aprendizado` tem gap ≤ −0,10 na tarefa A. O gap usado para essa decisão é o da métrica principal (média dos gaps pareados das 5 últimas tarefas) na semente 1000. Se nenhum nível chegar lá: registrar "regime sem perda; H_real, H_reg, H_part e H_orc não testáveis neste toy" e reportar só a figura de dose-resposta.
- **H_real, H_reg, H_part e o critério de esquecimento passam a ser avaliados no nível m\***, com as mesmas regras de antes (agora com 5 referências por janela). Os demais níveis são reportados como dose-resposta (`fig_dose.png`), sem critério de decisão.
- **Nova hipótese H_orc** (a vantagem do FedAvg sobre K=1 some quando os dois são comparados no mesmo nível de aprendizado da rede nova). **Critério**: no nível m\*, a média da métrica principal do `fedavg_iid` cai dentro do IC 95% da métrica principal de `central_aprendizado`, na tarefa A e na tarefa B (intervalos fechados; mesmo bootstrap). A comparação é com `central_aprendizado` porque ele é o controle calibrado para o mesmo nível de aprendizado da rede nova.
- H_part continua com o texto original, no nível m\*. Dado (a), o critério (i) compara `media_sem_particao` (agora com o mesmo orçamento por cliente) com `central_aprendizado`; a leitura de H_part deve citar essa mudança.

### Ordem de execução
1. Teste curto (semente 1000, só tarefa A, m ∈ {1, 10}; `fedavg_iid`, `sem_media_k1`, `central_passos`): se em m=10 o `central_passos` não perder plasticidade (gap ≤ −0,10), parar e mostrar o log.
2. Semente 1000: `fedavg_iid` por nível e tarefa, calibração de `central_aprendizado`, definição de m\*, seleção de λ (L2, SP) em m\*.
3. Completo: sementes 1 a 5, tarefas A, B e B_disjunta, todos os níveis (L2 e SP só em m\*).
4. Revisão do `auditor-metricas` antes de qualquer interpretação.

### Previsão do autor
Previsão do autor (2026-10-20, antes do teste curto). Confiança baixa.
Nível m*: espero os primeiros indícios de perda por volta de m=5, nível que não é testado. Entre os níveis testados, m* = 10. Em m=3, sem perda clara (gap do central_aprendizado acima de −0,10).
Tamanho: em m=10, o gap do sem_media_k1 na tarefa A fica perto de −0,4.
FedAvg: em m*, o fedavg_iid perde plasticidade, mas menos que o central_aprendizado. Previsão para H_orc: refuta.
H_real, H_reg, H_part e esquecimento na B_disjunta: sem previsão.

### 2026-10-20 · Errata dos Desvios v3, item (c), antes de qualquer execução
O item (c) diz que "a janela principal e a de H_part têm agora 5 referências". Isso é verdade para a janela principal de qualquer condição e para a janela de H_part (índices 7 a 11) apenas em `media_sem_particao` (T=12). Nas condições com T=30, a janela 7 a 11 só tem rede nova nos índices 8 e 11 (t mod 3 = 2), isto é, 2 referências. O código atual implementa o texto literal (2 referências nessas condições). **Decisão pendente do autor** antes do Passo 5: (i) manter 2 referências, ou (ii) acrescentar rede nova também nos índices 7 a 11 das condições de T=30 (custo: +3 redes novas por execução). Qualquer escolha será registrada aqui antes de rodar.

### 2026-10-20 · Leituras do código de análise registradas após a auditoria nº 5 (antes de qualquer execução)
1. **Sem regime (m\* nulo)**: as decisões saem "não avaliável". As figuras antigas, incluindo `fig_tradeoff.png`, são geradas no maior nível apenas como descrição, com aviso no título; o texto dos Desvios v3 ("reportar só a figura de dose-resposta") vale para qualquer interpretação.
2. **H_orc**: basta a média de `fedavg_iid` (n ≥ 1) cair no IC fechado de `central_aprendizado` (n ≥ 2); janela principal; sem pré-condição.
3. **`fig_dose.png`**: o eixo x é a acurácia média da rede nova da PRÓPRIA condição, nas mesmas tarefas e sementes da métrica principal. Para `media_sem_particao` a janela principal é 7 a 11 (T=12); para as outras, 25 a 29. A linha em −0,10 é o limiar do regime, definido só na tarefa A, e aparece nos três painéis como referência visual.
4. **`fig_k1_mecanismo.png`**: nível m=10, `sem_media_k1` contra `fedavg_iid`, dormentes (τ=0 e τ=0,025) da última camada oculta, norma e rank efetivo.
5. **Sanidade `sem_media_k1` × `central_passos`**: "coincidem" = ICs da métrica principal sobrepostos; com IC indefinido, "n/d".
6. **H_reg** sai "não avaliável" se existe um só candidato (L2 ou SP) e ele falha, porque o outro poderia atender.
7. **Exclusões lógicas a considerar na leitura**: em B, a pré-condição de H_part (inf de `fedavg_iid` > sup de `central_aprendizado`) torna H_orc impossível, e H_orc atendida implica H_part "sem vantagem a explicar". H_reg e H_real também se excluem.
8. **Análise só com sementes confirmatórias (1 a 5)**: `carregar` descarta e avisa qualquer outra semente na raiz de `results/`.

### 2026-10-20 · Decisão da Errata do item (c): opção (ii), tomada ANTES de qualquer execução do Passo 5
**Decisão do autor:** acrescentar rede nova de referência também nos índices 7 a 11 das condições de T=30, para que H_part compare as tarefas 8 a 12 com **5 referências pareadas em todas as condições**. Implementação: `tarefas_com_nova(T)` = t mod 3 = 2, as 5 últimas da condição e os índices 7 a 11. Com T=30 são 16 tarefas (2, 5, 7, 8, 9, 10, 11, 14, 17, 20, 23, 25 a 29); com T=12 as janelas coincidem (7 tarefas). A calibração usa as mesmas 16 tarefas do alvo. Custo: +3 redes novas por execução. Isto **substitui** o "(i) manter 2 referências" da Errata anterior.
- **Teste de que as redes novas não consomem o gerador do contínuo:** `test_redes_novas_extras_nao_alteram_o_continuo` roda `central_SP` (T=30) e `fedavg_iid` (T=12, B_disjunta) com todas as referências, só com as de t mod 3 = 2 e sem nenhuma, e exige igualdade bit a bit do contínuo (acurácia, norma, distância, dormentes, rank, sonda, cosseno e esquecimento). Passou.
- **Números do prompt v3 vs. a auditoria.** Os +0,13 (`fedavg_iid`) e +0,09 (`central_passos`), na tarefa A do modo rápido (semente 0, desenho antigo), foram informados pelo autor como a média do gap pareado nas tarefas 15 a 30. Os valores calculados nesta sessão, +0,148 (`fedavg_iid`) e +0,072 (`central_passos`), são a **métrica pré-registrada** (últimas 5 tarefas, com referências só em 27 e 30). Não há divergência nos dados: são janelas diferentes. A explicação do autor não foi reverificada nos CSVs, porque os arquivos exploratórios do desenho antigo foram esvaziados em 2026-10-20 (ver `notes.md`). Ambos os conjuntos são exploratórios (semente 0, desenho antigo) e não entram na análise confirmatória.

### 2026-10-20 · Registro do commit da previsão (antes do teste curto)
A "Previsão do autor" foi commitada sozinha no commit `77f0b55`. O parêntese desatualizado da regra de análise 4 foi corrigido antes, no commit `5d6f599`. O teste curto (`--fase teste_curto`) só foi lançado depois deste registro.

## Desvios v3.1 · Métricas do protocolo de Klein et al. (2026) (2026-10-20)
Registrado ANTES de mexer no código, depois do teste curto e antes do completo. O teste curto (semente 1000, tarefa A) rodou sem estas métricas; é exploratório.

### Motivo e status
Alinhar as medidas ao protocolo padrão proposto por Klein et al., "Plasticity Loss in Deep Reinforcement Learning: A Survey" (arXiv 2411.04832, Tabela 3). As métricas novas são **diagnósticas**: não entram nos critérios de H_orc, H_real, H_reg e H_part, nem na `tabela_decisao.md`.

### Pergunta exploratória
Na condição que perder plasticidade, a norma dos parâmetros e o maior autovalor da Hessiana sobem **antes** da queda da acurácia do contínuo? Isso testa a ordem da cadeia causal da Figura 1 do survey. Não tem critério de decisão nem previsão registrada.

### Métricas novas (colunas acrescentadas aos CSVs existentes; as atuais não mudam)
a) Norma L2 dos parâmetros por camada linear (l1, l2, l3), pesos e vieses separados, além da norma global que já existe.
b) Norma do gradiente por camada (pesos e vieses separados), em L1 e em L2, e a L2 global. É o gradiente da entropia cruzada da tarefa sobre todos os pontos da tarefa, numa passada só, sem passo de otimização.
c) Espectro do gradiente: matriz G com os gradientes por amostra, achatados, de 64 pontos fixos da tarefa (índices sorteados uma vez por semente e tarefa, com semente própria, os mesmos em todas as condições, níveis, momentos e redes). Registra: srank de G (δ = 0,01, Definição 4.1 do survey), cosseno médio entre pares de linhas e fração de pares com cosseno negativo. Os gradientes por amostra são sempre calculados vetorizados (`torch.func.vmap`).
d) Maior autovalor da Hessiana da perda da tarefa sobre todos os pontos, por power iteration com produtos Hessiana-vetor (autograd duplo), 30 iterações e vetor inicial com semente própria. Estimativa pelo quociente de Rayleigh (autovalor de maior módulo). Registra o valor e se convergiu: variação relativa < 1% nas últimas 5 iterações.
e) **`srank_feat`** (aprovado pelo autor): srank com δ = 0,01 das ativações da última camada oculta, ao lado do `rank_ef` atual (rank efetivo por entropia), que **não muda**. Assim o srank do gradiente e o das features usam a mesma definição.

### Momentos e onde medir
- **Início de cada tarefa**: com os rótulos ou dados já trocados e **depois do shrink-and-perturb** (aprovado pelo autor), antes do primeiro passo. **Fim de cada tarefa**: depois do treino.
- Em todas as condições e níveis m, no modelo contínuo (no FL, o modelo global) e em cada rede nova de referência, nos mesmos dois momentos, como linha de base. Colunas do contínuo com prefixos `ini_` e `fim_`; da rede nova, `nova_ini_` e `nova_fim_` (vazias nas tarefas sem rede nova).
- A semente 1000 (calibração) não recebe diagnósticos.
- Regras: tudo numa cópia (`deepcopy`), em modo de avaliação, sem tocar no otimizador e com gerador aleatório próprio (papéis novos de semente, acrescentados no fim da lista).

### Critério de custo (substitui a regra dos 10%, por decisão do autor)
O critério é o **tempo total do completo**, porque em m=1 a execução é curta e o custo relativo engana. Se o completo crescer mais de 25%, aplicam-se os cortes nesta ordem, parando assim que ficar abaixo: (a) Hessiana e espectro do gradiente num subconjunto fixo de 256 pontos da tarefa, os mesmos índices em todas as condições; (b) power iteration começando do autovetor da medição anterior da mesma rede, mantendo o critério de convergência; (c) Hessiana e espectro só nas tarefas de `tarefas_com_nova`. As normas de parâmetro e de gradiente por camada continuam em todos os pontos, com todos os dados. Os cortes aplicados serão registrados aqui, em bloco datado, antes do completo.

### Testes (acrescentados)
1. Num problema quadrático com Hessiana conhecida (regressão linear com 0,5 × MSE, de modo que H = XᵀX/n; o adendo original dizia MSE, o que daria 2XᵀX/n e foi corrigido pelo autor), a power iteration acerta o maior autovalor com erro < 1%.
2. A raiz da soma dos quadrados das normas L2 por camada é igual à norma L2 global, para gradiente e para parâmetros.
3. Uma condição curta rodada com e sem as métricas novas dá resultados idênticos no contínuo.
4. O cálculo das métricas não altera nenhum parâmetro, nem o estado do otimizador, nem o modo (train/eval) da rede original.

### Fora do escopo
`tabela_decisao.md` e as regras de decisão não mudam. A `fig_k1_mecanismo.png` é ampliada (painéis por tarefa de: norma dos parâmetros por camada, norma L2 do gradiente por camada no início da tarefa, srank do gradiente, maior autovalor da Hessiana no início e no fim) para `sem_media_k1`, `fedavg_iid`, `central_passos` e `central_aprendizado`, em m=10 e em m\*, com a rede nova como linha tracejada de referência.

### 2026-10-20 · Desvios v3.1: custo medido e cortes aplicados (antes do completo)
Medição feita com `python -m experiments.medir_custo` (resultado em `results/custo_diagnosticos.json`; tempos de treino lidos do log do teste curto).
- **Custo por medição completa** (27 colunas: normas, gradiente por camada, espectro de 64 pontos com `vmap`, Hessiana com 30 iterações): 0,17 a 0,19 s isolada; **0,42 s dentro de uma execução real**, com outras execuções em paralelo. A primeira chamada de cada processo custa ~1 s (aquecimento do `vmap`).
- **Execução real de `fedavg_iid` em m=1, T=30, semente 1** (92 medições): 163 s sem diagnósticos, 202 s com, **+23,6%**. Em m=1 o custo relativo é o mais enganoso, como previsto pelo autor; por isso o critério é o tempo total do completo.
- **Projeção do tempo total do completo** (255 execuções: 210 de T=30 e 45 de T=12; 15 combinações de tarefa e semente): base de treino ≈ 15,4 h de CPU (estimativa; o treino de `central_aprendizado`, L2 e SP é aproximado pelo de `central_passos`, e calibração, regime e seleção entram com uma constante grosseira de 1,5 h). Diagnósticos: +2,45 h com o custo medido em situação real, **+15,9%**; +7,1% com o custo isolado.
- **Decisão:** o acréscimo projetado fica abaixo de 25% mesmo no cenário conservador. **Nenhum corte foi aplicado**: (a) Hessiana e espectro continuam em todos os pontos da tarefa (Hessiana) e em 64 pontos (espectro); (b) partida a frio da power iteration, 30 iterações fixas; (c) Hessiana e espectro em todas as tarefas. As constantes `HESS_PONTOS`, `HESS_PARTIDA_QUENTE` e `DIAG_SO_REFERENCIAS` existem em `src/config.py` com os valores que desligam os cortes, e têm teste.
- **Limite da estimativa:** o tempo real do completo será medido e reportado no README. Se o completo real ficar mais de 25% acima do tempo equivalente sem diagnósticos, isso será anotado aqui.

### 2026-10-20 · Desvios v3.1: esclarecimentos após a auditoria do v3.1 (antes do completo)
1. O código grava também `grad_l1_global` (soma das normas L1 por tensor), além da L2 global descrita no item b.
2. A semente 1000 não recebe os diagnósticos de Klein, mas recebe a coluna `srank_feat`, que vem de `metricas_modelo` (item e). `srank_feat` é numericamente igual a `fim_srank_feat`.
3. `hess_max` é o autovalor de maior módulo **com sinal**; numa rede com ReLU a Hessiana pode ser indefinida e o valor sair negativo. `hess_conv = 1` significa que a estimativa estabilizou (variação relativa < 1% nas últimas 5 iterações), não que está exata; nos testes o critério aceitou estimativas a ~0,5% do valor verdadeiro. Se |λ máximo| e |λ mínimo| forem quase iguais, a iteração pode oscilar e `hess_conv` sai 0.
4. `csv_valido` passa a exigir a coluna `fim_hess_max` nos CSVs das sementes de avaliação, para um CSV gerado antes do v3.1 não ser reaproveitado. `srank_feat` usa `SRANK_DELTA` de `src/config.py`, o mesmo δ de `grad_srank`.
5. O critério de 25% foi aplicado sobre uma **projeção** (a auditoria mostrou que a decisão é robusta: seriam necessários ~0,66 s por medição, contra 0,42 s medidos, para passar de 25%). O completo não tem um tempo "equivalente sem diagnósticos" medido; o tempo real do completo é reportado no README, com a estimativa do acréscimo.

## Leituras após a auditoria dos resultados confirmatórios (2026-10-21)
**Este bloco foi escrito DEPOIS de ver os resultados** (os 255 CSVs das sementes 1 a 5, nível m\*=3). Não acrescenta nenhum critério e **não muda nenhum veredito** de H_reg, H_part, H_real e H_orc; só limita o que pode ser afirmado. A auditoria dos resultados (somente leitura) conferiu por aritmética 21 células da `tabela_decisao.md` e as regras literais.

1. **Esquecimento em B_disjunta está saturado.** Os valores finais ficam em 0,84 a 0,88 em todas as condições, a acurácia nas tarefas antigas fica em 0,10 a 0,13 (acaso) e `bwt` é exatamente o negativo do esquecimento. Não é erro de definição: é consequência dos dados (cada tarefa é uma função nova). O máximo possível (~0,89) é quase o limite (0,892), então o limite de 5 pontos percentuais **nunca restringiu nada**: "ganho válido" equivale a "melhora plasticidade", e a cláusula de esquecimento de H_real não foi usada. A seleção de λ pelo limite também foi vazia (todos os λ respeitaram). **Não se afirma nada sobre retenção nem sobre trade-off entre plasticidade e esquecimento.**
2. **λ na borda da grade.** L2 = 1e-4 (o menor) e SP = 0,8 (o mais fraco) foram escolhidos nas três tarefas; a acurácia contínua piora sempre que a regularização aumenta. O ótimo provavelmente está fora da grade. L2 e SP aqui são regularizações **fracas**, não ótimas.
3. **A sonda de alvos aleatórios não mede plasticidade neste toy.** O denominador (perda da rede nova) é praticamente constante entre condições (0,00073 na semente 1; mesmos alvos, mesmos lotes, mesma inicialização) e o numerador acompanha a norma das saídas, não o gap; a razão chega a 1e6 a 1e8. Fica como medida exploratória; reportar a perda da cópia e a norma, não a razão. A `fig_sonda.png` usa escala symlog.
4. **Controle `sem_media_k1` × `central_passos`.** Por ICs sobrepostos coincidem em 8 de 9 casos; em B, m=10 divergem (−0,009 contra −0,003). As diferenças pareadas por semente são sistemáticas em A, m=1 e na janela 7 a 11, e bimodais em A, m=10. **Não se afirma equivalência**; reporta-se como resultado, conforme o item (a) dos Desvios v3.
5. **H_part (ii) e a janela 7 a 11.** Essa janela é anterior à perda: no `central_aprendizado` em B o gap é +0,009 ali, enquanto a pré-condição (janela principal) existe. A falha de (i) em A não depende da janela. Para `media_sem_particao` (T=12) não há dado nas tarefas 25 a 29, então nada se diz sobre se a média sem partição protegeria até o fim.
6. **Teto em m=10.** A rede nova do `fedavg_iid` chega a 1,000 em todas as tarefas; o gap só pode ser ≤ 0 e `media_sem_particao` dá exatamente 0 nas 5 sementes (IC degenerado). Nenhuma decisão usa m=10; a dose-resposta ali é limitada pelo teto. Em m=3, B e B_disjunta também estão perto do teto.
7. **Referência própria.** Cada condição usa a própria rede nova (item 3 das escolhas operacionais); a rede nova de SP aprende menos (0,79 contra 0,83), então parte do gap positivo de SP em A vem da referência. Pela acurácia contínua das 5 últimas tarefas a diferença continua grande (0,938 contra 0,767) e nenhum veredito muda.
8. **Frações da classe majoritária após o balanceamento** (item (e) dos Desvios v3), média e máximo por tarefa: A 0,116 e 0,134; B 0,140 e 0,183; B_disjunta 0,144 e 0,196; nenhuma tarefa acima de 0,3. (Valores também em `notes.md`.)
9. **Tempo real.** Completo: 34 208 s (9,5 h) e total 38 593 s (10,7 h) de parede com 3 processos, contra ~6 h projetadas; as medições por execução incluem pausas do computador durante a noite (execuções com tempo de parede muito maior que as vizinhas) e um evento de falta de memória do sistema; o equivalente sem diagnósticos não foi medido.
10. **Horizonte da sonda em `media_sem_particao`:** a última sonda é a da tarefa 9 (T=12), contra a 29 nas demais.

## Incidente: pastas esvaziadas em 2026-10-20 e proteção de `results/` (registrado em 2026-10-21)
- **O que sumiu:** `results_v1_exploratorio/` (87 arquivos do desenho antigo: CSVs, matrizes e JSONs da semente 0, mais o pipeline completo interrompido) e o **conteúdo** de `logs/` (logs do `--quick` e do pipeline antigo). Foram esvaziadas às **16:45:20** (`logs/`) e **16:45:38** (`results_v1_exploratorio/`) de 2026-10-20, 18 s uma da outra; nenhuma outra pasta foi tocada.
- **Quem apagou: não se sabe.** Verificado: (i) nas transcrições da sessão principal e de todos os subagentes não há comando de remoção naquele instante (19:45Z); nenhuma ferramenta foi chamada entre ~16:25 e 16:45:43 local, e todos os subagentes estavam parados entre 16:24:23 e 16:53:51; (ii) nenhum código do repositório apaga essas pastas (os únicos `rmtree`/`remove` do projeto agem em diretórios temporários e em arquivos sintéticos de teste); (iii) `git reflog` e o histórico git não mostram remoção (os arquivos não eram rastreados); (iv) o `~/.bash_history` só tem `rm` de outros projetos. O formato (duas pastas, 18 s de intervalo) é compatível com dois comandos digitados à mão, mas **isso não foi confirmado**: o histórico de terminais abertos só é gravado ao fechar o shell.
- **Nenhum CSV confirmatório estava lá:** os 255 CSVs confirmatórios (nome `<cond>_m<nível>_<tarefa>_<semente>`) foram criados depois do início do completo (2026-10-20 23:14), 6,5 h após o esvaziamento; a pasta guardava só o desenho antigo (semente 0, sem nível `m`).
- **Integridade dos resultados confirmatórios:** `results/` no disco é idêntico ao commit `fa9d3f7`; nenhum arquivo foi modificado depois do fim do completo; o resumo por semente que o auditor leu (`resumo_por_semente.csv`) coincide com o recomputado dos CSVs atuais (diferença máxima 1,1e-16, 255 linhas, mesmas chaves). Foi gerado o `MANIFESTO_results.sha256` (793 arquivos) e commitado.
- **Medidas:** (a) o `CLAUDE.md` passou a proibir qualquer agente de apagar, mover, renomear, sobrescrever ou truncar arquivos em `results/`, `results_*` e `logs/` (e as mesmas linhas entraram nos três agentes); (b) depois de cada execução completa, o pipeline deixa `results/` somente leitura (`experiments/protecao.py`, arquivos 0444 e pastas 0555; reabrir exige `--reabrir-resultados` e autorização do autor); (c) `results/` foi protegida agora (793 arquivos) e `sha256sum -c MANIFESTO_results.sha256` confere. Código novo: `experiments/protecao.py`, com teste. Nenhum resultado, vereditos ou critério mudou.

## Análises exploratórias pós-resultados (2026-10-21) · definições escritas ANTES de calcular
**Todas as leituras abaixo são exploratórias e feitas depois de ver os resultados. Nenhum veredito confirmatório (H_reg, H_part, H_real, H_orc) é alterado por elas.**

### A. Adiantamentos (ordem temporal dos sinais e da acurácia)
- **Versão 1, por limiar: DEGENERADA e mantida no relatório.** Ela marcava o primeiro instante t ≥ 5 em que cada sinal passava de um limiar relativo ao começo; como o limiar era cruzado já no primeiro instante permitido (t = 5), o "adiantamento" resultava sempre em t_acc − 5 e não informa nada sobre quem vem antes de quem. É descartada como evidência, e fica no relatório (`analise_exploratoria_klein.md`) só como registro.
- **Versão 2, meia-excursão sem filtro: executada uma vez antes de estas definições existirem** (commit `1e3de74`, todas as condições e as 5 sementes, sem o filtro de perda real). Fica no relatório como "prévia". **Quem escreve estas definições já viu os resultados dessa versão.**
- **Versão 3 (definição final, a calcular uma única vez):**
  1. *Universo:* só combinações de (condição, nível, tarefa, semente) com **perda real**, isto é, métrica principal da semente (gap médio pareado nas 5 últimas tarefas) ≤ −0,10.
  2. *Percurso da acurácia:* |média da `acc_cont` das 5 últimas tarefas − média das 3 primeiras|. Se for **< 0,10, o adiantamento é "indefinido"** e a semente sai.
  3. *t50 de uma série:* o primeiro t em que a média móvel (janela de 3 tarefas, para trás) percorreu a metade do caminho entre o nível inicial (média das 3 primeiras) e o final (média das 5 últimas). Série sem excursão (|fim − início| < 1e-9) = indefinido.
  4. *Sinais:* norma global dos parâmetros (`fim_norma_global`), |`fim_hess_max`|, dormentes τ=0,025 da última camada oculta, `srank_feat` e `rank_ef`; todos medidos no **fim** da tarefa, o mesmo instante da `acc_cont`.
  5. *Adiantamento:* t50(acc) − t50(sinal), **por semente**; positivo = o sinal se moveu antes.
  6. *Relatório:* mediana entre sementes e IC 95% por bootstrap percentil da mediana (10 000 reamostragens, `default_rng(0)`), com o n de sementes definidas; com n < 3 o IC não é reportado.
  7. *Resolução temporal:* **uma tarefa** (índice t), com suavização por média móvel de 3 tarefas em todas as séries; diferenças menores que 1 a 2 tarefas não são interpretáveis.
  8. *Roda uma vez e reporta como sair.*

### B. Curvas completas e área do gap
- O gap só existe nas tarefas com rede nova (16 de 30; 7 de 12 em `media_sem_particao`). **Curva do gap por tarefa** = esses pontos, ligados por linha, média e IC 95% entre sementes, por condição e nível, para as três tarefas.
- **Área do gap** por (condição, nível, tarefa, semente) = integral trapezoidal do gap sobre o índice da tarefa, do primeiro ao último ponto observado (2 a 29; 2 a 11 com T=12), dividida pelo comprimento do intervalo (27; 9): ou seja, o gap médio ao longo de toda a sequência, ponderado pelo espaçamento. Relata-se média entre sementes e IC 95% por bootstrap. Limite: os pontos estão mais densos nas tarefas 7 a 11 e 25 a 29.

### C. Sonda em escala logarítmica
- A sonda original (razão cópia/nova) fica como está (invalidada como medida de plasticidade; ver "Leituras após a auditoria").
- **Variante pedida:** mediana entre os alvos de log10(perda_cópia + 1e-6) − log10(perda_nova + 1e-6). **Não é computável com os dados atuais:** os CSVs guardam só a **média das 3 perdas** (cópia e nova), não as perdas por alvo. Calcula-se a variante **Δlog = log10(média das perdas da cópia + 1e-6) − log10(média das perdas da nova + 1e-6)** nas tarefas de sonda (índices 4, 9, …, 29), por semente; reporta-se média entre sementes e IC 95%. Log da média não é média do log. A mediana entre alvos exige guardar a perda por alvo na próxima rodada.
- **A sonda de 300 passos satura:** a perda da rede nova chega a ~7e-4 e é praticamente constante entre condições, de modo que o denominador deixa de discriminar. Para a próxima rodada se propõe o número de passos em que a rede nova fica **entre 10% e 30% da perda inicial**. Para fixar esse número calcula-se, uma vez, num script à parte (sem tocar em `results/`): 20 alvos sorteados com sementes fora das de avaliação, rede nova treinada com o protocolo da sonda (Adam, lr 1e-3, lotes de 100, MSE), perda em todos os 1000 pontos a cada passo; reporta-se, entre alvos, a mediana [mín, máx] do primeiro passo em que perda/perda inicial ≤ 0,30 e ≤ 0,10.

### D. Limitações a registrar
- **O esquecimento em B_disjunta saturou por construção:** cada tarefa tem um professor novo e a entrada deslocada, então a função muda por completo; a rede cai ao acaso nas tarefas antigas assim que aprende a seguinte, e o "esquecimento" passa a medir quanto cada tarefa foi aprendida (plasticidade) e não retenção. O limite de 5 pontos percentuais nunca restringiu nada.

### E. Propostas para a próxima rodada (nada disto foi implementado nem rodado)
1. **B_disjunta com professor fixo e só a entrada deslocada**, para que a função seja a mesma e exista algo a reter. Junto, uma **medida de reaprendizado**: quantos passos a rede precisa para voltar a 90% da acurácia antiga numa tarefa velha.
2. **Aprendiz centralizado no estilo Hare & Tortoise:** uma rede rápida que volta para a média móvel exponencial (EMA) dos pesos a cada rodada, como controle para "a média por si só protege" (`media_sem_particao` quase não perde plasticidade).
3. **Partição não IID por Dirichlet na tarefa B** (α em {0,1; 1; 100}), para separar o efeito de dividir os dados do efeito de promediar.
4. **Sonda reformulada:** passos calibrados (10% a 30% da perda inicial), perda por alvo guardada, mediana do Δlog entre alvos.
5. Também sugeridos pelos dados: grade de λ estendida além da borda (L2 < 1e-4; SP > 0,8), mais de uma semente de calibração para m\*, e um nível entre 1 e 3.
