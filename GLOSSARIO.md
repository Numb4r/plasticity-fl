# Glossário

Uma linha por nome, em português simples, sem fórmulas. Os nomes são os que aparecem no código, nos CSVs, nas figuras e no `PREREGISTRO.md`.

## A ideia central

- **Plasticidade** — a capacidade de uma rede de continuar aprendendo tarefas novas depois de já ter aprendido muitas. Aqui é medida contra uma rede nova, não contra a estabilidade da memória.
- **Perda de plasticidade** — a rede que treinou por muito tempo aprende a tarefa nova pior do que uma rede recém-criada aprenderia.
- **Rede contínua (ou "contínuo")** — a única rede que atravessa todas as tarefas em sequência, sem nunca ser reiniciada.
- **Rede nova (ou "rede de referência")** — uma rede recém-sorteada, treinada só numa tarefa, com o mesmo protocolo da condição. Serve de régua: mostra o que seria possível aprender sem histórico.
- **Esquecimento** — o quanto a rede piora nas tarefas antigas depois de aprender as novas. É outra coisa, medida à parte, só na tarefa B_disjunta.
- **Toy** — "brinquedo": um problema pequeno e sintético, escolhido para ser barato e controlável, não para ser realista.

## Variáveis e letras

- **N** — número de pontos de dados por tarefa (1000).
- **D** — número de números que descrevem cada ponto (32).
- **T** — quantas tarefas a sequência tem (30; só 12 em `media_sem_particao`).
- **t** — o número da tarefa dentro da sequência, contando a partir de zero.
- **K** — número de clientes do aprendizado federado (10, ou 1 no `sem_media_k1`).
- **Cliente** — um dos "participantes" do FedAvg, que treina com a sua parte dos dados.
- **R** — número de rodadas de comunicação por tarefa (25 vezes o nível m).
- **Rodada** — um ciclo do FedAvg: cada cliente treina um pouco, o servidor faz a média e devolve o modelo.
- **S_local** — quantos passos de treino cada cliente dá por rodada (sempre 15, não importa quantos dados ele tenha).
- **Passo** — uma atualização da rede com um lote de 20 pontos.
- **Lote (batch)** — o pequeno grupo de pontos usado em cada passo (20).
- **Época** — uma passada completa pelos dados de quem treina.
- **m (nível de orçamento)** — o multiplicador de esforço por tarefa: 1, 3 ou 10. Quanto maior, mais rodadas e mais passos.
- **m\* (regime de teste)** — o menor nível em que o `central_aprendizado` já perde plasticidade na tarefa A. É onde as hipóteses são avaliadas.
- **λ (lambda)** — a força de uma regularização: de L2 (peso do encolhimento) ou de shrink-and-perturb (quanto a rede é encolhida).
- **τ (tau)** — o limiar abaixo do qual um neurônio é considerado dormente (0 ou 0,025).
- **δ** — a tolerância usada na definição de srank (0,01): quanto da soma dos valores singulares pode ficar de fora.

## Condições (as 7 formas de treinar)

- **`fedavg_iid`** — FedAvg com 10 clientes, cada um com 100 pontos sorteados ao acaso; o servidor faz a média ponderada. É a condição principal.
- **`sem_media_k1`** — um cliente só, com os 1000 pontos e o mesmo protocolo local. Separa o efeito da média do efeito de dividir os dados. Com S_local fixo, vira quase uma cópia do `central_passos`; é mantido como controle de sanidade.
- **`media_sem_particao`** — 10 clientes que têm todos os mesmos 1000 pontos; só a média está presente, sem divisão de dados. Usa só 12 tarefas porque custa 10 vezes mais.
- **`central_passos`** — treino centralizado com o mesmo número de passos que um cliente do FedAvg dá por tarefa. Controle de "mesmo esforço".
- **`central_aprendizado`** — treino centralizado com o número de passos ajustado para que a rede nova aprenda tanto quanto a do FedAvg. Controle de "mesmo nível de aprendizado".
- **`central_L2`** — o `central_aprendizado` com L2 (um puxão dos pesos para perto de zero). Mitigação conhecida.
- **`central_SP`** — o `central_aprendizado` com shrink-and-perturb no início de cada tarefa. Mitigação conhecida.
- **FedAvg** — o método em que cada cliente treina localmente e o servidor faz a média ponderada dos modelos.
- **Média ponderada** — média em que cada cliente pesa conforme o número de pontos que tem.
- **Adam** — o algoritmo que atualiza a rede. É recriado do zero a cada rodada (e a cada 15 passos nos centralizados), para não carregar memória de uma rodada para a outra.
- **L2** — penalidade que encolhe os pesos um pouco a cada passo.
- **Shrink-and-perturb (SP)** — no começo de cada tarefa, encolher todos os pesos e somar um pouco de ruído.
- **Calibração** — descobrir, na semente 1000, quantos passos o `central_aprendizado` precisa para a rede nova dele igualar a do FedAvg (dentro de ±0,03).
- **Seleção de λ** — escolher, na semente 1000, o melhor λ de L2 e de SP, respeitando o limite de esquecimento.

## Tarefas (os tipos de problema)

- **Tarefa A** — rótulos sorteados ao acaso a cada tarefa, sem estrutura: a rede só pode memorizar.
- **Tarefa B** — os rótulos vêm de uma rede "professora" nova a cada tarefa: há uma regra a aprender, e os clientes concordam sobre ela.
- **Tarefa B_disjunta** — como a B, mas cada tarefa tem os seus próprios pontos, em regiões diferentes do espaço. Só nela o esquecimento faz sentido.
- **Professor** — uma rede aleatória que, para cada ponto, escolhe a classe com a maior saída. Define os rótulos de B e B_disjunta.
- **Balanceamento do professor** — ajuste que impede uma classe de dominar: tira de cada saída do professor a sua média antes de escolher a classe.
- **Classe majoritária (`maj_frac`)** — a fração de pontos da tarefa que pertence à classe mais comum.
- **Partição** — como os pontos são repartidos entre os clientes.
- **Versão dos dados (`versao_dados`)** — número que muda quando a forma de gerar os dados muda; impede reaproveitar arquivos antigos por engano.

## Sementes

- **Semente** — o número que fixa todos os sorteios; o mesmo número dá exatamente o mesmo resultado.
- **Sementes confirmatórias (1 a 5)** — as cinco usadas na análise final.
- **Semente 1000** — usada só para calibrar, escolher λ e definir o regime de teste; nunca entra na avaliação.
- **Semente 0 do desenho antigo** — exploratória; ficou fora da análise confirmatória.
- **Papel (da semente)** — cada fonte de sorteio (dados, rótulos, rede contínua, rede nova...) tem o seu próprio papel, para que não se misturem.
- **Pareado** — comparado na mesma semente e na mesma tarefa, para que a dificuldade da tarefa não atrapalhe.

## Métricas (as colunas dos CSVs)

- **`tarefa`** — o número da tarefa (a partir de zero).
- **`tipo`** — qual das tarefas: A, B ou B_disjunta.
- **`cond`** — o nome da condição.
- **`nivel`** — o nível de orçamento m.
- **`semente`** — a semente da execução.
- **`acc_cont`** — acurácia da rede contínua nos pontos da tarefa, ao fim do treino dela.
- **`acc_nova`** — acurácia da rede nova de referência na mesma tarefa; vazia nas tarefas sem referência.
- **`gap`** — acurácia da contínua menos a da nova, na mesma tarefa. Positivo: o histórico ajudou. Negativo: perdeu plasticidade.
- **Métrica principal** — a média do `gap` nas 5 últimas tarefas, depois a média entre sementes.
- **`n_ref`** — quantas tarefas com rede nova entram numa média.
- **`norma`** — o tamanho total dos parâmetros da rede (uma medida de quanto os pesos cresceram).
- **`distancia`** — quanto os parâmetros se moveram durante a tarefa.
- **`dorm_tau0_l1`, `dorm_tau0_l2`** — fração de neurônios dormentes (nunca ativos) na primeira e na segunda camada oculta, com limiar zero.
- **`dorm_tau0025_l1`, `dorm_tau0025_l2`** — o mesmo, com limiar 0,025 (neurônios quase inativos também contam).
- **Neurônio dormente** — neurônio cuja ativação é desprezível perto da dos outros da mesma camada (definição de Sokar et al., 2023).
- **`rank_ef`** — rank efetivo da última camada oculta: quantas direções independentes a rede realmente usa. Cair indica perda de diversidade nas representações.
- **`cos_medio`** — o quanto as atualizações dos clientes apontam na mesma direção (de -1 a 1); vazio quando só há um cliente.
- **Sonda de alvos aleatórios** — teste em que a rede tenta imitar uma rede aleatória nova; compara com uma rede nova na mesma tarefa (Lyle et al., 2022).
- **`sonda_perda_copia`, `sonda_perda_nova`** — erro final da cópia da rede treinada e da rede nova ao imitar o mesmo alvo.
- **`sonda_razao`** — erro da cópia dividido pelo da rede nova; acima de 1 significa que a rede perdeu capacidade de aprender.
- **`acc_media_vistas`** — acurácia média, depois de cada tarefa, sobre todas as tarefas já vistas (B_disjunta).
- **`esquecimento`** — média de quanto a rede piorou, em cada tarefa antiga, em relação ao melhor que já fez nela (Chaudhry et al., 2018).
- **`bwt`** — média da diferença entre a acurácia atual numa tarefa antiga e a que a rede tinha logo ao aprendê-la; negativa quando há esquecimento (Lopez-Paz e Ranzato, 2017).
- **Matriz de acurácias** — tabela com a acurácia em cada tarefa j depois de treinar até a tarefa i (arquivo `_matriz.csv`).
- **Limite de esquecimento** — uma condição só conta como ganho de plasticidade se esquecer, no máximo, 5 pontos percentuais a mais que o `central_aprendizado`.

## Diagnósticos de Klein et al. (Adendo v3.1; só descrevem, não decidem nada)

- **Diagnóstico** — medida extra para entender *por que* uma rede perde plasticidade; não entra em nenhuma hipótese nem na `tabela_decisao.md`.
- **Klein et al. (2026)** — o survey de perda de plasticidade em aprendizado por reforço profundo que propõe esse conjunto padrão de medidas.
- **Cadeia causal** — a hipótese do survey de que a norma dos pesos e a curvatura sobem antes de a acurácia cair; a pergunta exploratória do projeto é se isso acontece aqui.
- **Início e fim da tarefa (`ini_`, `fim_`)** — os dois momentos de medição: logo antes do primeiro passo (depois do shrink-and-perturb, com os rótulos novos) e logo depois do treino.
- **`nova_ini_`, `nova_fim_`** — as mesmas medidas na rede nova de referência, como linha de base; vazias nas tarefas sem rede nova.
- **Cópia da rede** — todos os diagnósticos são calculados numa cópia, para nunca mexer na rede de verdade.
- **`norma_l1_w`, `norma_l1_b`, ... `norma_l3_b`** — tamanho dos pesos (w) e dos vieses (b) de cada uma das três camadas lineares; `norma_global` é o tamanho de tudo junto.
- **Gradiente** — a direção e a intensidade com que a rede "quer" mudar para errar menos na tarefa.
- **`grad_l2_l1_w`, ... (norma L2 do gradiente)** — intensidade do gradiente em cada camada; a L2 é sensível a valores extremos.
- **`grad_l1_l1_w`, ... (norma L1 do gradiente)** — a mesma intensidade somando valores absolutos; é mais robusta a valores extremos.
- **`grad_l2_global`, `grad_l1_global`** — o mesmo para todas as camadas juntas.
- **Espectro do gradiente** — olhar os gradientes de 64 pontos fixos da tarefa, um por ponto, para ver se apontam para direções variadas ou todos para a mesma.
- **Pontos fixos do espectro** — os 64 pontos são sorteados uma vez por semente e tarefa e são os mesmos em todas as condições, momentos e redes.
- **`grad_srank`** — quantas direções independentes os gradientes dos 64 pontos realmente usam; valor baixo indica gradientes redundantes.
- **srank** — o menor número de direções que juntas explicam quase tudo (99%) de uma matriz.
- **`srank_feat`** — o mesmo srank, mas das ativações da última camada oculta (as "features"); fica ao lado do `rank_ef`, que usa outra definição. É numericamente igual a `fim_srank_feat` (a mesma conta, repetida nos diagnósticos); só `srank_feat` existe também na semente 1000.
- **`grad_cos_medio`** — o quanto, em média, os gradientes de pontos diferentes apontam para o mesmo lado.
- **`grad_cos_neg`** — a fração de pares de pontos cujos gradientes puxam em sentidos opostos.
- **Hessiana** — a "curvatura" da perda: diz o quão rápido o erro muda quando se mexe nos pesos.
- **`hess_max`** — o autovalor da curvatura de maior módulo, **com sinal** (pode sair negativo numa rede com ReLU). Valor alto de curvatura positiva indica que passos pequenos já desestabilizam o treino.
- **`hess_conv`** — 1 se a estimativa de `hess_max` estabilizou (variação menor que 1% nas últimas 5 iterações), 0 se não. Estabilizar não garante estar exata: o critério aceita estimativas ainda a ~0,5% do valor verdadeiro.
- **Power iteration** — método iterativo que acha o maior valor de curvatura sem montar a matriz inteira (30 iterações).
- **Produto Hessiana-vetor** — a conta que a power iteration repete a cada passo, feita com diferenciação automática dupla.
- **`torch.func.vmap`** — ferramenta que calcula os gradientes dos 64 pontos de uma vez, em vez de um por um.
- **Cortes de custo (a, b, c)** — formas de baratear os diagnósticos, usadas só se o completo crescesse mais de 25%; nenhum foi necessário.
- **`medir_custo.py` e `custo_diagnosticos.json`** — o script e o arquivo com o custo medido dos diagnósticos e a projeção do acréscimo no tempo do completo.

## Hipóteses e decisão

- **H_reg** — a vantagem do FedAvg é só regularização conhecida: L2 ou SP no centralizado chegam ao mesmo resultado.
- **H_part** — a vantagem vem de dividir os dados entre clientes que discordam, não da média.
- **H_real** — existe um efeito do FedAvg além da regularização conhecida.
- **H_orc** — a vantagem do FedAvg sobre o centralizado some quando os dois são comparados no mesmo nível de aprendizado da rede nova.
- **Pré-condição de H_part** — só vale testar H_part se o `fedavg_iid` tiver mesmo uma vantagem a explicar.
- **"Sem vantagem a explicar"** — resultado em que não há diferença entre FedAvg e centralizado para explicar.
- **"Não avaliável"** — faltam dados ou sementes para decidir.
- **Ganho válido / ganho com violação de esquecimento / sem ganho** — a marcação de B_disjunta: melhorou a plasticidade respeitando o limite, melhorou mas esqueceu demais, ou não melhorou.
- **IC 95% (bootstrap)** — faixa em que o valor verdadeiro provavelmente está, calculada reamostrando as sementes dez mil vezes.
- **Janela** — o conjunto de tarefas usado numa média: as 5 últimas, ou as tarefas 8 a 12 (índices 7 a 11) em H_part.
- **Regime sem perda** — situação em que nenhum nível chega a perder plasticidade; nesse caso as hipóteses não podem ser testadas.

## Arquivos e pastas

- **`PREREGISTRO.md`** — as hipóteses e critérios, escritos antes dos resultados; mudanças só entram em "Desvios".
- **Desvios / Adendo / Errata** — registros datados de mudanças ou correções ao pré-registro.
- **Previsão do autor** — o palpite do autor sobre os resultados, escrito antes de rodar.
- **`CLAUDE.md`** — as regras do projeto e a divisão de trabalho entre os agentes.
- **`notes.md`** — observações soltas durante o trabalho (por exemplo, o balanço de classes).
- **`task_plan.md`** — o plano e o andamento das fases.
- **`results/`** — um CSV por condição, nível, tarefa e semente de avaliação.
- **`results/calibracao/`** — tudo da semente 1000 (calibração, seleção de λ, regime); fora da análise.
- **`results/calibracao/selecao/`** — as execuções de cada λ testado.
- **`results/teste/`** — execuções curtas de teste (`--T`), nunca misturadas com os resultados.
- **`results_v1_exploratorio/`** — o material do desenho antigo (hoje vazio).
- **`calibracao_<tarefa>_m<m>.json`** — resultado da calibração: passos escolhidos, erro, se ficou dentro de ±0,03.
- **`selecao_<L2|SP>_<tarefa>_m<m>.json`** — λ escolhido e se violou o limite de esquecimento.
- **`regime.json`** — qual é o m\* e os gaps que levaram a ele.
- **`*.params.json`** — os parâmetros exatos com que um CSV foi gerado.
- **`*_rodadas.csv`** — o cosseno entre clientes em cada rodada.
- **`*_matriz.csv`** — a matriz de acurácias da B_disjunta.
- **`tempo_*.json`** — tempos de execução de cada fase.
- **`plots/`** — scripts que fazem figuras e tabelas; **`figures/`** — as figuras e a tabela de decisão.
- **`logs/`** — o texto que cada execução imprimiu.
- **`src/`** — o código do experimento; **`experiments/`** — os comandos para rodar; **`tests/`** — os testes.

## Figuras e tabela

- **`fig_gap.png`** — o `gap` por tarefa, uma linha por condição.
- **`fig_norma.png`** — a norma dos parâmetros e a distância percorrida por tarefa.
- **`fig_sonda.png`** — a razão da sonda de alvos aleatórios ao longo das tarefas.
- **`fig_tradeoff.png`** — plasticidade contra esquecimento, um ponto por condição, só na B_disjunta.
- **`fig_dose.png`** — a figura principal: o `gap` em função do quanto a rede nova aprende, em cada nível m.
- **`fig_k1_mecanismo.png`** — compara as condições em m=10 (e em m\*) para ver se o colapso vem de neurônios mortos, perda de rank, explosão de norma ou aumento de curvatura; usa os diagnósticos do Adendo v3.1.
- **`tabela_decisao.md`** — a tabela final com as métricas por condição, o resultado de cada hipótese e as ressalvas.

## Agentes

- **`executor-experimentos`** — roda os experimentos e confere os arquivos gerados; nunca edita código.
- **`gerador-graficos`** — cria as figuras e tabelas; só mexe em `plots/` e `figures/`.
- **`auditor-metricas`** — revisor somente leitura; confere métricas, sementes e a interpretação antes de qualquer conclusão.
- **Sessão principal** — a única que edita `src/`, `experiments/` e `tests/`.

## Pessoas e referências citadas

- **Dohare et al. (2024)** e **Lyle et al. (2022)** — de onde vem a ideia de plasticidade como capacidade de seguir aprendendo, medida contra uma rede nova.
- **Sokar et al. (2023)** — a definição de neurônio dormente.
- **Ash e Adams (2020)** — o shrink-and-perturb.
- **Chaudhry et al. (2018)** e **Lopez-Paz e Ranzato (2017)** — as medidas de esquecimento e de BWT.
- **Klein et al. (2026)** — o survey cujas métricas entram no Adendo v3.1.
