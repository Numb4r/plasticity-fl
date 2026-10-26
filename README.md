# Plasticidade: FedAvg contra treino centralizado

> **Rodada 1 fechada:** tag git `rodada1-final` (código, dados, pré-registro e figuras exatamente como na rodada 1).
> **Rodadas 2 e 2b fechadas:** tag git `rodada2-final` (commit `8bd0ee0`). Resultados em `results_rodada2/` e `results_rodada2b/`, rodados no cluster Recod; análises posteriores estão nos Desvios do `PREREGISTRO.md`. Resumo na seção "Rodadas 2 e 2b" abaixo.

> **Dúvida sobre algum nome?** Veja o [GLOSSARIO.md](GLOSSARIO.md): uma linha em português simples para cada condição, variável, coluna, hipótese e figura.

Um experimento pequeno e pré-registrado para responder: **a vantagem do FedAvg sobre o treino centralizado em preservar plasticidade sobrevive a controles de passos, de partição dos dados e de regularização?**

## O que é "plasticidade" aqui

Plasticidade é a **capacidade de uma rede de continuar aprendendo**, medida contra uma rede nova: depois de treinar em muitas tarefas, a rede contínua aprende a tarefa seguinte tão bem quanto aprenderia uma rede recém-sorteada? É o sentido de Dohare et al. (2024) e Lyle et al. (2022). **Não** é o equilíbrio entre estabilidade e plasticidade da literatura de aprendizado federado contínuo (aprender sem esquecer).

A medida central é o **gap**: acurácia da rede contínua menos a da rede nova, na mesma tarefa (pareado). Gap negativo = perda de plasticidade. A **métrica principal** é o gap médio nas 5 últimas tarefas.

## Leitura rápida dos resultados (5 sementes, nível m\* = 3)

Tudo abaixo é do `PREREGISTRO.md` e da `figures/tabela_decisao.md`; a auditoria dos resultados conferiu a tabela contra os CSVs e as regras contra o texto do pré-registro. **O resultado é "nenhum critério atendido"** (H_reg, H_part, H_real e H_orc: todas "não atendidas", com margens largas), e isso é um resultado pré-registrado.

Gap médio nas 5 últimas tarefas de cada condição, m=3 (média entre sementes, IC 95% por bootstrap). **Atenção:** o `media_sem_particao` da rodada 1 rodou com **T=12 e a métrica dele é a das tarefas 8 a 12**; as demais usam as tarefas 26 a 30. As janelas não são comparáveis; a comparação em pé de igualdade está na seção das rodadas 2 e 2b.

| condição | tarefa A | tarefa B |
|---|---|---|
| `fedavg_iid` | −0,062 [−0,090; −0,033] | −0,001 [−0,004; +0,003] |
| `media_sem_particao` (**T=12, tarefas 8 a 12**) | −0,006 [−0,014; +0,006] | +0,009 [+0,007; +0,011] |
| `sem_media_k1` | −0,332 [−0,475; −0,238] | −0,066 [−0,076; −0,054] |
| `central_passos` | −0,463 [−0,583; −0,339] | −0,067 [−0,073; −0,062] |
| `central_aprendizado` | −0,452 [−0,585; −0,318] | −0,083 [−0,097; −0,069] |
| `central_L2` | +0,100 [+0,089; +0,113] | +0,028 [+0,026; +0,031] |
| `central_SP` | +0,148 [+0,142; +0,154] | +0,041 [+0,037; +0,044] |

O que **pode** ser afirmado (n = 5 sementes, um problema sintético pequeno):
- Em m=3, o `fedavg_iid` perde menos plasticidade que o `central_aprendizado` calibrado no mesmo nível de aprendizado da rede nova, em A e em B. **H_orc é refutada.**
- L2 e shrink-and-perturb, mesmo com λ na borda da grade (regularização fraca), preservam a plasticidade e ficam **acima** do `fedavg_iid`. H_reg e H_real não são atendidas.
- H_part não é atendida: a média sem partição (`media_sem_particao`, **T=12, tarefas 8 a 12**) parecia proteger também, e (i) falha na tarefa A. **A rodada 2 mostrou que isso era efeito da janela curta:** nas tarefas 26 a 30 ele perde tanto quanto o `sem_media_k1` (veja abaixo). O veredito da rodada 1 não muda.
- Há dose-resposta com o orçamento m nas condições sem regularização, na tarefa A; o regime de perda aparece a partir de m=3.

O que **não** pode ser afirmado: nada sobre esquecimento ou trade-off (o eixo está saturado, veja abaixo), nada sobre a sonda de alvos aleatórios como medida de plasticidade, que `sem_media_k1` é equivalente a `central_passos`, comparações em m=10 (rede nova no teto, 1,000), a robustez de m\* (definido numa só semente de calibração) nem generalizações para fora deste problema. Em B e B_disjunta a perda é pequena.

**Esquecimento.** Nas tarefas **A e B originais o esquecimento não é medido**, porque o mesmo X recebe rótulos conflitantes a cada tarefa. Ele só existe na tarefa **B_disjunta** (cada tarefa com os seus próprios pontos), e ali está **saturado**: 0,84 a 0,88 em todas as condições, com acurácia nas tarefas antigas ao nível do acaso. O limite de 5 pontos percentuais do Adendo 1 nunca restringiu nada, então "ganho válido" quer dizer só "melhora plasticidade".

**Análises exploratórias (feitas depois dos resultados; não alteram nenhum veredito).** As definições foram escritas no `PREREGISTRO.md` antes de calcular (seção "Análises exploratórias pós-resultados").
- *Diagnósticos de Klein et al. e ordem temporal* (`analise_exploratoria_klein.md`). Definição final (versão 3): só combinações com perda real (gap ≤ −0,10 nas 5 últimas tarefas) e percurso da acurácia ≥ 0,10; adiantamento por semente, mediana com IC bootstrap. Nas três condições que perdem plasticidade (`central_passos`, `central_aprendizado`, `sem_media_k1`; tarefa A, m=3 e m=10), o **rank efetivo** das ativações é o sinal que mais consistentemente se move antes da acurácia (adiantamento mediano de +2 a +14 tarefas; o IC exclui zero em 5 dos 6 casos). A **norma** adianta +1 a +10 tarefas, mas o IC toca ou inclui zero em 4 dos 6. Dormentes e `srank_feat` ficam entre 0 e +10. O **maior autovalor da Hessiana não adianta de forma consistente** (medianas de −6 a +9, com o sinal mudando entre condições). Entre condições, a magnitude da norma e da curvatura não separa quem perde: o `fedavg_iid` tem as maiores e perde menos. Em B e B_disjunta quase não há perda real (1 semente de 5), então nada a dizer ali. Duas versões anteriores da medida (por limiar, **degenerada**: sempre t = 5; e meia-excursão sem filtro, "prévia") estão no relatório como registro. Não prova causalidade.
- *Área do gap* (`figures/exploratorio_gap_area.md`, `figures/exploratorio_fig_gap_curvas.png`): o gap médio ao longo de toda a sequência (integral trapezoidal dos pontos com rede nova, dividida pelo intervalo). Ordena as condições como a métrica das 5 últimas tarefas, com menos amplitude. Tarefa A, m=3: `fedavg_iid` +0,030, `media_sem_particao` (T=12) +0,048, `sem_media_k1` −0,138, `central_passos` −0,209, `central_aprendizado` −0,251, `central_L2` +0,127, `central_SP` +0,152.
- *Sonda em escala logarítmica* (`figures/exploratorio_sonda_log.md`, `figures/exploratorio_fig_sonda_log.png`): Δlog = log10(perda da cópia + 1e-6) − log10(perda da rede nova + 1e-6), com as médias das 3 perdas por alvo (a **mediana entre alvos não é computável**: os CSVs só guardam a média). Δlog é positivo em **todas** as condições, inclusive nas que não perdem plasticidade (`media_sem_particao` em m=10, T=12, tarefas 8 a 12, gap 0, Δlog +7,5), e cresce com o nível e com a norma (tarefa A, `fedavg_iid`: +3,6, +5,4 e +7,9 em m = 1, 3, 10, com norma média 35, 115 e 280): a sonda acompanha a escala das saídas, não a plasticidade. Os 300 passos **saturam** a rede nova (perda final em 2,6% da inicial; ela passa de 30% a 10% da inicial entre os passos ~9 e ~46, `analise_exploratoria_sonda_passos.md`).

## Rodadas 2 e 2b: qual média protege a plasticidade? (sementes 6 a 10; 2b: 11 a 15; m\* = 3)

**Leitura principal (exploratória; escrita depois dos resultados, não é um veredito pré-registrado):** o FedAvg protege, e isso se reproduz na rodada 2; a média de cópias com os mesmos dados (`media_sem_particao`) **não** protege no longo prazo; o efeito parece vir da média entre clientes com dados diferentes. Base: tarefas 26 a 30, tarefa A, m=3, `fedavg_iid` −0,044 [−0,055; −0,033] contra `media_sem_particao` −0,326 [−0,430; −0,259] e `sem_media_k1` −0,366 [−0,491; −0,262]; na rodada 1 o `fedavg_iid` deu −0,062 [−0,090; −0,033]. Em B o mesmo padrão em escala menor (+0,003 contra −0,035 e −0,080). A rede contínua do `fedavg_iid` também fica num nível mais alto (`acc_cont` 0,785 contra 0,550 do `media_sem_particao`). Isto **não** foi testado com um desenho que isole o mecanismo (é a proposta da rodada 3), e a rede nova de cada condição tem nível diferente.

Pergunta: o que, na média do FedAvg, protege a plasticidade: a média no tempo (um aprendiz com média móvel), a média entre aprendizes ou a heterogeneidade dos dados? Tudo abaixo vem de `figures/rodada2_tabela_decisao.md`, `figures/rodada2b_tabela_decisao.md` e dos Desvios do `PREREGISTRO.md`; as duas rodadas foram auditadas antes de interpretar (nenhum P0). **Resultado principal: a hipótese central (H_tempo e H_dist) ficou "não avaliável" nas duas rodadas**, e isso é um resultado registrado, não uma falha.

Gap médio nas 5 últimas tarefas (**tarefas 26 a 30, janela comum a todas as condições**; nesta rodada o `media_sem_particao` roda com T=30), m=3, média entre sementes [IC 95% bootstrap]:

| condição | tarefa A | tarefa B | B_fixa |
|---|---|---|---|
| `fedavg_iid` | −0,044 [−0,055; −0,033] | +0,003 [−0,001; +0,008] | +0,067 [+0,057; +0,076] |
| `media_sem_particao` | −0,326 [−0,430; −0,259] | −0,035 [−0,050; −0,023] | +0,032 [+0,027; +0,037] |
| `sem_media_k1` | −0,366 [−0,491; −0,262] | −0,080 [−0,103; −0,058] | +0,078 [+0,068; +0,087] |
| `central_aprendizado` | −0,592 [−0,660; −0,479] | −0,095 [−0,104; −0,088] | +0,065 [+0,058; +0,073] |
| `central_HT` (= `central_HT_cal`, S=15) | +0,188 [+0,183; +0,194] | +0,051 [+0,048; +0,055] | +0,080 [+0,070; +0,088] |
| `central_lookahead` | +0,102 [+0,082; +0,125] | +0,030 [+0,024; +0,037] | — |
| `fedavg_dirichlet` α=100 | −0,080 [−0,104; −0,060] | −0,010 [−0,019; −0,000] | — |
| `fedavg_dirichlet` α=1 | +0,098 [+0,071; +0,127] | +0,032 [+0,024; +0,040] | — |
| `fedavg_dirichlet` α=0,1 | +0,130 [+0,110; +0,153] | +0,052 [+0,033; +0,074] | — |

**Rodada 2b** (`central_HT_cal2`, sementes 11 a 15, mesmas tarefas): o número de rodadas R foi calibrado na semente 1000 (A: R=86, erro −0,6%; B: R=89, erro +0,3%; μ = 0,99 e 0,995). Gap, tarefas 26 a 30: `central_HT_cal2` +0,146 [+0,142; +0,150] em A e +0,021 [+0,020; +0,023] em B; `media_sem_particao` −0,264 [−0,289; −0,239] e −0,034 [−0,043; −0,026]; `sem_media_k1` −0,301 [−0,369; −0,254] e −0,077 [−0,088; −0,065].

Vereditos pelas regras pré-registradas (nada foi ajustado depois dos dados):

| hipótese | rodada 2 | rodada 2b |
|---|---|---|
| H_tempo e H_dist (a média no tempo basta? / a média entre aprendizes faz algo a mais?) | **não avaliável**: calibração não atingida (S=15 deu perda 0,878 e S=30 deu 0,054, sem valor intermediário), P_vant falhou em A, rede nova fora de 10% | **não avaliável**: P_vant falhou em A (o IC do `media_sem_particao` não fica acima do do `sem_media_k1`); a calibração chegou a 10% |
| H_het (heterogeneidade piora o FedAvg) | **não atendida**: ordem reversa nas duas tarefas (gap cresce quando α diminui) | — |
| horizonte longo (T=100, 3 sementes, tarefa A) | **o FedAvg só atrasa a perda**: `fedavg_iid` colapsa na tarefa 44; `sem_media_k1` e `media_sem_particao` na 17; `central_aprendizado` na 14; `central_HT` não colapsa | — |

Previsão do autor (commit `ab3c8ab`): H_tempo/H_dist "não avaliável" não conta; H_het "sim" foi um erro; "o FedAvg só atrasa" foi um acerto.

O que as análises posteriores mostraram (exploratórias, `figures/rodada2_pos_resultados.md`):
- **A diferença do `media_sem_particao` entre rodadas é de janela e horizonte, não de código.** Na rodada 1 ele rodou com **T=12, tarefas 8 a 12** (−0,006 [−0,014; +0,006] em A); na mesma janela das tarefas 8 a 12 a rodada 2 dá −0,018 [−0,032; −0,000]. Nas tarefas 26 a 30, a janela principal desta comparação, ele dá −0,326, **perdendo tanto quanto o `sem_media_k1`** (−0,366). Toda citação do `media_sem_particao` da rodada 1 diz "T=12, tarefas 8 a 12".
- O `fedavg_iid` das duas rodadas concorda (A, tarefas 26 a 30: −0,062 e −0,044; B: −0,001 e +0,003).
- **H_het está contaminado pelo nível de aprendizado.** Com α menor, a rede nova **e** a contínua aprendem menos; a nova cai mais, e por isso o gap sobe. A "ordem reversa" do veredito reflete o nível das redes, não uma perda de plasticidade medida contra um nível comum:

  | tarefa | α | `acc_nova` | `loss_nova` | `acc_cont` | `loss_cont` | gap |
  |---|---|---|---|---|---|---|
  | A | 100 | 0,857 | 0,682 | 0,777 | 0,789 | −0,080 |
  | A | 1 | 0,660 | 1,128 | 0,758 | 0,744 | +0,098 |
  | A | 0,1 | 0,401 | 2,269 | 0,531 | 1,675 | +0,130 |
  | B | 100 | 0,968 | 0,178 | 0,958 | 0,194 | −0,010 |
  | B | 1 | 0,897 | 0,356 | 0,929 | 0,233 | +0,032 |
  | B | 0,1 | 0,673 | 1,331 | 0,725 | 1,020 | +0,052 |

  (Tarefas 26 a 30, sementes 6 a 10, médias entre sementes; ICs em `figures/rodada2_pos_resultados.md`.)
- No braço longo, o `central_HT` não colapsa, mas o nível da rede nova dele é outro (ver abaixo).

O que **não** pode ser afirmado: nada sobre H_tempo/H_dist; que o `central_HT` "protege mais" (o gap dele vem com rede nova de outro nível, e na 2b a `acc_nova` do `central_HT_cal2` fica 0,037 abaixo da do `media_sem_particao` em A); que existe um `central_HT_cal` diferente do `central_HT` na rodada 2 (a calibração falhou e os CSVs são idênticos); que a média no tempo "basta" (o `central_HT_cal2` treina mais rodadas e a rede contínua dele não está no mesmo nível); que os μ escolhidos são ótimos (quase todos caíram na borda da grade); qualquer leitura de plasticidade pela sonda v2 (não passou na validação: a perda da cópia é de 4 a 7 ordens de grandeza maior que a da rede nova) ou sinal no rank; que o gap da B_fixa mede só plasticidade (mistura transferência, e o gatilho de classe majoritária disparou em 21% das tarefas); comparar os passos de reaprendizado entre condições; somar as sementes da 2b com as da rodada 2.

Outros resultados descritivos (sem critério de decisão): na B_fixa a cópia reaprende em menos passos que a rede nova em todas as condições (economia de 58% a 66%, nenhuma execução censurada); o esquecimento final fica em 0,46 a 0,50. O `rank_ef` e o `srank_feat` discordam entre si (ver `figures/rodada2_tabela_decisao.md`, seção 4).

## Propostas para depois (status das propostas da rodada 1)

1. ~~B_disjunta com professor fixo e reaprendizado~~ — feito na rodada 2 (`B_fixa`); o gap dela mistura transferência.
2. ~~Controle Hare & Tortoise~~ — feito (`central_HT`, `central_HT_cal`, `central_HT_cal2`, `central_lookahead`); ver os vereditos acima.
3. ~~Partição Dirichlet na tarefa B~~ — feito, nas tarefas A e B.
4. ~~Sonda reformulada~~ — feita (20 passos, mediana do Δlog), mas **não validada** (ver Desvios).
5. Pendentes: grade de λ além da borda (L2 < 1e-4; SP > 0,8), mais de uma semente de calibração, um nível entre 1 e 3, e uma comparação de H_tempo/H_dist em que o `media_sem_particao` tenha de fato vantagem em A a explicar (P_vant) e a rede nova do HT chegue ao mesmo nível em perda **e** em acurácia.

## Como rodar

Python 3.11, PyTorch (CPU), numpy, pandas, matplotlib, pytest. Nenhum download de dados.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy pandas matplotlib pytest
.venv/bin/python -m pytest -q          # 61 testes, ~15 min (os diagnósticos rodam em todos os testes de semente de avaliação)
```

Execução (sempre em CPU, determinística; cada processo usa ~0,3 GB de memória, ajuste `--processos` à sua máquina):

```bash
# demo rápida: semente 1, só tarefa A, sem media_sem_particao (inclui a fase de calibração na semente 1000, todos os níveis)
.venv/bin/python -m experiments.pipeline --quick --processos 3

# completa: sementes 1 a 5, tarefas A, B e B_disjunta, níveis 1, 3 e 10
.venv/bin/python -m experiments.pipeline --fase completo --processos 3

# fases separadas
.venv/bin/python -m experiments.pipeline --fase teste_curto --processos 3   # semente 1000, tarefa A, m em {1,10}
.venv/bin/python -m experiments.pipeline --fase calibracao --processos 3    # calibração, regime m*, seleção de λ
```

A execução retoma de onde parou: CSVs completos e com os mesmos parâmetros são pulados. Figuras e tabela:

```bash
for s in fig_gap fig_norma fig_sonda fig_tradeoff fig_dose fig_k1_mecanismo tabela_decisao; do .venv/bin/python plots/$s.py; done
.venv/bin/python -m experiments.analise_exploratoria_klein            # exploratória: adiantamentos (v1, v2, v3)
.venv/bin/python -m experiments.calibrar_passos_sonda               # exploratória: passos da sonda
.venv/bin/python plots/exploratorio_gap_area.py; .venv/bin/python plots/exploratorio_sonda_log.py
```

### Rodadas 2 e 2b e o cluster

A rodada 2 é CPU, 1 thread por processo. Local: `python -m experiments.pipeline_r2 --fase plano|teste_curto|completo --processos 3` (o `completo` recusa começar sem um `teste_curto` com `OK`); a 2b: `python -m experiments.pipeline_r2b --fase ...`. Verificação: `python -m experiments.verificar_rodada2 --log <log>` e `python -m experiments.verificar_rodada2b --log <log>` (`VERIFICAÇÃO: OK` ou `FALHOU`). Tempos medidos no cluster (nó `dl-01`, 20 processos): rodada 2 completo 2326 s (39 min); 2b completo 290 s.

No cluster Recod (Slurm): `slurm/enviar.sh` envia o projeto para `~/plasticidade` (sem `--delete`); no headnode, `bash slurm/rodada2b.sh curto|longo` (ou `rodada2.sh`) submete o job com nome, tempo e processos certos e recusa o `longo` se o teste curto não deu `OK` ou se o longo já rodou; `slurm/buscar.sh --rodada 2|2b` traz os resultados; `slurm/verificar.sh --rodada 2|2b [--buscar]` verifica; `slurm/limpar_cluster.sh [--aplicar]` arquiva logs antigos (não apaga nada). Os scripts `sync_remote.sh` e `fetch_logs_from_remote.sh` do projeto `keys` **não** servem para este projeto (espelham outra pasta com `--delete`).

### Tempo medido

| execução | máquina | tempo de parede |
|---|---|---|
| teste curto (6 execuções, semente 1000, tarefa A, m = 1 e 10, 3 processos) | CPU, 12 núcleos | 1062 s (17,7 min); a execução de `fedavg_iid` em m=10 levou ~1019 s |
| `fedavg_iid`, m=1, T=30, sem diagnósticos / com diagnósticos | 2 processos em paralelo | 163 s / 202 s (+23,6%) |
| **completo** (255 execuções + calibração + seleção, 3 processos) | idem | **38 593 s = 643 min = 10,7 h** (calibração 73 min; sementes 1 a 5: 570 min) |
| **`--quick`** (semente 1, tarefa A, sem `media_sem_particao`, 3 processos, pasta de saída nova, código final) | idem | **3134 s = 52 min** (`fedavg_iid` da semente 1000 em m=1, 3 e 10: 18 min; calibração: 4,4 min; regime: 1,8 min; seleção de λ: 2,9 min; 14 execuções de avaliação: 24,8 min); pico de memória 321 MB |

O completo levou ~1,8× o tempo projetado (~6 h): as medições por execução incluem pausas do computador durante a noite e uma falta de memória do sistema. **O `--quick` foi cronometrado com o código final (52 min, saída 0, sem erros) e reproduz exatamente a rodada 1: os 14 CSVs da semente 1 saem idênticos aos de `results/` e a calibração dá os mesmos passos.** O tempo equivalente sem diagnósticos do completo não foi medido; a projeção do acréscimo dos diagnósticos foi de +7% a +16% (`results/custo_diagnosticos.json`).

## Figuras (`figures/`)

- **`fig_gap.png`** — o gap (contínua − nova) por tarefa, uma linha por condição, com média e IC 95% entre sementes, um painel por tipo de tarefa. Mostra a perda se acumulando ao longo das 30 tarefas nas condições sem regularização e a manutenção em L2/SP.
- **`fig_norma.png`** — norma dos parâmetros e distância percorrida por tarefa, por condição. Mostra que a norma cresce em todas as condições sem regularização, inclusive no FedAvg, que tem a maior.
- **`fig_sonda.png`** — razão da sonda de alvos aleatórios (cópia sobre rede nova) em escala symlog. **Atenção:** a razão acompanha a escala das saídas da rede e não o gap, e chega a 1e6; use como exploratória, não como medida de plasticidade (veja os Desvios).
- **`fig_tradeoff.png`** — plasticidade (eixo x) contra esquecimento final (eixo y) em B_disjunta, com IC 95% e a faixa do limite de 5 pontos percentuais. Os pontos ficam quase numa reta e todos perto do máximo possível: o esquecimento está saturado.
- **`fig_dose.png`** — a figura principal: o gap em função da acurácia da rede nova de cada condição, nos níveis m = 1, 3 e 10, um painel por tarefa. Mostra que o gap só vira negativo quando o orçamento é alto, e que em m=10 a rede nova satura em 1,000 (gaps ≤ 0). Os rótulos "m=" se sobrepõem nos painéis B e B_disjunta.
- **`fig_k1_mecanismo.png`** (m=10) e **`fig_k1_mecanismo_mstar.png`** (m=3) — por tarefa: dormentes, norma por camada, gradiente por camada, rank efetivo do gradiente, maior autovalor da Hessiana e a acurácia, para `sem_media_k1`, `fedavg_iid`, `central_passos` e `central_aprendizado`, com a rede nova tracejada. Serve à pergunta sobre o mecanismo da perda; os diagnósticos são exploratórios.
- **`fig_k1_por_semente.png`** (e `_mstar`) — as mesmas medidas, uma linha por semente, para comparar a ordem temporal sem o borrão da média.
- **`exploratorio_fig_gap_curvas.png`** e **`exploratorio_fig_sonda_log.png`** (+ tabelas `exploratorio_*.md`) — curvas do gap por tarefa e sonda em log, por nível e tarefa; **exploratórias**, escritas depois dos resultados.
- **Rodada 2 e 2b** (`figures/rodada2_*`, `rodada2b_*`): `rodada2_tabela_decisao.md` e `rodada2b_tabela_decisao.md` (métrica principal por condição com IC, vereditos, calibrações, P_vant, elegibilidade, previsão × resultado); `rodada2_gap_por_tarefa.png`, `rodada2_h_het.png`, `rodada2_longo_colapso.png`, `rodada2_reaprendizado.png`, `rodada2_rank.png`, `rodada2b_tabela_decisao_gap.png`; e `rodada2_pos_resultados.md` (análises exploratórias posteriores: janela do `media_sem_particao`, `fedavg_iid` entre rodadas, nível das redes em H_het).
- **`tabela_decisao.md`** — a métrica principal por condição e tarefa, as marcações de B_disjunta, a dose-resposta, a calibração, a seleção de λ, a sanidade `sem_media_k1` × `central_passos` e a linha final com os critérios atendidos (nenhum).

## O que há no repositório

- `PREREGISTRO.md` — hipóteses, critérios e **todos os desvios**, com data. As leituras escritas depois dos resultados estão marcadas como tais.
- `results_rodada2/` (155 CSVs + `calibracao/` da semente 1000) e `results_rodada2b/` (30 CSVs + `calibracao/`): resultados das rodadas 2 e 2b; `slurm/` os scripts do cluster; `logs/cluster/` os logs dos jobs.
- `src/` código (dados, modelo, métricas, diagnósticos, FedAvg, condições); `experiments/` comandos; `tests/` testes; `plots/` figuras; `results/` os 255 CSVs (+ matrizes de esquecimento, cossenos por rodada e `results/calibracao/` da semente 1000, que nunca entra na análise).
- `CLAUDE.md` e `.claude/agents/` — a divisão de trabalho entre a sessão principal (único editor de `src/`, `experiments/`, `tests/`) e os agentes `executor-experimentos`, `gerador-graficos` e `auditor-metricas` (somente leitura).
- `MANIFESTO_results.sha256` — hash de cada arquivo de `results/`.
- `notes.md`, `analise_exploratoria_klein.md`, `analise_exploratoria_sonda_passos.md`, `task_plan.md` — observações, análises exploratórias e plano.

## Limites

Cinco sementes; um problema sintético pequeno; m\* definido numa só semente de calibração; λ de L2 e SP na borda da grade; mitigações só em m=3; rede nova no teto em m=10; esquecimento e sonda sem poder de discriminação aqui. Rodadas 2 e 2b: cinco sementes (três no braço longo), calibrações feitas numa só semente, μ quase sempre na borda da grade, e a hipótese central ficou não avaliável. Nada disso foi ajustado depois de ver os resultados: o que mudou depois dos dados está em "Desvios" do pré-registro.
