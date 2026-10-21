# Plasticidade: FedAvg contra treino centralizado

> **Dúvida sobre algum nome?** Veja o [GLOSSARIO.md](GLOSSARIO.md): uma linha em português simples para cada condição, variável, coluna, hipótese e figura.

Um experimento pequeno e pré-registrado para responder: **a vantagem do FedAvg sobre o treino centralizado em preservar plasticidade sobrevive a controles de passos, de partição dos dados e de regularização?**

## O que é "plasticidade" aqui

Plasticidade é a **capacidade de uma rede de continuar aprendendo**, medida contra uma rede nova: depois de treinar em muitas tarefas, a rede contínua aprende a tarefa seguinte tão bem quanto aprenderia uma rede recém-sorteada? É o sentido de Dohare et al. (2024) e Lyle et al. (2022). **Não** é o equilíbrio entre estabilidade e plasticidade da literatura de aprendizado federado contínuo (aprender sem esquecer).

A medida central é o **gap**: acurácia da rede contínua menos a da rede nova, na mesma tarefa (pareado). Gap negativo = perda de plasticidade. A **métrica principal** é o gap médio nas 5 últimas tarefas.

## Leitura rápida dos resultados (5 sementes, nível m\* = 3)

Tudo abaixo é do `PREREGISTRO.md` e da `figures/tabela_decisao.md`; a auditoria dos resultados conferiu a tabela contra os CSVs e as regras contra o texto do pré-registro. **O resultado é "nenhum critério atendido"** (H_reg, H_part, H_real e H_orc: todas "não atendidas", com margens largas), e isso é um resultado pré-registrado.

Gap médio nas 5 últimas tarefas, m=3 (média entre sementes, IC 95% por bootstrap):

| condição | tarefa A | tarefa B |
|---|---|---|
| `fedavg_iid` | −0,062 [−0,090; −0,033] | −0,001 [−0,004; +0,003] |
| `media_sem_particao` | −0,006 [−0,014; +0,006] | +0,009 [+0,007; +0,011] |
| `sem_media_k1` | −0,332 [−0,475; −0,238] | −0,066 [−0,076; −0,054] |
| `central_passos` | −0,463 [−0,583; −0,339] | −0,067 [−0,073; −0,062] |
| `central_aprendizado` | −0,452 [−0,585; −0,318] | −0,083 [−0,097; −0,069] |
| `central_L2` | +0,100 [+0,089; +0,113] | +0,028 [+0,026; +0,031] |
| `central_SP` | +0,148 [+0,142; +0,154] | +0,041 [+0,037; +0,044] |

O que **pode** ser afirmado (n = 5 sementes, um problema sintético pequeno):
- Em m=3, o `fedavg_iid` perde menos plasticidade que o `central_aprendizado` calibrado no mesmo nível de aprendizado da rede nova, em A e em B. **H_orc é refutada.**
- L2 e shrink-and-perturb, mesmo com λ na borda da grade (regularização fraca), preservam a plasticidade e ficam **acima** do `fedavg_iid`. H_reg e H_real não são atendidas.
- H_part não é atendida: a média sem partição (`media_sem_particao`) também protege, e (i) falha na tarefa A.
- Há dose-resposta com o orçamento m nas condições sem regularização, na tarefa A; o regime de perda aparece a partir de m=3.

O que **não** pode ser afirmado: nada sobre esquecimento ou trade-off (o eixo está saturado, veja abaixo), nada sobre a sonda de alvos aleatórios como medida de plasticidade, que `sem_media_k1` é equivalente a `central_passos`, comparações em m=10 (rede nova no teto, 1,000), a robustez de m\* (definido numa só semente de calibração) nem generalizações para fora deste problema. Em B e B_disjunta a perda é pequena.

**Esquecimento.** Nas tarefas **A e B originais o esquecimento não é medido**, porque o mesmo X recebe rótulos conflitantes a cada tarefa. Ele só existe na tarefa **B_disjunta** (cada tarefa com os seus próprios pontos), e ali está **saturado**: 0,84 a 0,88 em todas as condições, com acurácia nas tarefas antigas ao nível do acaso. O limite de 5 pontos percentuais do Adendo 1 nunca restringiu nada, então "ganho válido" quer dizer só "melhora plasticidade".

**Diagnósticos de Klein et al. (exploratórios).** Veja `analise_exploratoria_klein.md`. Em resumo: nas condições que perdem plasticidade, a norma e o rank efetivo mudam algumas tarefas antes da queda da acurácia, mas a magnitude da norma e da curvatura **não** separa as condições (o `fedavg_iid` tem as maiores, e perde menos); o que separa são neurônios dormentes e rank das features. Não prova causalidade.

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
.venv/bin/python -m experiments.analise_exploratoria_klein --nivel 3 --tarefa A   # exploratória
```

### Tempo medido

| execução | máquina | tempo de parede |
|---|---|---|
| teste curto (6 execuções, semente 1000, tarefa A, m = 1 e 10, 3 processos) | CPU, 12 núcleos | 1062 s (17,7 min); a execução de `fedavg_iid` em m=10 levou ~1019 s |
| `fedavg_iid`, m=1, T=30, sem diagnósticos / com diagnósticos | 2 processos em paralelo | 163 s / 202 s (+23,6%) |
| **completo** (255 execuções + calibração + seleção, 3 processos) | idem | **38 593 s = 643 min = 10,7 h** (calibração 73 min; sementes 1 a 5: 570 min) |

O completo levou ~1,8× o tempo projetado (~6 h): as medições por execução incluem pausas do computador durante a noite e uma falta de memória do sistema. O `--quick` **não** foi cronometrado com o código final; a demo rápida inclui a calibração em m=10 (a parte mais lenta), então conte com cerca de 1 h. O tempo equivalente sem diagnósticos do completo não foi medido; a projeção do acréscimo dos diagnósticos foi de +7% a +16% (`results/custo_diagnosticos.json`).

## Figuras (`figures/`)

- **`fig_gap.png`** — o gap (contínua − nova) por tarefa, uma linha por condição, com média e IC 95% entre sementes, um painel por tipo de tarefa. Mostra a perda se acumulando ao longo das 30 tarefas nas condições sem regularização e a manutenção em L2/SP.
- **`fig_norma.png`** — norma dos parâmetros e distância percorrida por tarefa, por condição. Mostra que a norma cresce em todas as condições sem regularização, inclusive no FedAvg, que tem a maior.
- **`fig_sonda.png`** — razão da sonda de alvos aleatórios (cópia sobre rede nova) em escala symlog. **Atenção:** a razão acompanha a escala das saídas da rede e não o gap, e chega a 1e6; use como exploratória, não como medida de plasticidade (veja os Desvios).
- **`fig_tradeoff.png`** — plasticidade (eixo x) contra esquecimento final (eixo y) em B_disjunta, com IC 95% e a faixa do limite de 5 pontos percentuais. Os pontos ficam quase numa reta e todos perto do máximo possível: o esquecimento está saturado.
- **`fig_dose.png`** — a figura principal: o gap em função da acurácia da rede nova de cada condição, nos níveis m = 1, 3 e 10, um painel por tarefa. Mostra que o gap só vira negativo quando o orçamento é alto, e que em m=10 a rede nova satura em 1,000 (gaps ≤ 0). Os rótulos "m=" se sobrepõem nos painéis B e B_disjunta.
- **`fig_k1_mecanismo.png`** (m=10) e **`fig_k1_mecanismo_mstar.png`** (m=3) — por tarefa: dormentes, norma por camada, gradiente por camada, rank efetivo do gradiente, maior autovalor da Hessiana e a acurácia, para `sem_media_k1`, `fedavg_iid`, `central_passos` e `central_aprendizado`, com a rede nova tracejada. Serve à pergunta sobre o mecanismo da perda; os diagnósticos são exploratórios.
- **`fig_k1_por_semente.png`** (e `_mstar`) — as mesmas medidas, uma linha por semente, para comparar a ordem temporal sem o borrão da média.
- **`tabela_decisao.md`** — a métrica principal por condição e tarefa, as marcações de B_disjunta, a dose-resposta, a calibração, a seleção de λ, a sanidade `sem_media_k1` × `central_passos` e a linha final com os critérios atendidos (nenhum).

## O que há no repositório

- `PREREGISTRO.md` — hipóteses, critérios e **todos os desvios**, com data. As leituras escritas depois dos resultados estão marcadas como tais.
- `src/` código (dados, modelo, métricas, diagnósticos, FedAvg, condições); `experiments/` comandos; `tests/` testes; `plots/` figuras; `results/` os 255 CSVs (+ matrizes de esquecimento, cossenos por rodada e `results/calibracao/` da semente 1000, que nunca entra na análise).
- `CLAUDE.md` e `.claude/agents/` — a divisão de trabalho entre a sessão principal (único editor de `src/`, `experiments/`, `tests/`) e os agentes `executor-experimentos`, `gerador-graficos` e `auditor-metricas` (somente leitura).
- `notes.md`, `analise_exploratoria_klein.md`, `task_plan.md` — observações, análise exploratória e plano.

## Limites

Cinco sementes; um problema sintético pequeno; m\* definido numa só semente de calibração; λ de L2 e SP na borda da grade; mitigações só em m=3; rede nova no teto em m=10; esquecimento e sonda sem poder de discriminação aqui. Nada disso foi ajustado depois de ver os resultados: o que mudou depois dos dados está em "Desvios" do pré-registro.
