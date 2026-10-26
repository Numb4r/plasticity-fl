# Projeto: o FedAvg preserva plasticidade melhor que o treino centralizado?

Pergunta: a vantagem do FedAvg em preservar plasticidade sobrevive a controles de passos, de partição dos dados e de regularização?

Plasticidade aqui é a capacidade da rede de continuar aprendendo, medida contra uma rede nova (Dohare et al., 2024; Lyle et al., 2022). Não é o equilíbrio estabilidade-plasticidade do FL contínuo.

## Layout
- `src/`: código da biblioteca (dados, modelo, métricas, sondas, treino, federado, condições)
- `experiments/`: scripts de linha de comando (rodar, calibrar, selecionar)
- `tests/`: pytest
- `results/`: CSVs `<condicao>_<tarefa>_<semente>.csv` (saída dos experimentos)
- `plots/`: scripts de figuras e tabela; `figures/`: PNGs e `tabela_decisao.md`
- `PREREGISTRO.md`: hipóteses e critérios, congelado antes dos resultados

## Regras do projeto
- Código com comentários em português, simples, reprodutível (toda aleatoriedade sai de `src/sementes.py`).
- Não simplificar nem remover controles, sementes, sondas ou testes para encurtar o código.
- Não ajustar código depois de ver um resultado para "consertar" o resultado. Toda mudança vai para "Desvios" em `PREREGISTRO.md`, com data e motivo.
- Pedir revisão ao `auditor-metricas` antes de interpretar qualquer resultado.
- **Diário:** `DIARIO.md` (raiz) é o diário de bordo em texto corrido. Acrescentar uma entrada a cada etapa (implementação, auditoria, execução, resultado, análise, decisão do autor), com a data em que foi escrita, o que foi feito, o que mudou e por quê, os resultados e os insights. Ao falar de código, descrever em palavras quais funções foram afetadas. Não substitui o `PREREGISTRO.md`.

## Agentes
Definidos em `.claude/agents/`.

| Agente | Quando delegar | Pode editar |
|---|---|---|
| `executor-experimentos` | Rodar experimentos, varreduras de sementes, checar logs e se cada execução gerou o CSV esperado | nada (só lê e roda; os CSVs e logs são escritos pelos scripts) |
| `gerador-graficos` | Criar ou ajustar figuras e tabelas a partir de `results/` | só `plots/` e `figures/` |
| `auditor-metricas` | Depois de qualquer mudança em métricas, sondas, FedAvg ou controles, e sempre antes de interpretar resultados | nada (somente leitura) |

Regras de coordenação:
1. A sessão principal é a única que edita `src/`, `experiments/` e `tests/`.
2. Tudo o que não edita os mesmos arquivos roda em paralelo (por exemplo: o executor roda sementes enquanto o auditor revisa o código; o gerador-graficos só lê CSVs de execuções já terminadas).
3. Dois agentes nunca editam o mesmo arquivo ao mesmo tempo.
4. O executor roda uma semente por processo em paralelo, respeitando o número de núcleos.
5. Se o auditor apontar um problema depois de ver resultados, a correção vai para "Desvios" em `PREREGISTRO.md`; nenhum agente altera o pré-registro.
