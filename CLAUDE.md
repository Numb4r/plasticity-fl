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

## Proteção de `results/` (incidente de 2026-10-20; ver "Desvios" no PREREGISTRO.md)
- **Nenhum agente apaga, move, renomeia, sobrescreve ou trunca arquivos em `results/`** (incluindo `results/calibracao/`), em `results_*` (arquivos mortos) nem em `logs/`; e só o `gerador-graficos` regrava `plots/` e `figures/`, mais ninguém. "Limpar artefatos descartáveis" nunca vale para essas pastas. Exceção só com instrução escrita do autor citando o caminho.
- Depois de cada execução **completa**, o pipeline deixa `results/` **somente leitura** (`experiments/protecao.py`: arquivos 0444, pastas 0555). Para rodar de novo, use outra `--saida`; reabrir exige `--reabrir-resultados` e autorização do autor.
- `MANIFESTO_results.sha256` guarda o hash de cada arquivo de `results/`. Conferir com `sha256sum -c MANIFESTO_results.sha256`. Se mudar, é incidente: registrar em "Desvios".
- Agentes **leem** `results/`; o `gerador-graficos` escreve só em `plots/` e `figures/`; o `executor-experimentos` só roda os scripts (que gravam CSVs novos, nunca por cima de arquivos existentes); o `auditor-metricas` não escreve em lugar nenhum.
