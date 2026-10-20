# Plano: FedAvg vs centralizado, plasticidade (Dohare 2024, Lyle 2022)

## Objetivo
Responder, até quinta-feira, se a vantagem do FedAvg em preservar plasticidade sobrevive a controles de passos, de partição e de regularização.

## Fases
- [ ] F0 Ambiente: checar python/torch/uv/núcleos de CPU
- [ ] F1 Etapa 0: CLAUDE.md + .claude/agents/ (3 agentes)
- [ ] F2 Etapa 1: PREREGISTRO.md + git init + commit (ANTES de qualquer código de experimento)
- [ ] F3 Etapa 2 código: src/ (sementes, dados, modelo, métricas, sondas, treino, federado, condições)
- [ ] F4 Testes (6 pedidos) passando
- [ ] F5 Auditoria (auditor-metricas) do código ANTES de rodar
- [ ] F6 Rodar: fedavg_iid sem 0 -> calibrar central_aprendizado -> selecionar L2/SP -> congelar -> resto das sementes
- [ ] F7 Auditoria dos resultados (auditor-metricas) ANTES de interpretar
- [ ] F8 Etapa 3: figuras + tabela_decisao.md (gerador-graficos)
- [ ] F9 README (definição de plasticidade, --quick, completo, tempo medido)

## Decisões
- Único editor de src/, experiments/, tests/: sessão principal.
- Desvios do pré-registro só em PREREGISTRO.md > "Desvios".

## Erros
(nenhum ainda)

## Status
Fase F0/F1.
