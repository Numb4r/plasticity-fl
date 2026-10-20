# Plano: FedAvg vs centralizado, plasticidade (Dohare 2024, Lyle 2022)

## Objetivo
Responder, até quinta-feira, se a vantagem do FedAvg em preservar plasticidade sobrevive a controles de passos, de partição e de regularização.

## Fases
- [x] F0 Ambiente: checar python/torch/uv/núcleos de CPU
- [x] F1 Etapa 0: CLAUDE.md + .claude/agents/ (3 agentes)
- [x] F2 Etapa 1: PREREGISTRO.md + git init + commit (ANTES de qualquer código de experimento)
- [x] F3 Etapa 2 código: src/ (sementes, dados, modelo, métricas, sondas, treino, federado, condições)
- [x] F4 Testes (6 pedidos) passando
- [ ] F5 Auditoria (auditor-metricas) do código ANTES de rodar
- [ ] F6 Rodar: fedavg_iid sem 0 -> calibrar central_aprendizado -> selecionar L2/SP -> congelar -> resto das sementes
- [ ] F7 Auditoria dos resultados (auditor-metricas) ANTES de interpretar
- [ ] F8 Etapa 3: figuras + tabela_decisao.md (gerador-graficos)
- [ ] F9 README (definição de plasticidade, --quick, completo, tempo medido)

## Decisões
- Único editor de src/, experiments/, tests/: sessão principal.
- Desvios do pré-registro só em PREREGISTRO.md > "Desvios".

## Erros
- auditor-metricas/executor/gerador não carregam como subagent_type na sessão em que foram criados: reiniciar a sessão ou usar general-purpose com o corpo do agente.
- Python do sistema é 3.10; criado .venv com 3.11 (uv) e torch CPU 2.14.1.

## Status
F5: auditoria pré-execução em andamento (auditor-metricas indisponível na sessão; usando general-purpose/opus com o corpo do auditor).
