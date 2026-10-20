# Plano: FedAvg vs centralizado, plasticidade (Dohare 2024, Lyle 2022)

## Objetivo
Responder, até quinta-feira, se a vantagem do FedAvg em preservar plasticidade sobrevive a controles de passos, de partição e de regularização.

## Fases
- [x] F0 Ambiente: checar python/torch/uv/núcleos de CPU
- [x] F1 Etapa 0: CLAUDE.md + .claude/agents/ (3 agentes)
- [x] F2 Etapa 1: PREREGISTRO.md + git init + commit (ANTES de qualquer código de experimento)
- [x] F3 Etapa 2 código: src/ (sementes, dados, modelo, métricas, sondas, treino, federado, condições)
- [x] F4 Testes (6 pedidos) passando
- [x] F5 Auditoria (auditor-metricas) do código ANTES de rodar
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
Passo 5 PARADO: aguardando o usuário preencher "Previsão do autor" no PREREGISTRO.md (auditoria 4: sem P0/P1 no código). Passo 4 (figuras) com o gerador em andamento.

## Fila (ordem fixada pelo autor)
1. Revisão final curta (auditor) em andamento.
2. Autor preenche "Previsão do autor" -> commit SÓ da previsão -> anotar o hash no PREREGISTRO -> lançar teste_curto.
   Se central_passos m=10 não tiver gap <= -0,10: PARAR e mostrar o log.
3. Depois do teste_curto, ANTES do completo: Adendo v3.1 (métricas de Klein et al. 2026), nesta ordem:
   a) PREREGISTRO.md > Desvios (data de hoje) antes de mexer no código; métricas diagnósticas, fora dos critérios e da tabela_decisao;
   b) métricas em src/ (norma por camada, grad L1/L2 por camada, espectro do gradiente com 64 pontos, autovalor máx. da Hessiana por power iteration);
   c) 4 testes novos; d) medir custo em fedavg_iid m=1 (se >10%: 20 iterações e AVISAR antes de mudar qualquer outra coisa);
   e) fig_k1_mecanismo ampliada (gerador); f) revisão do auditor; g) só então o completo.
