# plots/ (esquema v3)

Rode da raiz do projeto (`--results` padrão `results`, `--saida` padrão `figures`):

    .venv/bin/python plots/fig_dose.py          # figures/fig_dose.png (FIGURA PRINCIPAL: dose-resposta por nível m)
    .venv/bin/python plots/fig_k1_mecanismo.py  # figures/fig_k1_mecanismo.png (m=10: dormentes, norma, rank; K=1 vs fedavg_iid)
    .venv/bin/python plots/fig_gap.py           # figures/fig_gap.png      (nível m*)
    .venv/bin/python plots/fig_norma.py         # figures/fig_norma.png    (nível m*)
    .venv/bin/python plots/fig_sonda.py         # figures/fig_sonda.png    (nível m*)
    .venv/bin/python plots/fig_tradeoff.py      # figures/fig_tradeoff.png (só B_disjunta, nível m*)
    .venv/bin/python plots/tabela_decisao.py    # figures/tabela_decisao.md

Teste com CSVs sintéticos em `tempfile.mkdtemp()` (não toca em `results/`; `PLAST_TEST_KEEP=1` mantém o diretório):

    .venv/bin/python plots/test_fake.py

## Código de saída
- 0: figuras/tabela geradas.
- 1: sem resultados em `results/` (diretório vazio, CSVs sem as colunas-chave/`gap`/`acc_nova`, ou só sementes fora de 1 a 5).
  Os scripts imprimem "sem resultados em results/" em stderr e saem antes de criar figuras ou a pasta de saída.

## Entrada
- `analise.carregar` lê só `results/*.csv` (sem subpastas): `<cond>_m<nivel>_<tarefa>_<semente>.csv`. Usa as colunas
  `cond`, `tipo`, `semente`, `tarefa`, `nivel` (nunca o nome do arquivo). Ignora `*_rodadas.csv`, `*_matriz.csv`,
  `*.params.json`. Mantém só as sementes 1 a 5 (`SEMENTES_AVALIACAO`, réplica de `src.config`) e avisa em stderr
  quais foram descartadas (0, 6, 1000 etc.), mesmo na raiz.
- `results/calibracao/regime.json` dá m*; `calibracao_<tarefa>_m<m>.json` e `selecao_<L2|SP>_<tarefa>_m<m*>.json`
  alimentam as tabelas (e) e (f).

## Nível de decisão
- Decisões (H_reg, H_part, H_real, H_orc, cláusula de esquecimento, rótulos de ganho) e figuras antigas usam m*.
- m* nulo ou regime.json ausente: decisões "não avaliável (regime sem perda; ...)"; figuras antigas no maior nível
  presente, com aviso no título; fig_dose e fig_k1_mecanismo continuam.
- Tabela (b)/(b')/(c)/(d): `n_ref` = nº de referências pareadas por semente na janela (2 vs 5 na janela 7..11).
- Aviso na linha final se `calibracao_<A|B>_m<m*>.json` tem ok=false.
- H_orc: no nível m*, média da métrica principal do fedavg_iid dentro do IC 95% fechado de central_aprendizado, em A e B.
- A linha final cita o nível de cada hipótese e separa "sem vantagem a explicar" de "dados insuficientes".
- As 27 ambiguidades e a leitura adotada estão no topo de `analise.py`.
  Destaque: a janela 7..11 de H_part tem só 2 referências (índices 8 e 11) nas condições de T=30; só `media_sem_particao` tem 5.

Com 1 semente o IC é indefinido: as figuras mostram só a média e a tabela escreve "IC indefinido (n=1)".
