# plots/ (esquema v3)

Rode da raiz do projeto (`--results` padrão `results`, `--saida` padrão `figures`):

    .venv/bin/python plots/fig_dose.py          # figures/fig_dose.png (FIGURA PRINCIPAL: dose-resposta por nível m)
    .venv/bin/python plots/fig_k1_mecanismo.py  # figures/fig_k1_mecanismo.png (m=10) e fig_k1_mecanismo_mstar.png (só se m* existir e != 10)
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
- Tabela (b)/(b')/(c)/(d): `n_ref` = nº de referências pareadas por semente na janela (5 em todas as condições,
  inclusive na janela 7..11). A tabela (b') só acrescenta uma nota se n_ref divergir entre condições.
- Aviso na linha final se `calibracao_<A|B>_m<m*>.json` tem ok=false.
- H_orc: no nível m*, média da métrica principal do fedavg_iid dentro do IC 95% fechado de central_aprendizado, em A e B.
- A linha final cita o nível de cada hipótese e separa "sem vantagem a explicar" de "dados insuficientes".
- As 27 ambiguidades e a leitura adotada estão no topo de `analise.py`.
  Destaque (ambiguidade 27, resolvida pela decisão (ii) da Errata, 2026-10-20): a rede nova de referência existe em
  t%3==2, nas 5 últimas e nos índices 7..11 (`tarefas_com_nova`: 16 tarefas em T=30, 7 em T=12); a janela 7..11 de
  H_part tem 5 referências pareadas em todas as condições.

Com 1 semente o IC é indefinido: as figuras mostram só a média e a tabela escreve "IC indefinido (n=1)".

## fig_k1_mecanismo (Desvios v3.1, diagnósticos de Klein et al. 2026)
- 4 condições (sem_media_k1, fedavg_iid, central_passos, central_aprendizado) x colunas A, B, B_disjunta; 14 linhas (+ a figura por semente abaixo):
  dormentes tau=0 e 0,025, norma, rank_ef (originais); norma dos pesos l1/l2/l3 (`fim_norma_l*_w`); norma L2 do
  gradiente l1/l2/l3 (`ini_grad_l2_l*_w`); srank do gradiente (`ini_grad_srank`); maior autovalor da Hessiana no início
  (`ini_hess_max`) e no fim (`fim_hess_max`); `acc_cont` na última linha, para comparar no tempo se norma/autovalor
  sobem antes da queda da acurácia.
- Rede nova, regra de desenho (escolha também escrita na legenda da figura):
  - linhas de INÍCIO (`nova_ini_*`): a rede nova é idêntica nas 4 condições (mesma semente de init_nova, sem SP em
    nenhuma) -> UMA linha cinza tracejada (marcador x) = média entre as 4 condições (por semente, depois bootstrap);
  - linhas de FIM (`nova_fim_*`: norma por camada, Hessiana no fim) e `acc_cont` (referência `acc_nova`): a rede nova é
    treinada com o protocolo da própria condição -> uma linha fina tracejada POR CONDIÇÃO (cor e marcador da condição,
    média e valores daquela condição, sem faixa de IC).
  Só nas tarefas com rede nova (`tarefas_com_nova`).
- Hessiana: marcador vazado = ao menos uma semente com `hess_conv == 0` naquele ponto (power iteration não convergiu);
  vale para o contínuo, para a linha da rede nova por condição (`nova_fim_hess_conv`) e para a cinza (`nova_ini_hess_conv`
  de qualquer condição). A entrada de legenda só aparece se houver ponto não convergido.
- `hess_max` é o autovalor de maior módulo COM SINAL: a média entre sementes pode ficar perto de 0 se o sinal muda.
- Norma dos parâmetros por camada: só pesos (os vieses estão nos CSVs: `norma_l*_b`).
- Tempo de medição: `ini_*` na tarefa t é medido ANTES de treinar t (estado após t-1 e após o shrink-and-perturb);
  `fim_*` e `acc_cont`, DEPOIS.
- `figures/fig_k1_por_semente.png` (e `_mstar.png`, mesma regra de níveis e de aviso): linhas fina por semente (sem média,
  sem IC), por condição (4) e tarefa (3), em trios de linhas `fim_hess_max`, `fim_norma_l2_w`, `acc_cont`, para ver semente
  a semente se norma/Hessiana sobem antes da queda da acurácia. Mesma regra de dados ausentes ("sem diagnósticos" em
  hess/norma; "sem dados" em acc_cont). Linha y=0 nas linhas da Hessiana; marcador vazado = `hess_conv == 0`.
- Colunas novas ausentes (execuções antigas) ou todas NaN: o painel diz "sem diagnósticos"; o script não quebra.
- m* nulo/ausente: só m=10, com aviso no título. Métricas diagnósticas: fora de regras de decisão e da tabela_decisao.md.
- Custo: cerca de 30 s por figura (bootstrap de 10 000 reamostragens por ponto); test_fake.py leva ~8 min.
