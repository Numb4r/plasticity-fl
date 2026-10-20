# plots/

Rode da raiz do projeto (`--results` padrão `results`, `--saida` padrão `figures`):

    .venv/bin/python plots/fig_gap.py          # figures/fig_gap.png      (painéis A, B, B_disjunta)
    .venv/bin/python plots/fig_norma.py        # figures/fig_norma.png    (norma e ‖Δθ‖ na tarefa (após SP))
    .venv/bin/python plots/fig_sonda.py        # figures/fig_sonda.png
    .venv/bin/python plots/fig_tradeoff.py     # figures/fig_tradeoff.png (só B_disjunta)
    .venv/bin/python plots/tabela_decisao.py   # figures/tabela_decisao.md

Teste com CSVs sintéticos em `tempfile.mkdtemp()` (não toca em `results/`; `PLAST_TEST_KEEP=1` mantém o diretório):

    .venv/bin/python plots/test_fake.py

## Entrada
`analise.carregar` lê só os CSVs principais `results/<cond>_<tipo>_<semente>.csv` e usa as colunas
`cond`, `tipo`, `semente`, `tarefa` (nunca o nome do arquivo). Ignora `*_rodadas.csv`, `*_matriz.csv`,
`*.params.json` e subpastas. B_disjunta tem as colunas extras `acc_media_vistas`, `esquecimento`, `bwt`;
em A e B elas não existem.

## Regras (Adendo 1 incluído)
- Esquecimento por semente = coluna `esquecimento` na última tarefa da condição (T-1; 11 em `media_sem_particao`), mesmo bootstrap da métrica principal.
- Limite: média_esq(cond) ≤ média_esq(central_aprendizado) + 0,05. Marcação em B_disjunta: "ganho válido", "ganho com violação de esquecimento", "sem ganho" (melhora = inf do IC da condição > sup do IC de central_aprendizado).
- H_real exige também que `fedavg_iid` respeite o limite; sem dados de B_disjunta fica "não avaliável (falta B_disjunta)".
- A linha final separa "Sem vantagem a explicar" (pré-condição de H_part falhou) de "Dados insuficientes" (falta dado ou IC indefinido).
- As 18 ambiguidades e a leitura adotada estão no topo de `analise.py`. Destaque: o Adendo diz "ganho com violação" para quem "passa o limite"; implementado como "ultrapassa".

Com 1 semente o IC é indefinido: as figuras mostram só a média e a tabela escreve "IC indefinido (n=1)".
