# Notas

## Balanço de classes dos professores (semente 0, checagem pedida pela auditoria nº 3, P2-6)
Classe majoritária / classes usadas, tarefas t = 0,1,2,5,10,20,29:
- B:          0.62/8, 0.28/9, 0.32/8, 0.41/7, 0.53/8, 0.44/7, 0.68/7
- B_disjunta: 0.54/8, 0.32/10, 0.28/9, 0.48/7, 0.43/6, 0.48/6, 0.69/7
Consequência: um preditor constante da classe majoritária já acerta 28% a 69%. Em B_disjunta, parte da
acurácia retida em tarefas antigas pode vir de prever a classe majoritária; ler esquecimento com essa ressalva.
O desenho (professor com init padrão) é pré-registrado e não foi alterado.

## Quick (1 semente, A e B_disjunta, sem media_sem_particao): 604 s de parede com 8 processos
Etapas: fedavg_iid sem 0 94,5 s; calibração 261,0 s; seleção 92,6 s; resto 153,8 s.
Calibração: A alvo 0,3873 -> 420 passos (8,4 épocas), erro -0,0008; B_disjunta alvo 0,8318 -> 405 passos (8,1 épocas), erro -0,0031.
Seleção: L2 lambda=1e-4 (A e B_disjunta); SP lambda=0,8 (A e B_disjunta). (Valores congelados na semente 0.)

## 2026-10-20 · Material exploratório do desenho antigo apagado (causa desconhecida)
`results_v1_exploratorio/` (87 arquivos: CSVs, matrizes e JSONs da semente 0 do desenho antigo, mais o pipeline completo interrompido) e `logs/` apareceram vazios por volta das 16:45. Nenhum código do repositório apaga essas pastas e a sessão não executou `rm`; pode ter sido limpeza manual. Nada disso alimenta a análise confirmatória (semente 0 e desenho antigo são exploratórios). Reproduzível: `git worktree add` do commit 0b1c019 e rodar `python -m experiments.pipeline --quick --tarefas A B_disjunta` (~10 min, CPU, determinístico). Os números que importam estão em PREREGISTRO.md (Desvios v3) e nesta sessão.
