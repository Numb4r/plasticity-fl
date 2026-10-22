#!/bin/bash
# Arruma a bagunça do cluster depois dos jobs de 2026-10-22. NÃO apaga nada: só move logs para um arquivo
# (logs/arquivo/ no cluster e logs/cluster/arquivo/ aqui) e restaura um único arquivo de tempo da rodada 2.
#   slurm/limpar_cluster.sh            # mostra o que faria (dry-run, padrão)
#   slurm/limpar_cluster.sh --aplicar  # faz
# O que faz:
#  1. recusa aplicar se houver job seu na fila/rodando;
#  2. restaura ~/plasticidade/results_rodada2/tempo_completo.json com a versão do commit c8c6143 (os jobs 115961 e
#     115991 rodaram o completo da rodada 2 por engano e o sobrescreveram com tempos falsos; nenhum CSV mudou);
#  3. arquiva os logs dos jobs que não valem: 115961 e 115991 (completo da rodada 2 rodado por engano, nome r2b),
#     115992 e 115993 (completos da 2b recusados pelo gate: o teste curto ainda não tinha dado OK);
#  4. renomeia o log do job 115995 (o completo VÁLIDO da 2b, nomeado r2b_teste_longo) para slurm-r2b_longo-115995.out.
# Não toca em results_rodada2/ (exceto o tempo_completo.json) nem em results_rodada2b/.
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
APLICAR=0; [[ "${1:-}" == "--aplicar" ]] && APLICAR=1
ALVO="yuri.faria@ssh.recod.ic.unicamp.br"
RUINS=(r2b_completo-115961 r2b_completo-115991 r2b_completo-115992 r2b_completo-115993)
VALIDO_ORIG="r2b_teste_longo-115995"; VALIDO_NOVO="r2b_longo-115995"
TEMPO_ORIG="$(git -C "$RAIZ" show c8c6143:results_rodada2/tempo_completo.json)" || { echo "não achei o commit c8c6143" >&2; exit 2; }

[[ $APLICAR -eq 0 ]] && echo "[dry-run] nada será alterado. Use --aplicar para executar."
FILA="$(ssh -o BatchMode=yes headnode 'squeue -u $USER -h' 2>/dev/null | grep -v "█\|░\|Welcome\|alias" | sed '/^ *$/d')"
if [[ -n "$FILA" ]]; then
    echo "Há jobs seus na fila/rodando:"; echo "$FILA"
    [[ $APLICAR -eq 1 ]] && { echo "Recuso aplicar com jobs ativos." >&2; exit 3; }
fi

echo "== 1. tempo_completo.json da rodada 2 no cluster"
ATUAL="$(ssh -o BatchMode=yes headnode 'cat ~/plasticidade/results_rodada2/tempo_completo.json' 2>/dev/null | tr -d ' \n')"
ORIG="$(echo "$TEMPO_ORIG" | tr -d ' \n')"
if [[ "$ATUAL" == "$ORIG" ]]; then echo "   já está correto."; else
    echo "   remoto: $ATUAL"; echo "   original (c8c6143): $ORIG"
    if [[ $APLICAR -eq 1 ]]; then
        ssh -o BatchMode=yes headnode 'cp ~/plasticidade/results_rodada2/tempo_completo.json ~/plasticidade/results_rodada2/tempo_completo.json.falso && cat > ~/plasticidade/results_rodada2/tempo_completo.json' <<<"$TEMPO_ORIG" 2>&1 | grep -v "█\|░\|Welcome\|alias\|^ *$"
        echo "   restaurado (cópia do falso em tempo_completo.json.falso)."
    fi
fi

echo "== 2. logs a arquivar (mover, não apagar)"
for n in "${RUINS[@]}"; do echo "   slurm-$n.out -> logs/arquivo/"; done
echo "== 3. renomear o log válido: slurm-$VALIDO_ORIG.out -> slurm-$VALIDO_NOVO.out"
if [[ $APLICAR -eq 1 ]]; then
    ssh -o BatchMode=yes headnode 'cd ~/plasticidade/logs && mkdir -p arquivo && for n in '"${RUINS[*]}"'; do [ -f slurm-$n.out ] && mv slurm-$n.out arquivo/; done; [ -f slurm-'"$VALIDO_ORIG"'.out ] && mv slurm-'"$VALIDO_ORIG"'.out slurm-'"$VALIDO_NOVO"'.out; ls' 2>&1 | grep -v "█\|░\|Welcome\|alias\|^ *$"
    mkdir -p "$RAIZ/logs/cluster/arquivo"
    for n in "${RUINS[@]}"; do [[ -f "$RAIZ/logs/cluster/slurm-$n.out" ]] && mv "$RAIZ/logs/cluster/slurm-$n.out" "$RAIZ/logs/cluster/arquivo/"; done
    [[ -f "$RAIZ/logs/cluster/slurm-$VALIDO_ORIG.out" ]] && mv "$RAIZ/logs/cluster/slurm-$VALIDO_ORIG.out" "$RAIZ/logs/cluster/slurm-$VALIDO_NOVO.out"
    echo "feito."
fi
