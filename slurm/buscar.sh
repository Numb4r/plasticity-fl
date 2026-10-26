#!/bin/bash
# Traz os resultados da rodada 2 de ~/plasticidade (cluster) para o projeto local.
#   results_rodada2/  (CSVs, JSONs de seleção/calibração, teste curto, tempo_*.json)  -> ./results_rodada2/
#   logs/             (saída dos jobs do Slurm, slurm-*.out)                          -> ./logs/cluster/
# Nunca apaga nada local (sem --delete) e só substitui um arquivo local se o remoto for mais novo (--update).
# Pode ser repetido durante a execução para trazer resultados parciais. Os originais ficam no cluster.
# Uso: slurm/buscar.sh [--rodada 2|2b|4a|4b] [--dry-run]     (padrão: 2; a 2b traz results_rodada2b/)
set -euo pipefail
REMOTE_USER="yuri.faria"
REMOTE_HOST="ssh.recod.ic.unicamp.br"
REMOTE_DIR="plasticidade"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

OPTS=(-avz --progress --update --exclude=__pycache__/ --exclude='*.pyc')
RODADA="2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) OPTS+=(--dry-run); echo "[dry-run] nada será transferido." ;;
        --rodada) RODADA="${2:?--rodada 2|2b|4a|4b}"; shift ;;
        *) echo "argumento desconhecido: $1" >&2; exit 2 ;;
    esac
    shift
done
case "$RODADA" in 2) PASTA="results_rodada2" ;; 2b) PASTA="results_rodada2b" ;; 4a) PASTA="results_rodada4a" ;; 4b) PASTA="results_rodada4b" ;; *) echo "rodada inválida: $RODADA" >&2; exit 2 ;; esac

mkdir -p "$LOCAL_DIR/$PASTA" "$LOCAL_DIR/logs/cluster"
echo "Buscando $PASTA/ ..."
rsync "${OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/$PASTA/" "$LOCAL_DIR/$PASTA/"
echo "Buscando logs do Slurm ..."
rsync "${OPTS[@]}" --include='slurm-*.out' --exclude='*' "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/cluster/"
echo "Busca completa. Confira com: slurm/verificar.sh --rodada $RODADA"
