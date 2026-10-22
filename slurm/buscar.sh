#!/bin/bash
# Traz os resultados da rodada 2 de ~/plasticidade (cluster) para o projeto local.
#   results_rodada2/  (CSVs, JSONs de seleção/calibração, teste curto, tempo_*.json)  -> ./results_rodada2/
#   logs/             (saída dos jobs do Slurm, slurm-*.out)                          -> ./logs/cluster/
# Nunca apaga nada local (sem --delete) e só substitui um arquivo local se o remoto for mais novo (--update).
# Pode ser repetido durante a execução para trazer resultados parciais. Os originais ficam no cluster.
# Uso: slurm/buscar.sh [--dry-run]
set -euo pipefail
REMOTE_USER="yuri.faria"
REMOTE_HOST="ssh.recod.ic.unicamp.br"
REMOTE_DIR="plasticidade"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

OPTS=(-avz --progress --update --exclude=__pycache__/ --exclude='*.pyc')
[[ "${1:-}" == "--dry-run" ]] && { OPTS+=(--dry-run); echo "[dry-run] nada será transferido."; }

mkdir -p "$LOCAL_DIR/results_rodada2" "$LOCAL_DIR/logs/cluster"
echo "Buscando results_rodada2/ ..."
rsync "${OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/results_rodada2/" "$LOCAL_DIR/results_rodada2/"
echo "Buscando logs do Slurm ..."
rsync "${OPTS[@]}" --include='slurm-*.out' --exclude='*' "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/cluster/"
echo "Busca completa. Confira com: .venv/bin/python -m experiments.verificar_rodada2 --log logs/cluster/<slurm-r2_completo-ID>.out"
