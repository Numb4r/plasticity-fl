#!/bin/bash
# Envia o projeto plasticidade para ~/plasticidade no cluster Recod.
# Mesmo host e mecanismo ssh/rsync do sync_remote.sh do projeto keys, mas SEM --delete e com destino próprio:
# nunca toca em ~/keys. Não envia .venv (o ambiente do cluster é próprio), .git, logs, figures nem resultados antigos.
# Uso: slurm/enviar.sh [--dry-run]
set -euo pipefail
REMOTE_USER="yuri.faria"
REMOTE_HOST="ssh.recod.ic.unicamp.br"
REMOTE_DIR="plasticidade"                      # relativo ao home remoto
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -n "$(git -C "$LOCAL_DIR" status --porcelain)" ]]; then
    echo "AVISO: há alterações não commitadas; o que será enviado não corresponde a um commit." >&2
fi
git -C "$LOCAL_DIR" rev-parse HEAD > "$LOCAL_DIR/slurm/COMMIT"     # o job registra qual commit rodou

OPTS=(-avz --progress
      --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ --exclude='*.pyc' --exclude=.pytest_cache/
      --exclude=logs/ --exclude=figures/ --exclude=results_v1_exploratorio/ --exclude=results_rodada2/)
[[ "${1:-}" == "--dry-run" ]] && { OPTS+=(--dry-run); echo "[dry-run] nada será transferido."; }

echo "Enviando $LOCAL_DIR/ → $REMOTE_USER@$REMOTE_HOST:~/$REMOTE_DIR/"
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p ~/$REMOTE_DIR/logs"
rsync "${OPTS[@]}" "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
echo "Envio completo."
