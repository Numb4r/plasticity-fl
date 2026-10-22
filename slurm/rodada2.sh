#!/bin/bash
# Submete a rodada 2 no cluster. Uso (no headnode): bash slurm/rodada2.sh curto|longo [--dry-run] [--forcar]
exec bash "$(dirname "$0")/_submeter.sh" 2 "$@"
