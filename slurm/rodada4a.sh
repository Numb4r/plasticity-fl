#!/bin/bash
# Submete a rodada 4a (técnicas distribuídas) no cluster. Uso (no headnode): bash slurm/rodada4a.sh curto|longo [--dry-run] [--forcar]
exec bash "$(dirname "$0")/_submeter.sh" 4a "$@"
