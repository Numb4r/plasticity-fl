#!/bin/bash
# Submete a rodada 3 (fatorial K x origem dos dados) no cluster. Uso (no headnode): bash slurm/rodada3.sh curto|longo [--dry-run] [--forcar]
exec bash "$(dirname "$0")/_submeter.sh" 3 "$@"
