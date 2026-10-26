#!/bin/bash
# Submete a rodada 4b (remédios de plasticidade no FedAvg) no cluster. Uso (no headnode): bash slurm/rodada4b.sh curto|longo [--dry-run] [--forcar]
exec bash "$(dirname "$0")/_submeter.sh" 4b "$@"
