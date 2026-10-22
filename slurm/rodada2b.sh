#!/bin/bash
# Submete a rodada 2b no cluster. Uso (no headnode): bash slurm/rodada2b.sh curto|longo [--dry-run] [--forcar]
#   curto -> job r2b_curto (teste curto, 1 h, 8 processos)
#   longo -> job r2b_longo (completo, 4 h, 20 processos); recusa se o teste curto não deu OK ou se o longo já rodou
exec bash "$(dirname "$0")/_submeter.sh" 2b "$@"
