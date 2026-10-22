#!/bin/bash
# Rodada 2 no cluster Recod. Uso (no headnode, dentro de ~/plasticidade):
#   sbatch --job-name=r2_teste_curto --time=01:00:00 slurm/rodada2.sbatch teste_curto 8
#   sbatch --job-name=r2_completo    --time=12:00:00 slurm/rodada2.sbatch completo 20
# Um nó só (dl-01), CPU apenas, 1 thread por processo: a rodada 2 é CPU e o pré-registro exige
# determinismo por processo; manter o teste curto e o completo na mesma CPU evita misturar aritméticas.
#SBATCH --partition=l40s
#SBATCH --nodelist=dl-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=12G
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
FASE="${1:?fase: teste_curto|completo}"
PROC="${2:-8}"
cd "$HOME/plasticidade"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PLAST_DISPOSITIVO=cpu PYTHONUNBUFFERED=1
echo "no=$(hostname) fase=$FASE processos=$PROC job=$SLURM_JOB_ID inicio=$(date -Is)"
echo "commit do código enviado: $(cat slurm/COMMIT 2>/dev/null)"
.venv/bin/python -m experiments.pipeline_r2 --fase "$FASE" --processos "$PROC"
COD=$?
echo "fim=$(date -Is) codigo_de_saida=$COD"
exit $COD
