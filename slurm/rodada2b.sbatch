#!/bin/bash
# Rodada 2b no cluster Recod. Uso (no headnode, dentro de ~/plasticidade):
#   (use bash slurm/rodada2b.sh curto|longo; o sbatch direto fica para depuração)
#   sbatch --job-name=r2b_teste_curto --time=01:00:00 slurm/rodada2b.sbatch teste_curto 8
#   sbatch --job-name=r2b_completo    --time=04:00:00 slurm/rodada2b.sbatch completo 20
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
case "$SLURM_JOB_NAME" in
    r2b_*) ;;
    *) echo "ERRO: este script é da rodada 'r2b' e o job se chama '$SLURM_JOB_NAME'. Use bash slurm/rodada2b.sh curto|longo." >&2; exit 4 ;;
esac
cd "$HOME/plasticidade"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PLAST_DISPOSITIVO=cpu PYTHONUNBUFFERED=1
echo "no=$(hostname) fase=$FASE processos=$PROC job=$SLURM_JOB_ID inicio=$(date -Is)"
echo "commit do código enviado: $(cat slurm/COMMIT 2>/dev/null)"
.venv/bin/python -m experiments.pipeline_r2b --fase "$FASE" --processos "$PROC"
COD=$?
echo "fim=$(date -Is) codigo_de_saida=$COD"
exit $COD
