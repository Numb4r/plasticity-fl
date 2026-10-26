#!/bin/bash
# Rodada 4 (4a e 4b) no cluster Recod. Use: bash slurm/rodada4a.sh curto|longo   ou   bash slurm/rodada4b.sh curto|longo
# (o sbatch direto fica para depuração: sbatch --job-name=r4a_curto --time=01:00:00 slurm/rodada4.sbatch 4a teste_curto 8)
# Um nó só (dl-01), CPU apenas, 1 thread por processo, como as rodadas 2 e 2b.
#SBATCH --partition=l40s
#SBATCH --nodelist=dl-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=12G
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
PARTE="${1:?parte: 4a|4b}"
FASE="${2:?fase: teste_curto|completo}"
PROC="${3:-8}"
ESPERADO="r${PARTE}_curto"; [[ "$FASE" == "completo" ]] && ESPERADO="r${PARTE}_longo"
if [[ "$SLURM_JOB_NAME" != "$ESPERADO" ]]; then
    echo "ERRO: a parte $PARTE, fase $FASE, pede o job '$ESPERADO' e este se chama '$SLURM_JOB_NAME'. Use bash slurm/rodada${PARTE}.sh curto|longo." >&2
    exit 4
fi
cd "$HOME/plasticidade"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PLAST_DISPOSITIVO=cpu PYTHONUNBUFFERED=1
echo "no=$(hostname) parte=$PARTE fase=$FASE processos=$PROC job=$SLURM_JOB_ID inicio=$(date -Is)"
echo "commit do código enviado: $(cat slurm/COMMIT 2>/dev/null)"
.venv/bin/python -m experiments.pipeline_r4 --parte "$PARTE" --fase "$FASE" --processos "$PROC"
COD=$?
echo "fim=$(date -Is) codigo_de_saida=$COD"
exit $COD
