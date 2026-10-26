#!/bin/bash
# Uso interno (chamado por slurm/rodada2.sh, rodada2b.sh, rodada4a.sh e rodada4b.sh): submete UM job com nome, tempo e processos
# corretos para a rodada e a fase pedidas. Rode no headnode, de qualquer pasta.
#   _submeter.sh <rodada: 2|2b|4a|4b> <fase: curto|longo> [--dry-run] [--forcar]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
RODADA="${1:?rodada 2|2b|4a|4b}"; FASE_ARG="${2:?fase curto|longo}"; shift 2
DRY=0; FORCAR=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --forcar) FORCAR=1 ;;
        *) echo "argumento desconhecido: $a" >&2; exit 2 ;;
    esac
done

case "$RODADA" in
    2)  PREFIXO="r2";  SBATCH="slurm/rodada2.sbatch";  PASTA="results_rodada2"
        T_CURTO="01:00:00"; T_LONGO="12:00:00" ;;
    2b) PREFIXO="r2b"; SBATCH="slurm/rodada2b.sbatch"; PASTA="results_rodada2b"
        T_CURTO="01:00:00"; T_LONGO="04:00:00" ;;
    4a) PREFIXO="r4a"; SBATCH="slurm/rodada4.sbatch"; PASTA="results_rodada4a"
        T_CURTO="01:00:00"; T_LONGO="12:00:00" ;;
    4b) PREFIXO="r4b"; SBATCH="slurm/rodada4.sbatch"; PASTA="results_rodada4b"
        T_CURTO="01:00:00"; T_LONGO="10:00:00" ;;
    *)  echo "rodada inválida: $RODADA (use 2, 2b, 4a ou 4b)" >&2; exit 2 ;;
esac
case "$FASE_ARG" in
    curto)             FASE="teste_curto"; NOME="${PREFIXO}_curto"; TEMPO="$T_CURTO"; PROC=8 ;;
    longo|completo)    FASE="completo";    NOME="${PREFIXO}_longo"; TEMPO="$T_LONGO"; PROC=20 ;;
    *) echo "fase inválida: $FASE_ARG (use curto ou longo)" >&2; exit 2 ;;
esac

cd "$RAIZ"
if [[ "$FASE" == "completo" && $FORCAR -eq 0 && -f "$PASTA/tempo_completo.json" ]]; then
    echo "A rodada $RODADA já tem $PASTA/tempo_completo.json (o longo já rodou)." >&2
    echo "Rodar de novo só retoma/pula CSVs válidos e SOBRESCREVE o registro de tempo. Para insistir: --forcar" >&2
    exit 3
fi
if [[ "$FASE" == "completo" && $FORCAR -eq 0 && ! -f "$PASTA/calibracao/teste_curto/RESULTADO.txt" ]]; then
    echo "O teste curto da rodada $RODADA ainda não rodou (sem $PASTA/calibracao/teste_curto/RESULTADO.txt). Rode primeiro: bash slurm/rodada${RODADA}.sh curto" >&2
    exit 3
fi
if [[ "$FASE" == "completo" && $FORCAR -eq 0 ]] && ! head -1 "$PASTA/calibracao/teste_curto/RESULTADO.txt" | grep -q '^OK'; then
    echo "O teste curto da rodada $RODADA não deu OK. Veja: head $PASTA/calibracao/teste_curto/RESULTADO.txt" >&2
    exit 3
fi

if [[ "$RODADA" == 4* ]]; then
    CMD=(sbatch --job-name="$NOME" --time="$TEMPO" "$SBATCH" "$RODADA" "$FASE" "$PROC")     # o sbatch da rodada 4 recebe a parte
else
    CMD=(sbatch --job-name="$NOME" --time="$TEMPO" "$SBATCH" "$FASE" "$PROC")
fi
echo "Rodada $RODADA, fase $FASE -> job '$NOME', tempo $TEMPO, $PROC processos, script $SBATCH"
if [[ $DRY -eq 1 ]]; then
    echo "[dry-run] ${CMD[*]}"
    exit 0
fi
"${CMD[@]}"
echo "Acompanhe: squeue -u \$USER   |   tail -f logs/slurm-${NOME}-<ID>.out"
