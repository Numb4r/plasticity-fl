#!/bin/bash
# Verifica a rodada 2 depois do cluster: opcionalmente traz os resultados, acha o log mais recente do
# completo e roda experiments.verificar_rodada2. Pode ser rodado de qualquer pasta.
# Uso: slurm/verificar.sh [--buscar] [caminho/do/log.out]
#   --buscar   roda slurm/buscar.sh antes (traz results_rodada2/ e os logs do Slurm)
#   sem log    usa o slurm-r2_completo-*.out mais recente em logs/cluster/
# Saída: o resultado do verificador (VERIFICAÇÃO: OK ou FALHOU) e código 0 (OK) ou 1 (FALHOU).
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ"

BUSCAR=0
LOG=""
for a in "$@"; do
    case "$a" in
        --buscar) BUSCAR=1 ;;
        *) LOG="$a" ;;
    esac
done

if [[ $BUSCAR -eq 1 ]]; then
    slurm/buscar.sh || { echo "Falha ao buscar os resultados; abortando." >&2; exit 2; }
fi

if [[ -z "$LOG" ]]; then
    LOG="$(ls -t logs/cluster/slurm-r2_completo-*.out 2>/dev/null | head -1)"
fi
if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "Não achei o log do completo em logs/cluster/. Rode com --buscar ou passe o caminho do log." >&2
    exit 2
fi
echo "Log: $LOG"
echo "Resultados: $RAIZ/results_rodada2"
.venv/bin/python -m experiments.verificar_rodada2 --saida results_rodada2 --log "$LOG"
