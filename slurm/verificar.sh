#!/bin/bash
# Verifica a rodada 2 depois do cluster: opcionalmente traz os resultados, acha o log mais recente do
# completo e roda experiments.verificar_rodada2. Pode ser rodado de qualquer pasta.
# Uso: slurm/verificar.sh [--rodada 2|2b|3|4a|4b] [--buscar] [caminho/do/log.out]   (padrão: rodada 2)
#   --buscar   roda slurm/buscar.sh antes (traz results_rodada2/ e os logs do Slurm)
#   sem log    usa o slurm-r2_completo-*.out mais recente em logs/cluster/
# Saída: o resultado do verificador (VERIFICAÇÃO: OK ou FALHOU) e código 0 (OK) ou 1 (FALHOU).
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ"

BUSCAR=0
LOG=""
RODADA="2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --buscar) BUSCAR=1 ;;
        --rodada) RODADA="${2:?--rodada 2|2b|3|4a|4b}"; shift ;;
        *) LOG="$1" ;;
    esac
    shift
done
case "$RODADA" in
    2)  MARCA="calibração B_fixa"; PASTA="results_rodada2";  MODULO="experiments.verificar_rodada2";  PADRAO="slurm-r2_longo-*.out slurm-r2_completo-*.out" ;;
    2b) MARCA="calibração central_HT_cal2"; PASTA="results_rodada2b"; MODULO="experiments.verificar_rodada2b"; PADRAO="slurm-r2b_longo-*.out slurm-r2b_completo-*.out" ;;
    3)  MARCA="RODADA4 parte=3 fase=completo"; PASTA="results_rodada3"; MODULO="experiments.verificar_rodada4 --parte 3"; PADRAO="slurm-r3_longo-*.out" ;;
    4a) MARCA="RODADA4 parte=4a fase=completo"; PASTA="results_rodada4a"; MODULO="experiments.verificar_rodada4 --parte 4a"; PADRAO="slurm-r4a_longo-*.out" ;;
    4b) MARCA="RODADA4 parte=4b fase=completo"; PASTA="results_rodada4b"; MODULO="experiments.verificar_rodada4 --parte 4b"; PADRAO="slurm-r4b_longo-*.out" ;;
    *)  echo "rodada inválida: $RODADA" >&2; exit 2 ;;
esac

if [[ $BUSCAR -eq 1 ]]; then
    slurm/buscar.sh --rodada "$RODADA" || { echo "Falha ao buscar os resultados; abortando." >&2; exit 2; }
fi

if [[ -z "$LOG" ]]; then
    # o log mais recente que TERMINOU (tem a linha 'tempos:'); se nenhum terminou, o mais recente de todos
    for f in $(cd logs/cluster 2>/dev/null && ls -t $PADRAO 2>/dev/null); do
        [[ -z "$LOG" ]] && LOG="logs/cluster/$f"
        if grep -q "tempos:" "logs/cluster/$f" && grep -q "$MARCA" "logs/cluster/$f"; then LOG="logs/cluster/$f"; break; fi
    done
fi
if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "Não achei o log do completo em logs/cluster/. Rode com --buscar ou passe o caminho do log." >&2
    exit 2
fi
echo "Log: $LOG"
echo "Resultados: $RAIZ/$PASTA"
.venv/bin/python -m $MODULO --saida "$PASTA" --log "$LOG"
