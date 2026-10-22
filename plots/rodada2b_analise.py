"""Análise da RODADA 2b (PREREGISTRO.md: "Rodada 2b · `central_HT_cal2`").

Reaproveita `rodada2_analise` SEM alterá-la: dentro de `_config_2b()` troca temporariamente as constantes
SEMENTES (11..15) e HTCAL (`central_HT_cal2`) e chama as funções da rodada 2 (mesmas regras: δ=0,02, equivalência
de dois lados, "HT acima", P_vant por tarefa, elegibilidade pela perda CE a 10%, acurácia ±0,03 só reportada,
operacionalizações 1 a 10 e confirmações de 2026-10-21/22). H_het, horizonte, B_fixa, Dirichlet e rank não entram.

DECISÕES DE LEITURA além do texto:
 a. Calibração da 2b: `calibracao/calibracao_HT2_{A,B}_m3.json` (campo `ok` booleano obrigatório; ausente = "falta
    calibracao_HT2_X_m3.json"; ok=false = "calibração não atingida"). μ vem de `selecao_HTcal2_{A,B}_m3.json`.
 b. Os textos de motivo que a rodada 2 gera citam `central_HT_cal` e `calibracao_HT_`; aqui são reescritos para
    `central_HT_cal2` e `calibracao_HT2_` (só troca de nome, a regra é a mesma).
 c. Previsão: só o item único "H_tempo e H_dist", julgado por `R.julgar_previsao` (itens de H_het e horizonte
    descartados; entram dados de enchimento "não avaliável" que nunca aparecem).
 d. As exploratórias da rodada 2 (central_HT, lookahead) saem vazias aqui e não são usadas.
CLI: python plots/rodada2b_analise.py --results results_rodada2b
"""
import contextlib
import json
import os
import re
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rodada2_analise as R  # noqa: E402

SEMENTES_2B = (11, 12, 13, 14, 15)
HTCAL2 = "central_HT_cal2"
CONDICOES_2B = (HTCAL2, "media_sem_particao", "sem_media_k1")
TIPOS = R.TIPOS_AB
PREVISAO_LITERAL = R.PREVISAO_LITERAL   # texto da rodada 2, inalterado
PREVISAO_ITEM = 'H_tempo não e H_dist não (desfecho "HT acima")'


@contextlib.contextmanager
def _config_2b():
    """Troca sementes e condição da rodada 2 só enquanto o bloco roda; restaura sempre."""
    antigo = (R.SEMENTES, R.HTCAL)
    R.SEMENTES, R.HTCAL = SEMENTES_2B, HTCAL2
    try:
        yield
    finally:
        R.SEMENTES, R.HTCAL = antigo


def _renomear(txt):
    txt = re.sub(r"central_HT_cal(?!2)", HTCAL2, txt)
    return txt.replace("calibracao_HT_", "calibracao_HT2_")


def carregar(results="results_rodada2b"):
    """CSVs da raiz de `results` (sementes 11 a 15; a 1000 e calibracao/ ficam de fora)."""
    with _config_2b(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="sementes descartadas")   # o texto da rodada 2 cita 6..10
        return R.carregar(results)


def carregar_calibracao(results="results_rodada2b"):
    """{tipo: {"calibracao": dict|None, "sel": dict|None}}; arquivo ausente/ilegível = None."""
    out = {}
    for t in TIPOS:
        out[t] = {}
        for chave, nome in (("calibracao", f"calibracao_HT2_{t}_m3.json"), ("sel", f"selecao_HTcal2_{t}_m3.json")):
            try:
                with open(os.path.join(results, "calibracao", nome), encoding="utf-8") as f:
                    out[t][chave] = json.load(f)
            except (OSError, ValueError):
                out[t][chave] = None
    return out


def h_tempo(df, calib=None):
    """Veredito de H_tempo/H_dist da 2b (mesmas regras da rodada 2, condição central_HT_cal2)."""
    ck = None if calib is None else {t: {"calibracao": calib[t]["calibracao"]} for t in TIPOS}
    with _config_2b():
        out = R.h_tempo(df, ck)
    out["veredito"] = _renomear(out["veredito"])
    out["motivos"] = [_renomear(m) for m in out["motivos"]]
    if out["motivo"]:
        out["motivo"] = _renomear(out["motivo"])
    return out


def julgar_previsao_item(ht):
    """Só o item 'H_tempo e H_dist' (o julgamento é o da rodada 2; os outros itens são descartados)."""
    col = {(c, "A"): dict(falta=[0]) for c in ("fedavg_iid", "sem_media_k1", "central_aprendizado")}
    item = R.julgar_previsao(ht, dict(geral="não avaliável (fora da 2b)"), col)[0]
    item["previsao"] = PREVISAO_ITEM
    return item


def resumo_serie(df, cond, tipo):
    with _config_2b():
        return R.resumo(df, cond, tipo)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Vereditos da rodada 2b (texto)")
    p.add_argument("--results", default="results_rodada2b")
    a = p.parse_args(argv)
    df = carregar(a.results) if os.path.isdir(a.results) else pd.DataFrame()
    if df.empty:
        print(f"sem resultados em {a.results}/ (nenhum CSV válido das sementes 11 a 15); nada foi calculado",
              file=sys.stderr)
        return 1
    ht = h_tempo(df, carregar_calibracao(a.results))
    print(f"h_tempo = h_dist: {ht['veredito']}")
    print(f"previsão (item único): {julgar_previsao_item(ht)['julgamento']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
