"""Verifica, depois do completo, se a rodada 2 terminou sem erro.

  python -m experiments.verificar_rodada2 --log logs/r2_completo.log

Confere: o log (saída 0, sem Traceback/ERROR/PARE), o teste curto (OK), cada CSV esperado do plano
(existe, nº de linhas, padrões de NaN, colunas novas), as matrizes e os arquivos de reaprendizado da
B_fixa, os JSONs de seleção de μ e de calibração, e o tempo. Imprime OK ou FALHOU (código de saída 0 ou 1).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.pipeline_r2 import DIR_PADRAO, nome_csv, plano_confirmatorio, plano_longo
from src.config import T_LONGO, tarefas_com_nova


def verificar(dir_r2: Path, log: Path | None) -> list[str]:
    erros = []
    if log is not None:
        if not log.exists():
            erros.append(f"log {log} não existe")
        else:
            txt = log.read_text(errors="ignore")
            if "Traceback" in txt or re.search(r"\bERROR\b", txt):
                erros.append("o log tem Traceback ou ERROR")
            if re.search(r"\bPARE\b", txt):
                erros.append("o log tem aviso PARE")
            if "tempos:" not in txt:
                erros.append("o log não chegou à linha final 'tempos:' (o completo não terminou)")
    res = dir_r2 / "calibracao" / "teste_curto" / "RESULTADO.txt"
    if not (res.exists() and res.read_text().startswith("OK")):
        erros.append("teste curto sem RESULTADO.txt OK")
    for m in ("HT", "lookahead", "HTcal"):
        for t in ("A", "B"):
            if not (dir_r2 / "calibracao" / f"selecao_{m}_{t}_m3.json").exists():
                erros.append(f"falta selecao_{m}_{t}_m3.json")
    for t in ("A", "B"):
        if not (dir_r2 / "calibracao" / f"calibracao_HT_{t}_m3.json").exists():
            erros.append(f"falta calibracao_HT_{t}_m3.json")
    if not (dir_r2 / "calibracao" / "calibracao_B_fixa_m3.json").exists():
        erros.append("falta calibracao_B_fixa_m3.json")
    plano = plano_confirmatorio() + plano_longo()
    for job in plano:
        cond, tipo, sem, T, alfa = job
        f = dir_r2 / nome_csv(job)
        if not f.exists():
            erros.append(f"falta {f.name}")
            continue
        d = pd.read_csv(f)
        if len(d) != T:
            erros.append(f"{f.name}: {len(d)} linhas, esperado {T}")
            continue
        ref = set(tarefas_com_nova(T, janela_hpart=T <= 30))
        if set(d.index[d["acc_nova"].notna()]) != ref or set(d.index[d["loss_nova"].notna()]) != ref:
            erros.append(f"{f.name}: redes novas fora das tarefas esperadas")
        if d[["acc_cont", "loss_cont", "norma", "rank_ef", "srank_feat"]].isna().any().any():
            erros.append(f"{f.name}: NaN em colunas que não podem ter")
        sonda = d.loc[[t for t in range(T) if t % 5 == 4], "sonda2_dlog_norm"]
        if sonda.isna().any():
            erros.append(f"{f.name}: sonda v2 sem valor em tarefa de sonda")
        if cond in ("central_HT", "central_HT_cal") and d["acc_tartaruga"].isna().any():
            erros.append(f"{f.name}: acc_tartaruga faltando")
        if cond == "fedavg_dirichlet" and d["part_max"].isna().any():
            erros.append(f"{f.name}: partições Dirichlet não registradas")
        if tipo == "B_fixa":
            for suf in ("_matriz.csv", "_reaprendizado.csv"):
                if not (dir_r2 / f.name.replace(".csv", suf)).exists():
                    erros.append(f"falta {f.name.replace('.csv', suf)}")
        if "fim_hess_max" not in d.columns or d["fim_hess_max"].isna().any():
            erros.append(f"{f.name}: diagnósticos de Klein faltando")
    return erros, len(plano)


def resumo_calibracoes(dir_r2: Path) -> list[str]:
    """Linhas informativas (não são erro): μ escolhidos, borda e a calibração do central_HT_cal.
    Calibração fora da tolerância é um RESULTADO (H_tempo/H_dist 'não avaliável'), não uma falha."""
    pasta, linhas = dir_r2 / "calibracao", []
    for t in ("A", "B"):
        arq = pasta / f"calibracao_HT_{t}_m3.json"
        if arq.exists():
            c = json.loads(arq.read_text())
            linhas.append(f"calibração central_HT_cal {t}: S={c['passos_locais']} passos locais/rodada, erro_rel={c['erro_rel']:+.3f}, "
                          f"ok={c['ok']}" + ("" if c["ok"] else "  <-- FORA DA TOLERÂNCIA (H_tempo e H_dist: não avaliável)"))
    for m in ("HT", "lookahead", "HTcal"):
        for t in ("A", "B"):
            arq = pasta / f"selecao_{m}_{t}_m3.json"
            if arq.exists():
                s = json.loads(arq.read_text())
                linhas.append(f"μ {m} {t}: {s['mu']:g}" + ("  [BORDA]" if s["borda"] else ""))
    arq = pasta / "calibracao_B_fixa_m3.json"
    if arq.exists():
        c = json.loads(arq.read_text())
        linhas.append(f"calibração central_aprendizado B_fixa: passos={c['passos']} ok={c['ok']}")
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=DIR_PADRAO)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    erros, n = verificar(Path(a.saida), Path(a.log) if a.log else None)
    tempo = Path(a.saida) / "tempo_completo.json"
    print(f"execuções esperadas: {n}")
    for linha in resumo_calibracoes(Path(a.saida)):
        print(linha)
    if tempo.exists():
        print("tempos:", json.loads(tempo.read_text()))
    if erros:
        print(f"VERIFICAÇÃO: FALHOU ({len(erros)} problemas)")
        for e in erros[:40]:
            print(" -", e)
        raise SystemExit(1)
    print("VERIFICAÇÃO: OK (todas as execuções, arquivos e checagens)")


if __name__ == "__main__":
    main()
