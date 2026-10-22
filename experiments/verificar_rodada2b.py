"""Verifica, depois do completo da rodada 2b, se terminou sem erro.

  python -m experiments.verificar_rodada2b --log logs/cluster/slurm-r2b_completo-<ID>.out

Confere: o log (sem Traceback/ERROR/PARE, com a linha final 'tempos:'), o teste curto (OK), os 30 CSVs do plano
(existem, linhas, padrões de NaN, R e S registrados nos params), os JSONs de calibração de R e de seleção de μ.
Imprime OK ou FALHOU (código 0 ou 1). Calibração fora de 10% NÃO é falha: é resultado (H_tempo/H_dist não avaliável).
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from experiments.pipeline_r2b import DIR_PADRAO, nome_csv, plano_confirmatorio
from src.config import tarefas_com_nova


def verificar(dir_r2b: Path, log: Path | None) -> tuple[list[str], int]:
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
            if "calibração central_HT_cal2" not in txt:
                erros.append("o log não é do completo da 2b (falta a linha 'calibração central_HT_cal2'); passe o log certo")
            if "tempos:" not in txt:
                erros.append("o log não chegou à linha final 'tempos:' (o completo não terminou)")
    res = dir_r2b / "calibracao" / "teste_curto" / "RESULTADO.txt"
    if not (res.exists() and res.read_text().startswith("OK")):
        erros.append("teste curto sem RESULTADO.txt OK")
    rodadas, mu_sel = {}, {}
    for t in ("A", "B"):
        for arq in (f"calibracao_HT2_{t}_m3.json", f"selecao_HTcal2_{t}_m3.json"):
            if not (dir_r2b / "calibracao" / arq).exists():
                erros.append(f"falta {arq}")
        arq = dir_r2b / "calibracao" / f"calibracao_HT2_{t}_m3.json"
        if arq.exists():
            rodadas[t] = json.loads(arq.read_text())["rodadas"]
    for t in ("A", "B"):
        arq = dir_r2b / "calibracao" / f"calibracao_HT2_{t}_m3.json"
        sel = dir_r2b / "calibracao" / f"selecao_HTcal2_{t}_m3.json"
        if arq.exists():
            c = json.loads(arq.read_text())
            if c.get("grade") != [75, 150] or len(c.get("tentativas", {})) != 76:
                erros.append(f"calibracao_HT2_{t}: grade {c.get('grade')} com {len(c.get('tentativas', {}))} valores, esperado [75, 150] e 76")
            anterior = Path("results_rodada2") / "calibracao" / f"calibracao_HT_{t}_m3.json"
            if anterior.exists() and abs(json.loads(anterior.read_text())["alvo_loss_nova_msp"] - c["alvo_loss_nova_msp"]) > 1e-9:
                erros.append(f"calibracao_HT2_{t}: alvo difere do alvo da rodada 2")
        for mu in (0.9, 0.99, 0.995, 0.999):
            if not (dir_r2b / "calibracao" / "selecao_mu" / f"central_HT_cal2_mu{mu:g}_m3_{t}_1000.csv").exists():
                erros.append(f"falta o CSV de seleção de μ={mu:g} ({t})")
        if sel.exists():
            mu_sel[t] = json.loads(sel.read_text())["mu"]
    plano = plano_confirmatorio()
    for job in plano:
        cond, tipo, sem, T = job
        f = dir_r2b / nome_csv(job)
        if not f.exists():
            erros.append(f"falta {f.name}")
            continue
        d = pd.read_csv(f)
        if len(d) != T:
            erros.append(f"{f.name}: {len(d)} linhas, esperado {T}")
            continue
        ref = set(tarefas_com_nova(T))
        if set(d.index[d["acc_nova"].notna()]) != ref or set(d.index[d["loss_nova"].notna()]) != ref:
            erros.append(f"{f.name}: redes novas fora das tarefas esperadas")
        if d[["acc_cont", "loss_cont", "norma", "rank_ef", "srank_feat"]].isna().any().any():
            erros.append(f"{f.name}: NaN em colunas que não podem ter")
        if d.loc[[t for t in range(T) if t % 5 == 4], "sonda2_dlog_norm"].isna().any():
            erros.append(f"{f.name}: sonda v2 sem valor em tarefa de sonda")
        if "fim_hess_max" not in d.columns or d["fim_hess_max"].isna().any():
            erros.append(f"{f.name}: diagnósticos de Klein faltando")
        p = json.loads(f.with_suffix(".params.json").read_text()) if f.with_suffix(".params.json").exists() else {}
        if cond == "central_HT_cal2":
            if d["acc_tartaruga"].isna().any():
                erros.append(f"{f.name}: acc_tartaruga faltando")
            if mu_sel.get(tipo) is not None and p.get("mu") != mu_sel[tipo]:
                erros.append(f"{f.name}: μ {p.get('mu')} não bate com a seleção ({mu_sel[tipo]})")
            if rodadas.get(tipo) is not None and (p.get("rodadas") != rodadas[tipo] or p.get("s_local") != 15):
                erros.append(f"{f.name}: params ({p.get('rodadas')} rodadas, S={p.get('s_local')}) não batem com a calibração (R={rodadas[tipo]}, S=15)")
        elif p.get("rodadas") != 75:
            erros.append(f"{f.name}: {p.get('rodadas')} rodadas, esperado 75")
    return erros, len(plano)


def resumo_calibracoes(dir_r2b: Path) -> list[str]:
    pasta, linhas = dir_r2b / "calibracao", []
    for t in ("A", "B"):
        arq = pasta / f"calibracao_HT2_{t}_m3.json"
        if arq.exists():
            c = json.loads(arq.read_text())
            linhas.append(f"calibração central_HT_cal2 {t}: R={c['rodadas']} rodadas/tarefa, erro_rel={c['erro_rel']:+.3f}, ok={c['ok']}"
                          + ("" if c["ok"] else "  <-- FORA DA TOLERÂNCIA (H_tempo e H_dist: não avaliável)"))
        arq = pasta / f"selecao_HTcal2_{t}_m3.json"
        if arq.exists():
            s = json.loads(arq.read_text())
            linhas.append(f"μ HTcal2 {t}: {s['mu']:g}" + ("  [BORDA]" if s["borda"] else ""))
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=DIR_PADRAO)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    erros, n = verificar(Path(a.saida), Path(a.log) if a.log else None)
    print(f"execuções esperadas: {n}")
    for linha in resumo_calibracoes(Path(a.saida)):
        print(linha)
    tempo = Path(a.saida) / "tempo_completo.json"
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
