"""Verifica, depois do completo da rodada 4 (4a ou 4b), se terminou sem erro.

  python -m experiments.verificar_rodada4 --parte 4a --log logs/cluster/slurm-r4a_longo-<ID>.out

Confere: o log (sem Traceback/ERROR/PARE; com a linha 'tempos:' e a marca da rodada 4), o teste curto (OK), a seleção
de hiperparâmetro e de R de cada condição e tarefa, cada CSV do plano (existe, 30 linhas, 16 redes novas, sem NaN
indevido, params batendo com a seleção) e as colunas próprias da técnica. Calibração fora de 10% NÃO é falha: é
resultado ("não avaliável"). Imprime OK ou FALHOU (código de saída 0 ou 1).
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from experiments.pipeline_r4 import DIRS, nome_csv, plano_confirmatorio
from src.condicoes_r4 import CATALOGO, arq_selecao
from src.config import RESET_CENTRAL, tarefas_com_nova

MARCA_LOG = "RODADA4 parte={parte} fase=completo"   # linha escrita pelo pipeline ao começar o completo da parte


def verificar(dir_r4: Path, parte: str, log: Path | None) -> tuple[list[str], int]:
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
            if MARCA_LOG.format(parte=parte) not in txt:
                erros.append(f"o log não é do completo da rodada 4, parte {parte} (falta a linha '{MARCA_LOG.format(parte=parte)}'); passe o log certo")
            if "tempos:" not in txt:
                erros.append("o log não chegou à linha final 'tempos:' (o completo não terminou)")
    res = dir_r4 / "calibracao" / "teste_curto" / "RESULTADO.txt"
    if not (res.exists() and res.read_text().startswith("OK")):
        erros.append("teste curto sem RESULTADO.txt OK")
    sel = {}
    for nome in CATALOGO[parte]:
        for tarefa in ("A", "B"):
            arq = arq_selecao(dir_r4, nome, tarefa)
            if not arq.exists():
                erros.append(f"falta {arq.name}")
            else:
                sel[(nome, tarefa)] = json.loads(arq.read_text())
    plano = plano_confirmatorio(parte)
    for job in plano:
        nome, tipo, sem = job
        f = dir_r4 / nome_csv(job)
        if not f.exists():
            erros.append(f"falta {f.name}")
            continue
        d = pd.read_csv(f)
        if len(d) != 30:
            erros.append(f"{f.name}: {len(d)} linhas, esperado 30")
            continue
        ref = set(tarefas_com_nova(30))
        if set(d.index[d["acc_nova"].notna()]) != ref or set(d.index[d["loss_nova"].notna()]) != ref:
            erros.append(f"{f.name}: redes novas fora das tarefas esperadas")
        if d[["acc_cont", "loss_cont", "norma", "rank_ef", "srank_feat"]].isna().any().any():
            erros.append(f"{f.name}: NaN em colunas que não podem ter")
        if "fim_hess_max" not in d.columns or d["fim_hess_max"].isna().any():
            erros.append(f"{f.name}: diagnósticos de Klein faltando")
        p = json.loads(f.with_suffix(".params.json").read_text()) if f.with_suffix(".params.json").exists() else {}
        s = sel.get((nome, tipo))
        if s is not None:
            esperado = RESET_CENTRAL * s["rodadas"] if p.get("familia") == "central" else s["rodadas"]
            obtido = p.get("passos") if p.get("familia") == "central" else p.get("rodadas")
            if obtido != esperado:
                erros.append(f"{f.name}: {obtido} (rodadas/passos) não bate com a seleção ({esperado})")
        if p.get("familia") == "fedx":
            tem = d[["encolhimento", "cos_medio"]].isna().any().any()
            if tem:
                erros.append(f"{f.name}: encolhimento/cosseno faltando")
            if json.loads(p["tec"]).get("tipo") == "gossip" and d[["acc_modelo_medio", "consenso"]].isna().any().any():
                erros.append(f"{f.name}: modelo médio/consenso do gossip faltando")
    return erros, len(plano)


def resumo_selecoes(dir_r4: Path, parte: str) -> list[str]:
    linhas = []
    for nome in CATALOGO[parte]:
        for tarefa in ("A", "B"):
            arq = arq_selecao(dir_r4, nome, tarefa)
            if arq.exists():
                s = json.loads(arq.read_text())
                aviso = "" if s["ok"] else "  <-- FORA DA TOLERÂNCIA (não avaliável)"
                borda = "  [BORDA]" if s.get("borda") else ""
                linhas.append(f"{nome} {tarefa}: R={s['rodadas']} erro_rel={s['erro_rel']:+.3f} hp={s['hp']}{borda}{aviso}")
    return linhas + resumo_redo(dir_r4, parte)


def resumo_redo(dir_r4: Path, parte: str) -> list[str]:
    """Quantos neurônios os ReDo reiniciaram (soma sobre tarefas, sementes e as duas tarefas). Zero = não acionado."""
    linhas = []
    for nome in CATALOGO[parte]:
        if "redo" not in nome:
            continue
        tot = {"redo_servidor": 0.0, "redo_cliente": 0.0}
        for f in dir_r4.glob(f"{nome}_m3_[AB]_*.csv"):
            d = pd.read_csv(f)
            for c in tot:
                tot[c] += float(d[c].sum()) if c in d else 0.0
        aviso = "  <-- REMÉDIO NÃO ACIONADO (0 reinícios)" if sum(tot.values()) == 0 else ""
        linhas.append(f"{nome}: neurônios reiniciados no total: servidor={tot['redo_servidor']:.0f} cliente={tot['redo_cliente']:.0f}{aviso}")
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", choices=["4a", "4b"], required=True)
    ap.add_argument("--saida", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    dir_r4 = Path(a.saida or DIRS[a.parte])
    erros, n = verificar(dir_r4, a.parte, Path(a.log) if a.log else None)
    print(f"execuções esperadas: {n}")
    for linha in resumo_selecoes(dir_r4, a.parte):
        print(linha)
    tempo = dir_r4 / "tempo_completo.json"
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
