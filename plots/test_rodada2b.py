"""Testes da análise da rodada 2b com CSVs SINTÉTICOS (tudo em diretórios temporários; nada em figures/).
Rodar da raiz: .venv/bin/python -m pytest plots/test_rodada2b.py -q"""
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import rodada2b_analise as B
import rodada2b_tabela as TB

SEM = (11, 12, 13, 14, 15)
PAD = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


def _gap(nivel, ruido, s, t):
    if not (t % 3 == 2 or t >= 25):
        return np.nan
    return nivel + (ruido * PAD[(s - 11) % 5] if t >= 25 else 0.0)


def _linhas(cond, tipo, s, nivel, ruido, loss):
    out = []
    for t in range(30):
        g = _gap(nivel, ruido, s, t)
        ref = not np.isnan(g)
        out.append(dict(tarefa=t, tipo=tipo, cond=cond, nivel=3, semente=s, acc_cont=0.95 + g if ref else 0.9,
                        acc_nova=0.95 if ref else np.nan, gap=g, loss_nova=loss if ref else np.nan,
                        loss_cont=0.1, norma=5.0))
    return out


def gerar(pasta, nivel=None, ruido=None, loss_ht=0.05, sem_falta=None, cal_ok=None):
    niv = {("media_sem_particao", t): -0.02 for t in "AB"}
    niv.update({("sem_media_k1", t): -0.30 for t in "AB"})
    niv.update({(B.HTCAL2, t): -0.02 for t in "AB"})
    niv.update(nivel or {})
    rui = {c: 0.004 for c in B.CONDICOES_2B}
    rui.update(ruido or {})
    for t in "AB":
        for s in SEM:
            for c in B.CONDICOES_2B:
                if (c, s) == sem_falta:
                    continue
                lin = _linhas(c, t, s, niv[(c, t)], rui[c], loss_ht if c == B.HTCAL2 else 0.05)
                pd.DataFrame(lin).to_csv(os.path.join(pasta, f"{c}_m3_{t}_{s}.csv"), index=False)
    # semente 1000 (calibração) deve ser ignorada
    pd.DataFrame(_linhas(B.HTCAL2, "A", 1000, 5.0, 0, 0.05)).to_csv(
        os.path.join(pasta, f"{B.HTCAL2}_m3_A_1000.csv"), index=False)
    cal_ok = {"A": True, "B": True, **(cal_ok or {})}
    os.makedirs(os.path.join(pasta, "calibracao"), exist_ok=True)
    for t in "AB":
        ok = cal_ok[t]
        for nome, d in ((f"calibracao_HT2_{t}_m3.json", dict(rodadas=100, ok=ok, erro_rel=0.04 if ok else 0.3,
                                                              alvo_loss_nova_msp=0.05, loss_ht=0.052,
                                                              tentativas={"75": 0.9, "100": 0.052})),
                        (f"selecao_HTcal2_{t}_m3.json", dict(mu=0.99, borda=False, notas={"0.9": 0.63, "0.99": 0.98}))):
            with open(os.path.join(pasta, "calibracao", nome), "w") as f:
                json.dump(d, f)
    return pasta


def _ht(**kw):
    p = tempfile.mkdtemp(prefix="r2b_sint_")
    try:
        gerar(p, **kw)
        return B.h_tempo(B.carregar(p), B.carregar_calibracao(p))
    finally:
        shutil.rmtree(p)


def _n(a, b):
    return {(B.HTCAL2, "A"): a, (B.HTCAL2, "B"): b}


def test_desfechos():
    assert _ht(nivel=_n(0.10, 0.10))["veredito"] == "HT acima"
    assert _ht(nivel=_n(-0.015, -0.015))["veredito"] == "H_tempo"
    assert _ht(nivel=_n(-0.15, -0.15))["veredito"] == "H_dist"
    v = _ht(nivel=_n(-0.02, -0.02), ruido={B.HTCAL2: 0.05})   # ruído grande só no HT: IC da diferença largo
    assert v["veredito"] == "intermediário", v["veredito"]
    v = _ht(nivel=_n(-0.015, 0.10))
    assert v["veredito"].startswith("misto") and "A: dentro" in v["veredito"] and "B: acima" in v["veredito"]
    assert v["tarefas"]["A"]["ht"]["n"] == 5   # semente 1000 ignorada


def test_nao_avaliavel():
    v = _ht(cal_ok={"A": False})
    assert v["veredito"] == "não avaliável (calibração não atingida)", v["veredito"]
    v = _ht(sem_falta=(B.HTCAL2, 13))
    assert "falta central_HT_cal2/A sementes 13" in v["veredito"] and v["veredito"].startswith("não avaliável")
    v = _ht(nivel={("media_sem_particao", t): -0.31 for t in "AB"})
    assert "P_vant falhou em A,B" in v["veredito"], v["veredito"]
    v = _ht(loss_ht=0.5)
    assert "nível de aprendizado diferente" in v["veredito"] and "central_HT_cal2" in v["veredito"], v["veredito"]
    assert v["tarefas"]["A"]["ce_ok"] is False


def test_previsao_item_unico():
    j = lambda **kw: B.julgar_previsao_item(_ht(**kw))   # noqa: E731
    r = j(nivel=_n(0.10, 0.10))
    assert r["julgamento"] == "acerto" and r["calculado"] == "HT acima"
    assert r["previsao"] == B.PREVISAO_ITEM
    assert j(nivel=_n(-0.015, 0.10))["julgamento"] == "acerto parcial"
    assert j(nivel=_n(-0.015, -0.015))["julgamento"] == "erro"
    assert j(nivel=_n(-0.15, -0.15))["julgamento"] == "erro"
    assert j(cal_ok={"B": False})["julgamento"] == "não conta (não avaliável)"


def test_cli_gera_md(tmp_path):
    p = tempfile.mkdtemp(prefix="r2b_sint_")
    try:
        gerar(p, nivel=_n(0.10, 0.10))
        md = tmp_path / "t.md"
        fig = tmp_path / "t.png"
        assert TB.main(["--results", p, "--saida", str(md), "--figura", str(fig)]) == 0
        txt = md.read_text(encoding="utf-8")
        for trecho in ("## 1. Calibração", "## 2. Métrica", "## 3. Diferença", "## 4. P_vant", "## 5. Previsão",
                       "HT acima", "acerto", "central_HT_cal2", "0.99"):
            assert trecho in txt, trecho
        assert fig.exists() and fig.stat().st_size > 0
        assert TB.main(["--results", str(tmp_path / "nao_existe"), "--saida", str(tmp_path / "z.md")]) == 1
        assert not (tmp_path / "z.md").exists()
    finally:
        shutil.rmtree(p)


def test_rodada2_inalterada():
    import rodada2_analise as R
    assert R.SEMENTES == (6, 7, 8, 9, 10) and R.HTCAL == "central_HT_cal"
    _ht()
    assert R.SEMENTES == (6, 7, 8, 9, 10) and R.HTCAL == "central_HT_cal"


def test_relatorio_a_d():
    p = tempfile.mkdtemp(prefix="r2b_sint_")
    try:
        gerar(p, nivel=_n(0.10, 0.10))
        txt, _ = TB.texto(p)
        # (a) diferença de acc_nova e ±0,03 + nota
        assert "diferença média de acc_nova" in txt and "| sim |" in txt
        assert "a perda CE da rede nova está igualada; a acurácia pode não estar; gap = acc_cont − acc_nova" in txt
        # (b) critério secundário literal
        assert "sec_dentro" in txt and "sec_acima" in txt and "Critério secundário" in txt
        # (c) avaliável: sem "não decisória"
        assert "não decisória" not in txt
        # (d) notas como tabela, sem repr de dict
        assert "μ=0.9: 0.630; μ=0.99: 0.980" in txt and "{'0.9'" not in txt
    finally:
        shutil.rmtree(p)
    p = tempfile.mkdtemp(prefix="r2b_sint_")
    try:
        gerar(p, cal_ok={"A": False})
        txt, _ = TB.texto(p)
        assert txt.count("| (não decisória)") + txt.count(" (não decisória) |") == 2
    finally:
        shutil.rmtree(p)
