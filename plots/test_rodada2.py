"""Teste da análise da rodada 2 com CSVs SINTÉTICOS (nenhum dado real; tudo em tempfile.mkdtemp()).
Rodar da raiz: .venv/bin/python plots/test_rodada2.py"""
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import rodada2_analise as R  # noqa: E402

PAD = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])   # padrão de variação entre as sementes 6..10


def _gap_t(nivel, ruido, s, t, T):
    """Gap da tarefa t: nível + ruído (só na janela principal); definido em t%3==2 e nas 5 últimas."""
    if not (t % 3 == 2 or t >= T - 5):
        return np.nan
    return nivel + (ruido * PAD[s - 6] if t >= T - 5 else 0.0)


def _linha(cond, tipo, s, t, gap, loss_nova=0.05, alfa=np.nan, ref_rank=10.0):
    ref = not np.isnan(gap)
    nova = 0.95
    d = dict(tarefa=t, tipo=tipo, cond=cond, nivel=3, semente=s, maj_frac=0.13,
             acc_cont=nova + gap if ref else 0.9, norma=5.0, distancia=1.0,
             dorm_tau0_l1=0.0, dorm_tau0_l2=0.0, dorm_tau0025_l1=0.0, dorm_tau0025_l2=0.0,
             rank_ef=ref_rank + 0.1 * t, srank_feat=3 * ref_rank + 0.2 * t, cos_medio=0.5,
             acc_nova=nova if ref else np.nan, gap=gap, esquecimento=np.nan, bwt=np.nan, acc_media_vistas=np.nan,
             loss_cont=0.1, loss_nova=loss_nova if ref else np.nan, acc_tartaruga=np.nan, mu=np.nan, alfa=alfa,
             part_min=np.nan, part_max=np.nan, part_vazios=np.nan, sonda_perda_copia=np.nan,
             sonda_perda_nova=np.nan, sonda_razao=np.nan)
    sonda = t % 5 == 4
    for k, lc20 in enumerate((0.5, 0.2, 0.9)):
        d[f"sonda2_Lc0_{k}"] = 1.0 if sonda else np.nan
        d[f"sonda2_Lc20_{k}"] = lc20 if sonda else np.nan
        d[f"sonda2_Ln0_{k}"] = 1.0 if sonda else np.nan
        d[f"sonda2_Ln20_{k}"] = 0.25 if sonda else np.nan
    d["sonda2_dlog_norm"] = 0.30103 if sonda else np.nan
    d["sonda2_dlog_raw"] = 0.1 if sonda else np.nan
    return d


def _salvar(pasta, nome, linhas):
    pd.DataFrame(linhas).to_csv(os.path.join(pasta, nome), index=False)


def gerar(pasta, nivel=None, ruido=None, loss_ht=0.05, het=None, longo=None, sem_ht_semente=None,
          com_longo=True, com_bfixa=True):
    """Gera o conjunto sintético. nivel[(cond, tipo)] = gap na janela; ruido[cond]; het[tipo]=(g0.1, g1, g100);
    longo[(cond, tipo)] = tarefa a partir da qual o gap cai a -0,4 (None = nunca)."""
    niv = {("fedavg_iid", t): -0.10 for t in "AB"}
    niv.update({("media_sem_particao", t): -0.02 for t in "AB"})
    niv.update({("sem_media_k1", t): -0.30 for t in "AB"})
    niv.update({("central_aprendizado", t): -0.25 for t in "AB"})
    niv.update({("central_HT", t): -0.02 for t in "AB"})
    niv.update({("central_lookahead", t): -0.05 for t in "AB"})
    niv.update(nivel or {})
    rui = {c: 0.004 for c in R.COND_T30}
    rui.update(ruido or {})
    het = het or {t: (-0.20, -0.10, -0.02) for t in "AB"}
    lo = {(c, t): None for c in R.COND_LONGO for t in "AB"}
    lo.update({("sem_media_k1", "A"): 20, ("central_aprendizado", "A"): 35, ("fedavg_iid", "A"): 60})
    lo.update(longo or {})
    for t in "AB":
        for s in R.SEMENTES:
            for c in R.COND_T30:
                if c == "central_HT" and s == sem_ht_semente:
                    continue
                lin = [_linha(c, t, s, k, _gap_t(niv[(c, t)], rui[c], s, k, 30),
                              loss_nova=loss_ht if c == "central_HT" else 0.05) for k in range(30)]
                _salvar(pasta, f"{c}_m3_{t}_{s}.csv", lin)
            for a, g in zip(R.ALFAS, het[t]):
                lin = [_linha("fedavg_dirichlet", t, s, k, _gap_t(g, 0.004, s, k, 30), alfa=a) for k in range(30)]
                _salvar(pasta, f"fedavg_dirichlet_a{a:g}_m3_{t}_{s}.csv", lin)
    if com_bfixa:
        for s in R.SEMENTES:
            for c in R.COND_BFIXA:
                lin = [_linha(c, "B_fixa", s, k, _gap_t(-0.05, 0.01, s, k, 30)) for k in range(30)]
                lin[-1].update(esquecimento=0.05 + 0.01 * PAD[s - 6], bwt=-0.05)
                _salvar(pasta, f"{c}_m3_B_fixa_{s}.csv", lin)
                pd.DataFrame({"i": [0], "j0": [1.0]}).to_csv(os.path.join(pasta, f"{c}_m3_B_fixa_{s}_matriz.csv"),
                                                              index=False)
                cens = c == "sem_media_k1"   # só este nunca volta ao limiar
                rea = [dict(tarefa_atual=ta, tarefa_antiga=tb, a_jj=1.0, limiar=0.9, acc_atual_j=0.2,
                            passos_copia=1500 if cens else 30, censurado_copia=cens,
                            passos_nova=300, censurado_nova=False) for ta in (9, 19, 29) for tb in (0, 5)]
                _salvar(pasta, f"{c}_m3_B_fixa_{s}_reaprendizado.csv", rea)
    if com_longo:
        for t in "AB":
            for c in R.COND_LONGO:
                for s in R.SEMENTES_LONGO:
                    on = lo[(c, t)]
                    on_s = None if on is None else on + (3 * (s - 6) if c == "fedavg_iid" else 0)
                    lin = []
                    for k in range(100):
                        g = np.nan
                        if k % 3 == 2 or k >= 95:
                            g = -0.4 if (on_s is not None and k >= on_s) else -0.01
                        lin.append(_linha(c, t, s, k, g))
                    _salvar(pasta, f"{c}_T100_m3_{t}_{s}.csv", lin)
    return pasta


def _novo():
    return tempfile.mkdtemp(prefix="r2_sint_")


def _veredito(**kw):
    p = _novo()
    try:
        gerar(p, **kw)
        return R.h_tempo(R.carregar(p))
    finally:
        shutil.rmtree(p)


def test_h_tempo_desfechos():
    casos = {
        "H_tempo": dict(nivel={("central_HT", t): -0.015 for t in "AB"}),
        "HT acima": dict(nivel={("central_HT", t): 0.10 for t in "AB"}),
        "H_dist": dict(nivel={("central_HT", t): -0.15 for t in "AB"}),
        "intermediário": dict(nivel={("central_HT", t): -0.02 for t in "AB"}, ruido={"central_HT": 0.05}),
    }
    for esperado, kw in casos.items():
        v = _veredito(**kw)
        assert v["veredito"] == esperado, (esperado, v["veredito"], v["tarefas"]["A"]["d"])
    v = _veredito(nivel={("central_HT", "A"): -0.015, ("central_HT", "B"): 0.10})
    assert v["veredito"].startswith("misto") and "A: dentro" in v["veredito"] and "B: acima" in v["veredito"], v
    # critério secundário literal e lookahead (exploratório, sem veredito)
    v = _veredito(nivel={("central_HT", t): 0.10 for t in "AB"})
    assert v["tarefas"]["A"]["sec_dentro"] is False and v["tarefas"]["A"]["sec_acima"] is False
    v = _veredito(nivel={("central_HT", t): -0.02 for t in "AB"})
    assert v["tarefas"]["A"]["sec_dentro"] is True and v["tarefas"]["A"]["sec_acima"] is False
    assert abs(v["lookahead"]["A"]["d"]["media"] - (-0.03)) < 1e-9
    print("ok h_tempo: dentro, HT acima, H_dist, intermediário, misto, secundário, lookahead")


def test_nao_avaliavel():
    v = _veredito(loss_ht=0.5)
    assert v["veredito"].startswith("não avaliável") and "nível de aprendizado diferente" in v["veredito"], v
    assert v["tarefas"]["A"]["acc_ok"] is True          # checagem secundária de acurácia só reportada
    v = _veredito(nivel={("media_sem_particao", t): -0.31 for t in "AB"})
    assert v["veredito"].startswith("não avaliável") and "P_vant falhou em A,B" in v["veredito"], v
    v = _veredito(sem_ht_semente=9)
    assert "falta central_HT/A sementes 9" in v["veredito"], v["veredito"]
    print("ok não avaliável: elegibilidade, P_vant, falta de semente")


def _het(het):
    p = _novo()
    try:
        gerar(p, het=het, com_longo=False, com_bfixa=False)
        return R.h_het(R.carregar(p))
    finally:
        shutil.rmtree(p)


def test_h_het():
    mono = (-0.20, -0.10, -0.02)
    r = _het({"A": mono, "B": mono})
    assert r["geral"] == "atendida" and r["tarefas"]["A"]["veredito"] == "atendida", r["geral"]
    r = _het({"A": mono, "B": (-0.004, 0.002, 0.003)})
    assert r["geral"] == "parcial" and r["tarefas"]["B"]["veredito"].startswith("inconclusivo"), r["geral"]
    r = _het({"A": (0.001, 0.002, 0.003), "B": (-0.004, 0.002, 0.003)})
    assert r["geral"] == "não atendida" and all(
        r["tarefas"][t]["veredito"].startswith("inconclusivo") for t in "AB")
    r = _het({"A": tuple(reversed(mono)), "B": tuple(reversed(mono))})
    assert r["tarefas"]["A"]["veredito"] == "H_het reversa" and r["geral"] == "não atendida", r["geral"]
    r = _het({"A": mono, "B": (-0.20, -0.02, -0.10)})   # não monotônica em B
    assert r["geral"] == "parcial" and r["tarefas"]["B"]["veredito"] == "não atendida"
    assert r["tarefas"]["A"]["acc_cont"][0.1]["n"] == 5 and r["tarefas"]["A"]["acc_nova"][100.0]["n"] == 5
    print("ok H_het: atendida, parcial, inconclusivo, reversa, não monotônica")


def test_colapso_e_horizonte():
    p = _novo()
    try:
        gerar(p, longo={("fedavg_iid", "A"): None})   # impede a perda
        df = R.carregar(p)
        col = R.colapso_tabela(df)
        assert R.julgar_horizonte(col) == "impede a perda", R.julgar_horizonte(col)
        assert col[("fedavg_iid", "A")]["frac_sem"] == 1.0 and col[("fedavg_iid", "A")]["mediana"] is None
        assert col[("sem_media_k1", "A")]["mediana"] == 20
        # sem colapso em B (nenhuma condição colapsa)
        assert all(col[(c, "B")]["frac_sem"] == 1.0 for c in R.COND_LONGO)
    finally:
        shutil.rmtree(p)
    p = _novo()
    try:
        gerar(p)   # padrão: fedavg_iid colapsa a partir de 60 (+3/semente), depois de k1 (20) e de ca (35)
        col = R.colapso_tabela(R.carregar(p))
        # k1: onset 20 (20%3==2, é referência) -> colapso em 20; fedavg_iid: onsets 60, 63, 66 -> refs 62, 65, 68
        assert col[("sem_media_k1", "A")]["mediana"] == 20, col[("sem_media_k1", "A")]
        assert col[("fedavg_iid", "A")]["mediana"] == 65 and col[("fedavg_iid", "A")]["frac_sem"] == 0.0
        assert R.julgar_horizonte(col) == "só atrasa a perda", R.julgar_horizonte(col)
    finally:
        shutil.rmtree(p)
    p = _novo()
    try:
        gerar(p, longo={("fedavg_iid", "A"): 5, ("central_aprendizado", "A"): None})
        col = R.colapso_tabela(R.carregar(p))
        j = R.julgar_horizonte(col)
        assert j.startswith("outro (") and "central_aprendizado não colapsa" in j, j
        gerar(p, longo={("fedavg_iid", "A"): 5})   # colapsa antes de k1 e ca -> outro
        j = R.julgar_horizonte(R.colapso_tabela(R.carregar(p)))
        assert j.startswith("outro (") and "fedavg_iid colapsa" in j, j
    finally:
        shutil.rmtree(p)
    print("ok colapso: com e sem; impede / só atrasa / outro")


def test_indice_colapso_unidade():
    ts = np.arange(2, 100, 3)
    assert R.indice_colapso(ts, np.zeros(33)) is None
    g = np.zeros(33)
    g[2:] = -0.3                      # média móvel (0,0,-0.3) = -0.10 -> colapsa no 3º ponto (t=8)
    assert R.indice_colapso(ts, g) == 8
    g = np.zeros(33)
    g[1] = -0.6                       # um ponto isolado em i=1 só entra na janela (1,2,3)... mas i>=2 exigido
    assert R.indice_colapso(ts, g) == 8   # janela (0,1,2): média -0.2
    print("ok índice de colapso (unidade)")


def test_semente_1000_ignorada():
    base = _novo()
    gerar(base)
    v0 = R.h_tempo(R.carregar(base))["veredito"]
    # planta a 1000 na raiz (valores absurdos) e em calibracao/
    lin = [_linha("central_HT", "A", 1000, k, _gap_t(5.0, 0.0, 6, k, 30)) for k in range(30)]
    _salvar(base, "central_HT_m3_A_1000.csv", lin)
    os.makedirs(os.path.join(base, "calibracao"))
    _salvar(os.path.join(base, "calibracao"), "central_HT_m3_A_6.csv", lin)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = R.carregar(base)
    assert any("1000" in str(x.message) for x in w), [str(x.message) for x in w]
    assert set(df["semente"].unique()) == set(R.SEMENTES)
    assert R.h_tempo(df)["veredito"] == v0
    assert len(df[(df["serie"] == "central_HT") & (df["tipo"] == "A") & (df["braco"] == "T30")]) == 30 * 5   # calibracao/ não lida
    shutil.rmtree(base)
    print("ok semente 1000 plantada na raiz e em calibracao/ ignorada com aviso")


def test_tabelas_auxiliares_e_cli():
    base, saida = _novo(), os.path.join(_novo(), "out")
    vazio = _novo()
    gerar(base)
    df = R.carregar(base)
    # sonda: mediana entre alvos de (0.301, -0.097, 0.556) = 0.301; raw = 0.1
    nome, t, a, b = R.sonda_tabela(df)[0]
    assert abs(a["media"] - 0.30103) < 1e-4 and abs(b["media"] - 0.1) < 1e-9, (a, b)
    # rank: grupos juntos, 2 medidas
    rk = {(n, t): (a, b) for n, t, a, b in R.rank_tabela(df)}
    assert ("grupo: com média", "A") in rk and ("grupo: sem média", "B") in rk
    assert rk[("grupo: sem média", "A")][0]["n"] == 5 and rk[("grupo: sem média", "A")][1]["media"] > 0
    # reaprendizado: k1 censurado (> 1500), as outras com economia = 1 - 30/300 = 0.90, limiar no 0: 0
    rea = {r["cond"]: r for r in R.reaprendizado_tabela(R.carregar_reaprendizado(base))}
    assert rea["sem_media_k1"]["txt_copia"] == "> 1500" and rea["sem_media_k1"]["cens_copia"] == 1.0
    assert rea["fedavg_iid"]["economia"] == "0.90" and rea["fedavg_iid"]["txt_copia"] == "30"
    # B_fixa
    bf = {c: (g, e, b) for c, g, e, b in R.bfixa_tabela(df)}
    assert abs(bf["fedavg_iid"][1]["media"] - 0.05) < 1e-9 and abs(bf["fedavg_iid"][2]["media"] + 0.05) < 1e-9
    # arquivos *_matriz/_reaprendizado não entram no principal; braço longo separado
    assert not (df["braco"] == "T100").all() and set(df["braco"]) == {"T30", "T100"}
    # CLI completa
    r = subprocess.run([sys.executable, os.path.join(AQUI, "rodada2_tabela.py"), "--results", base,
                        "--saida", saida], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ("rodada2_tabela_decisao.md", "rodada2_gap_por_tarefa.png", "rodada2_h_het.png",
              "rodada2_longo_colapso.png", "rodada2_reaprendizado.png", "rodada2_rank.png"):
        assert os.path.getsize(os.path.join(saida, f)) > 1000, f
    md = open(os.path.join(saida, "rodada2_tabela_decisao.md"), encoding="utf-8").read()
    for s in ("Previsão do autor × resultado", "só atrasa a perda", "H_het", "rank_ef", "> 1500"):
        assert s in md, s
    # resultados vazios: código 1, sem pasta/figuras
    out_vazio = os.path.join(vazio, "out")
    for res in (vazio, os.path.join(vazio, "nao_existe")):
        r = subprocess.run([sys.executable, os.path.join(AQUI, "rodada2_tabela.py"), "--results", res,
                            "--saida", out_vazio], capture_output=True, text=True)
        assert r.returncode == 1 and "sem resultados" in r.stderr, (r.returncode, r.stderr)
        assert not os.path.exists(out_vazio)
        r = subprocess.run([sys.executable, os.path.join(AQUI, "rodada2_analise.py"), "--results", res],
                           capture_output=True, text=True)
        assert r.returncode == 1
    r = subprocess.run([sys.executable, os.path.join(AQUI, "rodada2_analise.py"), "--results", base],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "h_tempo:" in r.stdout, r.stdout
    for d in (base, os.path.dirname(saida), vazio):
        shutil.rmtree(d, ignore_errors=True)
    print("ok sonda, rank, reaprendizado, B_fixa, CLI completa e resultados vazios (código 1)")


def main():
    for nome, f in sorted(globals().items()):
        if nome.startswith("test_") and callable(f):
            f()
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
