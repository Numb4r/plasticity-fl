"""Teste da análise da rodada 2 com CSVs SINTÉTICOS (nenhum dado real; tudo em tempfile.mkdtemp()).
Rodar da raiz: .venv/bin/python plots/test_rodada2.py"""
import json
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


RANK = {"central_HT_cal": 20.0, "central_HT": 100.0, "central_lookahead": 200.0}   # rank base por condição


def gerar(pasta, nivel=None, ruido=None, loss_ht=0.05, het=None, longo=None, sem_ht_semente=None,
          com_longo=True, com_bfixa=True, cal_ok=None, het_ruido=None):
    """Gera o conjunto sintético. nivel[(cond, tipo)] = gap na janela; ruido[cond]; het[tipo]=(g0.1, g1, g100);
    longo[(cond, tipo)] = tarefa a partir da qual o gap cai a -0,4 (None = nunca)."""
    niv = {("fedavg_iid", t): -0.10 for t in "AB"}
    niv.update({("media_sem_particao", t): -0.02 for t in "AB"})
    niv.update({("sem_media_k1", t): -0.30 for t in "AB"})
    niv.update({("central_aprendizado", t): -0.25 for t in "AB"})
    niv.update({("central_HT_cal", t): -0.02 for t in "AB"})
    niv.update({("central_HT", t): -0.12 for t in "AB"})
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
                if c == "central_HT_cal" and s == sem_ht_semente:
                    continue
                lin = [_linha(c, t, s, k, _gap_t(niv[(c, t)], rui[c], s, k, 30),
                              loss_nova=loss_ht if c == "central_HT_cal" else 0.05,
                              ref_rank=RANK.get(c, 10.0)) for k in range(30)]
                _salvar(pasta, f"{c}_m3_{t}_{s}.csv", lin)
            for i, (a, g) in enumerate(zip(R.ALFAS, het[t])):
                # het_ruido: ruído por alfa com padrão deslocado entre os alfa (a diferença pareada deixa de ser constante)
                sr = s if het_ruido is None else 6 + (s - 6 + 2 * i) % 5
                lin = [_linha("fedavg_dirichlet", t, s, k, _gap_t(g, het_ruido or 0.004, sr, k, 30), alfa=a)
                       for k in range(30)]
                _salvar(pasta, f"fedavg_dirichlet_a{a:g}_m3_{t}_{s}.csv", lin)
    cal_ok = {"A": True, "B": True, **(cal_ok or {})}
    os.makedirs(os.path.join(pasta, "calibracao"), exist_ok=True)
    for t in "AB":
        ok = cal_ok[t]
        for nome, d in ((f"calibracao_HT_{t}_m3.json", dict(ok=ok, passos_locais=60, erro_rel=0.04 if ok else 0.25,
                                                           alvo_loss_nova_msp=0.05, loss_ht=0.052)),
                        (f"selecao_HTcal_{t}_m3.json", dict(mu=0.9, borda=False)),
                        (f"selecao_HT_{t}_m3.json", dict(mu=0.99, borda=True)),
                        (f"selecao_lookahead_{t}_m3.json", dict(mu=0.5, borda=True))):
            with open(os.path.join(pasta, "calibracao", nome), "w") as f:
                json.dump(d, f)
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
        return R.h_tempo(R.carregar(p), R.carregar_calibracao(p))
    finally:
        shutil.rmtree(p)


def test_h_tempo_desfechos():
    casos = {
        "H_tempo": dict(nivel={("central_HT_cal", t): -0.015 for t in "AB"}),
        "HT acima": dict(nivel={("central_HT_cal", t): 0.10 for t in "AB"}),
        "H_dist": dict(nivel={("central_HT_cal", t): -0.15 for t in "AB"}),
        "intermediário": dict(nivel={("central_HT_cal", t): -0.02 for t in "AB"}, ruido={"central_HT_cal": 0.05}),
    }
    for esperado, kw in casos.items():
        v = _veredito(**kw)
        assert v["veredito"] == esperado, (esperado, v["veredito"], v["tarefas"]["A"]["d"])
    v = _veredito(nivel={("central_HT_cal", "A"): -0.015, ("central_HT_cal", "B"): 0.10})
    assert v["veredito"].startswith("misto") and "A: dentro" in v["veredito"] and "B: acima" in v["veredito"], v
    # critério secundário literal e lookahead (exploratório, sem veredito)
    v = _veredito(nivel={("central_HT_cal", t): 0.10 for t in "AB"})
    assert v["tarefas"]["A"]["sec_dentro"] is False and v["tarefas"]["A"]["sec_acima"] is False
    v = _veredito(nivel={("central_HT_cal", t): -0.02 for t in "AB"})
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
    assert "falta central_HT_cal/A sementes 9" in v["veredito"], v["veredito"]
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


def test_calibracao_nao_atingida():
    # ok=false em A (ou em B, ou nas duas) -> H_tempo e H_dist "não avaliável (calibração não atingida)"
    for cal in ({"A": False}, {"B": False}, {"A": False, "B": False}):
        v = _veredito(cal_ok=cal, nivel={("central_HT_cal", t): 0.10 for t in "AB"})   # sem a calibração seria "HT acima"
        assert v["veredito"] == "não avaliável (calibração não atingida)", v["veredito"]
        assert v["calib_falha"] == sorted(cal), v["calib_falha"]
        p = _novo()
        try:
            gerar(p, cal_ok=cal, com_longo=False, com_bfixa=False)
            r = R.resumo_vereditos(R.carregar(p), R.carregar_calibracao(p))
        finally:
            shutil.rmtree(p)
        assert r["h_tempo"] == r["h_dist"] == "não avaliável (calibração não atingida)", r
    # ordem e lista de todos os motivos: falta de dado > calibração > P_vant/elegibilidade
    v = _veredito(cal_ok={"A": False}, sem_ht_semente=9, loss_ht=0.5)
    assert v["veredito"].startswith("não avaliável (falta central_HT_cal/A sementes 9"), v["veredito"]
    assert v["veredito"].index("falta") < v["veredito"].index("calibração não atingida")
    v = _veredito(cal_ok={"B": False}, loss_ht=0.5, nivel={("media_sem_particao", t): -0.31 for t in "AB"})
    m = v["motivos"]
    assert m[0] == "calibração não atingida" and m[1].startswith("P_vant") and m[2].startswith("nível de aprendizado"), m
    # segundo filtro continua valendo com a calibração ok
    v = _veredito(loss_ht=0.5)
    assert v["veredito"].startswith("não avaliável") and "central_HT_cal" in v["veredito"] and v["calib_falha"] == []
    # JSON ausente com calibração pedida = falta de dado; sem pedir (calib=None) a checagem é pulada
    p = _novo()
    try:
        gerar(p, com_longo=False, com_bfixa=False)
        os.remove(os.path.join(p, "calibracao", "calibracao_HT_A_m3.json"))
        df = R.carregar(p)
        v = R.h_tempo(df, R.carregar_calibracao(p))["veredito"]
        assert v.startswith("não avaliável (falta calibracao_HT_A_m3.json"), v
        assert R.h_tempo(df)["veredito"] != v and R.h_tempo(df)["calib_checada"] is False
    finally:
        shutil.rmtree(p)
    print("ok calibração ok=false em A, B e nas duas; ordem e lista dos motivos; JSON ausente")


def test_exploratorios_e_tabela_calibracao():
    v = _veredito()   # HT_cal -0,02; HT -0,12; msp -0,02; k1 -0,30; lookahead -0,05
    for t in "AB":
        a, b = v["expl_ht_msp"][t], v["expl_ht_k1"][t]
        assert abs(a["d"]["media"] - (-0.10)) < 1e-9 and a["d"]["n"] == 5 and a["falta"] is None, a["d"]
        assert abs(b["d"]["media"] - 0.18) < 1e-9 and b["d"]["n"] == 5 and len(b["por_semente"]) == 5, b["d"]
        assert abs(b["d_acc_cont"]["media"] - 0.18) < 1e-9       # a rede nova é a mesma: d(gap) = d(acc_cont)
        assert all(abs(x - 0.18) < 0.02 for x in b["por_semente"].values)
        assert abs(v["lookahead"][t]["d"]["media"] - (-0.03)) < 1e-9
        assert v["tarefas"][t]["classe"] == "dentro"               # o veredito usa HT_cal, não o HT
    assert v["veredito"] == "H_tempo", v["veredito"]
    # os exploratórios não alteram o veredito: HT longe de msp e HT_cal equivalente
    assert _veredito(nivel={("central_HT", t): 0.5 for t in "AB"})["veredito"] == "H_tempo"
    # falta de semente do central_HT: o exploratório diz "falta", o veredito de HT_cal segue
    p = _novo()
    try:
        gerar(p, com_longo=False, com_bfixa=False)
        os.remove(os.path.join(p, "central_HT_m3_A_9.csv"))
        r = R.h_tempo(R.carregar(p), R.carregar_calibracao(p))
        assert "central_HT/A sementes 9" in r["expl_ht_msp"]["A"]["falta"] and r["veredito"] == "H_tempo"
    finally:
        shutil.rmtree(p)
    # tabela de calibração: S, erro_rel, ok, μ dos três (tabela_decisao.md)
    base, saida = _novo(), os.path.join(_novo(), "out")
    try:
        gerar(base, cal_ok={"B": False})
        r = subprocess.run([sys.executable, os.path.join(AQUI, "rodada2_tabela.py"), "--results", base,
                            "--saida", saida], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        md = open(os.path.join(saida, "rodada2_tabela_decisao.md"), encoding="utf-8").read()
        assert "| A | 60 | 0.040 | True | 0.050 | 0.052 | 0.9 | 0.99 [borda] | 0.5 [borda] |" in md, md[:3000]
        assert "| B | 60 | 0.250 | False |" in md
        assert "não avaliável (calibração não atingida)" in md and "(a) central_HT (15 passos" in md
        assert "(b) central_HT − sem_media_k1" in md and "6: " in md
    finally:
        shutil.rmtree(base)
        shutil.rmtree(os.path.dirname(saida))
    print("ok exploratórios (a), (b), lookahead e tabela de calibração")


def test_rank_grupo_com_ht_cal():
    p = _novo()
    try:
        gerar(p)
        rk = {(n, t): (a, b) for n, t, a, b in R.rank_tabela(R.carregar(p))}
    finally:
        shutil.rmtree(p)
    # rank_ef = base + 0,1 t (média das tarefas 25..29 = +2,7); grupo com média = {fedavg_iid 10, msp 10, HT_cal 20, dirichlet 10}
    assert abs(rk[("grupo: com média", "A")][0]["media"] - (12.5 + 2.7)) < 1e-9, rk[("grupo: com média", "A")][0]
    assert abs(rk[("grupo: com média", "A")][1]["media"] - (3 * 12.5 + 5.4)) < 1e-9   # srank_feat = 3 base + 0,2 t
    assert abs(rk[("grupo: sem média", "B")][0]["media"] - (10 + 2.7)) < 1e-9
    # central_HT e central_lookahead à parte, fora do grupo (não mexem no valor do grupo)
    assert abs(rk[("central_HT (fora do grupo)", "A")][0]["media"] - (100 + 2.7)) < 1e-9
    assert abs(rk[("central_lookahead (fora do grupo)", "B")][0]["media"] - (200 + 2.7)) < 1e-9
    assert ("central_HT_cal", "A") in rk and ("central_HT_cal (fora do grupo)", "A") not in rk
    assert R.GRUPO_COM == ("fedavg_iid", "media_sem_particao", "central_HT_cal", "fedavg_dirichlet")
    print("ok rank: grupo com média usa central_HT_cal; central_HT e lookahead fora do grupo")


def _fake_ht(v, cA="dentro", cB="dentro"):
    return dict(veredito=v, tarefas={"A": dict(classe=cA), "B": dict(classe=cB)})


def test_julgamento_previsao_proposto():
    col0 = {(c, "A"): R.stats_colapso({6: 30, 7: 30, 8: 30}) for c in ("fedavg_iid", "sem_media_k1", "central_aprendizado")}
    hh_ok = dict(geral="atendida")

    def jul(ht, hh=hh_ok, col=col0):
        return {x["item"].split(" (")[0]: x for x in R.julgar_previsao(ht, hh, col)}

    # regra 1 e 2: um só item para H_tempo/H_dist; só "HT acima" acerta
    j = jul(_fake_ht("HT acima", "acima", "acima"))
    assert len(R.julgar_previsao(_fake_ht("HT acima"), hh_ok, col0)) == 3
    assert j["H_tempo e H_dist"]["julgamento"] == "acerto" and "H_dist" not in [x["item"] for x in j.values()][1:]
    assert jul(_fake_ht("misto (A: acima, B: dentro)", "acima", "dentro"))["H_tempo e H_dist"]["julgamento"] == "acerto parcial"
    assert jul(_fake_ht("misto (A: dentro, B: abaixo)", "dentro", "abaixo"))["H_tempo e H_dist"]["julgamento"] == "erro"
    for v in ("H_tempo", "H_dist", "intermediário"):
        assert jul(_fake_ht(v))["H_tempo e H_dist"]["julgamento"] == "erro", v
    nv = jul(_fake_ht("não avaliável (calibração não atingida)"))["H_tempo e H_dist"]
    assert nv["julgamento"] == "não conta (não avaliável)" and nv["calculado"].startswith("não avaliável")
    # regra 4: H_het
    for g, esp in (("atendida", "acerto"), ("parcial", "acerto parcial"), ("não atendida", "erro"),
                   ("não avaliável (A: falta x)", "não conta (não avaliável)")):
        assert jul(_fake_ht("HT acima"), dict(geral=g))["H_het"]["julgamento"] == esp, g
    # H_het com ordem certa e IC pareado passando por 0 -> "não atendida" -> erro (ponta a ponta)
    p = _novo()
    try:
        certa = (-0.04, -0.02, 0.0)   # ordem prevista (100 > 1 > 0,1), mas o IC pareado passa por 0
        gerar(p, het={"A": certa, "B": certa}, het_ruido=0.08, com_longo=False, com_bfixa=False)
        hh = R.h_het(R.carregar(p))
    finally:
        shutil.rmtree(p)
    assert hh["geral"] == "não atendida" and hh["tarefas"]["A"]["d"]["inf"] <= 0, hh["tarefas"]["A"]["d"]
    assert hh["tarefas"]["A"]["gap"][100.0]["media"] > hh["tarefas"]["A"]["gap"][0.1]["media"]
    assert jul(_fake_ht("HT acima"), hh)["H_het"]["julgamento"] == "erro"
    # regra 3: censura à direita, mediana nas 3 sementes, empate não é "mais tarde"
    mk = lambda f, k, c: {("fedavg_iid", "A"): R.stats_colapso(f), ("sem_media_k1", "A"): R.stats_colapso(k),  # noqa: E731
                          ("central_aprendizado", "A"): R.stats_colapso(c)}
    cedo = {6: 20, 7: 20, 8: 20}
    # 2 de 3 sementes do fedavg sem colapso: mediana censurada = > T (mais tarde), mediana só dos colapsados = 70
    col = mk({6: 70, 7: None, 8: None}, cedo, cedo)
    assert col[("fedavg_iid", "A")]["mediana"] == 70 and col[("fedavg_iid", "A")]["mediana_cens"] == np.inf
    assert R.julgar_horizonte_proposta(col) == "só atrasa a perda"
    # 1 de 3 sementes sem colapso: mediana = a do meio (não é infinita)
    col = mk({6: 70, 7: 15, 8: None}, cedo, cedo)
    assert col[("fedavg_iid", "A")]["mediana_cens"] == 70 and R.julgar_horizonte_proposta(col) == "só atrasa a perda"
    # empate (mediana 20 = 20) não é "mais tarde"
    col = mk({6: 20, 7: 20, 8: 20}, cedo, cedo)
    assert R.julgar_horizonte_proposta(col).startswith("outro (")
    # empate entre censuradas (sem colapso mediano em fedavg e em k1) não é "mais tarde"
    col = mk({6: 70, 7: None, 8: None}, {6: 20, 7: None, 8: None}, cedo)
    assert R.julgar_horizonte_proposta(col).startswith("outro (") and "sem colapso (> T)" in R.julgar_horizonte_proposta(col)
    # nenhuma semente do fedavg colapsa e as outras sim: impede (não é o previsto)
    col = mk({6: None, 7: None, 8: None}, cedo, cedo)
    assert R.julgar_horizonte_proposta(col) == "impede a perda"
    assert jul(_fake_ht("HT acima"), col=col)["horizonte longo"]["julgamento"] == "erro"
    col = mk({6: 70, 7: None, 8: None}, cedo, cedo)
    assert jul(_fake_ht("HT acima"), col=col)["horizonte longo"]["julgamento"] == "acerto"
    # a regra já confirmada (mediana só entre as que colapsaram) segue descritiva e inalterada
    assert R.julgar_horizonte(mk({6: 70, 7: None, 8: None}, cedo, cedo)) == "só atrasa a perda"
    # falta de semente: não conta
    col = mk({6: 70, 7: None, 8: None}, cedo, cedo)
    col[("sem_media_k1", "A")] = R.stats_colapso({6: 20}, falta=[7, 8])
    assert jul(_fake_ht("HT acima"), col=col)["horizonte longo"]["julgamento"] == "não conta (não avaliável)"
    print("ok julgamento proposto: HT acima/misto/erro/não avaliável, H_het, censura à direita e empate")


def test_md_previsao_literal():
    sys.path.insert(0, AQUI)
    import rodada2_tabela as TB
    col0 = {(c, "A"): R.stats_colapso({6: 30, 7: 30, 8: 30}) for c in ("fedavg_iid", "sem_media_k1", "central_aprendizado")}
    md = "\n".join(TB.md_previsao(_fake_ht("HT acima"), dict(geral="atendida"), col0))
    assert "H_tempo não, H_dist não, H_het sim, FedAvg só atrasa" in md
    assert "PROPOSTA — aguarda confirmação do autor" in md and "| acerto |" in md and "| erro |" in md
    print("ok md_previsao escreve a previsão literal e o rótulo PROPOSTA")


def test_semente_1000_ignorada():
    base = _novo()
    gerar(base)
    v0 = R.h_tempo(R.carregar(base), R.carregar_calibracao(base))["veredito"]
    # planta a 1000 na raiz (valores absurdos) e em calibracao/
    lin = [_linha("central_HT_cal", "A", 1000, k, _gap_t(5.0, 0.0, 6, k, 30)) for k in range(30)]
    _salvar(base, "central_HT_cal_m3_A_1000.csv", lin)
    os.makedirs(os.path.join(base, "calibracao"), exist_ok=True)
    _salvar(os.path.join(base, "calibracao"), "central_HT_cal_m3_A_6.csv", lin)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = R.carregar(base)
    assert any("1000" in str(x.message) for x in w), [str(x.message) for x in w]
    assert set(df["semente"].unique()) == set(R.SEMENTES)
    assert R.h_tempo(df, R.carregar_calibracao(base))["veredito"] == v0
    assert len(df[(df["serie"] == "central_HT_cal") & (df["tipo"] == "A") & (df["braco"] == "T30")]) == 30 * 5   # calibracao/ não lida
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
    for s in ("Previsão do autor × resultado", "só atrasa a perda", "H_het", "rank_ef", "> 1500",
              "PROPOSTA — aguarda confirmação do autor", R.PREVISAO_LITERAL, "central_HT_cal", "borda",
              "Exploratórios", "(fora do grupo)"):
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
    assert r.returncode == 0 and "h_tempo:" in r.stdout and "h_dist:" in r.stdout, r.stdout
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
