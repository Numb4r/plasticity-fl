"""Testes da análise da rodada 3 com dados sintéticos em diretórios temporários (nada de results*/ nem figures/)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rodada3_analise as R  # noqa: E402
import rodada3_tabela as T  # noqa: E402

BASE = {"fedavg_iid": 0.10, "particao_iid_k2": 0.03, "particao_iid_k5": 0.06, "particao_iid_k20": 0.14,
        "copias_k2": 0.0, "copias_k5": 0.0, "copias_k10": 0.0, "copias_k20": 0.0, "sem_media_k1": 0.0,
        "particao_epocas_k2": 0.02, "particao_epocas_k5": 0.04, "particao_epocas_k20": 0.08}
REF = [3, 8, 12, 25, 26, 27, 28, 29]   # tarefas com gap (as da janela mais algumas)


def fazer(dir_, gaps=None, loss=None, calib_ruim=(), sem_arquivo=(), sem_semente=(), sem_tarefa=(), sem_calib=()):
    """Escreve CSVs e JSONs sintéticos. gaps/loss: {cond: valor} (ou {(cond, tipo): valor}) sobre o BASE."""
    g = dict(BASE)
    g.update(gaps or {})
    rng = np.random.default_rng(1)
    os.makedirs(os.path.join(dir_, "calibracao"), exist_ok=True)
    for c in R.CONDICOES:
        for t in R.TIPOS:
            for s in R.SEMENTES:
                if (c, s) in sem_arquivo:
                    continue
                gv = g[(c, t)] if (c, t) in g else g[c]
                lv = (loss or {}).get((c, t), (loss or {}).get(c, 0.50))
                linhas = []
                for tarefa in range(30):
                    if tarefa in REF and not (c, s, tarefa) in sem_tarefa:
                        gap = gv + rng.normal(0, 0.002)
                    else:
                        gap = np.nan
                    linhas.append(dict(tarefa=tarefa, cond=c, acc_cont=0.9, acc_nova=0.8, loss_nova=lv + rng.normal(0, 0.001),
                                       loss_cont=0.2, gap=gap, cos_medio=0.3, encolhimento=0.8, rank_ef=10.0,
                                       srank_feat=9.0))
                pd.DataFrame(linhas).to_csv(os.path.join(dir_, f"{c}_m3_{t}_{s}.csv"), index=False)
            if c != "fedavg_iid" and (c, t) not in sem_calib:
                with open(os.path.join(dir_, "calibracao", f"selecao_hp_{c}_{t}_m3.json"), "w") as f:
                    json.dump(dict(rodadas=75, ok=(c, t) not in calib_ruim, erro_rel=0.01), f)
    return str(dir_)


def rodar(tmp_path, **kw):
    d = fazer(tmp_path, **kw)
    df = R.carregar(d)
    return df, R.carregar_calibracao(d), R.decidir(df, R.carregar_calibracao(d))


# ---------------------------------------------------------------- base e P_rep

def test_base_tudo_atendida(tmp_path):
    df, calib, dec = rodar(tmp_path)
    assert dec["p_rep"]["ok"] is True
    assert [dec["hip"][h]["geral"] for h in ("H_dados", "H_cópias", "H_K")] == ["atendida"] * 3
    assert len(R.serie(df, "fedavg_iid", "A")) == 8
    assert all(abs(v - 0.10) < 0.01 for v in R.serie(df, "fedavg_iid", "A").values())


def test_p_rep_falhando_derruba_tudo(tmp_path):
    _, _, dec = rodar(tmp_path, gaps={("fedavg_iid", "B"): 0.04})   # 0,04 − 0 < δ em B
    assert dec["p_rep"]["ok"] is False
    for h in dec["hip"].values():
        assert h["geral"] == R.MSG_PREP
    assert {j["julgamento"] for j in R.julgar_previsao(dec)} == {"não conta"}


def test_p_rep_exige_inf_acima_de_delta(tmp_path):
    _, _, dec = rodar(tmp_path, gaps={"fedavg_iid": 0.05})   # média = δ: inf < δ
    assert dec["p_rep"]["ok"] is False


# ---------------------------------------------------------------- H_dados

def test_h_dados_nao_atendida(tmp_path):
    cop = {"copias_k5": 0.06, "copias_k10": 0.10, "copias_k20": 0.14}
    _, _, dec = rodar(tmp_path, gaps=cop)   # copias = particao: d = 0
    assert dec["hip"]["H_dados"]["geral"] == "não atendida"


def test_h_dados_parcial_k10_e_mais_um(tmp_path):
    # K=5 e K=10 valem; K=20 não (copias_k20 igual à partição)
    _, _, dec = rodar(tmp_path, gaps={"copias_k20": 0.14})
    assert dec["hip"]["H_dados"]["por"]["A"]["veredito"] == "parcial"
    assert dec["hip"]["H_dados"]["geral"] == "parcial"


def test_h_dados_sem_k10_nao_e_parcial(tmp_path):
    # vale em K=5 e K=20, mas não em K=10: "não atendida"
    _, _, dec = rodar(tmp_path, gaps={"copias_k10": 0.10})
    assert dec["hip"]["H_dados"]["geral"] == "não atendida"


def test_h_dados_k2_so_reportado(tmp_path):
    _, _, dec = rodar(tmp_path, gaps={"copias_k2": 0.03})   # K=2 não vale, mas fora dos critérios
    assert dec["hip"]["H_dados"]["geral"] == "atendida"
    assert dec["hip"]["H_dados"]["por"]["A"]["porK"][2]["res"] == "não vale"


# ---------------------------------------------------------------- H_cópias

def test_h_copias_nao_atendida(tmp_path):
    _, _, dec = rodar(tmp_path, gaps={"copias_k5": 0.1, "copias_k10": 0.1, "copias_k20": 0.1, "fedavg_iid": 0.2})
    assert dec["hip"]["H_cópias"]["geral"] == "não atendida"


def test_h_copias_parcial(tmp_path):
    _, _, dec = rodar(tmp_path, gaps={"copias_k20": 0.2})   # K=5 e 10 dentro de ±δ; K=20 fora
    assert dec["hip"]["H_cópias"]["geral"] == "parcial"


def test_h_copias_fora_em_qualquer_direcao(tmp_path):
    _, _, dec = rodar(tmp_path, gaps={"copias_k5": -0.2, "copias_k10": -0.2, "copias_k20": -0.2})
    assert dec["hip"]["H_cópias"]["geral"] == "não atendida"


def test_h_copias_ic_na_borda_fora(tmp_path):
    # média perto de δ: sup do IC > δ => não vale
    _, _, dec = rodar(tmp_path, gaps={"copias_k5": 0.052, "copias_k10": 0.052, "copias_k20": 0.052})
    assert dec["hip"]["H_cópias"]["geral"] == "não atendida"


def test_tarefas_diferentes_parcial_geral(tmp_path):
    _, _, dec = rodar(tmp_path, gaps={("copias_k5", "B"): 0.3, ("copias_k10", "B"): 0.3, ("copias_k20", "B"): 0.3,
                                      ("fedavg_iid", "B"): 0.3})
    h = dec["hip"]["H_cópias"]
    assert h["vers"] == {"A": "atendida", "B": "não atendida"}
    assert h["geral"] == "parcial"


# ---------------------------------------------------------------- H_K

def test_h_k_nao_atendida_ordem(tmp_path):
    _, _, dec = rodar(tmp_path, gaps={"particao_iid_k20": 0.05, "particao_iid_k5": 0.15})
    assert dec["hip"]["H_K"]["geral"] == "não atendida"


def _por_semente(df, cond, tipo, valores):
    """Sobrescreve o gap da janela de `cond` com um valor por semente (na ordem de R.SEMENTES)."""
    for s, v in zip(R.SEMENTES, valores):
        m = (df["cond"] == cond) & (df["tipo"] == tipo) & (df["semente"] == s) & df["gap"].notna()
        df.loc[m, "gap"] = v
    return df


def test_h_k_ordem_ok_mas_ic_nao_acima_de_zero(tmp_path):
    d = fazer(tmp_path)
    df, calib = R.carregar(d), R.carregar_calibracao(d)
    # médias 0,098 < 0,100 < 0,103 (ordem ok), mas gap(20) − gap(5) = ±0,03 alternando: IC contém 0
    _por_semente(df, "particao_iid_k5", "A", [0.07, 0.13] * 4)
    _por_semente(df, "particao_iid_k20", "A", [0.10, 0.10] * 4)
    _por_semente(df, "fedavg_iid", "A", [0.099, 0.101] * 4)
    x = R.h_k(df, calib, "A")
    assert x["t"]["ordem"] is False   # médias 0,10 / 0,099 / 0,10: ordem estrita falha
    _por_semente(df, "particao_iid_k5", "A", [0.07, 0.12, 0.08, 0.12, 0.09, 0.11, 0.07, 0.12])    # média 0,0975
    _por_semente(df, "particao_iid_k20", "A", [0.12, 0.07, 0.13, 0.08, 0.12, 0.10, 0.11, 0.09])   # média 0,1025
    x = R.h_k(df, calib, "A")
    assert x["t"]["ordem"] is True and x["t"]["ic_acima0"] is False
    assert x["veredito"] == "não atendida"


def test_h_k_sem_faixa_dinamica(tmp_path):
    # as três médias a menos de 0,01 de 0 (P_rep precisa passar: fedavg_iid sobe? não: usa k1 negativa)
    g = {"particao_iid_k5": 0.002, "fedavg_iid": 0.004, "particao_iid_k20": 0.006, "sem_media_k1": -0.2}
    _, _, dec = rodar(tmp_path, gaps=g)
    # P_rep: fedavg 0,004 − (−0,2) > δ; copias ficam em 0 (nível de H_cópias irrelevante aqui)
    assert dec["p_rep"]["ok"] is True
    h = dec["hip"]["H_K"]
    assert h["vers"] == {"A": R.SEM_FAIXA, "B": R.SEM_FAIXA}
    assert h["geral"] == "parcial"   # "inconclusivo" cai nos "demais casos"


def test_h_k_inconclusivo_e_atendida_e_parcial(tmp_path):
    g = {("particao_iid_k5", "A"): 0.002, ("fedavg_iid", "A"): 0.004, ("particao_iid_k20", "A"): 0.006,
         ("sem_media_k1", "A"): -0.2}
    _, _, dec = rodar(tmp_path, gaps=g)
    assert dec["hip"]["H_K"]["vers"]["A"] == R.SEM_FAIXA
    assert dec["hip"]["H_K"]["vers"]["B"] == "atendida"
    assert dec["hip"]["H_K"]["geral"] == "parcial"


# ---------------------------------------------------------------- calibração, nível, dados faltando

def test_calibracao_ok_false(tmp_path):
    _, _, dec = rodar(tmp_path, calib_ruim={("copias_k5", "A")})
    # a calibração vale nas duas tarefas: B também fica não avaliável para H_dados e H_cópias
    h = dec["hip"]["H_dados"]
    assert h["vers"]["A"].startswith("não avaliável") and h["vers"]["B"].startswith("não avaliável")
    assert "calibração não atingida" in h["vers"]["A"]
    assert h["geral"].startswith("não avaliável")
    assert dec["hip"]["H_K"]["geral"] == "atendida"   # H_K não usa copias_k5


def test_calibracao_arquivo_ausente(tmp_path):
    _, _, dec = rodar(tmp_path, sem_calib={("copias_k10", "B")})
    assert "falta" in dec["hip"]["H_cópias"]["vers"]["A"] and "selecao_hp_copias_k10_B_m3.json" in \
        dec["hip"]["H_cópias"]["vers"]["A"]


def test_nivel_de_aprendizado_diferente(tmp_path):
    _, _, dec = rodar(tmp_path, loss={"copias_k20": 0.56})   # 12% acima do fedavg (0,50)
    h = dec["hip"]["H_dados"]
    assert h["vers"]["A"] == "não avaliável (K=20: não avaliável (nível de aprendizado diferente))"
    assert h["geral"].startswith("não avaliável")
    assert dec["hip"]["H_K"]["geral"] == "atendida"


def test_nivel_dentro_de_10_por_cento_e_elegivel(tmp_path):
    df, calib, dec = rodar(tmp_path, loss={"copias_k20": 0.54})
    assert R.elegibilidade(df, calib, "copias_k20", "A")["elegivel"]
    assert dec["hip"]["H_dados"]["geral"] == "atendida"


def test_acc_nova_so_reportada(tmp_path):
    d = fazer(tmp_path)
    df = R.carregar(d)
    df.loc[df["cond"] == "copias_k5", "acc_nova"] = 0.5   # muito diferente: não muda a elegibilidade
    calib = R.carregar_calibracao(d)
    e = R.elegibilidade(df, calib, "copias_k5", "A")
    assert e["elegivel"] and e["acc_ok"] is False
    assert R.decidir(df, calib)["hip"]["H_dados"]["geral"] == "atendida"


def test_falta_semente(tmp_path):
    _, _, dec = rodar(tmp_path, sem_arquivo={("copias_k10", 19)})
    v = dec["hip"]["H_dados"]["vers"]["A"]
    assert v.startswith("não avaliável") and "semente(s) 19" in v
    assert dec["hip"]["H_K"]["geral"] == "atendida"   # H_K não usa copias_k10


def test_falta_semente_do_fedavg_derruba_p_rep(tmp_path):
    _, _, dec = rodar(tmp_path, sem_arquivo={("fedavg_iid", 16)})
    assert dec["p_rep"]["ok"] is None and dec["p_rep"]["msg"].startswith("não avaliável (falta")
    assert dec["hip"]["H_K"]["geral"].startswith("não avaliável")


def test_falta_tarefa_da_janela(tmp_path):
    _, _, dec = rodar(tmp_path, sem_tarefa={("particao_iid_k5", 20, 27)})
    assert dec["hip"]["H_K"]["vers"]["A"].startswith("não avaliável")
    assert "tarefa(s) da janela" in dec["hip"]["H_K"]["vers"]["A"]


def test_semente_1000_e_descartada(tmp_path):
    d = fazer(tmp_path)
    pd.read_csv(os.path.join(d, "copias_k5_m3_A_16.csv")).to_csv(os.path.join(d, "copias_k5_m3_A_1000.csv"), index=False)
    with pytest.warns(UserWarning, match="descartada"):
        df = R.carregar(d)
    assert 1000 not in set(df["semente"])


# ---------------------------------------------------------------- relatórios à parte, épocas, mecanismo

def test_relato_gap20_menos_gap5(tmp_path):
    df, calib, dec = rodar(tmp_path)
    x = dec["hip"]["H_K"]["por"]["A"]["t"]
    assert x["ic_acima0"] is True and x["c"]["inf"] > 0
    assert abs(x["c"]["media"] - 0.08) < 0.01


def test_epocas_sem_veredito_e_contrastes(tmp_path):
    df, calib, _ = rodar(tmp_path, gaps={"particao_epocas_k20": 0.12})
    b = R.braco_epocas(df, calib, "A")
    assert "veredito" not in b
    assert b["dk"][5]["c"]["media"] == pytest.approx(0.04, abs=0.01)
    assert b["dk"][10]["c"]["media"] == pytest.approx(0.10, abs=0.01)   # fedavg_iid − copias_k10
    assert b["tend"]["ordem"] is True


def test_mecanismo_junta_rank_e_srank(tmp_path):
    df, _, _ = rodar(tmp_path)
    m = R.mecanismo(df, "A")
    assert set(m[("copias", 5)]) == {"cos_medio", "encolhimento", "rank_ef", "srank_feat"}
    assert m[("particao_iid", 10)]["cos_medio"]["media"] == pytest.approx(0.3)
    assert ("sem_media_k1", 1) in m


# ---------------------------------------------------------------- previsão do autor

def _prev(dec):
    return {j["item"]: j["julgamento"] for j in R.julgar_previsao(dec)}


def test_previsao_acerta(tmp_path):
    g = {"copias_k5": 0.2, "copias_k10": 0.2, "copias_k20": 0.2}   # d_K < 0 (H_dados) e longe de k1 (H_cópias)
    _, _, dec = rodar(tmp_path, gaps=g)
    assert dec["hip"]["H_dados"]["geral"] == "não atendida" and dec["hip"]["H_cópias"]["geral"] == "não atendida"
    assert _prev(dec) == {"H_dados": "acerta", "H_cópias": "acerta", "H_K": "acerta"}


def test_previsao_erra(tmp_path):
    _, _, dec = rodar(tmp_path)   # base: H_dados, H_cópias, H_K todas atendidas
    assert _prev(dec) == {"H_dados": "erra", "H_cópias": "erra", "H_K": "acerta"}
    _, _, dec2 = rodar(tmp_path / "x", gaps={"particao_iid_k20": 0.05, "particao_iid_k5": 0.15})
    assert _prev(dec2)["H_K"] == "erra"


def test_previsao_acerta_em_parte_e_nao_conta(tmp_path):
    _, _, dec = rodar(tmp_path, gaps={"copias_k20": 0.14})   # H_dados parcial
    assert _prev(dec)["H_dados"] == "acerta em parte"
    _, _, dec2 = rodar(tmp_path / "y", loss={"copias_k20": 0.7})
    assert _prev(dec2)["H_dados"] == "não conta" and _prev(dec2)["H_cópias"] == "não conta"


# ---------------------------------------------------------------- tabela e figura

def test_tabela_e_figura_em_tmp(tmp_path):
    d = fazer(tmp_path / "res")
    saida, fig = tmp_path / "out" / "t.md", tmp_path / "out" / "f.png"
    assert T.main(["--results", d, "--saida", str(saida), "--figura", str(fig)]) == 0
    txt = saida.read_text(encoding="utf-8")
    for s in ("P_rep", "H_dados", "H_cópias", "H_K", "épocas", "rank_ef", "srank_feat", "Previsão", "(n=8)"):
        assert s in txt
    assert fig.exists() and fig.stat().st_size > 0


def test_sem_resultados_nao_cria_nada(tmp_path):
    assert T.main(["--results", str(tmp_path / "nada"), "--saida", str(tmp_path / "o" / "t.md")]) == 1
    assert not (tmp_path / "o").exists()
    assert R.main(["--results", str(tmp_path / "nada")]) == 1
