"""Testes da rodada 3: partição em K clientes, cópias, braço de épocas locais fixas, condições, plano e pipeline."""
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import experiments.pipeline_r4 as p
import src.tecnicas_fed as tf
from src import config
from src.condicoes import Condicao, condicao_base, executar, particoes_da_condicao
from src.condicoes_r4 import CATALOGO, CATALOGO_3, condicao_r4
from src.dados import Sequencia, particionar, particionar_k
from src.federado import treinar_fedavg
from src.modelo import criar_mlp
from src.sementes import gerador, semente
from src.tecnicas_fed import iniciar_extra, treinar_fedx


@pytest.fixture(autouse=True)
def uma_thread():
    ant = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(ant)


def _igual(a, b):
    return all(torch.equal(x, z) for x, z in zip(a.state_dict().values(), b.state_dict().values()))


def test_particionar_k_iid_cobre_tudo_sem_sobreposicao_e_k10_e_o_iid_antigo():
    for K in (2, 5, 10, 25):
        partes = particionar_k("iid", 1, K)
        assert len(partes) == K and all(len(q) == 1000 // K for q in partes)
        assert torch.equal(torch.cat(partes).sort().values, torch.arange(1000))
    assert all(torch.equal(a, b) for a, b in zip(particionar_k("iid", 3, 10), particionar("iid", 3)))
    assert not torch.equal(particionar_k("iid", 3, 5)[0], particionar_k("iid", 4, 5)[0])           # depende da semente
    with pytest.raises(ValueError):
        particionar_k("outro", 1, 5)


def test_copias_sao_k_copias_dos_mesmos_1000_pontos_e_k10_e_a_particao_total():
    for K in (2, 5, 10, 25):
        partes = particionar_k("copias", 1, K)
        assert len(partes) == K and all(torch.equal(q, torch.arange(1000)) for q in partes)
    assert all(torch.equal(a, b) for a, b in zip(particionar_k("copias", 1, 10), particionar("total", 1)))


def _fedx(tec, rodadas=2, sem=1, t=0):
    X, y = Sequencia("A", sem, 3).dados(t)
    part = particionar_k(tec["modo"], sem, tec["K"]) if "K" in tec else particionar("iid", sem)
    m = criar_mlp(semente(sem, "init_continua"))
    cos = treinar_fedx(m, iniciar_extra(tec, m), X, y, part, gerador(sem, "embaralha_continua", t), rodadas, tec)
    return m, cos, (X, y, part)


def test_copias_k10_e_o_fedavg_com_a_particao_total_bit_a_bit_e_k10_iid_e_o_fedavg_iid():
    m, cos, (X, y, part) = _fedx({"tipo": "fedavg", "K": 10, "modo": "copias"})
    ref = criar_mlp(semente(1, "init_continua"))
    cos_ref = treinar_fedavg(ref, X, y, particionar("total", 1), gerador(1, "embaralha_continua", 0), 2)
    assert _igual(m, ref) and cos == cos_ref
    m2, cos2, _ = _fedx({"tipo": "fedavg", "K": 10, "modo": "iid"})
    ref2 = criar_mlp(semente(1, "init_continua"))
    cos_ref2 = treinar_fedavg(ref2, X, y, particionar("iid", 1), gerador(1, "embaralha_continua", 0), 2)
    assert _igual(m2, ref2) and cos2 == cos_ref2


def test_passos_locais_por_cliente_e_um_adam_por_cliente_por_rodada(monkeypatch):
    contagem = {"passos": 0, "adams": 0}
    step, init = torch.optim.Adam.step, torch.optim.Adam.__init__
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, *a, **k: (contagem.__setitem__("passos", contagem["passos"] + 1), step(self, *a, **k))[1])
    monkeypatch.setattr(torch.optim.Adam, "__init__", lambda self, *a, **k: (contagem.__setitem__("adams", contagem["adams"] + 1), init(self, *a, **k))[0])
    for tec, passos_por_cliente in (({"tipo": "fedavg", "K": 2, "modo": "iid"}, 15), ({"tipo": "fedavg", "K": 25, "modo": "iid"}, 15),
                                    ({"tipo": "fedavg", "K": 2, "modo": "iid", "S": 75}, 75), ({"tipo": "fedavg", "K": 25, "modo": "iid", "S": 6}, 6),
                                    ({"tipo": "fedavg", "K": 5, "modo": "copias"}, 15)):
        contagem.update(passos=0, adams=0)
        _fedx(tec, rodadas=2)
        assert contagem["passos"] == tec["K"] * passos_por_cliente * 2 and contagem["adams"] == tec["K"] * 2, tec


def test_k25_tem_40_pontos_2_lotes_inteiros_e_usa_o_mesmo_amostrador_das_demais(monkeypatch):
    chamadas = {"geral": 0, "ext": 0}
    geral, ext = tf.treinar_passos_geral, tf.treinar_passos_ext
    monkeypatch.setattr(tf, "treinar_passos_geral", lambda *a, **k: (chamadas.__setitem__("geral", chamadas["geral"] + 1), geral(*a, **k))[1])
    monkeypatch.setattr(tf, "treinar_passos_ext", lambda *a, **k: (chamadas.__setitem__("ext", chamadas["ext"] + 1), ext(*a, **k))[1])
    m, _, (X, y, part) = _fedx({"tipo": "fedavg", "K": 25, "modo": "iid"}, rodadas=2)
    assert all(len(q) == 40 for q in part) and 40 % config.BATCH == 0 and 40 // config.BATCH == 2
    assert chamadas["geral"] == 0 and chamadas["ext"] == 25 * 2                # nenhum cliente cai no caminho geral
    assert all(torch.isfinite(v).all() for v in m.state_dict().values())


def test_braco_de_epocas_fixas_tem_s_igual_a_teto_de_3n_sobre_20_e_k10_nao_existe():
    esperado = {2: 75, 5: 30, 25: 6}
    for K, S in esperado.items():
        c = condicao_r4(f"particao_epocas_k{K}", "A", Path("/tmp/x"), rodadas=3)
        assert json.loads(c.tec)["S"] == S and c.s_local == S and json.loads(c.tec)["K"] == K
    assert "particao_epocas_k10" not in CATALOGO_3 and "particao_iid_k10" not in CATALOGO_3        # uniões: K=10 é o fedavg_iid
    m10, _, _ = _fedx({"tipo": "fedavg", "K": 10, "modo": "iid", "S": 15})                           # 3 épocas de 100 pontos = 15 passos; em K = 25, 6 passos = 3 épocas exatas de 40 pontos
    assert _igual(m10, _fedx({"tipo": "fedavg"})[0])


def test_condicoes_da_rodada_3_tec_particao_e_k(tmp_path):
    c = condicao_r4("particao_iid_k5", "A", tmp_path, rodadas=4)
    assert (c.familia, c.particao, json.loads(c.tec)) == ("fedx", "iid", {"K": 5, "modo": "iid", "tipo": "fedavg"})
    c = condicao_r4("copias_k25", "B", tmp_path, rodadas=4)
    assert (c.particao, json.loads(c.tec)["K"], json.loads(c.tec)["modo"]) == ("copias", 25, "copias")
    assert condicao_r4("sem_media_k1", "A", tmp_path, rodadas=4).familia == "fl"
    assert particoes_da_condicao(condicao_r4("fedavg_iid", "A", tmp_path, rodadas=4), 1)[0].shape == (100,)
    assert len(particoes_da_condicao(condicao_r4("copias_k5", "A", tmp_path, rodadas=4), 1)) == 5


@pytest.mark.parametrize("nome", [n for n in CATALOGO_3 if n != "fedavg_iid"])
def test_executar_cada_condicao_da_rodada_3_em_miniatura(nome, tmp_path):
    cond = condicao_r4(nome, "A", tmp_path, rodadas=2, T=3)
    df = executar(cond, "A", 1, tmp_path, diagnosticos=False)
    assert len(df) == 3 and df[["acc_cont", "loss_cont", "norma", "acc_nova", "loss_nova"]].notna().all().all()
    if cond.familia == "fedx":
        assert df[["encolhimento", "cos_medio"]].notna().all().all()
    assert json.loads((tmp_path / f"{nome}_m3_A_1.params.json").read_text())["nivel"] == 3


def test_plano_da_rodada_3_tem_192_execucoes_sementes_16_a_23_e_custo_razoavel():
    plano = p.plano_confirmatorio("3")
    nomes = [p.nome_csv(j) for j in plano]
    assert len(CATALOGO["3"]) == 12 and len(plano) == 12 * 2 * 8 == 192 and len(set(nomes)) == 192
    assert {j[2] for j in plano} == set(range(16, 24)) and 1000 not in {j[2] for j in plano}
    e = p.estimar("3")
    assert 5 < e["cpu_horas"][0] < e["cpu_horas"][1] < 100


def test_gate_e_verificador_da_rodada_3(tmp_path):
    import experiments.verificar_rodada4 as v
    with pytest.raises(SystemExit):
        p.fase_completo(None, tmp_path, "3")
    erros, n = v.verificar(tmp_path, "3", None)
    assert n == 192 and any("teste curto" in e for e in erros) and any("selecao_hp_copias_k5_A" in e for e in erros)
    log = tmp_path / "log.txt"
    log.write_text("INFO: tempos: {}\nINFO RODADA4 parte=4a fase=completo iniciada\n")                 # log de outra parte
    assert any("não é do completo" in e for e in v.verificar(tmp_path, "3", log)[0])
    log.write_text("INFO: tempos: {}\nINFO RODADA4 parte=3 fase=completo iniciada\n")
    assert not any("não é do completo" in e for e in v.verificar(tmp_path, "3", log)[0])
