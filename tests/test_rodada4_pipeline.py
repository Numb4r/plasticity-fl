"""Testes do pipeline da rodada 4: calibração de R, escolha de hiperparâmetro, plano, trava, verificador e o
teste curto em miniatura (sem rodar o experimento real)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import experiments.pipeline_r4 as p
from experiments.calibrar_r4 import calibrar_R
from src import config
from src.condicoes import aux_gerador, avaliar_cond, executar, particoes_da_condicao, pasta_calib, treinar_na_tarefa
from src.condicoes_r4 import CATALOGO_4A, CATALOGO_4B, COMBINACOES, arq_selecao, condicao_r4, tem_hp
from src.dados import Sequencia, particionar
from src.modelo import criar_mlp
from src.sementes import gerador, semente
from src.tecnicas_fed import iniciar_extra


@pytest.fixture(autouse=True)
def uma_thread():
    ant = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(ant)


def _alvo(tmp_path, valor=1.0, T=3):
    arq = tmp_path / "alvo.csv"
    pd.DataFrame({"loss_nova": [valor] * T}).to_csv(arq, index=False)
    return arq


def _treino_direto(cond, tarefa, R):
    """Perda média da rede nova depois de EXATAMENTE R rodadas, treinada como a execução real (T=3: refs 0, 1 e 2)."""
    from dataclasses import replace
    from src.config import RESET_CENTRAL, SEMENTE_CALIB
    from src.condicoes import _sp
    seq, tec = Sequencia(tarefa, SEMENTE_CALIB, 3), (json.loads(cond.tec) if cond.tec else None)
    part = particoes_da_condicao(cond, SEMENTE_CALIB)
    perdas = []
    for i, t in enumerate(config.tarefas_com_nova(3)):
        X, y = seq.dados(t)
        nova = criar_mlp(semente(SEMENTE_CALIB, "calib_nova", i))
        extra = iniciar_extra(tec, nova) if cond.familia == "fedx" else None
        _sp(cond, nova, SEMENTE_CALIB, i, 1)
        c = replace(cond, passos=RESET_CENTRAL * R) if cond.familia == "central" else replace(cond, rodadas=R)
        treinar_na_tarefa(c, nova, X, y, part, gerador(SEMENTE_CALIB, "calib_embaralha", i), extra, aux_gerador(cond, SEMENTE_CALIB, i, 1))
        perdas.append(avaliar_cond(cond, nova, extra, X, y)[1])
    return float(np.mean(perdas))


@pytest.mark.parametrize("nome,hp", [("fedprox", 0.1), ("sem_media_k1", None), ("central_aprendizado", None),
                                     ("gossip_anel", None), ("fedrep", None), ("fed_sp", 0.8), ("fed_redo_fed", 0.5),
                                     ("particao_iid_k25", None), ("copias_k5", None), ("particao_epocas_k5", None)])
def test_calibrar_R_e_exata_e_escolhe_o_menor_erro(nome, hp, tmp_path):
    cond = condicao_r4(nome, "A", tmp_path, hp=hp, rodadas=3, T=3)
    cal = calibrar_R(cond, "A", tmp_path, tmp_path / "c.json", grade=(2, 3, 4), T=3, alvo_csv=_alvo(tmp_path))
    for R in (2, 3, 4):                                                       # a leitura em R é o treino direto de R rodadas
        assert cal["tentativas"][R] == pytest.approx(_treino_direto(cond, "A", R), abs=1e-9)
    perdas = {int(k): v for k, v in cal["tentativas"].items()}
    alvo = _alvo(tmp_path, perdas[3])                                         # alvo = perda em R=3: escolhe R=3 com erro 0
    cal = calibrar_R(cond, "A", tmp_path, tmp_path / "c.json", grade=(2, 3, 4), T=3, alvo_csv=alvo)
    assert cal["rodadas"] == 3 and abs(cal["erro_rel"]) < 1e-12 and cal["ok"] is True
    longe = calibrar_R(cond, "A", tmp_path, tmp_path / "c.json", grade=(2, 3, 4), T=3, alvo_csv=_alvo(tmp_path, 1e6))
    assert longe["ok"] is False and longe["alvo_fora_do_alcance"] is True
    assert len(cal["alvo_sha256"]) == 64 and json.loads((tmp_path / "c.json").read_text())["condicao"] == nome


def _escrever_cand(d, nome, tarefa, hp, R, erro, ok, acc):
    arq = p.arq_calibracao(d, nome, tarefa, hp)
    arq.parent.mkdir(parents=True, exist_ok=True)
    arq.write_text(json.dumps({"rodadas": R, "erro_rel": erro, "ok": ok}))
    if ok:
        pasta = pasta_calib(d) / "selecao_hp"
        pasta.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"acc_cont": [0.0] * 25 + [acc] * 5}).to_csv(pasta / f"{nome}{p._sufixo_hp(hp)}_m3_{tarefa}_1000.csv", index=False)


def test_escolher_hp_maior_acuracia_entre_os_ok_empate_o_primeiro_e_sem_ok_o_menor_erro(tmp_path):
    grade = config.GRADE_FEDPROX_MU
    for hp, (R, erro, ok, acc) in zip(grade, [(80, 0.01, True, 0.6), (70, 0.2, False, 0.99), (90, -0.05, True, 0.8), (60, 0.0, True, 0.8)]):
        _escrever_cand(tmp_path, "fedprox", "A", hp, R, erro, ok, acc)
    r = p.escolher_hp("fedprox", "A", tmp_path)
    assert r["hp"] == grade[2] and r["rodadas"] == 90 and r["ok"] is True and r["borda"] is False     # não-ok fica fora; empate: o 1º
    assert json.loads(arq_selecao(tmp_path, "fedprox", "A").read_text())["hp"] == grade[2]
    for hp, (R, erro) in zip(grade, [(80, 0.3), (70, -0.2), (90, 0.25), (60, 0.5)]):
        _escrever_cand(tmp_path, "fedprox", "B", hp, R, erro, False, 0.0)
    r = p.escolher_hp("fedprox", "B", tmp_path)
    assert r["hp"] == grade[1] and r["ok"] is False                                                    # menor |erro|, marcado ok=false
    _escrever_cand(tmp_path, "fedprox", "A", grade[0], 80, 0.0, True, 0.99)
    assert p.escolher_hp("fedprox", "A", tmp_path)["borda"] is True and p.escolher_hp("fedprox", "A", tmp_path)["hp"] == grade[0]


def test_condicao_r4_le_a_selecao_combinacoes_e_voto(tmp_path):
    def sel(nome, hp, R, ok=True):
        for t in "AB":
            arq_selecao(tmp_path, nome, t).parent.mkdir(parents=True, exist_ok=True)
            arq_selecao(tmp_path, nome, t).write_text(json.dumps({"hp": hp, "rodadas": R, "ok": ok, "erro_rel": 0.0}))
    sel("fed_l2", 1e-4, 80); sel("fed_sp", 0.8, 70); sel("fed_l2init", 1e-3, 90); sel("fed_redo_local", 0.025, 60)
    sel("fed_redo_fed", 0.1, 85); sel("fed_l2_sp", None, 101, ok=False); sel("fedprox", 0.01, 77); sel("central_l2", 1e-5, 66)
    c = condicao_r4("fed_l2_sp", "A", tmp_path)
    assert json.loads(c.tec) == {"tipo": "fedavg", "wd": 1e-4} and c.sp_lambda == 0.8 and c.rodadas == 101 and c.calibracao_ok is False
    c = condicao_r4("fed_redo_ambos", "A", tmp_path, rodadas=5)
    assert json.loads(c.tec)["redo_local"] == {"tau": 0.025, "cada": 5} and json.loads(c.tec)["redo_fed"]["tau"] == 0.1
    c = condicao_r4("fed_redo_fed_voto", "A", tmp_path, hp=0.8, rodadas=5)                  # tau vem do ReDo federado já escolhido
    assert json.loads(c.tec)["redo_fed"] == {"tau": 0.1, "cada": 5, "modo": "voto", "q": 0.8}
    c = condicao_r4("fedprox", "B", tmp_path)
    assert json.loads(c.tec)["mu"] == 0.01 and c.rodadas == 77 and c.familia == "fedx" and c.particao == "iid"
    c = condicao_r4("central_l2", "A", tmp_path)
    assert (c.familia, c.passos, c.wd) == ("central", 15 * 66, 1e-5)
    assert condicao_r4("fedavg_iid", "A", tmp_path).rodadas == 75
    assert condicao_r4("diloco_h150", "A", tmp_path, rodadas=50).tec == json.dumps(
        {"H": 150, "lr_ext": 0.7, "mom": 0.9, "tipo": "diloco"}, sort_keys=True)
    with pytest.raises(FileNotFoundError):
        condicao_r4("scaffold", "A", tmp_path)                                                  # sem seleção e sem R explícito
    assert tem_hp("fedprox") and tem_hp("fed_l2_sp") and not tem_hp("scaffold") and not tem_hp("fedavg_iid")
    assert set(COMBINACOES) <= set(CATALOGO_4B)


def test_plano_4a_4b_e_estimativa():
    for parte, n in (("4a", 11), ("4b", 12)):
        plano = p.plano_confirmatorio(parte)
        nomes = [p.nome_csv(j) for j in plano]
        assert len(plano) == n * 2 * 8 and len(set(nomes)) == len(nomes)
        assert {j[2] for j in plano} == set(range(24, 32)) and 1000 not in {j[2] for j in plano}
        assert plano[0][0] == "fedavg_iid"
        e = p.estimar(parte)
        assert 5 < e["cpu_horas"][0] < e["cpu_horas"][1] < 200
    assert "gossip_completo" not in CATALOGO_4A and not {"fed_sp_cliente", "fed_sp_ambos"} & set(CATALOGO_4B)  # SP cliente = servidor: unido


def test_o_completo_recusa_sem_teste_curto_ok(tmp_path):
    with pytest.raises(SystemExit):
        p.fase_completo(None, tmp_path, "4a")
    pasta = tmp_path / "calibracao" / "teste_curto"
    pasta.mkdir(parents=True)
    (pasta / "RESULTADO.txt").write_text("PARE\nPARE: algo\n")
    with pytest.raises(SystemExit):
        p.fase_completo(None, tmp_path, "4b")


def test_verificador_acusa_o_que_falta_e_pega_o_log_da_rodada_errada(tmp_path):
    import experiments.verificar_rodada4 as v
    erros, n = v.verificar(tmp_path, "4a", None)
    assert n == 176 and any("teste curto" in e for e in erros) and any("selecao_hp_fedprox_A" in e for e in erros)
    log = tmp_path / "log.txt"
    log.write_text("INFO: tempos: {}\nINFO RODADA4 parte=4a fase=completo iniciada\n")
    erros, _ = v.verificar(tmp_path, "4a", log)
    assert not any("não é do completo" in e for e in erros)
    for errado in ("INFO: tempos: {}\nINFO calibração central_HT_cal2 A hp=0.1: R=80\n",       # log da rodada 2b
                   "INFO: tempos: {}\nINFO RODADA4 parte=4b fase=completo iniciada\n"):      # log da outra parte
        log.write_text(errado)
        assert any("não é do completo" in e for e in v.verificar(tmp_path, "4a", log)[0])
    log.write_text("PARE: algo\n")
    assert any("PARE" in e for e in v.verificar(tmp_path, "4a", log)[0])


# ---------------------------------------------------------------- teste curto de ponta a ponta, em miniatura
class _Serial:
    """Executor síncrono: roda cada job na hora, sem processos (só para o teste em miniatura)."""
    def submit(self, fn, *args):
        class _F:
            def __init__(self, v):
                self._v = v
            def result(self):
                return self._v
        return _F(fn(*args))


def _miniatura(monkeypatch):
    monkeypatch.setattr(p, "TC_R_PADRAO", 2)
    monkeypatch.setattr(p, "TC_R", {"diloco_h50": 1, "diloco_h150": 1})
    return p


@pytest.mark.parametrize("parte", ["3", "4a", "4b"])
def test_teste_curto_em_miniatura_pare_e_ok_e_libera_o_completo(parte, tmp_path, monkeypatch):
    _miniatura(monkeypatch)
    monkeypatch.setattr(p, "LIMITE_APRENDE", 0.99)                         # com 2 rodadas o fedavg_iid não chega perto disso
    assert p.fase_teste_curto(_Serial(), tmp_path, parte) == 3
    res = (tmp_path / "calibracao" / "teste_curto" / "RESULTADO.txt").read_text()
    assert res.startswith("PARE") and "fedavg_iid (R pequeno) não aprende" in res
    assert "sanidades" in res and "vs fedavg_iid: reproduzem" in res
    assert "calibração em miniatura" in res
    with pytest.raises(SystemExit):
        p.fase_completo(None, tmp_path, parte)
    monkeypatch.setattr(p, "LIMITE_APRENDE", 0.0)
    assert p.fase_teste_curto(_Serial(), tmp_path, parte) == 0               # a pasta é refeita
    res = (tmp_path / "calibracao" / "teste_curto" / "RESULTADO.txt").read_text()
    assert res.startswith("OK") and "PARE:" not in res
    monkeypatch.setattr(p, "fase_selecao", lambda ex, d, pt: (_ for _ in ()).throw(RuntimeError("gate abriu")))
    with pytest.raises(RuntimeError, match="gate abriu"):
        p.fase_completo(None, tmp_path, parte)


def test_teste_curto_acusa_remedio_que_nao_age(tmp_path, monkeypatch):
    _miniatura(monkeypatch)
    monkeypatch.setattr(p, "LIMITE_APRENDE", 0.0)
    monkeypatch.setitem(p.TC_HP, "fed_l2", 0.0)                            # L2 com lambda 0 = FedAvg: não age
    assert p.fase_teste_curto(_Serial(), tmp_path, "4b") == 3
    assert "fed_l2: resultado idêntico ao do fedavg_iid" in (tmp_path / "calibracao" / "teste_curto" / "RESULTADO.txt").read_text()


def test_calibrar_R_ignora_perda_nao_finita(tmp_path, monkeypatch):
    import experiments.calibrar_r4 as c4
    estado = {"r": 0}
    perdas = {1: 0.5, 2: float("nan"), 3: 1.0}

    def falso(cond, nova, X, y, particoes, g, extra, aux, cb):
        for r in (1, 2, 3):
            estado["r"] = r
            cb(nova, extra, r)

    monkeypatch.setattr(c4, "treinar_na_tarefa", falso)
    monkeypatch.setattr(c4, "avaliar_cond", lambda *a, **k: (0.0, perdas[estado["r"]]))
    cond = condicao_r4("sem_media_k1", "A", tmp_path, rodadas=3, T=3)
    cal = c4.calibrar_R(cond, "A", tmp_path, tmp_path / "c.json", grade=(1, 2, 3), T=3, alvo_csv=_alvo(tmp_path, 0.95))
    assert cal["rodadas"] == 3 and cal["ok"] is True                           # o R=2 (NaN) nunca é escolhido, nem com o alvo ao lado


def test_cache_da_calibracao_so_vale_com_o_mesmo_alvo(tmp_path):
    arq_alvo = p.arq_alvo(tmp_path, "A")
    arq_alvo.parent.mkdir(parents=True)
    pd.DataFrame({"loss_nova": [1.0] * 3}).to_csv(arq_alvo, index=False)
    import hashlib
    sha = hashlib.sha256(arq_alvo.read_bytes()).hexdigest()
    arq = p.arq_calibracao(tmp_path, "fedprox", "A", 0.1)
    arq.write_text(json.dumps({"rodadas": 99, "erro_rel": 0.0, "ok": True, "alvo_sha256": sha}))
    assert p.job_calibrar("fedprox", "A", 0.1, str(tmp_path))["rodadas"] == 99     # mesmo alvo: reaproveita
    arq.write_text(json.dumps({"rodadas": 99, "erro_rel": 0.0, "ok": True, "alvo_sha256": "outro"}))
    with pytest.raises(Exception):                                                  # alvo diferente: recalibra (aqui falha: T=30 esperado)
        p.job_calibrar("fedprox", "A", 0.1, str(tmp_path))
