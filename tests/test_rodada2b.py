"""Testes da rodada 2b: central_HT_cal2 (R calibrado), calibração de R, seleção de μ, plano, gate e verificação."""
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

import src.condicoes as cs
from src import config
from src.dados import Sequencia, particionar
from src.media_tempo import clonar_estado, treinar_ht
from src.metricas import perda_ce
from src.modelo import criar_mlp
from src.sementes import gerador, semente


def _executar(cond, sem=1):
    return cs.executar(cond, "A", sem, Path(tempfile.mkdtemp()), diagnosticos=False)


def test_condicao_r2_central_ht_cal2_le_r_e_mu_e_o_override_vale_para_todas(tmp_path):
    pasta = tmp_path / "calibracao"
    pasta.mkdir(parents=True)
    (pasta / "calibracao_HT2_A_m3.json").write_text(json.dumps({"rodadas": 101, "ok": False}))
    (pasta / "selecao_HTcal2_A_m3.json").write_text(json.dumps({"mu": 0.995}))
    c = cs.condicao_r2("central_HT_cal2", "A", tmp_path)
    assert (c.familia, c.rodadas, c.s_local, c.mu, c.calibracao_ok, c.T) == ("ht", 101, 15, 0.995, False, 30)
    c = cs.condicao_r2("central_HT_cal2", "A", tmp_path, mu=0.9, rodadas=90)          # explícito: não lê JSON
    assert (c.rodadas, c.mu, c.s_local) == (90, 0.9, 15)
    assert cs.condicao_r2("sem_media_k1", "A", tmp_path, rodadas=90).rodadas == 90      # o override vale para as outras
    assert cs.condicao_r2("sem_media_k1", "A", tmp_path).rodadas == 75                  # e o padrão não mudou
    assert cs.condicao_r2("media_sem_particao", "B", tmp_path).rodadas == 75


def test_ht_cal2_com_mu_zero_e_r_diferente_de_75_reproduz_o_sem_media_k1_com_o_mesmo_r():
    k1 = cs.condicao_r2("sem_media_k1", "A", Path("."), T=3, rodadas=4)
    ht = cs.condicao_r2("central_HT_cal2", "A", Path("."), T=3, rodadas=4, mu=0.0)
    a, b = _executar(k1), _executar(ht)
    cols = [c for c in a.columns if c not in ("cond", "mu", "acc_tartaruga")]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)
    ht9 = _executar(replace(ht, mu=0.9))
    assert not a["acc_cont"].equals(ht9["acc_cont"]) or not a["loss_cont"].equals(ht9["loss_cont"])   # μ age


def test_callback_do_treinar_ht_so_le_e_recebe_as_rodadas_em_ordem():
    seq = Sequencia("A", 1, 3)
    X, y = seq.dados(0)
    part = particionar("um", 1)

    def treinar(callback):
        m = criar_mlp(semente(1, "init_nova", 0))
        extra = {"tartaruga": clonar_estado(m)}
        treinar_ht(m, extra, X, y, part, gerador(1, "embaralha_nova", 0), 5, 0.5, ao_fim_de_rodada=callback)
        return m

    vistas = []
    com = treinar(lambda m, r: vistas.append((r, perda_ce(m, X, y))))
    sem = treinar(None)
    assert [r for r, _ in vistas] == [1, 2, 3, 4, 5]
    assert all(torch.equal(a, b) for a, b in zip(com.state_dict().values(), sem.state_dict().values()))
    assert vistas[-1][1] == perda_ce(sem, X, y)                                           # a leitura da última rodada = o modelo final


def _fake_ref(tmp_path, alvo, n=3):
    ref = tmp_path / "msp.csv"
    pd.DataFrame({"loss_nova": [alvo] * n}).to_csv(ref, index=False)
    return ref


def test_calibrar_ht2_e_exato_escolhe_o_menor_erro_e_marca_ok(tmp_path):
    from experiments.calibrar_ht2 import calibrar_ht2
    grade = (3, 4, 5, 6)
    def cal(alvo):
        return calibrar_ht2("A", tmp_path, T=3, grade=grade, referencia=_fake_ref(tmp_path, alvo), saida=tmp_path / "c.json")
    r = cal(1000.0)                                                                       # alvo enorme: o R de maior perda
    perdas = {int(k): v for k, v in r["tentativas"].items()}
    assert set(perdas) == set(grade) and r["rodadas"] == max(perdas, key=perdas.get) and r["ok"] is False
    assert r["alvo_fora_do_alcance"] is True
    # exatidão: a perda média no R escolhido é a do MESMO código de treino da execução real, rede a rede
    seq, part = Sequencia("A", config.SEMENTE_CALIB, 3), particionar("um", config.SEMENTE_CALIB)
    perdas_reais = []
    for i, t in enumerate(config.tarefas_com_nova(3)):
        X, y = seq.dados(t)
        m = criar_mlp(semente(config.SEMENTE_CALIB, "calib_nova", i))
        treinar_ht(m, {"tartaruga": clonar_estado(m)}, X, y, part, gerador(config.SEMENTE_CALIB, "calib_embaralha", i), 5, 0.0)
        perdas_reais.append(perda_ce(m, X, y))
    assert abs(perdas[5] - sum(perdas_reais) / 3) < 1e-9                                  # R=5 treinado direto
    r = cal(perdas[5])                                                                    # alvo = perda em R=5 exatamente: R=5, ok
    assert r["rodadas"] == 5 and r["ok"] is True and abs(r["erro_rel"]) < 1e-12
    r = cal(perdas[4])                                                                    # alvo = perda em R=4 exatamente
    assert r["rodadas"] == 4 and abs(r["erro_rel"]) < 1e-12
    r = cal(perdas[6] * 1.5)                                                              # longe de qualquer R: escolhe o mais próximo, ok=False
    melhor = min(perdas, key=lambda k: (abs(perdas[k] / (perdas[6] * 1.5) - 1), k))
    assert r["rodadas"] == melhor and (r["ok"] is False) == (abs(perdas[melhor] / (perdas[6] * 1.5) - 1) > 0.10)
    assert json.loads((tmp_path / "c.json").read_text())["rodadas"] == melhor              # o JSON é o da última calibração


def test_escolher_mu_2b_grade_empate_menor_e_borda(tmp_path):
    import experiments.pipeline_r2b as p
    pasta = tmp_path / "calibracao" / "selecao_mu"
    pasta.mkdir(parents=True)
    def fake(accs):
        for mu, a in zip(config.MUS_2B, accs):
            pd.DataFrame({"acc_cont": [a] * 30}).to_csv(pasta / f"central_HT_cal2_mu{mu:g}_m3_A_1000.csv", index=False)
    assert config.MUS_2B == (0.9, 0.99, 0.995, 0.999)
    fake([0.5, 0.6, 0.9, 0.7])
    r = p.escolher_mu("A", tmp_path)
    assert r["mu"] == 0.995 and r["borda"] is False
    fake([0.5, 0.6, 0.7, 0.9])
    assert p.escolher_mu("A", tmp_path)["borda"] is True
    fake([0.8, 0.9, 0.9, 0.6])                                                            # empate: o menor
    assert p.escolher_mu("A", tmp_path)["mu"] == 0.99
    assert json.loads((tmp_path / "calibracao" / "selecao_HTcal2_A_m3.json").read_text())["mu"] == 0.99


def test_plano_da_2b_tem_30_execucoes_sementes_11_a_15_sem_colisao():
    import experiments.pipeline_r2b as p
    plano = p.plano_confirmatorio()
    nomes = [p.nome_csv(j) for j in plano]
    assert len(plano) == 30 and len(set(nomes)) == 30
    assert {j[2] for j in plano} == {11, 12, 13, 14, 15} and {j[1] for j in plano} == {"A", "B"} and {j[3] for j in plano} == {30}
    assert {j[0] for j in plano} == {"central_HT_cal2", "media_sem_particao", "sem_media_k1"}
    assert sum(1 for j in plano if j[0] == "central_HT_cal2") == 10
    assert "central_HT_cal2_m3_A_11.csv" in nomes
    assert not {6, 7, 8, 9, 10, 1000} & {j[2] for j in plano}                             # sementes novas, sem a 1000
    assert config.R_CAL2 == tuple(range(75, 151))
    e = p.estimar()
    assert 0.5 < e["cpu_horas"][0] < e["cpu_horas"][1] < 10


def test_completo_da_2b_recusa_sem_teste_curto_ok_e_verificador_acusa_o_que_falta(tmp_path):
    import experiments.pipeline_r2b as p
    import experiments.verificar_rodada2b as v
    with pytest.raises(SystemExit):
        p.fase_completo(None, tmp_path)
    pasta = tmp_path / "calibracao" / "teste_curto"
    pasta.mkdir(parents=True)
    (pasta / "RESULTADO.txt").write_text("PARE\nPARE: algo\n")
    with pytest.raises(SystemExit):
        p.fase_completo(None, tmp_path)
    erros, n = v.verificar(tmp_path, None)
    assert n == 30 and any("falta central_HT_cal2_m3_A_11.csv" in e for e in erros) and any("teste curto" in e for e in erros)
    assert any("calibracao_HT2_A_m3.json" in e for e in erros)
    log = tmp_path / "log.txt"
    log.write_text("INFO ok\nPARE: central_HT_cal2 não aprende\n")
    erros, _ = v.verificar(tmp_path, log)
    assert any("PARE" in e for e in erros) and any("não chegou" in e for e in erros)


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
    import experiments.pipeline_r2b as p
    monkeypatch.setattr(cs, "RODADAS_BASE", 1)                 # R = 3 rodadas para as condições de 75 (k1, msp)
    monkeypatch.setattr(p, "R_TC", 4)                          # e R de teste 4 para o HT_cal2 (≠ 3)
    return p


def test_teste_curto_da_2b_em_miniatura_pare_e_ok(tmp_path, monkeypatch):
    p = _miniatura(monkeypatch)
    monkeypatch.setattr(p, "LIMITE_APRENDE", 0.99)             # com poucas rodadas o HT_cal2 não chega perto disso
    assert p.fase_teste_curto(_Serial(), tmp_path) == 3
    res = (tmp_path / "calibracao" / "teste_curto" / "RESULTADO.txt").read_text()
    assert res.startswith("PARE") and "central_HT_cal2 não aprende" in res
    assert "sanidade μ=0 (R=4): idêntico ao sem_media_k1" in res and "calibração HT2 em miniatura" in res
    with pytest.raises(SystemExit):                            # com PARE o completo recusa
        p.fase_completo(None, tmp_path)
    monkeypatch.setattr(p, "LIMITE_APRENDE", 0.0)              # agora o OK (a pasta é refeita)
    assert p.fase_teste_curto(_Serial(), tmp_path) == 0
    res = (tmp_path / "calibracao" / "teste_curto" / "RESULTADO.txt").read_text()
    assert res.startswith("OK") and "PARE:" not in res
    monkeypatch.setattr(p, "fase_selecao", lambda ex, d: (_ for _ in ()).throw(RuntimeError("gate abriu")))
    with pytest.raises(RuntimeError, match="gate abriu"):
        p.fase_completo(None, tmp_path)
