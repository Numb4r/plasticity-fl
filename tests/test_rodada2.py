"""Testes da rodada 2: médias no tempo, Dirichlet, B_fixa, reaprendizado, sonda v2, plano e verificação."""
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.condicoes as cs
from src import config
from src.dados import Sequencia, particionar_dirichlet
from src.federado import media_ponderada, treinar_fedavg, treino_local
from src.media_tempo import clonar_estado, misturar, treinar_ht, treinar_lookahead
from src.modelo import MLP, criar_mlp
from src.reaprendizado import passos_para_limiar
from src.sementes import PAPEIS, gerador, semente
from src.treino import treinar_passos_geral


def _dados(sem=1):
    seq = Sequencia("A", sem)
    return seq.dados(0)


def _estado_igual(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def _cond_k1(T=3, rodadas=3):
    return replace(cs.condicao_base("sem_media_k1", 3), T=T, rodadas=rodadas, rodada=2, sonda_versao=2)


# ---------- sementes e tarefas de referência ----------

def test_papeis_novos_vao_no_fim_e_os_antigos_nao_mudam():
    assert PAPEIS[:14] == ("dados", "rotulos", "professor", "particao", "init_continua", "init_nova",
                           "embaralha_continua", "embaralha_nova", "sp_ruido", "sonda_alvo", "sonda_nova",
                           "sonda_embaralha", "calib_nova", "calib_embaralha")
    assert PAPEIS[14:18] == ("media_tarefa", "dados_tarefa", "diag_indices", "diag_hess_v0")
    assert PAPEIS[18:] == ("particao_dirichlet", "ref_centro", "reaprende", "reaprende_nova")


def test_tarefas_com_nova_no_braco_longo_nao_tem_a_janela_7_a_11():
    esperado = sorted({t for t in range(100) if t % 3 == 2} | set(range(95, 100)))
    assert config.tarefas_com_nova(100, janela_hpart=False) == esperado and len(esperado) == 36
    assert not {7, 9, 10} & set(esperado)                                      # 8 e 11 entram por t mod 3 = 2
    assert config.tarefas_com_nova(30) == [2, 5, 7, 8, 9, 10, 11, 14, 17, 20, 23, 25, 26, 27, 28, 29]
    assert config.tarefas_com_nova(12) == [2, 5, 7, 8, 9, 10, 11]             # rodada 1 inalterada


# ---------- central_HT e central_lookahead ----------

def test_ht_e_lookahead_com_mu_zero_reproduzem_o_sem_media_k1_bit_a_bit():
    base = _cond_k1()
    k1 = cs.executar(base, "A", 1, Path(tempfile.mkdtemp()), diagnosticos=False)
    for nome, fam in (("central_HT", "ht"), ("central_lookahead", "lookahead")):
        d = cs.executar(replace(base, nome=nome, familia=fam, mu=0.0), "A", 1, Path(tempfile.mkdtemp()), diagnosticos=False)
        cols = [c for c in k1.columns if c not in ("cond", "mu", "acc_tartaruga")]
        pd.testing.assert_frame_equal(k1[cols], d[cols], check_exact=True)


def test_misturar_formula_e_casos_extremos():
    a, b = criar_mlp(1), criar_mlp(2)
    ta, la = clonar_estado(a), clonar_estado(b)
    assert _estado_igual(misturar(ta, la, 0.0), la)             # mu = 0: exatamente a lebre
    assert _estado_igual(misturar(ta, la, 1.0), ta)             # mu = 1: exatamente a tartaruga
    m = misturar(ta, la, 0.9)
    for k in ta:
        assert torch.allclose(m[k], 0.9 * ta[k] + 0.1 * la[k], atol=1e-6)


def test_lookahead_uma_rodada_e_a_mistura_da_tartaruga_com_a_lebre():
    X, y = _dados()
    todos = torch.arange(config.N)
    m = criar_mlp(5)
    inicial = clonar_estado(m)
    esperado_lebre = treino_local(MLP(), inicial, X, y, todos, torch.Generator().manual_seed(3))
    treinar_lookahead(m, X, y, [todos], torch.Generator().manual_seed(3), 1, 0.9)
    assert _estado_igual(clonar_estado(m), misturar(inicial, esperado_lebre, 0.9))
    for k in inicial:                                            # o deslocamento encolhe por (1 - mu)
        assert torch.allclose(m.state_dict()[k] - inicial[k], 0.1 * (esperado_lebre[k] - inicial[k]), atol=1e-6)


def test_ht_tartaruga_e_media_movel_exponencial_da_trajetoria_da_lebre():
    X, y = _dados()
    todos = torch.arange(config.N)
    mu = 0.5
    # simulação independente do algoritmo, com o mesmo gerador
    g_sim = torch.Generator().manual_seed(7)
    lebre_sim = criar_mlp(5).state_dict()
    lebre_sim = {k: v.clone() for k, v in lebre_sim.items()}
    tart_sim = {k: v.clone() for k, v in lebre_sim.items()}
    local = MLP()
    for _ in range(3):
        lebre_sim = treino_local(local, lebre_sim, X, y, todos, g_sim)
        tart_sim = misturar(tart_sim, lebre_sim, mu)
    m = criar_mlp(5)
    extra = {"tartaruga": clonar_estado(m)}
    treinar_ht(m, extra, X, y, [todos], torch.Generator().manual_seed(7), 3, mu)
    assert _estado_igual(clonar_estado(m), lebre_sim) and _estado_igual(extra["tartaruga"], tart_sim)
    assert not _estado_igual(clonar_estado(m), extra["tartaruga"])          # lebre e tartaruga divergem na tarefa
    # na tarefa seguinte a lebre volta para a tartaruga (e só aí)
    g2 = torch.Generator().manual_seed(9)
    inicio = {k: v.clone() for k, v in extra["tartaruga"].items()}
    esperado = treino_local(MLP(), inicio, X, y, todos, torch.Generator().manual_seed(9))
    treinar_ht(m, extra, X, y, [todos], g2, 1, mu)
    assert _estado_igual(clonar_estado(m), esperado)


# ---------- fedavg_dirichlet ----------

def test_particao_dirichlet_cobre_tudo_sem_sobreposicao_e_e_deterministica():
    sem = 3
    y = Sequencia("B", sem).dados(0)[1]
    for alfa in config.ALFAS:
        p = particionar_dirichlet(sem, 0, y, alfa)
        todos = torch.cat(p).sort().values
        assert len(p) == 10 and torch.equal(todos.cpu(), torch.arange(config.N))
        assert all(torch.equal(a, b) for a, b in zip(p, particionar_dirichlet(sem, 0, y, alfa)))
    # α grande ~ IID (tamanhos próximos e quase todas as classes por cliente); α pequeno = poucas classes
    def resumo(alfa):
        p = particionar_dirichlet(sem, 0, y, alfa)
        tam = [len(x) for x in p]
        classes = [len(set(y[x].tolist())) for x in p if len(x)]
        return max(tam) - min(tam), np.mean(classes), max(tam)
    grande, pequeno = resumo(100.0), resumo(0.1)
    assert grande[0] < 80 and grande[1] > 8.5
    assert pequeno[1] < 5.5 and pequeno[2] > grande[2]
    assert not torch.equal(particionar_dirichlet(sem, 0, y, 0.1)[0], particionar_dirichlet(sem, 1, y, 0.1)[0])  # refeita por tarefa


def test_particao_dirichlet_segue_a_regra_de_arredondamento_do_pre_registro():
    sem, t, alfa, c = 2, 4, 0.5, 3
    y = Sequencia("B", sem).dados(t)[1]
    p = particionar_dirichlet(sem, t, y, alfa)
    n_c = int((y == c).sum())
    rng = np.random.default_rng(semente(sem, "particao_dirichlet", t, c))
    pr = rng.dirichlet(alfa * np.ones(10))
    cortes = np.round(np.cumsum(pr) * n_c).astype(int)
    cortes[-1] = n_c
    esperado = np.diff(np.concatenate([[0], cortes]))
    obtido = [int((y[x] == c).sum()) for x in p]
    assert obtido == esperado.tolist()


@pytest.mark.parametrize("n", [37, 5, 20, 100])
def test_treinar_passos_geral_n_qualquer_passos_exatos_lotes_de_20(n, monkeypatch):
    X, y = _dados()
    indices = torch.arange(n)
    lotes, passos = [], {"n": 0}
    from src import treino
    original = treino.nn.CrossEntropyLoss

    class Espiao(original):
        def forward(self, entrada, alvo):
            lotes.append(len(alvo))
            return super().forward(entrada, alvo)

    monkeypatch.setattr(treino.nn, "CrossEntropyLoss", Espiao)
    copia = indices.clone()
    m = criar_mlp(1)
    assert treinar_passos_geral(m, X, y, indices, 15, 20, torch.Generator().manual_seed(0)) == 15
    assert lotes == [20] * 15 and torch.equal(indices, copia)


def test_fedavg_ignora_cliente_vazio_e_pesa_pelo_numero_de_pontos():
    X, y = _dados()
    p = [torch.arange(0, 30), torch.arange(0, 0), torch.arange(30, 100)]
    m = criar_mlp(4)
    inicial = clonar_estado(m)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    treinar_fedavg(m, X, y, p, g1, 1, geral=True)
    from src.treino import treinar_passos_geral as tpg
    estados = []
    for idx in (p[0], p[2]):
        local = MLP()
        local.load_state_dict(inicial)
        tpg(local, X, y, idx, 15, 20, g2)
        estados.append({k: v.detach().clone() for k, v in local.state_dict().items()})
    esperado = media_ponderada(estados, [30, 70])
    assert _estado_igual(clonar_estado(m), esperado)


def test_fedavg_iid_da_rodada_2_reproduz_a_rodada_1_bit_a_bit():
    """O código novo não mexe no caminho da rodada 1: três primeiras tarefas de results/fedavg_iid_m3_A_1.csv."""
    ref = pd.read_csv("results/fedavg_iid_m3_A_1.csv").iloc[:3]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)          # a rodada 1 rodou com 1 thread por processo; com mais threads a soma muda de ordem
    try:
        d = cs.executar(replace(cs.condicao_base("fedavg_iid", 3), T=3), "A", 1, Path(tempfile.mkdtemp()), diagnosticos=False)
    finally:
        torch.set_num_threads(threads)
    for col in ("acc_cont", "norma", "distancia", "rank_ef", "dorm_tau0_l1", "dorm_tau0_l2", "maj_frac", "cos_medio"):
        assert np.allclose(d[col].values, ref[col].values, rtol=0, atol=1e-9, equal_nan=True), col


# ---------- sonda v2 ----------

def test_sonda_v2_perdas_por_alvo_20_passos_e_estatistica_pedida(monkeypatch):
    from src.sondas import SONDA2_COLUNAS, sonda_v2
    X, y = _dados()
    modelo = criar_mlp(3)
    antes = clonar_estado(modelo)
    chamadas = {"n": 0}
    original = torch.optim.Adam.step

    def contando(self, *a, **k):
        chamadas["n"] += 1
        return original(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", contando)
    out = sonda_v2(modelo, X, 1, 4)
    assert chamadas["n"] == 3 * 2 * config.SONDA_PASSOS_R2 == 120            # 3 alvos x (cópia + nova) x 20 passos
    assert set(out) == set(SONDA2_COLUNAS) and all(math.isfinite(v) for v in out.values())
    assert _estado_igual(clonar_estado(modelo), antes)                      # o original não foi tocado
    norm = [math.log10(out[f"sonda2_Lc20_{k}"] / out[f"sonda2_Lc0_{k}"]) - math.log10(out[f"sonda2_Ln20_{k}"] / out[f"sonda2_Ln0_{k}"]) for k in range(3)]
    raw = [math.log10(out[f"sonda2_Lc20_{k}"] + 1e-6) - math.log10(out[f"sonda2_Ln20_{k}"] + 1e-6) for k in range(3)]
    assert out["sonda2_dlog_norm"] == pytest.approx(float(np.median(norm)))
    assert out["sonda2_dlog_raw"] == pytest.approx(float(np.median(raw)))
    assert all(out[f"sonda2_Lc20_{k}"] < out[f"sonda2_Lc0_{k}"] for k in range(3))   # 20 passos reduzem a perda


# ---------- B_fixa e reaprendizado ----------

def test_b_fixa_a_funcao_e_a_mesma_e_independe_da_ordem_de_acesso():
    s1, s2 = Sequencia("B_fixa", 1, T=12), Sequencia("B_fixa", 1, T=12)
    for t in (7, 0, 11, 3):                                      # ordem diferente de acesso
        s2.dados(t)
    for t in range(12):
        X, y = s1.dados(t)
        assert torch.equal(y, s2.dados(t)[1])
        assert torch.equal(y, s1.rotular(X))                     # o mesmo professor e o mesmo centro em qualquer entrada
    X0, X5 = s1.dados(0)[0], s1.dados(5)[0]
    assert torch.equal(s1.rotular(torch.cat([X0, X5]))[1000:], s1.dados(5)[1])   # a função não depende da tarefa
    for sem in (1, 2, 3):
        seq = Sequencia("B_fixa", sem, T=12)
        assert max(seq.maj_frac(t) for t in range(12)) < 0.5


def test_passos_para_limiar_zero_censurado_multiplo_de_15_e_nao_toca_no_original():
    seq = Sequencia("A", 1)
    X, y = seq.dados(0)
    m = criar_mlp(2)
    todos = torch.arange(config.N)
    treinar_passos_geral(m, X, y, todos, 450, 20, torch.Generator().manual_seed(0))
    acc = (m(X).argmax(1) == y).float().mean().item()
    antes = clonar_estado(m)
    assert passos_para_limiar(m, X, y, 0.5 * acc, torch.Generator().manual_seed(1))[:2] == (0, False)   # já no limiar
    ps, cen, _ = passos_para_limiar(criar_mlp(9), X, y, 0.9 * acc, torch.Generator().manual_seed(1), teto=900)
    assert (cen and ps == 900) or (not cen and ps % 15 == 0 and 0 < ps <= 900)
    assert passos_para_limiar(criar_mlp(9), X, y, 1.01, torch.Generator().manual_seed(1), teto=30) == (30, True, pytest.approx(0.1, abs=0.1))
    assert _estado_igual(clonar_estado(m), antes)


def test_executar_b_fixa_grava_matriz_e_reaprendizado(tmp_path):
    cond = _cond_k1(T=12, rodadas=2)
    df = cs.executar(cond, "B_fixa", 1, tmp_path, diagnosticos=False)
    base = "sem_media_k1_m3_B_fixa_1"
    assert (tmp_path / f"{base}_matriz.csv").exists()
    r = pd.read_csv(tmp_path / f"{base}_reaprendizado.csv")
    assert len(r) == 2 and set(r["tarefa_antiga"]) == {0, 5} and set(r["tarefa_atual"]) == {9}   # só t=9 existe com T=12
    assert (r["passos_copia"] % 15 == 0).all() and (r["passos_nova"] % 15 == 0).all()
    assert np.allclose(r["limiar"], 0.9 * r["a_jj"]) and "esquecimento" in df.columns


# ---------- colunas novas e braço longo ----------

def test_executar_registra_colunas_da_rodada_2():
    base = _cond_k1(T=3, rodadas=2)
    ht = cs.executar(replace(base, nome="central_HT", familia="ht", mu=0.9), "A", 1, Path(tempfile.mkdtemp()), diagnosticos=False)
    assert ht["acc_tartaruga"].notna().all() and (ht["mu"] == 0.9).all() and ht["alfa"].isna().all()
    dr = cs.executar(replace(base, nome="fedavg_dirichlet", particao="dirichlet", alfa=0.1, geral=True, familia="fl"),
                     "B", 1, Path(tempfile.mkdtemp()), diagnosticos=False)
    assert (dr["alfa"] == 0.1).all() and dr["part_max"].min() >= 100 and dr["acc_tartaruga"].isna().all()
    assert dr["part_min"].le(dr["part_max"]).all() and dr["part_vazios"].between(0, 9).all()
    for d in (ht, dr):
        assert (d["loss_nova"].isna() == d["acc_nova"].isna()).all() and d["loss_cont"].notna().all()


def test_braco_longo_so_tem_rede_nova_em_t_mod_3_e_nas_5_ultimas(tmp_path):
    cond = replace(cs.condicao_base("central_passos", 3), T=40, passos=15, rodada=2, sonda_versao=2)
    df = cs.executar(cond, "A", 1, tmp_path, diagnosticos=False)
    esperado = config.tarefas_com_nova(40, janela_hpart=False)
    assert list(df.index[df["acc_nova"].notna()]) == esperado
    assert {7, 9, 10}.isdisjoint(esperado)
    assert df["sonda2_dlog_norm"].notna().sum() == len([t for t in range(40) if t % 5 == 4])


# ---------- central_HT_cal: passos locais calibrados ----------

def test_treino_local_com_passos_faz_o_numero_pedido_e_o_padrao_nao_muda(monkeypatch):
    contagem = {"n": 0, "novos": 0}
    original, original_init = torch.optim.Adam.step, torch.optim.Adam.__init__

    def contando(self, *a, **k):
        contagem["n"] += 1
        return original(self, *a, **k)

    def init_contando(self, *a, **k):
        contagem["novos"] += 1
        return original_init(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", contando)
    monkeypatch.setattr(torch.optim.Adam, "__init__", init_contando)
    X, y = _dados()
    todos = torch.arange(len(X))
    local, estado = MLP(), criar_mlp(semente(1, "init_continua")).state_dict()
    treino_local(local, estado, X, y, todos, gerador(1, "embaralha_continua", 0))
    assert (contagem["n"], contagem["novos"]) == (15, 1)
    contagem.update(n=0, novos=0)
    treino_local(local, estado, X, y, todos, gerador(1, "embaralha_continua", 0), passos=45)
    assert (contagem["n"], contagem["novos"]) == (45, 3)                      # Adam recriado a cada 15 passos


def test_ht_cal_com_15_passos_e_o_central_ht_bit_a_bit_e_com_30_passos_muda():
    base = replace(_cond_k1(), nome="central_HT", familia="ht", mu=0.9)
    ht = cs.executar(base, "A", 1, Path(tempfile.mkdtemp()), diagnosticos=False)
    cal15 = cs.executar(replace(base, nome="central_HT_cal", s_local=15), "A", 1, Path(tempfile.mkdtemp()), diagnosticos=False)
    cols = [c for c in ht.columns if c not in ("cond",)]
    pd.testing.assert_frame_equal(ht[cols], cal15[cols], check_exact=True)
    cal30 = cs.executar(replace(base, nome="central_HT_cal", s_local=30), "A", 1, Path(tempfile.mkdtemp()), diagnosticos=False)
    assert not ht["loss_nova"].dropna().equals(cal30["loss_nova"].dropna())  # o S age na rede nova (e na contínua)
    assert not ht["loss_cont"].equals(cal30["loss_cont"])


def test_condicao_r2_central_ht_cal_le_calibracao_e_mu(tmp_path):
    pasta = tmp_path / "calibracao"
    pasta.mkdir(parents=True)
    (pasta / "calibracao_HT_A_m3.json").write_text(json.dumps({"passos_locais": 60, "ok": False}))
    (pasta / "selecao_HTcal_A_m3.json").write_text(json.dumps({"mu": 0.99}))
    c = cs.condicao_r2("central_HT_cal", "A", tmp_path)
    assert (c.familia, c.s_local, c.mu, c.calibracao_ok, c.rodadas) == ("ht", 60, 0.99, False, 75)
    c = cs.condicao_r2("central_HT_cal", "A", tmp_path, mu=0.5, s_local=30)           # explícito (teste curto) não lê JSON
    assert (c.s_local, c.mu) == (30, 0.5)


def test_calibrar_ht_escolhe_o_s_de_menor_erro_relativo_e_marca_ok(tmp_path, monkeypatch):
    from experiments import calibrar_ht as ch
    ref = tmp_path / "msp.csv"

    def calibrar(alvo):
        pd.DataFrame({"loss_nova": [alvo] * 3}).to_csv(ref, index=False)           # T=3: refs t=0..2 (3 últimas)
        return ch.calibrar_ht("A", tmp_path, T=3, passos_max=2 * 75 * 15, referencia=ref, saida=tmp_path / "c.json")

    r = calibrar(1000.0)                                                           # alvo enorme: o S de maior perda (15)
    assert r["passos_locais"] == 15 and r["ok"] is False and r["alvo_fora_do_alcance"] is False
    perdas = {int(k): v for k, v in r["tentativas"].items()}
    assert set(perdas) == {15, 30} and perdas[30] < perdas[15]                     # mais passos, menos perda
    r = calibrar(1e-9)                                                             # alvo inalcançável: o S máximo, fora do alcance
    assert r["passos_locais"] == 30 and r["ok"] is False and r["alvo_fora_do_alcance"] is True
    r = calibrar(perdas[30] * 1.05)                                                # 5% acima da perda em S=30: S=30, ok
    assert r["passos_locais"] == 30 and r["ok"] is True and abs(r["erro_rel"] + 0.0476) < 0.01
    assert json.loads((tmp_path / "c.json").read_text())["passos_locais"] == 30


# ---------- condições, seleção de μ, plano, gate e verificação ----------

def test_condicao_r2_le_passos_da_rodada_1_e_mu_da_selecao(tmp_path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    (r1 / "calibracao").mkdir(parents=True)
    (r2 / "calibracao").mkdir(parents=True)
    (r1 / "calibracao" / "calibracao_A_m3.json").write_text(json.dumps({"passos": 1200, "ok": True}))
    (r2 / "calibracao" / "calibracao_B_fixa_m3.json").write_text(json.dumps({"passos": 999, "ok": True}))
    (r2 / "calibracao" / "selecao_HT_B_m3.json").write_text(json.dumps({"mu": 0.9}))
    (r2 / "calibracao" / "selecao_HT_A_m3.json").write_text(json.dumps({"mu": 0.5}))
    assert cs.condicao_r2("central_aprendizado", "A", r2, dir_r1=r1).passos == 1200
    assert cs.condicao_r2("central_aprendizado", "B_fixa", r2, dir_r1=r1).passos == 999
    assert cs.condicao_r2("central_HT", "A", r2).mu == 0.5 and cs.condicao_r2("central_HT", "B_fixa", r2).mu == 0.9
    c = cs.condicao_r2("fedavg_dirichlet", "B", r2, alfa=0.1)
    assert c.alfa == 0.1 and c.geral and c.rodadas == 75 and c.sonda_versao == 2 and c.T == 30
    assert cs.condicao_r2("media_sem_particao", "A", r2).T == 30                       # rodada 2: T=30, não 12
    assert cs.condicao_r2("central_HT", "A", r2, mu=0.0).mu == 0.0                     # mu explícito vence


def test_escolher_mu_maior_acuracia_empate_menor_e_borda(tmp_path):
    import experiments.pipeline_r2 as p
    pasta = tmp_path / "calibracao" / "selecao_mu"
    pasta.mkdir(parents=True)
    def fake(accs):
        for mu, a in zip(config.MUS, accs):
            pd.DataFrame({"acc_cont": [a] * 30}).to_csv(pasta / f"central_HT_mu{mu:g}_m3_A_1000.csv", index=False)
    fake([0.9, 0.7, 0.6])
    r = p.escolher_mu("HT", "A", tmp_path)
    assert r["mu"] == 0.5 and r["borda"] is True
    fake([0.6, 0.8, 0.7])
    assert p.escolher_mu("HT", "A", tmp_path)["mu"] == 0.9 and p.escolher_mu("HT", "A", tmp_path)["borda"] is False
    fake([0.8, 0.8, 0.5])                                                               # empate: o menor
    assert p.escolher_mu("HT", "A", tmp_path)["mu"] == 0.5
    assert json.loads((tmp_path / "calibracao" / "selecao_HT_A_m3.json").read_text())["mu"] == 0.5
    for mu, a in zip(config.MUS, [0.5, 0.6, 0.9]):                                      # método HTcal: arquivos e JSON próprios
        pd.DataFrame({"acc_cont": [a] * 30}).to_csv(pasta / f"central_HT_cal_mu{mu:g}_m3_A_1000.csv", index=False)
    assert p.escolher_mu("HTcal", "A", tmp_path)["mu"] == 0.99
    assert (tmp_path / "calibracao" / "selecao_HTcal_A_m3.json").exists()


def test_plano_da_rodada_2_tem_as_execucoes_do_pre_registro():
    import experiments.pipeline_r2 as p
    conf, longo = p.plano_confirmatorio(), p.plano_longo()
    assert len(conf) == 125 and len(longo) == 30
    nomes = [p.nome_csv(j) for j in conf + longo]
    assert len(set(nomes)) == len(nomes)                                                 # sem colisão de arquivos
    assert {j[2] for j in conf} == {6, 7, 8, 9, 10} and {j[2] for j in longo} == {6, 7, 8}
    assert sum(1 for j in conf if j[0] == "fedavg_dirichlet") == 30 and {j[1] for j in conf if j[0] == "fedavg_dirichlet"} == {"A", "B"}
    assert {j[4] for j in conf if j[0] == "fedavg_dirichlet"} == {0.1, 1.0, 100.0}
    assert sum(1 for j in conf if j[0] == "central_HT_cal") == 10 and {j[1] for j in conf if j[0] == "central_HT_cal"} == {"A", "B"}
    assert not any(j[0] == "central_HT_cal" for j in longo)                           # braço longo e B_fixa seguem com central_HT
    assert sum(1 for j in conf if j[1] == "B_fixa") == 25 and any(j[:2] == ("central_HT", "B_fixa") for j in conf)
    assert not any(j[0] == "central_lookahead" and j[1] == "B_fixa" for j in conf)
    assert all(j[3] == 100 and j[0] != "central_lookahead" for j in longo) and "fedavg_dirichlet_a0.1_m3_A_6.csv" in nomes
    lo, hi = p.estimar()["cpu_horas"]
    assert 3.0 < lo < hi < 40


def test_completo_recusa_comecar_sem_teste_curto_ok(tmp_path):
    import experiments.pipeline_r2 as p
    with pytest.raises(SystemExit):
        p.fase_completo(None, tmp_path)
    pasta = tmp_path / "calibracao" / "teste_curto"
    pasta.mkdir(parents=True)
    (pasta / "RESULTADO.txt").write_text("PARE\nPARE: algo\n")
    with pytest.raises(SystemExit):
        p.fase_completo(None, tmp_path)


def test_verificador_acusa_o_que_falta_e_nao_quebra(tmp_path):
    import experiments.verificar_rodada2 as v
    erros, n = v.verificar(tmp_path, None)
    assert n == 155 and any("falta fedavg_iid_m3_A_6.csv" in e for e in erros) and any("teste curto" in e for e in erros)
    log = tmp_path / "log.txt"
    log.write_text("INFO ok\nPARE: central_HT não aprende\n")
    erros, _ = v.verificar(tmp_path, log)
    assert any("PARE" in e for e in erros) and any("não chegou" in e for e in erros)


# ---------- teste curto de ponta a ponta, em miniatura (poucas rodadas, diretório temporário) ----------

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
    import experiments.pipeline_r2 as p
    monkeypatch.setattr(cs, "RODADAS_BASE", 1)                 # R = 3 rodadas por tarefa (em vez de 75)
    monkeypatch.setattr(p, "T_LONGO", 40)                       # smoke do braço longo com T=40
    return p


def test_teste_curto_em_miniatura_dispara_pare_quando_o_ht_nao_aprende(tmp_path, monkeypatch):
    p = _miniatura(monkeypatch)
    monkeypatch.setattr(p, "LIMITE_APRENDE", 0.99)             # com 3 rodadas o HT não chega perto disso
    codigo = p.fase_teste_curto(_Serial(), tmp_path)
    res = (tmp_path / "calibracao" / "teste_curto" / "RESULTADO.txt").read_text()
    assert codigo == 3 and res.startswith("PARE") and "central_HT não aprende" in res
    assert "sanidade μ=0: HT e lookahead idênticos ao sem_media_k1" in res        # os outros critérios passaram
    assert "T=40: 40 linhas" in res and "B_fixa reaprendizado:" in res and "dirichlet _a0.1_tc A" in res
    with pytest.raises(SystemExit):                                               # com PARE o completo recusa
        p.fase_completo(None, tmp_path)


def test_teste_curto_em_miniatura_da_ok_e_libera_o_completo(tmp_path, monkeypatch):
    p = _miniatura(monkeypatch)
    monkeypatch.setattr(p, "LIMITE_APRENDE", 0.0)
    assert p.fase_teste_curto(_Serial(), tmp_path) == 0
    res = (tmp_path / "calibracao" / "teste_curto" / "RESULTADO.txt").read_text()
    assert res.startswith("OK") and "PARE:" not in res
    # o gate do completo abre; para aqui, antes de rodar a seleção, para não executar o experimento
    monkeypatch.setattr(p, "fase_selecao", lambda ex, d: (_ for _ in ()).throw(RuntimeError("gate abriu")))
    with pytest.raises(RuntimeError, match="gate abriu"):
        p.fase_completo(None, tmp_path)
