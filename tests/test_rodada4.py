"""Testes da rodada 4: cada técnica distribuída (4a) e cada remédio de plasticidade (4b).

Estes testes não olham dado real: usam a sequência sintética A com 3 tarefas e poucas rodadas.
A igualdade bit a bit só vale com 1 thread (como nas execuções reais)."""
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import src.tecnicas_fed as tf
from src import config
from src.condicoes import avaliar_cond, executar, params_cond
from src.condicoes_r4 import CATALOGO_4A, CATALOGO_4B, condicao_r4
from src.dados import Sequencia, particionar
from src.federado import media_ponderada, treinar_fedavg
from src.metricas import acuracia, fracao_dormentes, perda_ce
from src.modelo import MLP, criar_mlp
from src.sementes import PAPEIS, gerador, semente
from src.tecnicas_fed import (
    avaliar_tecnica, consenso, iniciar_extra, matriz_mistura, misturar_clientes, treinar_fedx, vizinhos, _passo_externo,
)
from src.tecnicas_local import (
    mascara_media, mascara_voto, montar_ajuste, redo_reiniciar, scores_neuronios, treinar_passos_ext,
)
from src.treino import shrink_and_perturb, treinar_passos


@pytest.fixture(autouse=True)
def uma_thread():
    ant = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(ant)


def _base(sem=1, t=0):
    X, y = Sequencia("A", sem, 3).dados(t)
    return X, y, particionar("iid", sem), criar_mlp(semente(sem, "init_continua"))


def _aux(sem=1, t=0, quem=0):
    return {"redo": gerador(sem, "redo_reinicio", t, quem), "sp": gerador(sem, "sp_ruido", t, quem)}


def _rodar(tec, rodadas=2, sem=1, t=0):
    X, y, part, m = _base(sem, t)
    extra = iniciar_extra(tec, m)
    cos = treinar_fedx(m, extra, X, y, part, gerador(sem, "embaralha_continua", t), rodadas, tec, _aux(sem, t))
    return m, extra, cos


def _igual(a: MLP, b: MLP) -> bool:
    return all(torch.equal(x, z) for x, z in zip(a.state_dict().values(), b.state_dict().values()))


def _maxdif(a: MLP, b: MLP) -> float:
    return max((x - z).abs().max().item() for x, z in zip(a.state_dict().values(), b.state_dict().values()))


def _fedavg_ref(rodadas=2, sem=1, t=0):
    X, y, part, m = _base(sem, t)
    cos = treinar_fedavg(m, X, y, part, gerador(sem, "embaralha_continua", t), rodadas)
    return m, cos


def test_papel_novo_vai_no_fim_e_os_antigos_nao_mudam():
    assert PAPEIS[-1] == "redo_reinicio" and PAPEIS[:22][-1] == "reaprende_nova"
    assert semente(1, "dados") == semente(1, "dados")


# ---------------------------------------------------------------- laço local e ganchos
def test_treinar_passos_ext_sem_ganchos_e_igual_ao_treinar_passos():
    X, y, part, _ = _base()
    a, b = criar_mlp(5), criar_mlp(5)
    treinar_passos(a, X, y, part[0], 30, 20, 15, gerador(1, "embaralha_continua"), wd=1e-3)
    treinar_passos_ext(b, X, y, part[0], 30, 20, 15, gerador(1, "embaralha_continua"), wd=1e-3)
    assert _igual(a, b)


def test_ajustes_de_gradiente_seguem_as_formulas():
    X, y, part, m = _base()
    ref = {n: torch.randn_like(p) for n, p in m.named_parameters()}
    corr = {n: torch.randn_like(p) for n, p in m.named_parameters()}
    theta0 = {n: torch.randn_like(p) for n, p in m.named_parameters()}
    F.cross_entropy(m(X[:20]), y[:20]).backward()
    g0 = {n: p.grad.clone() for n, p in m.named_parameters()}
    montar_ajuste(m, prox=(0.3, ref), corr=corr, l2init=(0.05, theta0))()
    for n, p in m.named_parameters():
        esperado = g0[n] + 0.3 * (p - ref[n]) + corr[n] + 0.05 * (p - theta0[n])
        assert torch.allclose(p.grad, esperado, atol=1e-6)


def test_so_params_congela_o_resto():
    X, y, part, m = _base()
    antes = {n: p.detach().clone() for n, p in m.named_parameters()}
    treinar_passos_ext(m, X, y, part[0], 5, 20, 5, gerador(1, "embaralha_continua"), so_params=list(m.l3.parameters()))
    depois = dict(m.named_parameters())
    assert torch.equal(antes["l1.weight"], depois["l1.weight"]) and torch.equal(antes["l2.weight"], depois["l2.weight"])
    assert not torch.equal(antes["l3.weight"], depois["l3.weight"])


# ---------------------------------------------------------------- 4a
def test_fedavg_da_rodada_4_reproduz_o_treinar_fedavg_bit_a_bit():
    m, extra, cos = _rodar({"tipo": "fedavg"})
    ref, cos_ref = _fedavg_ref()
    assert _igual(m, ref) and cos == cos_ref
    assert len(extra["metricas_rodada"]) == 2 and all(0 < e["enc"] <= 1.0 + 1e-6 for e in extra["metricas_rodada"])


def test_fedprox_mu_zero_e_fedavg_e_mu_positivo_prende_o_cliente_ao_global():
    m0, _, _ = _rodar({"tipo": "fedprox", "mu": 0.0}, rodadas=1)
    assert _igual(m0, _fedavg_ref(1)[0])
    ini = criar_mlp(semente(1, "init_continua"))
    d = lambda m: torch.cat([(a - b).flatten() for a, b in zip(m.parameters(), ini.parameters())]).norm().item()
    m1, _, _ = _rodar({"tipo": "fedprox", "mu": 10.0}, rodadas=1)
    assert d(m1) < d(m0)


def test_scaffold_primeira_rodada_e_fedavg_e_as_variaveis_de_controle_sao_as_pedidas():
    m, extra, _ = _rodar({"tipo": "scaffold"}, rodadas=1)
    assert _igual(m, _fedavg_ref(1)[0])                                    # c = c_k = 0 na rodada 1
    est = extra["estado_tecnica"]
    X, y, part, ini = _base()
    for k in (0, 7):                                                       # c_k = gradiente cheio do cliente no modelo recebido
        g = tf._grad_cheio(ini, X, y, part[k])
        assert all(torch.allclose(g[n], est["ck"][k][n], atol=1e-7) for n in g)
    for n in est["c"]:                                                     # c = média dos c_k, logo sum_k (c - c_k) = 0
        assert torch.allclose(est["c"][n], sum(c[n] for c in est["ck"]) / 10, atol=1e-7)
        assert torch.allclose(sum(est["c"][n] - c[n] for c in est["ck"]), torch.zeros_like(est["c"][n]), atol=1e-6)
    m2, _, _ = _rodar({"tipo": "scaffold"}, rodadas=2)
    assert not _igual(m2, _fedavg_ref(2)[0])                               # a partir da rodada 2 a correção age


def test_fedrep_cabeca_local_corpo_na_media_e_avaliacao_por_cliente():
    tec = {"tipo": "fedrep"}
    m, extra, _ = _rodar(tec, rodadas=2)
    cabecas = extra["cabecas"]
    assert not torch.equal(cabecas[0]["weight"], cabecas[1]["weight"])    # cada cliente tem a sua cabeça
    assert torch.allclose(m.l3.weight, sum(c["weight"] for c in cabecas) / 10, atol=1e-6)   # l3 do global só representa
    X, y, _, _ = _base()
    acc, perda = avaliar_tecnica(m, extra, tec, X, y)
    tmp, accs, perdas = MLP(), [], []
    for c in cabecas:
        tmp.load_state_dict(m.state_dict())
        tmp.l3.load_state_dict(c)
        accs.append(acuracia(tmp, X, y))
        perdas.append(perda_ce(tmp, X, y))
    assert acc == pytest.approx(np.mean(accs)) and perda == pytest.approx(np.mean(perdas))


def test_fedrep_treina_5_passos_na_cabeca_e_10_no_corpo(monkeypatch):
    chamadas = []
    original = tf.treinar_passos_ext

    def espia(modelo, X, y, p, n, batch, reset, g, **kw):
        chamadas.append((n, len(kw["so_params"])))
        return original(modelo, X, y, p, n, batch, reset, g, **kw)

    monkeypatch.setattr(tf, "treinar_passos_ext", espia)
    _rodar({"tipo": "fedrep"}, rodadas=1)
    assert chamadas[:2] == [(5, 2), (10, 4)] and len(chamadas) == 20       # cabeça (l3: 2 tensores), corpo (l1, l2: 4)


def test_topologias_do_gossip_sao_duplamente_estocasticas_e_simetricas():
    for topo, grau in (("anel", 3), ("grau4", 5), ("completo", 10)):
        W = matriz_mistura(topo, 10)
        assert torch.allclose(W.sum(0), torch.ones(10, dtype=torch.float64)) and torch.allclose(W.sum(1), torch.ones(10, dtype=torch.float64))
        assert torch.equal(W, W.T) and all(len(v) == grau for v in vizinhos(topo, 10))


def test_gossip_completo_e_fedavg_bit_a_bit_e_a_media_dos_pesos_e_o_modelo():
    m, extra, cos = _rodar({"tipo": "gossip", "topologia": "completo"})
    ref, cos_ref = _fedavg_ref()
    assert _igual(m, ref) and cos == cos_ref and consenso(extra) < 1e-9
    mg, ex, _ = _rodar({"tipo": "gossip", "topologia": "anel"})
    assert consenso(ex) > 0
    medio = media_ponderada(ex["clientes"], [1] * 10)
    assert all(torch.allclose(medio[k], v, atol=1e-7) for k, v in mg.state_dict().items())


def test_mistura_preserva_a_media_e_o_consenso_chega_mais_rapido_em_grafo_mais_denso():
    torch.manual_seed(0)
    base = [{"w": torch.randn(7, 5)} for _ in range(10)]

    def dist(estados):
        vs = torch.stack([e["w"].flatten() for e in estados])
        return (vs - vs.mean(0)).norm(dim=1).mean().item()

    media0 = torch.stack([e["w"] for e in base]).mean(0)
    finais = {}
    for topo in ("anel", "grau4", "completo"):
        est, ds = base, [dist(base)]
        for _ in range(6):
            est = misturar_clientes(est, topo)
            ds.append(dist(est))
            assert torch.allclose(torch.stack([e["w"] for e in est]).mean(0), media0, atol=1e-6)
        assert all(b <= a + 1e-7 for a, b in zip(ds, ds[1:]))
        finais[topo] = ds[-1]
    assert finais["completo"] < 1e-6 and finais["grau4"] < finais["anel"]


def test_diloco_lr1_sem_momento_e_fedavg_e_conta_os_passos_e_os_adams(monkeypatch):
    m, _, _ = _rodar({"tipo": "diloco", "H": 15, "lr_ext": 1.0, "mom": 0.0})
    assert _maxdif(m, _fedavg_ref()[0]) < 1e-7
    passos, adams = {"n": 0}, {"n": 0}
    step, init = torch.optim.Adam.step, torch.optim.Adam.__init__
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, *a, **k: (passos.__setitem__("n", passos["n"] + 1), step(self, *a, **k))[1])
    monkeypatch.setattr(torch.optim.Adam, "__init__", lambda self, *a, **k: (adams.__setitem__("n", adams["n"] + 1), init(self, *a, **k))[0])
    _rodar({"tipo": "diloco", "H": 50, "lr_ext": 0.7, "mom": 0.9}, rodadas=1)
    assert passos["n"] == 10 * 50 and adams["n"] == 10                     # H passos por cliente, um Adam contínuo por rodada


def test_passo_externo_de_nesterov_igual_ao_sgd_do_torch():
    torch.manual_seed(0)
    x0 = torch.randn(6, dtype=torch.float32)
    alvos = [torch.randn(6), torch.randn(6), torch.randn(6)]
    p = torch.nn.Parameter(x0.clone())
    opt = torch.optim.SGD([p], lr=0.7, momentum=0.9, nesterov=True)
    estado, buf = {"w": x0.clone()}, None
    for a in alvos:
        opt.zero_grad()
        p.grad = (p.detach() - a).clone()                                  # pseudo-gradiente = antes - agregado
        opt.step()
        estado, buf = _passo_externo(estado, {"w": estado["w"] - (estado["w"] - a)}, buf, 0.7, 0.9)
    assert torch.allclose(estado["w"], p.detach(), atol=1e-5)


# ---------------------------------------------------------------- 4b: remédios
def test_l2_local_e_o_mesmo_weight_decay_do_central_l2():
    m, _, _ = _rodar({"tipo": "fedavg", "wd": 1e-2}, rodadas=2)
    norma = lambda mm: torch.cat([p.flatten() for p in mm.parameters()]).norm().item()
    assert norma(m) < norma(_rodar({"tipo": "fedavg"}, rodadas=2)[0])
    X, y, part, _ = _base()
    a, b = criar_mlp(3), criar_mlp(3)
    treinar_passos(a, X, y, part[0], 15, 20, 15, gerador(1, "embaralha_continua"), wd=1e-2)
    treinar_passos_ext(b, X, y, part[0], 15, 20, 15, gerador(1, "embaralha_continua"), wd=1e-2)
    assert _igual(a, b)


def test_l2init_puxa_para_a_init_da_rede_e_lam_zero_e_fedavg():
    ini = criar_mlp(semente(1, "init_continua"))
    dist = lambda mm: torch.cat([(a - b).flatten() for a, b in zip(mm.parameters(), ini.parameters())]).norm().item()
    forte, _, _ = _rodar({"tipo": "fedavg", "l2init": 50.0}, rodadas=2)
    livre, _, _ = _rodar({"tipo": "fedavg"}, rodadas=2)
    assert dist(forte) < dist(livre)
    zero, _, _ = _rodar({"tipo": "fedavg", "l2init": 0.0}, rodadas=2)
    assert _igual(zero, livre)


def test_scores_dormentes_batem_com_fracao_dormentes_e_o_redo_reinicia_so_os_dormentes():
    X, y, _, m = _base()
    with torch.no_grad():
        m.l1.weight[:5] = -1.0
        m.l1.bias[:5] = -50.0                                              # neurônios 0..4 da camada 1: ativação exatamente 0
    s1, s2 = scores_neuronios(m, X)
    h1, h2, _ = m.ativacoes(X)
    for tau in (0.0, 0.025, 0.1):
        assert (s1 <= tau).float().mean().item() == fracao_dormentes(h1, tau)
        assert (s2 <= tau).float().mean().item() == fracao_dormentes(h2, tau)
    antes = {k: v.clone() for k, v in m.state_dict().items()}
    logits = m(X).detach().clone()
    masc = [s1 <= 0.0, torch.zeros(128, dtype=torch.bool)]
    n = redo_reiniciar(m, masc, gerador(1, "redo_reinicio"))
    assert n == int(masc[0].sum()) >= 5
    idx = masc[0].nonzero().flatten()
    lim = 1 / np.sqrt(32)
    assert (m.l1.weight[idx].abs() <= lim).all() and (m.l1.bias[idx] == 0).all() and (m.l2.weight[:, idx] == 0).all()
    resto = (~masc[0]).nonzero().flatten()
    assert torch.equal(m.l1.weight[resto], antes["l1.weight"][resto]) and torch.equal(m.l3.weight, antes["l3.weight"])
    assert torch.allclose(m(X), logits, atol=1e-5)                         # neurônios mortos: a função não muda


def test_mascaras_de_redo_federado_por_media_e_por_voto():
    s = [[torch.tensor([0.0, 1.0, 0.04]), torch.zeros(2)], [torch.tensor([0.0, 2.0, 0.0]), torch.ones(2)],
         [torch.tensor([1.0, 2.0, 0.0]), torch.ones(2)], [torch.tensor([1.0, 1.0, 0.0]), torch.ones(2)]]
    pesos = [100] * 4
    media = mascara_media(s, pesos, 0.1)[0]
    assert media.tolist() == [False, False, True]                          # médias 0,5; 1,5; 0,01
    voto = mascara_voto(s, pesos, 0.1, 0.5)[0]
    assert voto.tolist() == [True, False, True]                            # 2/4 dormentes no neurônio 0, 4/4 no 2
    assert mascara_voto(s, pesos, 0.1, 0.8)[0].tolist() == [False, False, True]


def test_redo_local_age_so_no_cliente_e_o_federado_so_no_servidor_no_ritmo_certo(monkeypatch):
    X, y, part, m = _base()
    alvos = []
    original = tf.redo_reiniciar
    monkeypatch.setattr(tf, "redo_reiniciar", lambda mod, masc, g: (alvos.append(id(mod)), original(mod, masc, g))[1])
    for tec, esperado_global, esperado_total in (
        ({"tipo": "fedavg", "redo_local": {"tau": 0.5, "cada": 5}}, 0, 10),    # r=0 de 3 rodadas: 10 clientes
        ({"tipo": "fedavg", "redo_fed": {"tau": 0.5, "cada": 5, "modo": "media"}}, 1, 1),
        ({"tipo": "fedavg", "redo_fed": {"tau": 0.5, "cada": 1, "modo": "media"}}, 3, 3),
    ):
        alvos.clear()
        mm = criar_mlp(semente(1, "init_continua"))
        treinar_fedx(mm, iniciar_extra(tec, mm), X, y, part, gerador(1, "embaralha_continua"), 3, tec, _aux())
        assert len(alvos) == esperado_total and alvos.count(id(mm)) == esperado_global


def test_sp_por_rodada_e_encolhimento_e_sp_cliente_e_servidor_sao_a_mesma_coisa(monkeypatch):
    monkeypatch.setattr(tf, "SP_RUIDO", 0.0)
    m, _, _ = _rodar({"tipo": "fedavg", "sp_rodada": 0.5}, rodadas=1)
    ref, _ = _fedavg_ref(1)
    assert all(torch.allclose(a, 0.5 * b, atol=1e-6) for a, b in zip(m.state_dict().values(), ref.state_dict().values()))
    # união SP cliente = SP servidor: o encolhimento é linear e comuta com a média ponderada
    ests = [{"w": torch.randn(4, 3)} for _ in range(10)]
    a = media_ponderada([{"w": 0.7 * e["w"]} for e in ests], [100] * 10)["w"]
    b = 0.7 * media_ponderada(ests, [100] * 10)["w"]
    assert torch.allclose(a, b, atol=1e-6)
    # e o ruído independente de K clientes tem variância sigma^2/K depois da média
    g = torch.Generator().manual_seed(0)
    ruidos = torch.stack([torch.randn(20000, generator=g) * 0.01 for _ in range(10)]).mean(0)
    assert ruidos.std().item() == pytest.approx(0.01 / np.sqrt(10), rel=0.05)


def test_avaliar_cond_de_condicoes_antigas_e_o_modelo_e_o_gossip_e_media_dos_clientes(tmp_path):
    X, y, _, m = _base()
    cond = condicao_r4("sem_media_k1", "A", tmp_path, rodadas=2, T=3)
    assert avaliar_cond(cond, m, None, X, y) == (acuracia(m, X, y), perda_ce(m, X, y))
    tec = {"tipo": "gossip", "topologia": "anel"}
    mg, extra, _ = _rodar(tec)
    cond = condicao_r4("gossip_anel", "A", tmp_path, rodadas=2, T=3)
    acc, perda = avaliar_cond(cond, mg, extra, X, y)
    tmp = MLP()
    ac = []
    for c in extra["clientes"]:
        tmp.load_state_dict(c)
        ac.append(acuracia(tmp, X, y))
    assert acc == pytest.approx(np.mean(ac))


# ---------------------------------------------------------------- condições e integração
@pytest.mark.parametrize("nome", [n for n in list(CATALOGO_4A) + list(CATALOGO_4B) if n not in ("fedavg_iid",)])
def test_executar_cada_condicao_da_rodada_4_em_miniatura(nome, tmp_path):
    from experiments.pipeline_r4 import TC_HP
    hp = TC_HP.get(nome, {"fedprox": 0.1}.get(nome))
    R = 1 if nome == "diloco_h150" else 2
    cond = condicao_r4(nome, "A", tmp_path, hp=hp, rodadas=R, T=3)
    df = executar(cond, "A", 1, tmp_path, diagnosticos=False)
    assert len(df) == 3 and df[["acc_cont", "loss_cont", "norma", "acc_nova", "loss_nova"]].notna().all().all()
    assert np.isfinite(df[["loss_cont", "loss_nova"]].to_numpy(float)).all()
    if cond.familia == "fedx":
        assert df["encolhimento"].notna().all() and df["cos_medio"].notna().all()
        assert (df["consenso"].notna().all()) == (json.loads(cond.tec)["tipo"] == "gossip")
    assert (tmp_path / f"{nome}_m3_A_1.params.json").exists()


def test_params_nao_ganham_campo_novo_nas_condicoes_antigas_e_fedx_reproduz(tmp_path):
    from src.condicoes import condicao_base, csv_valido
    antiga = replace(condicao_base("sem_media_k1", 3), T=3, rodadas=2, rodada=2, sonda_versao=2)
    assert "tec" not in params_cond(antiga)
    calib = tmp_path / "calibracao"                                         # semente 1000: não exige diagnósticos
    executar(antiga, "A", 1000, calib)
    assert csv_valido(antiga, "A", 1000, calib) is True
    assert "tec" not in json.loads((calib / "sem_media_k1_m3_A_1000.params.json").read_text())
    nova = condicao_r4("fedprox", "A", tmp_path, hp=0.1, rodadas=2, T=3)
    executar(nova, "A", 1000, calib)
    assert csv_valido(nova, "A", 1000, calib) is True and csv_valido(replace(nova, rodadas=3), "A", 1000, calib) is False
    assert json.loads(params_cond(nova)["tec"]) == {"mu": 0.1, "tipo": "fedprox"}
    a = executar(nova, "A", 1, tempfile.mkdtemp(), diagnosticos=False)
    b = executar(nova, "A", 1, tempfile.mkdtemp(), diagnosticos=False)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_sp_por_tarefa_no_fedx_encolhe_a_partir_da_segunda_tarefa_e_a_rede_nova_recebe_o_mesmo_protocolo(tmp_path):
    base = executar(condicao_r4("fedavg_iid", "A", tmp_path, rodadas=2, T=3), "A", 1, tempfile.mkdtemp(), diagnosticos=False)
    sp = executar(condicao_r4("fed_sp", "A", tmp_path, hp=0.5, rodadas=2, T=3), "A", 1, tempfile.mkdtemp(), diagnosticos=False)
    assert sp.loc[0, "acc_cont"] == base.loc[0, "acc_cont"] and sp.loc[1, "acc_cont"] != base.loc[1, "acc_cont"]
    assert sp.loc[0, "acc_nova"] != base.loc[0, "acc_nova"]                 # a rede nova leva SP desde a primeira tarefa
