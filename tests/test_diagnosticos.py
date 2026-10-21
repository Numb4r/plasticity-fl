"""Testes das métricas diagnósticas de Klein et al. (Desvios v3.1)."""
import copy
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src import config
from src import condicoes as cs
from src import diagnosticos as dg
from src.dados import Sequencia
from src.metricas import srank, vetor_parametros
from src.modelo import criar_mlp


def _dados(sem=1, t=0):
    return Sequencia("A", sem).dados(t)


# ---------- os 4 testes pedidos no adendo ----------

def test_power_iteration_acerta_o_maior_autovalor_num_problema_quadratico():
    """Regressão linear com 0,5 x MSE: a Hessiana é X^T X / n, conhecida."""
    g = torch.Generator().manual_seed(0)
    n, d = 400, 8
    X = torch.randn(n, d, generator=g) * torch.tensor([3.0, 1.5, 1, 1, 0.8, 0.6, 0.5, 0.4])
    y = torch.randn(n, generator=g)
    w = torch.zeros(d, requires_grad=True)
    verdadeiro = torch.linalg.eigvalsh(X.T @ X / n).max().item()
    lam, convergiu, _ = dg.autovalor_max(lambda: 0.5 * F.mse_loss(X @ w, y), [w],
                                          torch.Generator().manual_seed(1), 30, 0.01)
    assert abs(lam - verdadeiro) / verdadeiro < 0.01 and convergiu
    _, nao, _ = dg.autovalor_max(lambda: 0.5 * F.mse_loss(X @ w, y), [w],
                                 torch.Generator().manual_seed(1), 3, 0.01)
    assert not nao                                           # poucas iterações: não pode declarar convergência


def test_raiz_da_soma_dos_quadrados_por_camada_e_a_norma_global():
    X, y = _dados()
    modelo = criar_mlp(9)
    p = dg.normas_parametros(modelo)
    por_camada = [v for k, v in p.items() if k.startswith("norma_l")]
    assert len(por_camada) == 6                                   # l1, l2, l3 x pesos, vieses
    assert math.isclose(math.sqrt(sum(v ** 2 for v in por_camada)), p["norma_global"], rel_tol=1e-6)
    assert math.isclose(p["norma_global"], vetor_parametros(modelo).norm().item(), rel_tol=1e-6)
    g = dg.normas_gradiente(modelo, X, y)
    l2 = [v for k, v in g.items() if k.startswith("grad_l2_l")]
    l1 = [v for k, v in g.items() if k.startswith("grad_l1_l")]
    assert len(l2) == len(l1) == 6
    assert math.isclose(math.sqrt(sum(v ** 2 for v in l2)), g["grad_l2_global"], rel_tol=1e-6)
    assert math.isclose(sum(l1), g["grad_l1_global"], rel_tol=1e-6)
    cop = copy.deepcopy(modelo)                                   # gradiente global calculado por fora
    plano = torch.cat([x.flatten() for x in torch.autograd.grad(F.cross_entropy(cop(X), y), cop.parameters())])
    assert math.isclose(plano.norm().item(), g["grad_l2_global"], rel_tol=1e-5)
    assert math.isclose(plano.abs().sum().item(), g["grad_l1_global"], rel_tol=1e-5)


def test_condicao_curta_com_e_sem_metricas_da_o_mesmo_continuo():
    casos = [(replace(cs.condicao_base("central_passos"), nome="central_SP", T=8, passos=15, sp_lambda=0.8), "A"),
             (replace(cs.condicao_base("fedavg_iid"), T=8, rodadas=2), "B_disjunta")]
    for cond, tipo in casos:
        com = cs.executar(cond, tipo, 1, Path(tempfile.mkdtemp()), diagnosticos=True)
        sem = cs.executar(cond, tipo, 1, Path(tempfile.mkdtemp()), diagnosticos=False)
        assert set(sem.columns) < set(com.columns) and len(com.columns) > len(sem.columns)
        pd.testing.assert_frame_equal(com[sem.columns], sem, check_exact=True)   # colunas antigas intactas
        assert ((com["fim_norma_global"] - com["norma"]).abs() / com["norma"] < 1e-5).all()   # coerente com a norma já existente (arredondamento float32)


def test_medir_nao_altera_parametros_otimizador_modo_nem_rng():
    X, y = _dados()
    modelo = criar_mlp(3)
    modelo.train()
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    F.cross_entropy(modelo(X[:20]), y[:20]).backward()
    opt.step()
    antes_p = [p.detach().clone() for p in modelo.parameters()]
    antes_g = [p.grad.clone() for p in modelo.parameters()]
    antes_opt = copy.deepcopy(opt.state_dict())
    rng = torch.get_rng_state()
    dg.medir(modelo, X, y, 1, 0, 1, 0, "fim_")
    assert all(torch.equal(a, b) for a, b in zip(antes_p, modelo.parameters()))      # parâmetros
    assert all(torch.equal(a, p.grad) for a, p in zip(antes_g, modelo.parameters()))  # .grad intocado
    depois = opt.state_dict()
    assert depois["param_groups"] == antes_opt["param_groups"]
    assert all(torch.equal(antes_opt["state"][k][c], depois["state"][k][c])
               for k in antes_opt["state"] for c in ("exp_avg", "exp_avg_sq", "step"))   # estado do Adam
    assert modelo.training                                                           # modo da original
    assert torch.equal(rng, torch.get_rng_state())                                   # RNG global intocado


# ---------- validação das peças ----------

def test_gradientes_por_amostra_vetorizados_batem_com_o_laco():
    X, y = _dados()
    modelo = criar_mlp(4)
    G = dg.gradientes_por_amostra(modelo, X[:6], y[:6])
    assert G.shape == (6, sum(p.numel() for p in modelo.parameters()))
    for i in range(6):
        gi = torch.autograd.grad(F.cross_entropy(modelo(X[i:i + 1]), y[i:i + 1]), modelo.parameters())
        assert torch.allclose(G[i], torch.cat([x.flatten() for x in gi]), rtol=1e-4, atol=1e-6)


def test_srank_segue_a_definicao_do_survey():
    M = torch.diag(torch.tensor([8.0, 1.0, 1.0]))               # valores singulares 8, 1, 1 (soma 10)
    assert srank(M, 0.01) == 3                                    # precisa de 99%: os três
    assert srank(M, 0.15) == 2                                    # 85%: 8+1 = 90% já basta
    assert srank(M, 0.25) == 1                                    # 75%: só o primeiro (80%)
    assert srank(torch.zeros(4, 4), 0.01) == 0.0


def test_espectro_do_gradiente_casos_conhecidos():
    iguais = torch.ones(5, 7)
    e = dg.espectro_gradiente(iguais)
    assert e["grad_srank"] == 1 and abs(e["grad_cos_medio"] - 1) < 1e-6 and e["grad_cos_neg"] == 0
    opostos = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    e = dg.espectro_gradiente(opostos)
    assert abs(e["grad_cos_medio"] + 1) < 1e-6 and e["grad_cos_neg"] == 1.0


def test_cortes_de_custo_mantem_o_esquema_e_as_normas(monkeypatch):
    X, y = _dados()
    modelo = criar_mlp(5)
    completo, _ = dg.medir(modelo, X, y, 1, 0, 0, 0, "ini_")
    sem_hess, _ = dg.medir(modelo, X, y, 1, 0, 0, 0, "ini_", completo=False)          # corte (c)
    assert set(completo) == set(sem_hess) == set(dg.chaves("ini_"))
    assert math.isnan(sem_hess["ini_hess_max"]) and not math.isnan(sem_hess["ini_grad_l2_global"])
    assert sem_hess["ini_grad_l2_global"] == completo["ini_grad_l2_global"]               # normas: todos os dados
    monkeypatch.setattr(config, "HESS_PONTOS", 256)                                       # corte (a)
    c256, _ = dg.medir(modelo, X, y, 1, 0, 0, 0, "ini_")
    assert c256["ini_hess_max"] > 0 and c256["ini_grad_l2_global"] == completo["ini_grad_l2_global"]
    monkeypatch.setattr(config, "HESS_PARTIDA_QUENTE", True)                              # corte (b)
    _, v = dg.medir(modelo, X, y, 1, 0, 0, 0, "ini_")
    quente, _ = dg.medir(modelo, X, y, 1, 0, 1, 0, "fim_", v)
    assert abs(quente["fim_hess_max"] - completo["ini_hess_max"]) / completo["ini_hess_max"] < 0.05


def test_pontos_do_espectro_sao_os_mesmos_em_todas_as_medicoes():
    """Mesma semente e tarefa => mesmos 64 índices, em qualquer momento e para qualquer rede."""
    X, y = _dados()
    a, _ = dg.medir(criar_mlp(1), X, y, 1, 0, 0, 0, "ini_")
    b, _ = dg.medir(criar_mlp(1), X, y, 1, 0, 1, 1, "x_")          # outro momento e outra "quem"
    assert math.isclose(a["ini_grad_srank"], b["x_grad_srank"]) and math.isclose(a["ini_grad_cos_medio"], b["x_grad_cos_medio"], rel_tol=1e-6)
    c, _ = dg.medir(criar_mlp(1), X, y, 1, 1, 0, 0, "ini_")        # outra tarefa: índices diferentes
    assert not math.isclose(a["ini_grad_cos_medio"], c["ini_grad_cos_medio"], rel_tol=1e-9)


# ---------- reforços apontados pela auditoria do v3.1 ----------

def _quadratica(diag, iters, seed=1):
    w = torch.zeros(len(diag), requires_grad=True)
    d = torch.tensor(diag)
    return dg.autovalor_max(lambda: 0.5 * (d * w * w).sum(), [w], torch.Generator().manual_seed(seed), iters, 0.01)[:2]


def test_autovalor_dominante_negativo_sai_com_sinal():
    lam, conv = _quadratica([-3.0, 1.0], 30)
    assert abs(lam + 3.0) < 0.03 and conv                     # hess_max pode ser negativo (maior módulo)


def test_hess_conv_e_zero_quando_nao_estabilizou():
    lam, conv = _quadratica([1.0, 0.5], 7, seed=1)            # poucas iterações, partida desfavorável
    assert not conv
    lam, conv = _quadratica([1.0, 0.5], 30, seed=1)
    assert conv and abs(lam - 1.0) < 0.01


def test_v_ini_de_norma_zero_nao_quebra():
    w = torch.zeros(3, requires_grad=True)
    lam, conv, v = dg.autovalor_max(lambda: 0.5 * (torch.tensor([2.0, 1.0, 0.5]) * w * w).sum(), [w],
                                    torch.Generator().manual_seed(0), 30, 0.01, v_ini=[torch.zeros(3)])
    assert abs(lam - 2.0) < 0.02 and conv and abs(sum(x.norm() ** 2 for x in v).item() - 1) < 1e-5


def test_semente_de_calibracao_sai_sem_diagnosticos_mas_com_srank_feat(tmp_path):
    cond = replace(cs.condicao_base("central_passos"), T=2, passos=15)
    df = cs.executar(cond, "A", config.SEMENTE_CALIB, tmp_path / "calibracao")
    assert "srank_feat" in df.columns and "fim_hess_max" not in df.columns
    df = cs.executar(cond, "A", 1, tmp_path)
    assert "fim_hess_max" in df.columns and "nova_ini_hess_max" in df.columns


def test_ordem_das_medicoes_inicio_fim_e_redes_novas(tmp_path):
    sem_sp = replace(cs.condicao_base("central_passos"), T=8, passos=15)
    com_sp = replace(sem_sp, nome="central_SP", sp_lambda=0.8)
    a = cs.executar(sem_sp, "A", 1, tmp_path / "a")
    b = cs.executar(com_sp, "A", 1, tmp_path / "b")
    # sem SP o início da tarefa t é exatamente o fim da tarefa t-1; com SP o início vem DEPOIS do SP
    assert ((a["ini_norma_global"].iloc[1:].values - a["fim_norma_global"].iloc[:-1].values) ** 2 < 1e-9).all()
    assert (b["ini_norma_global"].iloc[1:].values < 0.9 * b["fim_norma_global"].iloc[:-1].values + 0.5).all()
    assert (b["ini_norma_global"].iloc[1:].values != b["fim_norma_global"].iloc[:-1].values).all()
    com_nova = set(config.tarefas_com_nova(8))
    for t in range(8):                                      # colunas nova_* só onde há rede nova
        assert a.loc[t, "nova_ini_hess_max"] == a.loc[t, "nova_ini_hess_max"] if t in com_nova else math.isnan(a.loc[t, "nova_ini_hess_max"])
        assert (not math.isnan(a.loc[t, "nova_fim_norma_global"])) == (t in com_nova)


def test_csv_valido_exige_diagnosticos_nas_sementes_de_avaliacao(tmp_path):
    cond = replace(cs.condicao_base("central_passos"), T=2, passos=15)
    cs.executar(cond, "A", 1, tmp_path, diagnosticos=False)         # como um CSV de antes do v3.1
    assert not cs.csv_valido(cond, "A", 1, tmp_path)
    cs.executar(cond, "A", 1, tmp_path, diagnosticos=True)
    assert cs.csv_valido(cond, "A", 1, tmp_path)
    cs.executar(cond, "A", config.SEMENTE_CALIB, tmp_path / "calibracao")   # a 1000 nunca tem diagnósticos
    assert cs.csv_valido(cond, "A", config.SEMENTE_CALIB, tmp_path / "calibracao")


def test_srank_feat_usa_o_mesmo_delta_e_e_igual_a_fim_srank_feat(tmp_path):
    df = cs.executar(replace(cs.condicao_base("central_passos"), T=2, passos=15), "A", 1, tmp_path)
    assert (df["srank_feat"] == df["fim_srank_feat"]).all()          # mesma conta: documentado no glossário
    assert config.SRANK_DELTA == 0.01
