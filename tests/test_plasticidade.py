"""Testes simples dos pontos que mais podem invalidar o experimento."""
import copy

import torch

from src import config
from src.condicoes import condicao_base, treinar_na_tarefa
from src.dados import gerar_X, particionar, rotulos_tarefa
from src.federado import media_ponderada, treinar_fedavg
from src.metricas import fracao_dormentes, rank_efetivo
from src.modelo import criar_mlp
from src.sementes import PAPEIS, semente
from src.sondas import sonda_alvos_aleatorios


def _estado(modelo):
    return {k: v.detach().clone() for k, v in modelo.state_dict().items()}


def _iguais(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_sonda_nao_altera_a_rede_original():
    X = gerar_X(0)
    modelo = criar_mlp(123)
    antes = _estado(modelo)
    sonda_alvos_aleatorios(modelo, X, 0, 4)
    assert _iguais(antes, _estado(modelo))
    assert all(p.grad is None for p in modelo.parameters())


def test_media_de_modelos_identicos_devolve_o_mesmo_modelo():
    estado = _estado(criar_mlp(7))
    media = media_ponderada([copy.deepcopy(estado) for _ in range(10)], [100] * 10)
    for k in estado:
        assert torch.allclose(media[k], estado[k], atol=1e-6)


def test_media_e_ponderada_pelo_numero_de_amostras():
    a, b = _estado(criar_mlp(1)), _estado(criar_mlp(2))
    media = media_ponderada([a, b], [100, 300])
    for k in a:
        assert torch.allclose(media[k], 0.25 * a[k] + 0.75 * b[k], atol=1e-6)


def test_com_k1_a_media_devolve_o_modelo_do_cliente():
    estado = _estado(criar_mlp(9))
    media = media_ponderada([estado], [1000])
    assert _iguais(media, estado)


def test_camada_toda_zero_da_100_por_cento_dormentes():
    zeros = torch.zeros(1000, 128)
    for tau in config.TAUS:
        assert fracao_dormentes(zeros, tau) == 1.0


def test_dormentes_sokar_conta_so_neuronios_sem_ativacao():
    h = torch.ones(10, 4)
    h[:, 0] = 0.0                       # 1 neurônio de 4 nunca ativa
    assert fracao_dormentes(h, 0.0) == 0.25
    h[:, 1] = 0.01 * h[:, 2].mean()     # score ~0,0199: dormente só com tau=0,025
    assert fracao_dormentes(h, 0.0) == 0.25
    assert fracao_dormentes(h, 0.025) == 0.5


def test_rank_efetivo_de_ativacoes_constantes_e_1():
    h = torch.ones(50, 8)
    assert abs(rank_efetivo(h) - 1.0) < 1e-4


def test_rede_nova_usa_semente_diferente_da_continua():
    for sem in (0, 1, 2):
        s_cont = semente(sem, "init_continua")
        for t in range(config.T_PADRAO):
            assert semente(sem, "init_nova", t) != s_cont
    p_cont = torch.cat([p.flatten() for p in criar_mlp(semente(0, "init_continua")).parameters()])
    p_nova = torch.cat([p.flatten() for p in criar_mlp(semente(0, "init_nova", 2)).parameters()])
    assert not torch.equal(p_cont, p_nova)


def test_papeis_e_sementes_base_dao_sementes_distintas():
    todas = {semente(s, p, i) for s in range(5) for p in PAPEIS for i in range(3)}
    assert len(todas) == 5 * len(PAPEIS) * 3


def test_rotulos_sao_deterministicos_e_mudam_a_cada_tarefa():
    X = gerar_X(3)
    assert torch.equal(rotulos_tarefa("B", 3, 5, X), rotulos_tarefa("B", 3, 5, X))
    assert not torch.equal(rotulos_tarefa("B", 3, 5, X), rotulos_tarefa("B", 3, 6, X))
    assert not torch.equal(rotulos_tarefa("A", 3, 5, X), rotulos_tarefa("A", 3, 6, X))


def test_particoes():
    iid = particionar("iid", 0)
    assert len(iid) == 10 and all(len(p) == 100 for p in iid)
    assert len(torch.cat(iid).unique()) == 1000          # sem sobreposição
    assert [len(p) for p in particionar("um", 0)] == [1000]
    total = particionar("total", 0)
    assert len(total) == 10 and all(torch.equal(p, total[0]) for p in total)


def test_central_passos_faz_o_mesmo_numero_de_passos_de_um_cliente_do_fedavg(monkeypatch):
    contagem = {"passos": 0}
    passo_original = torch.optim.Adam.step

    def contando(self, *a, **k):
        contagem["passos"] += 1
        return passo_original(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", contando)
    X, y = gerar_X(0), rotulos_tarefa("A", 0, 0, gerar_X(0))
    g = torch.Generator().manual_seed(0)

    treinar_fedavg(criar_mlp(1), X, y, particionar("iid", 0), g)
    passos_por_cliente = contagem["passos"] / config.CLIENTES

    contagem["passos"] = 0
    treinar_na_tarefa(condicao_base("central_passos"), criar_mlp(1), X, y, 0, None, g)
    assert passos_por_cliente == contagem["passos"] == config.PASSOS_CLIENTE == 375


def test_central_recria_o_adam_no_ritmo_de_uma_rodada(monkeypatch):
    criados = {"n": 0}
    init_original = torch.optim.Adam.__init__

    def contando(self, *a, **k):
        criados["n"] += 1
        return init_original(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "__init__", contando)
    X, y = gerar_X(0), rotulos_tarefa("A", 0, 0, gerar_X(0))
    treinar_na_tarefa(condicao_base("central_passos"), criar_mlp(1), X, y, 0, None,
                      torch.Generator().manual_seed(0))
    assert criados["n"] == config.RODADAS     # 375 passos / 15 = 25 Adams


def test_fedavg_k1_global_igual_ao_cliente(monkeypatch):
    """Com 1 cliente e 1 rodada, o modelo global vira exatamente o modelo do cliente."""
    import src.federado as federado
    from src.modelo import MLP

    monkeypatch.setattr(federado, "RODADAS", 1)
    X, y = gerar_X(0), rotulos_tarefa("A", 0, 0, gerar_X(0))
    part = particionar("um", 0)
    inicial = criar_mlp(5)
    esperado = federado.treino_local(
        MLP().to(X.device), _estado(inicial), X, y, part[0], torch.Generator().manual_seed(4))
    global_ = copy.deepcopy(inicial)
    federado.treinar_fedavg(global_, X, y, part, torch.Generator().manual_seed(4))
    assert _iguais(_estado(global_), esperado)


def test_rank_efetivo_de_k_direcoes_iguais_e_k():
    h = torch.eye(8)[:, :4].repeat(5, 1)          # 4 colunas ortogonais de mesma norma
    assert abs(rank_efetivo(h) - 4.0) < 1e-3


def test_fedavg_recria_o_adam_a_cada_rodada_de_cada_cliente(monkeypatch):
    criados = {"n": 0}
    init_original = torch.optim.Adam.__init__

    def contando(self, *a, **k):
        criados["n"] += 1
        return init_original(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "__init__", contando)
    X, y = gerar_X(0), rotulos_tarefa("A", 0, 0, gerar_X(0))
    treinar_fedavg(criar_mlp(1), X, y, particionar("iid", 0), torch.Generator().manual_seed(0))
    assert criados["n"] == config.CLIENTES * config.RODADAS


def test_cliente_fl_faz_15_passos_por_rodada_com_batch_20(monkeypatch):
    lotes = []
    from src import treino
    original = treino.nn.CrossEntropyLoss

    class Espiao(original):
        def forward(self, entrada, alvo):
            lotes.append(len(alvo))
            return super().forward(entrada, alvo)

    monkeypatch.setattr(treino.nn, "CrossEntropyLoss", Espiao)
    from src.federado import treino_local
    from src.modelo import MLP
    X, y = gerar_X(0), rotulos_tarefa("A", 0, 0, gerar_X(0))
    treino_local(MLP().to(X.device), _estado(criar_mlp(1)), X, y, particionar("iid", 0)[0],
                 torch.Generator().manual_seed(0))
    assert lotes == [config.BATCH] * config.PASSOS_RODADA


def test_shrink_and_perturb():
    from src.treino import shrink_and_perturb
    modelo = criar_mlp(3)
    antes = torch.cat([p.detach().flatten() for p in modelo.parameters()])
    shrink_and_perturb(modelo, 0.5, 0.0, torch.Generator().manual_seed(0))   # sem ruído: só encolhe
    depois = torch.cat([p.detach().flatten() for p in modelo.parameters()])
    assert torch.allclose(depois, 0.5 * antes)
    shrink_and_perturb(modelo, 0.0, 0.01, torch.Generator().manual_seed(0))  # lambda 0: só ruído
    ruido = torch.cat([p.detach().flatten() for p in modelo.parameters()])
    assert abs(ruido.std().item() - 0.01) < 1e-3


def test_l2_encolhe_a_norma_mais_que_sem_l2():
    from src.metricas import norma_parametros
    from src.treino import treinar_passos
    X, y = gerar_X(0), rotulos_tarefa("A", 0, 0, gerar_X(0))
    todos = torch.arange(config.N, device=X.device)
    normas = {}
    for wd in (0.0, 1e-2):
        m = criar_mlp(1)
        treinar_passos(m, X, y, todos, 150, config.BATCH, config.RESET_CENTRAL,
                       torch.Generator().manual_seed(0), wd=wd)
        normas[wd] = norma_parametros(m)
    assert normas[1e-2] < normas[0.0]


def test_condicoes_congeladas_leem_o_json_e_recusam_calibracao_ruim(tmp_path):
    import json
    from src.condicoes import carregar_condicao
    (tmp_path / "calibracao_A.json").write_text(json.dumps({"passos": 120, "ok": True}))
    (tmp_path / "selecao_L2_A.json").write_text(json.dumps({"lambda": 1e-3}))
    (tmp_path / "selecao_SP_A.json").write_text(json.dumps({"lambda": 0.6}))
    assert carregar_condicao("central_aprendizado", "A", tmp_path).passos == 120
    assert carregar_condicao("central_L2", "A", tmp_path).wd == 1e-3
    assert carregar_condicao("central_SP", "A", tmp_path).sp_lambda == 0.6
    (tmp_path / "calibracao_A.json").write_text(json.dumps({"passos": 120, "ok": False}))
    import pytest
    with pytest.raises(RuntimeError):
        carregar_condicao("central_aprendizado", "A", tmp_path)


def test_csv_valido_exige_mesmos_parametros_e_numero_de_tarefas(tmp_path):
    from dataclasses import replace
    from src.condicoes import csv_valido, executar
    cond = replace(condicao_base("central_passos"), T=2, passos=15)
    executar(cond, "A", 0, tmp_path)
    assert csv_valido(cond, "A", 0, tmp_path)
    assert not csv_valido(replace(cond, passos=30), "A", 0, tmp_path)   # parâmetro mudou
    assert not csv_valido(replace(cond, T=3), "A", 0, tmp_path)         # T diferente


# ---------- Adendo 1: B_disjunta e esquecimento ----------

def test_b_disjunta_tem_x_proprio_com_media_de_norma_3():
    from src.dados import Sequencia, mu_tarefa
    seq = Sequencia("B_disjunta", 0)
    X0, y0 = seq.dados(0)
    X1, y1 = seq.dados(1)
    assert X0.shape == (config.N, config.D) and not torch.equal(X0, X1)
    assert abs(mu_tarefa(0, 4).norm().item() - 3.0) < 1e-5
    assert not torch.equal(mu_tarefa(0, 0), mu_tarefa(0, 1))
    assert (y0 != y1).any()
    # com 1000 pontos, a média amostral fica perto de mu_t (erro ~ 1/sqrt(1000) por coordenada)
    assert (X1.mean(0).cpu() - mu_tarefa(0, 1)).abs().max() < 0.2
    # nas tarefas A e B o X é o mesmo em todas as tarefas
    sb = Sequencia("B", 0)
    assert torch.equal(sb.dados(0)[0], sb.dados(5)[0])


def test_esquecimento_e_zero_para_rede_que_nao_muda():
    import numpy as np
    from src.dados import Sequencia
    from src.esquecimento import linha_da_matriz, metricas_esquecimento
    seq, modelo, T = Sequencia("B_disjunta", 0), criar_mlp(1), 6
    a = np.full((T, T), np.nan)
    for i in range(T):                       # a rede nunca é treinada: nada muda entre linhas
        a[i] = linha_da_matriz(modelo, seq, i, T)
        m = metricas_esquecimento(a, i)
        if i > 0:
            assert m["esquecimento"] == 0.0 and m["bwt"] == 0.0


def test_esquecimento_e_bwt_em_matriz_conhecida():
    import numpy as np
    from src.esquecimento import metricas_esquecimento
    n = np.nan
    a = np.array([[0.9, n, n],
                  [0.7, 0.8, n],
                  [0.5, 0.6, 0.9]])
    m = metricas_esquecimento(a, 2)
    # j=0: max(a00,a10)=0.9 -> 0.9-0.5=0.4 ; j=1: max(a11)=0.8 -> 0.8-0.6=0.2
    assert abs(m["esquecimento"] - 0.3) < 1e-9
    # bwt: (0.5-0.9 + 0.6-0.8)/2 = -0.3
    assert abs(m["bwt"] + 0.3) < 1e-9
    assert abs(m["acc_media_vistas"] - (0.5 + 0.6 + 0.9) / 3) < 1e-9


def test_matriz_tem_diagonal_igual_a_acuracia_de_treino_no_fim_de_cada_tarefa(tmp_path):
    import pandas as pd
    from dataclasses import replace
    from src.condicoes import executar
    cond = replace(condicao_base("central_passos"), T=4, passos=30)
    df = executar(cond, "B_disjunta", 0, tmp_path)
    mat = pd.read_csv(tmp_path / "central_passos_B_disjunta_0_matriz.csv", index_col="i")
    diag = [mat.iloc[i, i] for i in range(4)]
    assert all(abs(d - a) < 1e-9 for d, a in zip(diag, df["acc_cont"]))   # CSV perde a 16ª casa
    assert mat.iloc[0, 1:].isna().all() and mat.iloc[3].notna().all()   # triangular
    assert {"acc_media_vistas", "esquecimento", "bwt"} <= set(df.columns)


def test_tarefas_a_e_b_nao_medem_esquecimento(tmp_path):
    from dataclasses import replace
    from src.condicoes import executar
    df = executar(replace(condicao_base("central_passos"), T=2, passos=15), "A", 0, tmp_path)
    assert "esquecimento" not in df.columns
    assert not (tmp_path / "central_passos_A_0_matriz.csv").exists()


def test_csv_valido_detecta_csv_truncado_e_params_ausentes(tmp_path):
    import pandas as pd
    from dataclasses import replace
    from src.condicoes import csv_valido, executar
    cond = replace(condicao_base("central_passos"), T=3, passos=15)
    executar(cond, "A", 0, tmp_path)
    csv = tmp_path / "central_passos_A_0.csv"
    params = tmp_path / "central_passos_A_0.params.json"
    assert csv_valido(cond, "A", 0, tmp_path)
    pd.read_csv(csv).head(2).to_csv(csv, index=False)        # CSV truncado, params iguais
    assert not csv_valido(cond, "A", 0, tmp_path)
    executar(cond, "A", 0, tmp_path)
    params.unlink()                                          # params ausente
    assert not csv_valido(cond, "A", 0, tmp_path)


def test_esquecimento_separa_chaudhry_de_bwt_e_i0_e_nan():
    import numpy as np
    from src.esquecimento import metricas_esquecimento
    n = np.nan
    # a[1][0]=0.95 > a[0][0]=0.9: o máximo de Chaudhry (0.95) difere da diagonal (BWT).
    # a[2][1]=0.9 > max anterior da tarefa 1 (0.8): esquecimento negativo para j=1.
    a = np.array([[0.90, n, n],
                  [0.95, 0.80, n],
                  [0.60, 0.90, 0.85]])
    m = metricas_esquecimento(a, 2)
    # Chaudhry: j=0: max(a00,a10)=0.95 -> 0.95-0.60=0.35 ; j=1: max(a11)=0.80 -> 0.80-0.90=-0.10
    assert abs(m["esquecimento"] - 0.125) < 1e-9
    # BWT: (0.60-0.90 + 0.90-0.80)/2 = -0.10  (e não é -esquecimento)
    assert abs(m["bwt"] + 0.10) < 1e-9
    m0 = metricas_esquecimento(a, 0)
    assert np.isnan(m0["esquecimento"]) and np.isnan(m0["bwt"]) and m0["acc_media_vistas"] == 0.90


def test_fedavg_em_b_disjunta_usa_x_t_e_aplica_a_particao(tmp_path):
    from dataclasses import replace
    from src.condicoes import executar
    from src.dados import Sequencia
    cond = replace(condicao_base("fedavg_iid"), T=3)
    df = executar(cond, "B_disjunta", 0, tmp_path)
    seq = Sequencia("B_disjunta", 0)
    assert len(df) == 3 and df["acc_nova"].notna().sum() == 1          # rede nova na tarefa 2
    assert not torch.equal(seq.dados(0)[0], seq.dados(2)[0])
    assert (tmp_path / "fedavg_iid_B_disjunta_0_rodadas.csv").exists()   # 10 clientes
    assert df["rank_ef"].notna().all() and df["esquecimento"].iloc[1:].notna().all()


def test_sementes_antigas_nao_mudaram_com_os_papeis_novos():
    assert PAPEIS[:14] == (
        "dados", "rotulos", "professor", "particao",
        "init_continua", "init_nova", "embaralha_continua", "embaralha_nova",
        "sp_ruido", "sonda_alvo", "sonda_nova", "sonda_embaralha",
        "calib_nova", "calib_embaralha")
    assert PAPEIS[14:] == ("media_tarefa", "dados_tarefa")
