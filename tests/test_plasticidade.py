"""Testes simples dos pontos que mais podem invalidar o experimento."""
import copy
import tempfile
from pathlib import Path

import torch

from src import config
from src.condicoes import condicao_base, treinar_na_tarefa
from src.dados import gerar_X, particionar, rotulos_tarefa
from src.federado import media_ponderada, treinar_fedavg
from src.metricas import fracao_dormentes, rank_efetivo
from src.modelo import criar_mlp
from src.sementes import PAPEIS, gerador, semente
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

    treinar_fedavg(criar_mlp(1), X, y, particionar("iid", 0), g, config.RODADAS_BASE)
    passos_por_cliente = contagem["passos"] / config.CLIENTES

    contagem["passos"] = 0
    treinar_na_tarefa(condicao_base("central_passos"), criar_mlp(1), X, y, None, g)
    assert passos_por_cliente == contagem["passos"] == config.PASSOS_CLIENTE == 375


def test_central_recria_o_adam_no_ritmo_de_uma_rodada(monkeypatch):
    criados = {"n": 0}
    init_original = torch.optim.Adam.__init__

    def contando(self, *a, **k):
        criados["n"] += 1
        return init_original(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "__init__", contando)
    X, y = gerar_X(0), rotulos_tarefa("A", 0, 0, gerar_X(0))
    treinar_na_tarefa(condicao_base("central_passos"), criar_mlp(1), X, y, None,
                      torch.Generator().manual_seed(0))
    assert criados["n"] == config.RODADAS_BASE     # 375 passos / 15 = 25 Adams


def test_fedavg_k1_global_igual_ao_cliente(monkeypatch):
    """Com 1 cliente e 1 rodada, o modelo global vira exatamente o modelo do cliente."""
    import src.federado as federado
    from src.modelo import MLP

    X, y = gerar_X(0), rotulos_tarefa("A", 0, 0, gerar_X(0))
    part = particionar("um", 0)
    inicial = criar_mlp(5)
    esperado = federado.treino_local(
        MLP().to(X.device), _estado(inicial), X, y, part[0], torch.Generator().manual_seed(4))
    global_ = copy.deepcopy(inicial)
    federado.treinar_fedavg(global_, X, y, part, torch.Generator().manual_seed(4), 1)
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
    treinar_fedavg(criar_mlp(1), X, y, particionar("iid", 0), torch.Generator().manual_seed(0),
                   config.RODADAS_BASE)
    assert criados["n"] == config.CLIENTES * config.RODADAS_BASE


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


def test_condicoes_congeladas_leem_o_json_e_marcam_calibracao_ruim(tmp_path):
    import json
    from src.condicoes import carregar_condicao
    pasta = tmp_path / "calibracao"
    pasta.mkdir()
    (pasta / "calibracao_A_m3.json").write_text(json.dumps({"passos": 120, "ok": True}))
    (pasta / "selecao_L2_A_m3.json").write_text(json.dumps({"lambda": 1e-3}))
    (pasta / "selecao_SP_A_m3.json").write_text(json.dumps({"lambda": 0.6}))
    c = carregar_condicao("central_aprendizado", "A", 3, tmp_path)
    assert c.passos == 120 and c.nivel == 3 and c.calibracao_ok
    assert carregar_condicao("central_L2", "A", 3, tmp_path).wd == 1e-3
    assert carregar_condicao("central_SP", "A", 3, tmp_path).sp_lambda == 0.6
    # calibração fora de ±0,03 não interrompe (Desvios v3): usa o valor e marca ok=False
    (pasta / "calibracao_A_m3.json").write_text(json.dumps({"passos": 120, "ok": False}))
    assert not carregar_condicao("central_aprendizado", "A", 3, tmp_path).calibracao_ok
    import pytest
    with pytest.raises(FileNotFoundError):                 # outro nível: sem calibração congelada
        carregar_condicao("central_aprendizado", "A", 10, tmp_path)


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
    mat = pd.read_csv(tmp_path / "central_passos_m1_B_disjunta_0_matriz.csv", index_col="i")
    diag = [mat.iloc[i, i] for i in range(4)]
    assert all(abs(d - a) < 1e-9 for d, a in zip(diag, df["acc_cont"]))   # CSV perde a 16ª casa
    assert mat.iloc[0, 1:].isna().all() and mat.iloc[3].notna().all()   # triangular
    assert {"acc_media_vistas", "esquecimento", "bwt"} <= set(df.columns)


def test_tarefas_a_e_b_nao_medem_esquecimento(tmp_path):
    from dataclasses import replace
    from src.condicoes import executar
    df = executar(replace(condicao_base("central_passos"), T=2, passos=15), "A", 0, tmp_path)
    assert "esquecimento" not in df.columns
    assert not (tmp_path / "central_passos_m1_A_0_matriz.csv").exists()


def test_csv_valido_detecta_csv_truncado_e_params_ausentes(tmp_path):
    import pandas as pd
    from dataclasses import replace
    from src.condicoes import csv_valido, executar
    cond = replace(condicao_base("central_passos"), T=3, passos=15)
    executar(cond, "A", 0, tmp_path)
    csv = tmp_path / "central_passos_m1_A_0.csv"
    params = tmp_path / "central_passos_m1_A_0.params.json"
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
    assert len(df) == 3 and df["acc_nova"].notna().sum() == 3          # T=3: todas são "últimas 5"
    assert not torch.equal(seq.dados(0)[0], seq.dados(2)[0])
    assert (tmp_path / "fedavg_iid_m1_B_disjunta_0_rodadas.csv").exists()   # 10 clientes
    assert df["rank_ef"].notna().all() and df["esquecimento"].iloc[1:].notna().all()


def test_sementes_antigas_nao_mudaram_com_os_papeis_novos():
    assert PAPEIS[:14] == (
        "dados", "rotulos", "professor", "particao",
        "init_continua", "init_nova", "embaralha_continua", "embaralha_nova",
        "sp_ruido", "sonda_alvo", "sonda_nova", "sonda_embaralha",
        "calib_nova", "calib_embaralha")
    assert PAPEIS[14:] == ("media_tarefa", "dados_tarefa")


# ---------- Desvios v3: S_local fixo, níveis, rede nova, sementes, balanceamento ----------

def test_tres_modos_de_fl_fazem_o_mesmo_numero_de_passos_por_cliente(monkeypatch):
    contagem = {"passos": 0}
    original = torch.optim.Adam.step

    def contando(self, *a, **k):
        contagem["passos"] += 1
        return original(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", contando)
    X, y = gerar_X(1), rotulos_tarefa("A", 1, 0, gerar_X(1))
    por_cliente = {}
    for modo in ("iid", "um", "total"):
        contagem["passos"] = 0
        part = particionar(modo, 1)
        treinar_fedavg(criar_mlp(1), X, y, part, torch.Generator().manual_seed(0), 1)   # 1 rodada
        por_cliente[modo] = contagem["passos"] / len(part)
    assert por_cliente == {"iid": 15, "um": 15, "total": 15} and config.S_LOCAL == 15


def test_nivel_m_multiplica_rodadas_e_passos():
    for m in config.NIVEIS:
        assert condicao_base("fedavg_iid", m).rodadas == 25 * m
        assert condicao_base("media_sem_particao", m).rodadas == 25 * m
        assert condicao_base("central_passos", m).passos == 375 * m


def test_tarefas_com_nova():
    assert config.tarefas_com_nova(30) == [2, 5, 7, 8, 9, 10, 11, 14, 17, 20, 23, 25, 26, 27, 28, 29]
    assert config.tarefas_com_nova(12) == [2, 5, 7, 8, 9, 10, 11]


def test_gap_so_existe_onde_ha_rede_nova_e_e_pareado(tmp_path):
    from dataclasses import replace
    from src.condicoes import executar
    cond = replace(condicao_base("central_passos"), T=9, passos=15)
    df = executar(cond, "A", 1, tmp_path)
    com_nova = config.tarefas_com_nova(9)
    assert list(df.index[df["acc_nova"].notna()]) == com_nova == list(df.index[df["gap"].notna()])
    ok = df.dropna(subset=["gap"])
    assert ((ok["acc_cont"] - ok["acc_nova"]) - ok["gap"]).abs().max() < 1e-12
    assert (df["nivel"] == 1).all() and df["maj_frac"].between(0.0, 1.0).all()


def test_sementes_de_calibracao_e_avaliacao_sao_disjuntas(tmp_path):
    import pytest
    from dataclasses import replace
    from src.condicoes import executar
    assert config.SEMENTE_CALIB not in config.SEMENTES_AVALIACAO
    assert 0 not in config.SEMENTES_AVALIACAO            # a semente 0 do desenho antigo é exploratória
    papeis = {semente(config.SEMENTE_CALIB, p, i) for p in PAPEIS for i in range(30)}
    avalia = {semente(s, p, i) for s in config.SEMENTES_AVALIACAO for p in PAPEIS for i in range(30)}
    assert not papeis & avalia
    cond = replace(condicao_base("central_passos"), T=1, passos=15)
    with pytest.raises(ValueError):                      # 1000 fora de .../calibracao/
        executar(cond, "A", config.SEMENTE_CALIB, tmp_path)
    with pytest.raises(ValueError):                      # semente de avaliação dentro de .../calibracao/
        executar(cond, "A", 1, tmp_path / "calibracao")
    executar(cond, "A", config.SEMENTE_CALIB, tmp_path / "calibracao")     # permitido


def test_professor_balanceado_em_b_e_b_disjunta():
    from src.dados import Sequencia
    for tipo in ("B", "B_disjunta"):
        for sem in (1, 2, 3, 4, 5, config.SEMENTE_CALIB):
            seq = Sequencia(tipo, sem)
            assert all(seq.maj_frac(t) < 0.3 for t in range(10)), (tipo, sem)
            assert len(seq.dados(0)[1].unique()) >= 8        # nada de tarefa com 1 ou 2 classes


def test_csv_valido_rejeita_outro_nivel_e_outra_versao_dos_dados(tmp_path):
    from dataclasses import replace
    from src.condicoes import csv_valido, executar
    cond = replace(condicao_base("central_passos"), T=2, passos=15)
    executar(cond, "A", 1, tmp_path)
    assert csv_valido(cond, "A", 1, tmp_path)
    assert not csv_valido(replace(cond, versao_dados=1), "A", 1, tmp_path)
    assert not csv_valido(replace(cond, rodadas=75, nivel=3), "A", 1, tmp_path)   # arquivo é de outro nível


def test_calibracao_incremental_igual_a_treinos_separados():
    from src.treino import treinar_passos
    from src.metricas import acuracia
    X, y = gerar_X(1000), rotulos_tarefa("A", 1000, 0, gerar_X(1000))
    todos = torch.arange(config.N, device=X.device)
    lidas = {}
    m = criar_mlp(7)
    treinar_passos(m, X, y, todos, 90, config.BATCH, config.RESET_CENTRAL,
                   torch.Generator().manual_seed(3),
                   ao_fim_de_bloco=lambda k: lidas.__setitem__(k, acuracia(m, X, y)))
    assert sorted(lidas) == [15, 30, 45, 60, 75, 90]
    for k in (15, 45, 90):                                # inclui cruzar a fronteira de época (50 lotes)
        m2 = criar_mlp(7)
        treinar_passos(m2, X, y, todos, k, config.BATCH, config.RESET_CENTRAL,
                       torch.Generator().manual_seed(3))
        assert abs(acuracia(m2, X, y) - lidas[k]) < 1e-12


def test_calibrar_nao_interrompe_fora_da_tolerancia(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pandas as pd
    import experiments.calibrar as cal
    pasta = tmp_path / "calibracao"
    pasta.mkdir()
    tarefas = config.tarefas_com_nova(config.T_PADRAO)
    nova = np.full(config.T_PADRAO, np.nan)
    nova[tarefas] = 0.99                                   # alvo inalcançável com 45 passos
    pd.DataFrame({"tarefa": range(config.T_PADRAO), "acc_nova": nova}).to_csv(
        pasta / "fedavg_iid_m1_A_1000.csv", index=False)
    monkeypatch.setattr(cal, "PASSOS_MAX_CALIB", 45)
    res = cal.calibrar("A", 1, tmp_path)
    assert res["passos"] in (15, 30, 45) and res["ok"] is False
    assert abs(res["alvo_acc_nova_fedavg"] - 0.99) < 1e-12        # alvo = média das acc_nova (16 tarefas com T=30)
    assert len(res["tentativas"]) == 3 and res["versao_dados"] == config.VERSAO_DADOS
    assert json.loads((pasta / "calibracao_A_m1.json").read_text())["ok"] is False


def test_selecao_respeita_o_limite_de_esquecimento(tmp_path, monkeypatch):
    import pandas as pd
    import experiments.selecionar as sel

    def falso(acc, esq):
        return pd.DataFrame({"acc_cont": [acc] * 5, "esquecimento": [esq] * 5})

    tabela = {"central_aprendizado": falso(0.5, 0.10), "_lam0.0001": falso(0.90, 0.30),
              "_lam0.001": falso(0.80, 0.14), "_lam0.01": falso(0.60, 0.05)}
    monkeypatch.setattr(sel, "_rodar", lambda cond, tarefa, pasta, sufixo="":
                        tabela[sufixo] if sufixo else tabela[cond.nome])
    pasta = tmp_path / "calibracao"
    pasta.mkdir()
    import json
    (pasta / "calibracao_B_disjunta_m3.json").write_text(json.dumps({"passos": 90, "ok": True}))
    r = sel.selecionar("B_disjunta", "L2", 3, tmp_path)
    assert r["lambda"] == 1e-3 and not r["violacao"]       # 0.90 viola (0.30 > 0.10+0.05); 0.80 respeita
    # nenhum respeita: escolhe o de menor esquecimento e marca violação
    tabela.update({"_lam0.001": falso(0.80, 0.40), "_lam0.01": falso(0.60, 0.20)})
    r = sel.selecionar("B_disjunta", "L2", 3, tmp_path)
    assert r["lambda"] == 1e-2 and r["violacao"]
    # em A não há esquecimento medido: vale só a acurácia
    (pasta / "calibracao_A_m3.json").write_text(json.dumps({"passos": 90, "ok": True}))
    tabela.update({"_lam0.4": falso(0.70, 0.0), "_lam0.6": falso(0.60, 0.0), "_lam0.8": falso(0.50, 0.0)})
    assert sel.selecionar("A", "SP", 3, tmp_path)["lambda"] == 0.4


def test_regime_de_teste_e_o_menor_nivel_com_gap_ate_menos_0_10(tmp_path):
    import json
    import pandas as pd
    from experiments.pipeline import definir_regime
    pasta = tmp_path / "calibracao"
    pasta.mkdir()

    def csv(m, gap):
        n = [float("nan")] * 25 + [gap] * 5
        pd.DataFrame({"tarefa": range(30), "gap": n}).to_csv(
            pasta / f"central_aprendizado_m{m}_A_1000.csv", index=False)

    for m, g in ((1, 0.12), (3, -0.05), (10, -0.40)):
        csv(m, g)
    assert definir_regime(tmp_path)["m_star"] == 10
    csv(3, -0.10)
    assert definir_regime(tmp_path)["m_star"] == 3         # -0,10 conta (<=)
    for m in config.NIVEIS:
        csv(m, 0.05)
    r = definir_regime(tmp_path)
    assert r["m_star"] is None and "não testáveis" in r["nota"]
    assert json.loads((pasta / "regime.json").read_text())["m_star"] is None


def test_tarefa_a_nao_e_centralizada_e_b_usa_logits_centralizados():
    from src.dados import Sequencia
    y_a = rotulos_tarefa("A", 1, 3, gerar_X(1))
    esperado = torch.randint(0, config.CLASSES, (config.N,), generator=gerador(1, "rotulos", 3))
    assert torch.equal(y_a.cpu(), esperado)                          # A: sorteio puro, sem professor
    X = gerar_X(1)
    prof = criar_mlp(semente(1, "professor", 3))
    with torch.no_grad():
        bruto = prof(X).argmax(1)
    assert not torch.equal(Sequencia("B", 1).dados(3)[1], bruto)     # B: argmax dos logits centralizados


def test_adam_recriado_por_rodada_em_um_e_total(monkeypatch):
    criados = {"n": 0}
    init_original = torch.optim.Adam.__init__

    def contando(self, *a, **k):
        criados["n"] += 1
        return init_original(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "__init__", contando)
    X, y = gerar_X(1), rotulos_tarefa("A", 1, 0, gerar_X(1))
    for modo, k in (("um", 1), ("total", config.CLIENTES)):
        criados["n"] = 0
        treinar_fedavg(criar_mlp(1), X, y, particionar(modo, 1), torch.Generator().manual_seed(0), 4)
        assert criados["n"] == 4 * k                                  # um Adam novo por cliente por rodada


def test_params_das_centralizadas_nao_trazem_rodadas_nem_s_local():
    c = condicao_base("central_passos", 10)
    assert c.rodadas is None and c.s_local is None and c.passos == 3750
    f = condicao_base("fedavg_iid", 10)
    assert f.rodadas == 250 and f.s_local == 15


def test_redes_novas_extras_nao_alteram_o_continuo(monkeypatch):
    """As redes novas de referência não podem consumir o gerador nem o estado da rede contínua:
    com e sem as referências extras, tudo o que é do contínuo sai bit a bit igual."""
    import pandas as pd
    from dataclasses import replace
    import src.condicoes as cs
    original = cs.tarefas_com_nova          # guarda a função real: o monkeypatch troca em cada rodada
    colunas = ["acc_cont", "norma", "distancia", "dorm_tau0_l1", "dorm_tau0_l2", "dorm_tau0025_l1",
               "dorm_tau0025_l2", "rank_ef", "cos_medio", "sonda_razao", "sonda_perda_copia",
               "sonda_perda_nova", "maj_frac"]
    casos = [
        (replace(condicao_base("central_passos"), nome="central_SP", T=30, passos=15, sp_lambda=0.8), "A"),      # SP + T=30 inteiro
        (replace(condicao_base("fedavg_iid"), T=12, rodadas=2), "B_disjunta"),            # FL + matriz de esquecimento
    ]
    for cond, tipo in casos:
        completo = original(cond.T)
        reduzido = [t for t in completo if t % 3 == 2]            # sem as extras
        resultados = {}
        for nome, lista in (("completo", completo), ("reduzido", reduzido), ("nenhuma", [])):
            monkeypatch.setattr(cs, "tarefas_com_nova", lambda T, lista=lista: lista)
            resultados[nome] = cs.executar(cond, tipo, 1, Path(tempfile.mkdtemp()))
        for nome in ("reduzido", "nenhuma"):
            a, b = resultados["completo"], resultados[nome]
            pd.testing.assert_frame_equal(a[colunas], b[colunas], check_exact=True)
            if tipo == "B_disjunta":
                for col in ("esquecimento", "bwt", "acc_media_vistas"):
                    pd.testing.assert_series_equal(a[col], b[col], check_exact=True)
        assert resultados["completo"]["acc_nova"].notna().sum() == len(completo) > len(reduzido)
        assert resultados["nenhuma"]["acc_nova"].notna().sum() == 0
