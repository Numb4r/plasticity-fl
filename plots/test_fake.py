"""Roda tudo contra CSVs sintéticos do esquema v3 (em tempfile.mkdtemp(), nunca em results/) e confere
arquivos gerados e vereditos. Defina PLAST_TEST_KEEP=1 para manter o diretório temporário."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "plots"))
import analise  # noqa: E402

T_COND = {"fedavg_iid": 30, "sem_media_k1": 30, "media_sem_particao": 12, "central_passos": 30,
          "central_aprendizado": 30, "central_L2": 30, "central_SP": 30}
SEMENTES = range(1, 6)
SAIDAS = ("fig_gap.png", "fig_norma.png", "fig_sonda.png", "fig_tradeoff.png", "fig_dose.png",
          "fig_k1_mecanismo.png", "tabela_decisao.md")
ACC_NOVA = {1: 0.80, 3: 0.88, 10: 0.95}   # a rede nova melhora com o orçamento (eixo x da dose)
OFFSET_COND = {c: 0.004 * i for i, c in enumerate(T_COND)}   # acc_nova difere por condição (pega x lido da errada)
FORA = 0.3                                # armadilha: gap/esquecimento fora do nível m* são deslocados


def _v(v, t):
    return v(t) if callable(v) else v


def _escrever_json(pasta, nome, obj):
    os.makedirs(os.path.join(pasta, "calibracao"), exist_ok=True)
    with open(os.path.join(pasta, "calibracao", nome), "w", encoding="utf-8") as f:
        json.dump(obj, f)


def gerar_calib(pasta, m_star, a_m10_acima=True, regime=True, calib_ok=True):
    """regime.json, calibracao_<tarefa>_m<m>.json e selecao_<L2|SP>_<tarefa>_m<m*>.json."""
    if regime:
        _escrever_json(pasta, "regime.json", dict(
            m_star=m_star, gap_central_aprendizado_A={"1": -0.01, "3": -0.12 if m_star else -0.02, "10": -0.3},
            nota=None if m_star else analise.MSG_SEM_REGIME))
    for t in ("A", "B", "B_disjunta"):
        for m in analise.NIVEIS:
            acima = a_m10_acima or not (t == "A" and m == 10)
            _escrever_json(pasta, f"calibracao_{t}_m{m}.json", {
                "alvo_acc_nova_fedavg": ACC_NOVA[m] if acima else 0.88, "passos": 15 * m * 30, "epocas": 5 * m,
                "acc_central": ACC_NOVA[m], "erro": 0.01 if calib_ok else 0.08, "ok": calib_ok, "alvo_acima_de_0.90": bool(acima and m == 10)
                if t == "A" and m == 10 else ACC_NOVA[m] > 0.9})
    if m_star:
        for met, lam in (("L2", 1e-3), ("SP", 0.6)):
            for t in ("A", "B", "B_disjunta"):
                # formato real de experiments/selecionar.py: notas e esquecimento são dicts {λ: valor};
                # esquecimento_ref é float (null e esquecimento {} fora de B_disjunta)
                bd = t == "B_disjunta"
                _escrever_json(pasta, f"selecao_{met}_{t}_m{m_star}.json", {
                    "tarefa": t, "metodo": met, "nivel": m_star, "lambda": lam,
                    "notas": {"0.0001": 0.90, str(lam): 0.93, "0.5": 0.70},
                    "esquecimento": {"0.0001": 0.20, str(lam): 0.10, "0.5": 0.04} if bd else {},
                    "esquecimento_ref": 0.10 if bd else None, "violacao": False})


# Espelha src/diagnosticos.py::chaves (CAMADAS = l1,l2,l3); os prefixos ini_/fim_/nova_ini_/nova_fim_ vêm de src/condicoes.py.
# test_conferir_esquema() lê o texto desses dois arquivos (sem importar torch) e confere os nomes.
COLS_DIAG = ([f"norma_{c}_{t}" for c in ("l1", "l2", "l3") for t in "wb"] + ["norma_global"]
             + [f"grad_{n}_{c}_{t}" for n in ("l2", "l1") for c in ("l1", "l2", "l3") for t in "wb"]
             + ["grad_l2_global", "grad_l1_global", "grad_srank", "grad_cos_medio", "grad_cos_neg",
                "hess_max", "hess_conv", "srank_feat"])
PREFIXOS_DIAG = ("ini_", "fim_", "nova_ini_", "nova_fim_")


def conferir_esquema():
    """Os nomes de COLS_DIAG e dos prefixos existem em src/diagnosticos.py e src/condicoes.py."""
    with open(os.path.join(RAIZ, "src", "diagnosticos.py"), encoding="utf-8") as f:
        diag = f.read()
    with open(os.path.join(RAIZ, "src", "condicoes.py"), encoding="utf-8") as f:
        cond = f.read()
    assert 'CAMADAS = ("l1", "l2", "l3")' in diag
    for trecho in ('f"norma_{c}_{t}" for c in CAMADAS for t in "wb"', '"norma_global"',
                   'f"grad_{n}_{c}_{t}" for n in ("l2", "l1") for c in CAMADAS for t in "wb"',
                   '"grad_l2_global", "grad_l1_global", "grad_srank", "grad_cos_medio", "grad_cos_neg"',
                   '"hess_max", "hess_conv", "srank_feat"'):
        assert trecho in diag, trecho
    for pref in PREFIXOS_DIAG:
        assert f'"{pref}"' in cond, pref
    assert len(COLS_DIAG) == 6 + 1 + 12 + 5 + 3


def colunas_diag(d, tem, rng):
    """Colunas novas (Desvios v3.1) sintéticas: ini_/fim_ no contínuo e nova_ini_/nova_fim_ onde há rede nova.
    Norma e Hessiana sobem com a tarefa; hess_conv = 0 em parte dos pontos (marcador vazado)."""
    T = len(d)
    t = np.arange(T)
    novas = {}
    for pref, nova in zip(PREFIXOS_DIAG, (False, False, True, True)):
        for c in COLS_DIAG:
            if c == "hess_conv":
                v = (rng.random(T) > 0.3).astype(float)
            elif c == "hess_max":
                v = 5 + 0.5 * t + rng.normal(0, .2, T)
                v = np.where((t == 4) | (t == 13), -2.0 - 0.1 * t, v)   # autovalor de maior módulo pode ser negativo
            else:
                v = 1 + 0.02 * t + rng.normal(0, .02, T)
            if nova:
                v = np.where(tem, v * 0.7, np.nan)
            novas[pref + c] = v
    novas["srank_feat"] = 40.0
    return pd.concat([d, pd.DataFrame(novas, index=d.index)], axis=1)


def gerar(pasta, m_star, sementes, tipos, gap, esq, rng, niveis=analise.NIVEIS, diag=True):
    """gap[(cond, tipo)] = gap no nível m* (número ou função de t), padrão 0,05; fora de m* soma FORA.
    esq[cond] = esquecimento final em B_disjunta em m* (padrão 0,10); em t < T-1 vale nível + 0,3.
    L2 e SP só existem em m*. Com m_star None, o nível 'central' dos gaps é 3 e L2/SP não existem."""
    os.makedirs(pasta, exist_ok=True)
    ref = m_star or 3
    comum = {(tp, s): rng.normal(0, 0.02) for tp in analise.TIPOS for s in [0, 6] + list(SEMENTES) + [1000]}
    comum_e = {s: rng.normal(0, 0.02) for s in [0, 6] + list(SEMENTES) + [1000]}
    for cond, T in T_COND.items():
        t = np.arange(T)
        for m in niveis:
            if cond in ("central_L2", "central_SP") and (m_star is None or m != m_star):
                continue
            for tipo in tipos:
                for s in sementes:
                    desloc = 0.0 if m == ref else FORA
                    lvl = np.array([_v(gap.get((cond, tipo), 0.05), x) for x in t], dtype=float) + desloc
                    tem = (t % 3 == 2) | (t >= T - 5) | ((t >= 7) & (t <= 11))   # decisão (ii): 5 referências nas duas janelas
                    acc_nova = np.where(tem, ACC_NOVA[m] + OFFSET_COND[cond], np.nan)
                    g = np.where(tem, lvl + comum[(tipo, s)] + rng.normal(0, 0.001, T), np.nan)
                    d = pd.DataFrame(dict(
                        tarefa=t, tipo=tipo, cond=cond, nivel=m, semente=s,
                        maj_frac=0.2, acc_cont=acc_nova - g, norma=10 + 0.1 * t + rng.normal(0, .1, T),
                        distancia=3 + rng.normal(0, .1, T),
                        dorm_tau0_l1=0.1, dorm_tau0_l2=0.1, dorm_tau0025_l1=0.2, dorm_tau0025_l2=0.2,
                        rank_ef=50.0, cos_medio=0.3, acc_nova=acc_nova, gap=g,
                        sonda_perda_copia=1.0, sonda_perda_nova=1.0,
                        sonda_razao=np.where(t % 5 == 4, 1 + 0.02 * t + rng.normal(0, .02, T), np.nan)))
                    if diag and s in analise.SEMENTES_AVALIACAO:
                        d = colunas_diag(d, tem, rng)
                    base = os.path.join(pasta, f"{cond}_m{m}_{tipo}_{s}")
                    if tipo == analise.TIPO_BD:
                        ne = esq.get(cond, 0.10) + (0.0 if m == ref else FORA)
                        e = np.where(t == T - 1, ne, ne + 0.3) + comum_e[s] + rng.normal(0, 0.001, T)
                        d["acc_media_vistas"] = 0.8
                        d["esquecimento"] = np.where(t == 0, np.nan, e)
                        d["bwt"] = np.where(t == 0, np.nan, -e)
                        mat = pd.DataFrame(np.tril(np.full((T, T), 0.8)), columns=[f"j{j}" for j in range(T)])
                        mat.insert(0, "i", range(T))
                        mat.to_csv(base + "_matriz.csv", index=False)
                        with open(base + ".params.json", "w", encoding="utf-8") as f:
                            f.write("{}")
                    d.to_csv(base + ".csv", index=False)
                    d.head(2).to_csv(base + "_rodadas.csv", index=False)  # deve ser ignorado


def plantar_1000(pasta):
    """Semente 1000 com números absurdos na RAIZ (deve ser descartada) e em calibracao/ (nem é lida)."""
    d = pd.read_csv(os.path.join(pasta, "fedavg_iid_m3_A_1.csv"))
    d["semente"], d["gap"] = 1000, 0.9
    d.to_csv(os.path.join(pasta, "fedavg_iid_m3_A_1000.csv"), index=False)
    os.makedirs(os.path.join(pasta, "calibracao"), exist_ok=True)
    d.to_csv(os.path.join(pasta, "calibracao", "fedavg_iid_m3_A_1000.csv"), index=False)


def plantar_sementes(pasta, sementes):
    """Sementes fora de 1..5 (0 exploratória, 6 etc.) na RAIZ, com números absurdos: devem ser descartadas."""
    for s in sementes:
        for cond in ("fedavg_iid", "central_aprendizado"):
            d = pd.read_csv(os.path.join(pasta, f"{cond}_m3_A_1.csv"))
            d["semente"], d["gap"] = s, 0.9
            d.to_csv(os.path.join(pasta, f"{cond}_m3_A_{s}.csv"), index=False)


SCRIPTS = ("fig_gap", "fig_norma", "fig_sonda", "fig_tradeoff", "fig_dose", "fig_k1_mecanismo", "tabela_decisao")


def rodar(pasta, saida):
    for s in SCRIPTS:
        subprocess.run([sys.executable, f"plots/{s}.py", "--results", pasta, "--saida", saida],
                       cwd=RAIZ, check=True)
    for f in SAIDAS:
        assert os.path.getsize(os.path.join(saida, f)) > 0, f


def cenario(base, nome, m_star, sementes, tipos, gap, esq, rng, figuras=False, **kw):
    p = os.path.join(base, nome)
    gerar(p, m_star, sementes, tipos, gap, esq, rng)
    gerar_calib(p, m_star, **kw)
    if figuras:
        rodar(p, os.path.join(base, "saida_" + nome))
    df, regime = analise.carregar(p), analise.carregar_regime(p)
    linha, expl = analise.criterios_atendidos(df, regime)
    print(f"[{nome}] {linha}\n    {expl}")
    return p, df, regime, linha, expl


def main():
    base = tempfile.mkdtemp(prefix="plast_fake_")
    print("diretório temporário:", base)
    rng = np.random.default_rng(1)
    TP = ["A", "B", analise.TIPO_BD]
    BD = analise.TIPO_BD
    try:
        # ---- 1: m*=3, H_orc FALSO (fedavg bem acima de central_aprendizado), H_real verdadeira
        gap1 = {("fedavg_iid", t): .20 for t in TP}
        gap1.update({("sem_media_k1", BD): .12, ("central_SP", BD): .30,
                     ("central_L2", "A"): .06, ("central_L2", "B"): .06, ("central_L2", BD): .06,
                     ("central_SP", "A"): .07, ("central_SP", "B"): .07})
        esq1 = {"fedavg_iid": .08, "sem_media_k1": .12, "central_SP": .30, "central_L2": .20}
        p1, df, regime, linha, expl = cenario(base, "orc_falso", 3, SEMENTES, TP, gap1, esq1, rng, figuras=True)
        assert regime["m_star"] == 3 and regime["gaps"][3] == -0.12
        assert "i" not in df.columns and "j0" not in df.columns and set(df["nivel"]) == {1, 3, 10}
        tam = df.groupby(["cond", "tipo", "nivel", "semente"]).size()
        assert len(tam) == (5 * 9 + 2 * 3) * 5          # 5 conds × 3 tipos × 3 níveis + L2/SP × 3 tipos (só m*)
        for (cond, *_), n in tam.items():
            assert n == T_COND[cond], (cond, n)
        assert set(df[df.cond.isin(["central_L2", "central_SP"])]["nivel"]) == {3}   # níveis sem L2/SP ok
        assert df[df["tipo"] != BD]["esquecimento"].isna().all()
        # janela com 5 referências pareadas, em todas as condições, tarefas, níveis e sementes
        for (c, tp, m), g in df.groupby(["cond", "tipo", "nivel"]):
            assert (analise.n_referencias(df, c, tp, m) == 5).all(), (c, tp, m)
            # janela 7..11: 5 referências pareadas em TODAS as condições (decisão (ii) da Errata)
            assert (analise.n_referencias(df, c, tp, m, analise.JANELA_PART) == 5).all(), (c, tp, m)
        # semente 1000 plantada na raiz e em calibracao/: nenhum número muda
        # eixo x de fig_dose = acc_nova da PRÓPRIA condição (acc_nova difere entre condições no gerador)
        import matplotlib.pyplot as plt
        import fig_dose
        fg, ax = plt.subplots()
        fig_dose._painel(ax, df, "A", 3)
        for cond in analise.CONDICOES_DOSE:
            est = analise.ESTILOS[cond]
            ls = [l for l in ax.lines if l.get_marker() == est["marker"] and l.get_markersize() == 8]
            assert len(ls) == 1, cond
            assert np.allclose(ls[0].get_xdata(), [ACC_NOVA[m] + OFFSET_COND[cond] for m in analise.NIVEIS],
                               atol=1e-9), (cond, ls[0].get_xdata())
        plt.close(fg)
        assert len({round(OFFSET_COND[c], 6) for c in analise.CONDICOES_DOSE}) == 5
        # sementes fora de 1..5 (1000, 0 e 6) plantadas na raiz: nenhum número muda e o aviso sai
        r0 = analise.resumo(df, "fedavg_iid", "A", 3)
        plantar_1000(p1)
        plantar_sementes(p1, (0, 6))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df1 = analise.carregar(p1)
        assert any("0, 6, 1000" in str(x.message) for x in w), [str(x.message) for x in w]
        assert df1.equals(df) and set(df1["semente"]) == set(analise.SEMENTES_AVALIACAO)
        saida_p = os.path.join(base, "saida_orc_falso_plantadas")
        pr = subprocess.run([sys.executable, "plots/tabela_decisao.py", "--results", p1, "--saida", saida_p],
                            cwd=RAIZ, check=True, capture_output=True, text=True)
        assert "descartadas" in pr.stderr and "0, 6, 1000" in pr.stderr, pr.stderr
        assert open(os.path.join(saida_p, "tabela_decisao.md"), encoding="utf-8").read() == open(
            os.path.join(base, "saida_orc_falso", "tabela_decisao.md"), encoding="utf-8").read()
        for s_ in (0, 6):
            for c_ in ("fedavg_iid", "central_aprendizado"):
                os.remove(os.path.join(p1, f"{c_}_m3_A_{s_}.csv"))
        assert analise.resumo(df1, "fedavg_iid", "A", 3) == r0
        assert analise.criterios_atendidos(df1, regime)[0] == linha
        os.remove(os.path.join(p1, "fedavg_iid_m3_A_1000.csv"))
        # nível: gaps fora de m* são deslocados em +0,3 e não podem vazar para a decisão
        assert abs(analise.resumo(df, "fedavg_iid", "A", 3)["media"] - .20) < .04
        assert abs(analise.resumo(df, "fedavg_iid", "A", 1)["media"] - .50) < .04
        assert abs(analise.resumo(df, "fedavg_iid", "A", 10)["media"] - .50) < .04
        for c in T_COND:
            r = analise.resumo_esq(df, c, 3)
            assert r["n"] == 5 and abs(r["media"] - esq1.get(c, .10)) < 0.04, (c, r)
        assert abs(analise.limite_esquecimento(df, 3) - 0.15) < 0.04
        assert analise.rotulo_ganho(df, "fedavg_iid", regime) == "ganho válido"
        assert analise.rotulo_ganho(df, "sem_media_k1", regime) == "ganho válido"
        assert analise.rotulo_ganho(df, "central_SP", regime) == "ganho com violação de esquecimento"
        assert analise.rotulo_ganho(df, "central_L2", regime) == "sem ganho"
        assert analise.rotulo_ganho(df, "central_aprendizado", regime) == "sem ganho"
        assert expl[3].startswith("H_orc: não atendida") and expl[2] == "H_real: atendida"
        assert linha.startswith("Critérios atendidos: H_real (m=3)") and "nível m*=3" in linha, linha
        assert "H_orc" not in linha.split(".")[0]
        assert "H_part: não atendida" in expl[1]
        md = open(os.path.join(base, "saida_orc_falso", "tabela_decisao.md"), encoding="utf-8").read()
        for trecho in ("m* = 3", "regime.json", "ganho com violação de esquecimento", "esquecimento não é medido",
                       "(d) Dose-resposta", "(e) Calibração", "(f) Seleção", "(g) Sanidade", "(h) Linha final",
                       "esquecimento médio final", "Tarefa B_disjunta", "m=1 | m=3 | m=10"):
            assert trecho in md, trecho
        assert "já é um resultado" not in md          # alvo passa de 0,90 em m=10
        assert "| A | 3 |" in md and "sim" in md.split("(g) Sanidade")[1]
        # n_ref visível: 5 na janela principal e na 7..11, em todas as condições; sem nota de assimetria
        secb = md.split("## (b)")[1].split("## (c)")[0]
        b_princ, b_jan = secb.split("## (b')")
        assert "| fedavg_iid | " in b_princ and "n_ref=5" in b_princ and "n_ref=2" not in b_princ
        lin = {l.split("|")[1].strip(): l for l in b_jan.splitlines() if l.startswith("| ")}
        for c_ in ("fedavg_iid", "central_aprendizado", "media_sem_particao"):
            assert "n_ref=5" in lin[c_] and "n_ref=2" not in lin[c_], c_
        assert "Nota: n_ref" not in md
        # a nota de assimetria é condicional: aparece se uma condição perder referências na janela 7..11
        import tabela_decisao
        assert tabela_decisao._nota_assimetria_nref(df, 3) == []
        dfa = df.copy()
        dfa.loc[(dfa.cond == "fedavg_iid") & dfa.tarefa.isin([7, 9, 10]), ["gap", "acc_nova"]] = np.nan
        nota = " ".join(tabela_decisao._nota_assimetria_nref(dfa, 3))
        assert "Nota: n_ref" in nota and "fedavg_iid/A: n_ref=2" in nota, nota
        assert "n_ref=5" in md.split("## (c)")[1].split("## (d)")[0]
        # seleção de λ no formato real (dicts): tabela por λ e flag de violação
        sec_f = md.split("## (f)")[1].split("## (g)")[0]
        assert "**L2, tarefa B_disjunta, m=3**" in sec_f and "violação = False" in sec_f
        assert "| 0.0001 | 0.9000 | 0.2000 | NÃO |" in sec_f.replace("| 0.0001 |", "| 0.0001 |")
        assert "(escolhido)" in sec_f and "| 0.5 | 0.7000 | 0.0400 | sim |" in sec_f
        assert "n/d (limite só medido em B_disjunta)" in sec_f      # tarefas A e B
        assert "calibração fora de ±0,03" not in md                 # calib ok=true: sem aviso
        assert "Traceback" not in md

        # ---- 1b: H_orc VERDADEIRO (central_aprendizado no mesmo nível que fedavg), H_part: sem vantagem
        gap1b = dict(gap1)
        gap1b.update({("central_aprendizado", t): .20 for t in TP})
        _, df, regime, linha, expl = cenario(base, "orc_ok", 3, SEMENTES, TP, gap1b, esq1, rng,
                                             figuras=True, a_m10_acima=False)
        assert expl[3] == "H_orc: atendida", expl[3]
        assert "H_orc (m=3)" in linha and "H_real (m=3)" in linha, linha
        assert "Sem vantagem a explicar: H_part" in linha and "Dados insuficientes" not in linha, linha
        md = open(os.path.join(base, "saida_orc_ok", "tabela_decisao.md"), encoding="utf-8").read()
        assert "não passa de 0,90 nem em m=10: já é um resultado" in md
        # sanidade K1 vs central_passos: ambos .05 em m*, ICs sobrepostos
        a, b = analise.resumo(df, "sem_media_k1", "A", 3), analise.resumo(df, "central_passos", "A", 3)
        assert analise._sobrepoe(a, b)

        # ---- 1c: H_orc FALSO só na tarefa A (central_aprendizado A bem acima de fedavg_iid A, B igual)
        gap1c = dict(gap1)
        gap1c.update({("central_aprendizado", "A"): .40, ("central_aprendizado", "B"): .20})
        _, df, regime, linha, expl = cenario(base, "orc_falso_so_em_A", 3, SEMENTES, TP, gap1c, esq1, rng)
        assert expl[3].startswith("H_orc: não atendida") and "A=False, B=True" in expl[3], expl[3]

        # ---- 1d: calibração com ok=false em A -> aviso na linha final da tabela
        _, df, regime, linha, expl = cenario(base, "calib_ruim", 3, SEMENTES, TP, gap1, esq1, rng,
                                             figuras=True, calib_ok=False)
        md = open(os.path.join(base, "saida_calib_ruim", "tabela_decisao.md"), encoding="utf-8").read()
        final = md.split("## (h) Linha final")[1]
        assert ("calibração fora de ±0,03 no nível m*: premissa de H_orc (mesmo nível de aprendizado) "
                "enfraquecida") in final, final
        assert "tarefa A, B" in final and "ok" in md

        # ---- 1e: H_part reprova por (i): media_sem_particao longe de central_aprendizado, (ii) vale
        gap1e = dict(gap1)
        gap1e.update({("fedavg_iid", "B"): .20, ("central_aprendizado", "B"): lambda t: .05 if t >= 20 else .20,
                      ("media_sem_particao", "B"): .05})
        _, df, regime, linha, expl = cenario(base, "part_i", 3, SEMENTES, TP, gap1e, esq1, rng)
        assert expl[1] == "H_part: não atendida ((i)=False, (ii)=True)", expl[1]
        assert "H_part (m=3)" not in linha, linha

        # ---- 2: m* nulo -> tudo não avaliável; figuras antigas no maior nível com aviso
        gap2 = {("fedavg_iid", t): .20 for t in TP}
        p2, df, regime, linha, expl = cenario(base, "m_nulo", None, SEMENTES, TP, gap2, {}, rng, figuras=True)
        txt = "não avaliável (regime sem perda; H_real, H_reg, H_part e H_orc não testáveis neste toy)"
        assert regime["m_star"] is None and regime["presente"]
        assert linha == "Critérios atendidos: " + txt, linha
        assert all(e.endswith(txt) for e in expl) and len(expl) == 4
        assert analise.rotulo_ganho(df, "fedavg_iid", regime) == txt
        assert analise.nivel_exibicao(df, regime)[0] == 10 and "AVISO" in analise.nivel_exibicao(df, regime)[1]
        md = open(os.path.join(base, "saida_m_nulo", "tabela_decisao.md"), encoding="utf-8").read()
        assert "m* = null" in md and "maior nível disponível (m=10)" in md and txt in md
        assert set(df.cond) == {"fedavg_iid", "sem_media_k1", "media_sem_particao", "central_passos",
                                "central_aprendizado"}     # sem L2/SP em lugar nenhum

        # ---- 2b: sem regime.json -> mesmo tratamento, motivo "regime.json ausente"
        os.remove(os.path.join(p2, "calibracao", "regime.json"))
        regime = analise.carregar_regime(p2)
        linha, expl = analise.criterios_atendidos(df, regime)
        assert linha == "Critérios atendidos: não avaliável (regime.json ausente)", linha

        # ---- 3: quick (1 semente, A e B_disjunta) -> dados insuficientes, nunca "sem vantagem"
        _, df, regime, linha, expl = cenario(base, "quick", 3, [1], ["A", BD], gap1, esq1, rng, figuras=True)
        assert "Dados insuficientes" in linha and "Sem vantagem" not in linha, linha
        assert analise.rotulo_ganho(df, "fedavg_iid", regime) == "não avaliável (IC indefinido)"
        assert expl[3].startswith("H_orc: não avaliável")

        # ---- 4: gap muda entre janelas 7..11 e 25..29 em m*; H_part vale; H_orc falso (central .05)
        gap4 = {("fedavg_iid", "A"): .20, ("fedavg_iid", BD): .20,
                ("fedavg_iid", "B"): lambda t: .05 if t < 20 else .20,
                ("central_L2", "A"): .06, ("central_L2", "B"): .06, ("central_SP", "A"): .07,
                ("central_SP", "B"): .07}
        _, df, regime, linha, expl = cenario(base, "janelas", 3, SEMENTES, TP, gap4, {}, rng)
        m_pad = analise.metrica_por_semente(df, "fedavg_iid", "B", 3)
        m_7_11 = analise.metrica_por_semente(df, "fedavg_iid", "B", 3, analise.JANELA_PART)
        assert len(m_pad) == 5 and abs(m_pad.mean() - .20) < .04, m_pad.mean()
        assert abs(m_7_11.mean() - .05) < .04, m_7_11.mean()
        assert analise.metrica_por_semente(df, "media_sem_particao", "A", 3).equals(
            analise.metrica_por_semente(df, "media_sem_particao", "A", 3, analise.JANELA_PART))
        d = df[(df.cond == "fedavg_iid") & (df.tipo == "B") & (df.semente == 1) & (df.nivel == 3)
               & (df.tarefa.between(7, 11))]
        assert abs(m_7_11[1] - d["gap"].dropna().mean()) < 1e-12 and d["gap"].notna().sum() == 5
        assert linha.startswith("Critérios atendidos: H_part (m=3), H_real (m=3)"), linha

        # ---- 5: fedavg viola o limite de esquecimento -> H_real NÃO conta
        esq5 = {"fedavg_iid": .30, "central_aprendizado": .10}
        _, df, regime, linha, expl = cenario(base, "violacao", 3, SEMENTES, TP, gap1, esq5, rng)
        assert analise.rotulo_ganho(df, "fedavg_iid", regime) == "ganho com violação de esquecimento"
        assert expl[2] == "H_real: ganho de plasticidade com esquecimento acima do limite", expl[2]
        assert "H_real: ganho de plasticidade com esquecimento acima do limite" in linha
        assert "H_real (m=3)" not in linha

        # ---- 6: H_reg vale (L2 dentro do IC do fedavg em A e B); H_real não
        gap6 = {("fedavg_iid", t): .20 for t in TP}
        gap6.update({("central_L2", "A"): .20, ("central_L2", "B"): .20, ("central_SP", "A"): .07,
                     ("central_SP", "B"): .07})
        _, df, regime, linha, expl = cenario(base, "reg", 3, SEMENTES, TP, gap6, {}, rng)
        assert expl[0] == "H_reg: atendida por central_L2", expl[0]
        assert linha.startswith("Critérios atendidos: H_reg (m=3)"), linha
        assert expl[2] == "H_real: não atendida"

        # ---- 7: sem B_disjunta -> H_real não avaliável (falta B_disjunta)
        _, df, regime, linha, expl = cenario(base, "semBD", 3, SEMENTES, ["A", "B"], gap1, esq1, rng)
        assert expl[2] == "H_real: não avaliável (falta B_disjunta)", expl[2]
        assert "Dados insuficientes: H_real (falta B_disjunta)" in linha, linha

        # ---- 8: m*=1 (nível mínimo) e L2/SP só nele; decisões usam só esse nível
        _, df, regime, linha, expl = cenario(base, "m1", 1, SEMENTES, TP, gap1, esq1, rng)
        assert set(df[df.cond == "central_L2"]["nivel"]) == {1} and "m*=1" in linha
        assert linha.startswith("Critérios atendidos: H_real (m=1)"), linha

        # ---- 9: results/ vazio, sem colunas ou só com sementes descartadas: código 1, mensagem, nada criado
        vazio = os.path.join(base, "r_vazio")
        os.makedirs(vazio)
        semcol = os.path.join(base, "r_semcol")
        os.makedirs(semcol)
        pd.DataFrame({"x": [1, 2]}).to_csv(os.path.join(semcol, "fedavg_iid_m3_A_1.csv"), index=False)
        so0 = os.path.join(base, "r_so_semente0")
        gerar(so0, 3, [0], ["A"], {}, {}, rng)
        for nome in ("r_vazio", "r_semcol", "r_so_semente0"):
            for sc in SCRIPTS:
                saida = os.path.join(base, "saida_" + nome)
                pr = subprocess.run([sys.executable, f"plots/{sc}.py", "--results", os.path.join(base, nome),
                                     "--saida", saida], cwd=RAIZ, capture_output=True, text=True)
                assert pr.returncode == 1, (nome, sc, pr.returncode, pr.stderr)
                assert "sem resultados em" in pr.stderr and "Traceback" not in pr.stderr, (nome, sc, pr.stderr)
                assert not os.path.exists(saida), (nome, sc)     # nenhuma figura nem pasta vazia
        assert analise.bootstrap_ic([1, 2, 3, 4]) == analise.bootstrap_ic([1, 2, 3, 4])

        # ---- 10: fig_k1_mecanismo com diagnósticos: m*=3 gera as duas figuras; m*=10 e m* nulo só a de m=10
        import matplotlib.pyplot as plt
        import fig_k1_mecanismo as fk
        s_orc = os.path.join(base, "saida_orc_falso")
        assert os.path.getsize(os.path.join(s_orc, "fig_k1_mecanismo_mstar.png")) > 0
        d10 = analise.carregar(os.path.join(base, "orc_falso"))
        assert "ini_hess_max" in d10.columns and "nova_fim_hess_max" in d10.columns
        assert fk.nao_convergiu(d10, "fedavg_iid", "A", 10, "ini_hess_max")   # hess_conv=0 detectado
        cn = fk.curva_nova(d10, "nova_ini_grad_srank", "A", 10)
        assert set(cn["tarefa"]) == {t for t in range(30) if t % 3 == 2 or t >= 25 or 7 <= t <= 11}
        assert d10[d10["tarefa"] == 0]["nova_ini_hess_max"].isna().all()
        conferir_esquema()
        assert (d10["fim_hess_max"] < 0).any() and (d10["ini_hess_max"] < 0).any()
        for nome_k in ("fig_k1_por_semente.png", "fig_k1_por_semente_mstar.png"):
            assert os.path.getsize(os.path.join(s_orc, nome_k)) > 0, nome_k
        for nome, ms, esperado in (("k1_m10", 10, False), ("k1_nulo", None, False)):
            pk = os.path.join(base, nome)
            gerar(pk, ms, SEMENTES, TP, {}, {}, rng)
            gerar_calib(pk, ms)
            sk = os.path.join(base, "saida_" + nome)
            pr = subprocess.run([sys.executable, "plots/fig_k1_mecanismo.py", "--results", pk, "--saida", sk],
                                cwd=RAIZ, capture_output=True, text=True)
            assert pr.returncode == 0, pr.stderr
            assert os.path.getsize(os.path.join(sk, "fig_k1_mecanismo.png")) > 0
            assert not os.path.exists(os.path.join(sk, "fig_k1_mecanismo_mstar.png")), nome
            assert os.path.getsize(os.path.join(sk, "fig_k1_por_semente.png")) > 0
            assert not os.path.exists(os.path.join(sk, "fig_k1_por_semente_mstar.png")), nome
        # execução antiga: sem as colunas novas, não quebra e as linhas dizem "sem diagnósticos"
        pk = os.path.join(base, "k1_antigo")
        gerar(pk, 3, SEMENTES, TP, {}, {}, rng, diag=False)
        gerar_calib(pk, 3)
        d_ant = analise.carregar(pk)
        assert "ini_hess_max" not in d_ant.columns
        sk = os.path.join(base, "saida_k1_antigo")
        pr = subprocess.run([sys.executable, "plots/fig_k1_mecanismo.py", "--results", pk, "--saida", sk],
                            cwd=RAIZ, capture_output=True, text=True)
        assert pr.returncode == 0 and "Traceback" not in pr.stderr, pr.stderr
        assert os.path.getsize(os.path.join(sk, "fig_k1_mecanismo.png")) > 0
        assert os.path.getsize(os.path.join(sk, "fig_k1_mecanismo_mstar.png")) > 0
        assert os.path.getsize(os.path.join(sk, "fig_k1_por_semente.png")) > 0
        assert os.path.getsize(os.path.join(sk, "fig_k1_por_semente_mstar.png")) > 0
        textos = []
        orig = fk.painel
        def espiao(ax, *a, **k):
            orig(ax, *a, **k)
            textos.extend(t.get_text() for t in ax.texts)
        fk.painel = espiao
        fk.figura(d_ant, 10, "", sk, "x.png")
        fk.painel = orig
        assert textos.count(fk.SEM_DIAG) == 3 * (len(fk.DIAG) - 1) and "sem dados" not in textos, textos
        # por semente, execução antiga: hess e norma dizem "sem diagnósticos" (2 linhas x 4 condições x 3 tarefas)
        textos.clear()
        origs = fk.painel_semente
        def espiao_s(ax, *a, **k):
            r = origs(ax, *a, **k)
            textos.extend(t.get_text() for t in ax.texts)
            return r
        fk.painel_semente = espiao_s
        fk.figura_semente(d_ant, 10, "", sk, "x.png")
        fk.painel_semente = origs
        assert textos.count(fk.SEM_DIAG) == 2 * 4 * 3 and "sem dados" not in textos, textos
        # com diagnósticos nenhum painel diz "sem diagnósticos"; o marcador vazado é DESENHADO (não só detectado)
        textos.clear()
        chamadas, retornos, legendas = [], [], []
        orig_vaz, orig_leg = fk._vazado, fk.legenda_fig
        def espiao_vaz(ax, pts, marker, cor, tam=8):
            chamadas.append((len(pts), cor, tam))
            n0 = len(ax.lines)
            orig_vaz(ax, pts, marker, cor, tam)
            ln = ax.lines[-1]
            assert len(ax.lines) == n0 + 1 and ln.get_markerfacecolor() == "white" and ln.get_markeredgecolor() == cor
            assert len(ln.get_xdata()) == len(pts) > 0
        def espiao_painel(ax, *a, **k):
            retornos.append(orig(ax, *a, **k))
            textos.extend(t.get_text() for t in ax.texts)
            return retornos[-1]
        def espiao_leg(fig, *a, **k):
            legendas.append(a)
            return orig_leg(fig, *a, **k)
        fk._vazado, fk.painel, fk.legenda_fig = espiao_vaz, espiao_painel, espiao_leg
        fk.figura(d10, 10, "", sk, "x.png")
        fk._vazado, fk.painel, fk.legenda_fig = orig_vaz, orig, orig_leg
        assert fk.SEM_DIAG not in textos, textos
        assert chamadas and any(c[2] == 8 for c in chamadas) and any(c[2] == 6 for c in chamadas), chamadas   # contínuo e nova
        assert any(retornos) and legendas[0][2] is True      # entrada de legenda só porque há ponto não convergido
        # cinza (início): marcador vazado cinza também desenhado
        assert any(c[1] == fk.CINZA for c in chamadas)
        # sem nenhum hess_conv == 0 não há marcador vazado nem entrada na legenda
        d_ok = d10.copy()
        for c in [c for c in d_ok.columns if c.endswith("hess_conv")]:
            d_ok[c] = 1.0
        chamadas.clear(); retornos.clear(); legendas.clear()
        fk._vazado, fk.painel, fk.legenda_fig = espiao_vaz, espiao_painel, espiao_leg
        fk.figura(d_ok, 10, "", sk, "x.png")
        fk._vazado, fk.painel, fk.legenda_fig = orig_vaz, orig, orig_leg
        assert not chamadas and not any(retornos) and legendas[0][2] is False, (chamadas, legendas)
        # por semente: marcador vazado desenhado, 1 linha fina por semente (5) num painel de acc_cont
        fk.figura_semente(d10, 10, "", sk, "x.png")
        fig_t, ax_t = plt.subplots()
        assert fk.painel_semente(ax_t, d10, "fedavg_iid", "A", 10, "fim_hess_max", True, True) is True
        assert len([l for l in ax_t.lines if l.get_linewidth() == 0.7]) == 5
        assert any(l.get_markerfacecolor() == "white" for l in ax_t.lines)
        assert min(min(l.get_ydata()) for l in ax_t.lines if l.get_linewidth() == 0.7) < 0   # hess negativo plotado
        fig_t, ax_t = plt.subplots()
        assert fk.painel_semente(ax_t, d10, "fedavg_iid", "A", 10, "acc_cont", False, False) is False
        assert len(ax_t.lines) == 5
        # linhas de fim: rede nova POR CONDIÇÃO (4 tracejadas finas); início: UMA cinza
        fig_t, ax_t = plt.subplots()
        fk.painel(ax_t, d10, "A", 10, "fim_norma_l2_w", "nova_fim_norma_l2_w", False, True)
        assert len([l for l in ax_t.lines if l.get_linewidth() == 0.9]) == 4
        assert not [l for l in ax_t.lines if l.get_color() == fk.CINZA]
        fig_t, ax_t = plt.subplots()
        fk.painel(ax_t, d10, "A", 10, "ini_hess_max", "nova_ini_hess_max", True, True)
        assert len([l for l in ax_t.lines if l.get_color() == fk.CINZA and l.get_linewidth() == 1.3]) == 1
        assert not [l for l in ax_t.lines if l.get_linewidth() == 0.9]
        fig_t, ax_t = plt.subplots()
        fk.painel(ax_t, d10, "A", 10, "acc_cont", "acc_nova", False, False)
        assert len([l for l in ax_t.lines if l.get_linewidth() == 0.9]) == 4
        plt.close("all")
        print("OK: 7 arquivos gerados (fig_dose e fig_k1_mecanismo inclusos), fig_k1 com/sem diagnósticos e vereditos conferidos")
    finally:
        if os.environ.get("PLAST_TEST_KEEP"):
            print("mantido:", base)
        else:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
