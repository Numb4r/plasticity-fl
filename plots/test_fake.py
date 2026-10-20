"""Roda tudo contra CSVs sintéticos do esquema v3 (em tempfile.mkdtemp(), nunca em results/) e confere
arquivos gerados e vereditos. Defina PLAST_TEST_KEEP=1 para manter o diretório temporário."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

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
FORA = 0.3                                # armadilha: gap/esquecimento fora do nível m* são deslocados


def _v(v, t):
    return v(t) if callable(v) else v


def _escrever_json(pasta, nome, obj):
    os.makedirs(os.path.join(pasta, "calibracao"), exist_ok=True)
    with open(os.path.join(pasta, "calibracao", nome), "w", encoding="utf-8") as f:
        json.dump(obj, f)


def gerar_calib(pasta, m_star, a_m10_acima=True, regime=True):
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
                "acc_central": ACC_NOVA[m], "erro": 0.01, "ok": True, "alvo_acima_de_0.90": bool(acima and m == 10)
                if t == "A" and m == 10 else ACC_NOVA[m] > 0.9})
    if m_star:
        for met, lam in (("L2", 1e-3), ("SP", 0.6)):
            for t in ("A", "B", "B_disjunta"):
                _escrever_json(pasta, f"selecao_{met}_{t}_m{m_star}.json", {
                    "lambda": lam, "notas": {"x": 0.9}, "esquecimento": 0.1, "esquecimento_ref": 0.1,
                    "violacao": False})


def gerar(pasta, m_star, sementes, tipos, gap, esq, rng, niveis=analise.NIVEIS):
    """gap[(cond, tipo)] = gap no nível m* (número ou função de t), padrão 0,05; fora de m* soma FORA.
    esq[cond] = esquecimento final em B_disjunta em m* (padrão 0,10); em t < T-1 vale nível + 0,3.
    L2 e SP só existem em m*. Com m_star None, o nível 'central' dos gaps é 3 e L2/SP não existem."""
    os.makedirs(pasta, exist_ok=True)
    ref = m_star or 3
    comum = {(tp, s): rng.normal(0, 0.02) for tp in analise.TIPOS for s in list(SEMENTES) + [1000]}
    comum_e = {s: rng.normal(0, 0.02) for s in list(SEMENTES) + [1000]}
    for cond, T in T_COND.items():
        t = np.arange(T)
        for m in niveis:
            if cond in ("central_L2", "central_SP") and (m_star is None or m != m_star):
                continue
            for tipo in tipos:
                for s in sementes:
                    desloc = 0.0 if m == ref else FORA
                    lvl = np.array([_v(gap.get((cond, tipo), 0.05), x) for x in t], dtype=float) + desloc
                    tem = (t % 3 == 2) | (t >= T - 5)          # esquema v3: 5 referências na janela
                    acc_nova = np.where(tem, ACC_NOVA[m], np.nan)
                    g = np.where(tem, lvl + comum[(tipo, s)] + rng.normal(0, 0.001, T), np.nan)
                    d = pd.DataFrame(dict(
                        tarefa=t, tipo=tipo, cond=cond, nivel=m, semente=s,
                        maj_frac=0.2, acc_cont=acc_nova - g, norma=10 + 0.1 * t + rng.normal(0, .1, T),
                        distancia=3 + rng.normal(0, .1, T),
                        dorm_tau0_l1=0.1, dorm_tau0_l2=0.1, dorm_tau0025_l1=0.2, dorm_tau0025_l2=0.2,
                        rank_ef=50.0, cos_medio=0.3, acc_nova=acc_nova, gap=g,
                        sonda_perda_copia=1.0, sonda_perda_nova=1.0,
                        sonda_razao=np.where(t % 5 == 4, 1 + 0.02 * t + rng.normal(0, .02, T), np.nan)))
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


def rodar(pasta, saida):
    for s in ("fig_gap", "fig_norma", "fig_sonda", "fig_tradeoff", "fig_dose", "fig_k1_mecanismo",
              "tabela_decisao"):
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
            # janela 7..11: 5 referências só em media_sem_particao (T=12); nas de T=30 só 8 e 11 (t%3==2)
            esperado = 5 if c == "media_sem_particao" else 2
            assert (analise.n_referencias(df, c, tp, m, analise.JANELA_PART) == esperado).all(), (c, tp, m)
        # semente 1000 plantada na raiz e em calibracao/: nenhum número muda
        r0 = analise.resumo(df, "fedavg_iid", "A", 3)
        plantar_1000(p1)
        df1 = analise.carregar(p1)
        assert df1.equals(df) and 1000 not in set(df1["semente"])
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

        # ---- 1c: H_orc FALSO por limite exato fora do IC (fedavg abaixo de central_aprendizado)
        gap1c = dict(gap1)
        gap1c.update({("central_aprendizado", "A"): .40, ("central_aprendizado", "B"): .20})
        _, df, regime, linha, expl = cenario(base, "orc_A_fora", 3, SEMENTES, TP, gap1c, esq1, rng)
        assert expl[3].startswith("H_orc: não atendida") and "A=False, B=True" in expl[3], expl[3]

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
        assert abs(m_7_11[1] - d["gap"].dropna().mean()) < 1e-12 and d["gap"].notna().sum() == 2
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

        assert analise.bootstrap_ic([1, 2, 3, 4]) == analise.bootstrap_ic([1, 2, 3, 4])
        print("OK: 7 arquivos gerados (fig_dose e fig_k1_mecanismo inclusos) e vereditos conferidos")
    finally:
        if os.environ.get("PLAST_TEST_KEEP"):
            print("mantido:", base)
        else:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
