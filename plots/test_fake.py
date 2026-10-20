"""Roda tudo contra CSVs sintéticos (em tempfile.mkdtemp(), nunca em results/) e confere
arquivos gerados e vereditos das regras. Defina PLAST_TEST_KEEP=1 para manter o diretório temporário."""
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
SAIDAS = ("fig_gap.png", "fig_norma.png", "fig_sonda.png", "fig_tradeoff.png", "tabela_decisao.md")


def _nivel(v, t):
    return v(t) if callable(v) else v


def gerar(pasta, sementes, tipos, gap, esq, rng):
    """gap[(cond, tipo)] = nível do gap (número ou função de t); padrão 0,05.
    esq[cond] = esquecimento final em B_disjunta; padrão 0,10. Em t < T-1 o esquecimento é
    nível + 0,3 (armadilha: só o valor da última tarefa deve contar)."""
    os.makedirs(pasta, exist_ok=True)
    # deslocamento comum por (tipo, semente): pareia as condições e dá IC com largura realista
    comum = {(tp, s): rng.normal(0, 0.02) for tp in analise.TIPOS for s in range(10)}
    comum_e = {s: rng.normal(0, 0.02) for s in range(10)}
    for cond, T in T_COND.items():
        t = np.arange(T)
        for tipo in tipos:
            for s in sementes:
                lvl = np.array([_nivel(gap.get((cond, tipo), 0.05), x) for x in t], dtype=float)
                tem = t % 3 == 2
                acc_nova = np.where(tem, 0.9, np.nan)
                g = np.where(tem, lvl + comum[(tipo, s)] + rng.normal(0, 0.001, T), np.nan)
                d = pd.DataFrame(dict(
                    tarefa=t, tipo=tipo, cond=cond, semente=s,
                    acc_cont=acc_nova - g, norma=10 + 0.1 * t + rng.normal(0, .1, T),
                    distancia=3 + rng.normal(0, .1, T),
                    dorm_tau0_l1=0.1, dorm_tau0_l2=0.1, dorm_tau0025_l1=0.2, dorm_tau0025_l2=0.2,
                    rank_ef=50.0, cos_medio=0.3, acc_nova=acc_nova, gap=g,
                    sonda_perda_copia=1.0, sonda_perda_nova=1.0,
                    sonda_razao=np.where(t % 5 == 4, 1 + 0.02 * t + rng.normal(0, .02, T), np.nan)))
                base = os.path.join(pasta, f"{cond}_{tipo}_{s}")
                if tipo == analise.TIPO_BD:
                    nivel_e = esq.get(cond, 0.10)
                    e = np.where(t == T - 1, nivel_e, nivel_e + 0.3) + comum_e[s] + rng.normal(0, 0.001, T)
                    d["acc_media_vistas"] = 0.8
                    d["esquecimento"] = np.where(t == 0, np.nan, e)
                    d["bwt"] = np.where(t == 0, np.nan, -e)
                    # arquivos que devem ser ignorados: matriz TxT, params.json
                    m = pd.DataFrame(np.tril(np.full((T, T), 0.8)), columns=[f"j{j}" for j in range(T)])
                    m = m.where(np.tril(np.ones((T, T))) > 0)
                    m.insert(0, "i", range(T))
                    m.to_csv(base + "_matriz.csv", index=False)
                    with open(base + ".params.json", "w", encoding="utf-8") as f:
                        f.write("{}")
                d.to_csv(base + ".csv", index=False)
                d.head(2).to_csv(base + "_rodadas.csv", index=False)  # deve ser ignorado
    os.makedirs(os.path.join(pasta, "selecao"), exist_ok=True)
    pd.DataFrame(dict(x=[1])).to_csv(os.path.join(pasta, "selecao", "lixo.csv"), index=False)


def rodar(pasta, saida):
    for s in ("fig_gap", "fig_norma", "fig_sonda", "fig_tradeoff", "tabela_decisao"):
        subprocess.run([sys.executable, f"plots/{s}.py", "--results", pasta, "--saida", saida],
                       cwd=RAIZ, check=True)
    for f in SAIDAS:
        assert os.path.getsize(os.path.join(saida, f)) > 0, f


def cenario(base, nome, sementes, tipos, gap, esq, rng):
    p = os.path.join(base, nome)
    gerar(p, sementes, tipos, gap, esq, rng)
    rodar(p, os.path.join(base, "saida_" + nome))
    df = analise.carregar(p)
    linha, expl = analise.criterios_atendidos(df)
    print(f"[{nome}] {linha}\n    {expl}")
    return p, df, linha, expl


def main():
    base = tempfile.mkdtemp(prefix="plast_fake_")
    print("diretório temporário:", base)
    rng = np.random.default_rng(1)
    TP = ["A", "B", analise.TIPO_BD]
    try:
        # ---- cenário 1: fedavg bem acima; msp ~ central; B_disjunta com ganho válido e com violação
        gap1 = {("fedavg_iid", t): .20 for t in TP}
        gap1.update({("sem_media_k1", analise.TIPO_BD): .12, ("central_SP", analise.TIPO_BD): .30,
                     ("central_L2", "A"): .06, ("central_L2", "B"): .06, ("central_L2", analise.TIPO_BD): .06,
                     ("central_SP", "A"): .07, ("central_SP", "B"): .07})
        esq1 = {"fedavg_iid": .08, "sem_media_k1": .12, "central_SP": .30, "central_L2": .20}
        p1, df, linha, expl = cenario(base, "cheio", range(5), TP, gap1, esq1, rng)
        # matriz/rodadas/params/subpasta ignorados: só colunas dos CSVs principais, T linhas exatas
        assert "i" not in df.columns and "j0" not in df.columns
        assert set(df["cond"]) == set(T_COND) and set(df["tipo"]) == set(TP)
        tam = df.groupby(["cond", "tipo", "semente"]).size()
        assert len(tam) == 7 * 3 * 5
        for (cond, _, _), n in tam.items():
            assert n == T_COND[cond], (cond, n)
        # A e B não têm as colunas do esquecimento (NaN); nome do arquivo não importa
        assert df[df["tipo"] != analise.TIPO_BD]["esquecimento"].isna().all()
        os.rename(os.path.join(p1, "fedavg_iid_A_0.csv"), os.path.join(p1, "zz_qualquer.csv"))
        df2 = analise.carregar(p1)
        assert df2.sort_values(["cond", "tipo", "semente", "tarefa"]).reset_index(drop=True).equals(
            df.sort_values(["cond", "tipo", "semente", "tarefa"]).reset_index(drop=True))
        os.rename(os.path.join(p1, "zz_qualquer.csv"), os.path.join(p1, "fedavg_iid_A_0.csv"))
        # esquecimento = valor de T-1 (11 em media_sem_particao), não o das tarefas anteriores
        for c, T in T_COND.items():
            r = analise.resumo_esq(df, c)
            assert r["n"] == 5 and abs(r["media"] - esq1.get(c, .10)) < 0.04, (c, r)
        assert abs(analise.limite_esquecimento(df) - 0.15) < 0.04
        assert analise.rotulo_ganho(df, "fedavg_iid") == "ganho válido"
        assert analise.rotulo_ganho(df, "sem_media_k1") == "ganho válido"
        assert analise.rotulo_ganho(df, "central_SP") == "ganho com violação de esquecimento"
        assert analise.rotulo_ganho(df, "central_L2") == "sem ganho"
        assert analise.rotulo_ganho(df, "central_aprendizado") == "sem ganho"
        assert linha == "Critérios atendidos: H_real", linha
        assert "H_part: não atendida" in expl[1]
        md = open(os.path.join(base, "saida_cheio", "tabela_decisao.md"), encoding="utf-8").read()
        assert "ganho com violação de esquecimento" in md and "esquecimento não é medido" in md

        # ---- cenário 2: tudo igual -> H_reg vale; H_part sem vantagem a explicar (não é "insuficiente")
        _, df, linha, expl = cenario(base, "iguais", range(5), TP, {}, {}, rng)
        assert "H_part: não avaliável (sem vantagem a explicar)" in expl[1]
        assert "Sem vantagem a explicar: H_part" in linha and "Dados insuficientes" not in linha, linha
        assert linha.startswith("Critérios atendidos: H_reg"), linha

        # ---- cenário 3: quick, 1 semente, A e B_disjunta -> dados insuficientes, nunca "sem vantagem"
        _, df, linha, expl = cenario(base, "quick", [0], ["A", analise.TIPO_BD], gap1, esq1, rng)
        assert "Dados insuficientes" in linha and "Sem vantagem" not in linha, linha
        assert "falta B_disjunta" not in expl[2]  # B_disjunta existe; falta é de A/B e IC
        assert analise.rotulo_ganho(df, "fedavg_iid") == "não avaliável (IC indefinido)"

        # ---- cenário 4: gap MUDA entre janelas 7..11 e 25..29 -> pega erro de janela; H_part vale
        gap4 = {("fedavg_iid", "A"): .20, ("fedavg_iid", analise.TIPO_BD): .20,
                ("fedavg_iid", "B"): lambda t: .05 if t < 20 else .20,
                ("central_L2", "A"): .06, ("central_L2", "B"): .06, ("central_SP", "A"): .07,
                ("central_SP", "B"): .07}
        _, df, linha, expl = cenario(base, "janelas", range(5), TP, gap4, {}, rng)
        m_pad = analise.metrica_por_semente(df, "fedavg_iid", "B")
        m_7_11 = analise.metrica_por_semente(df, "fedavg_iid", "B", analise.JANELA_PART)
        assert len(m_pad) == 5 and abs(m_pad.mean() - .20) < .04, m_pad.mean()
        assert abs(m_7_11.mean() - .05) < .04, m_7_11.mean()
        # media_sem_particao (T=12): janela padrão = 7..11
        assert analise.metrica_por_semente(df, "media_sem_particao", "A").equals(
            analise.metrica_por_semente(df, "media_sem_particao", "A", analise.JANELA_PART))
        # só tarefas 8 e 11 (t%3==2) entram na janela 7..11
        d = df[(df.cond == "fedavg_iid") & (df.tipo == "B") & (df.semente == 0) & (df.tarefa.between(7, 11))]
        assert abs(m_7_11[0] - d["gap"].dropna().mean()) < 1e-12 and d["gap"].notna().sum() == 2
        assert linha == "Critérios atendidos: H_part, H_real", linha

        # ---- cenário 5: B_disjunta com violação de esquecimento em fedavg_iid -> H_real NÃO conta
        esq5 = {"fedavg_iid": .30, "central_aprendizado": .10}
        _, df, linha, expl = cenario(base, "violacao", range(5), TP, gap1, esq5, rng)
        assert analise.rotulo_ganho(df, "fedavg_iid") == "ganho com violação de esquecimento"
        assert expl[2] == "H_real: ganho de plasticidade com esquecimento acima do limite", expl[2]
        assert linha.startswith("Critérios atendidos: nenhum"), linha
        assert "H_real: ganho de plasticidade com esquecimento acima do limite" in linha

        # ---- cenário 6: H_reg vale (L2 dentro do IC do fedavg em A e B); H_real não
        gap6 = {("fedavg_iid", t): .20 for t in TP}
        gap6.update({("central_L2", "A"): .20, ("central_L2", "B"): .20, ("central_SP", "A"): .07,
                     ("central_SP", "B"): .07})
        _, df, linha, expl = cenario(base, "reg", range(5), TP, gap6, {}, rng)
        assert expl[0] == "H_reg: atendida por central_L2", expl[0]
        assert linha.startswith("Critérios atendidos: H_reg"), linha
        assert expl[2] == "H_real: não atendida"

        # ---- cenário 7: sem B_disjunta -> H_real não avaliável (falta B_disjunta)
        _, df, linha, expl = cenario(base, "semBD", range(5), ["A", "B"], gap1, esq1, rng)
        assert expl[2] == "H_real: não avaliável (falta B_disjunta)", expl[2]
        assert "Dados insuficientes: H_real (falta B_disjunta)" in linha, linha

        assert analise.bootstrap_ic([1, 2, 3, 4]) == analise.bootstrap_ic([1, 2, 3, 4])
        print("OK: 5 arquivos gerados e vereditos conferidos nos 7 cenários")
    finally:
        if os.environ.get("PLAST_TEST_KEEP"):
            print("mantido:", base)
        else:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
