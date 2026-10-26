"""Pipeline das rodadas 3 (parte "3": fatorial K x origem dos dados) e 4 (partes 4a, técnicas distribuídas, e 4b, remédios de plasticidade no FedAvg).

Implementado ANTES do pré-registro, por decisão do autor (2026-10-26); o pré-registro vem depois dos resultados
da rodada 3 e antes de qualquer execução desta rodada. Nada aqui foi rodado em dados reais.

  --parte 3|4a|4b  --fase plano         plano e estimativa de custo (não roda nada)
                 --fase teste_curto   semente 1000, T=12, R pequeno: cobre cada técnica/remédio e aplica os PAREs
                 --fase completo      calibração de R e escolha de hiperparâmetros (semente 1000) -> sementes 24 a 31
Resultados em results_rodada4a/ e results_rodada4b/ (a semente 1000 em .../calibracao/). 1 thread por processo.
O `completo` recusa começar sem um `teste_curto` com OK. Jobs com CSV válido são pulados (retomável).
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.calibrar_r4 import arq_alvo, calibrar_R
from src.condicoes import _base_arquivo, csv_valido, executar, pasta_calib
from src.condicoes_r4 import CATALOGO, COMBINACOES, DEPENDE, arq_selecao, condicao_r4, tem_hp
from src.config import NIVEL_R2, R_BASE_R4, SEMENTE_CALIB, SEMENTES_R3, SEMENTES_R4, T_PADRAO

logger = logging.getLogger(__name__)
DIRS = {"3": "results_rodada3", "4a": "results_rodada4a", "4b": "results_rodada4b"}
SEMENTES_PARTE = {"3": SEMENTES_R3, "4a": SEMENTES_R4, "4b": SEMENTES_R4}
LIMITE_APRENDE = 0.2        # teste curto: PARE se o fedavg_iid (R pequeno) fica abaixo disto (o acaso é 0,1)
FALHAS: list[str] = []
# segundos de CPU de uma execução de T=30 com diagnósticos, R = 75. MEDIDOS só fedavg_iid, sem_media_k1 e
# central_aprendizado (rodadas 2 e 2b, nó dl-01); os das técnicas e remédios novos são SUPOSIÇÕES por analogia
CUSTO_S = {"fedavg_iid": 400, "fedprox": 440, "scaffold": 540, "fedrep": 400, "gossip_anel": 410, "gossip_grau4": 410,
           "diloco_h15": 400, "diloco_h50": 1330, "diloco_h150": 4000, "sem_media_k1": 130, "central_aprendizado": 85,
           "fed_l2": 410, "fed_l2init": 430, "fed_redo_local": 470, "fed_sp": 400, "fed_redo_fed": 450, "fed_l2_sp": 420,
           "fed_l2init_sp": 440, "fed_redo_ambos": 520, "fed_sp_rodada": 400, "fed_redo_fed_voto": 450, "central_l2": 90,
           # rodada 3: o custo vai com K x S (K = 10, S = 15 custa 400 s); suposições por analogia
           "particao_iid_k2": 160, "particao_iid_k5": 250, "particao_iid_k20": 700, "copias_k2": 160, "copias_k5": 250,
           "copias_k10": 400, "copias_k20": 700, "particao_epocas_k2": 400, "particao_epocas_k5": 400, "particao_epocas_k20": 430}
R_ESTIMADO = (60, 110)      # R calibrado suposto (a grade é 50 a 200)


def dir_de(parte: str) -> Path:
    return Path(DIRS[parte])


def plano_confirmatorio(parte: str) -> list[tuple]:
    """(condição, tarefa, semente): sementes 16 a 23 na rodada 3 e 24 a 31 na rodada 4, tarefas A e B."""
    return [(c, tipo, sem) for tipo in ("A", "B") for sem in SEMENTES_PARTE[parte] for c in CATALOGO[parte]]


def nome_csv(job: tuple) -> str:
    cond, tipo, sem = job
    return f"{cond}_m{NIVEL_R2}_{tipo}_{sem}.csv"


def _sufixo_hp(hp) -> str:
    return "" if hp is None else (f"_hp{hp:g}" if isinstance(hp, (int, float)) else "_hp" + "-".join(f"{v:g}" for v in hp.values()))


def _resultado(f) -> None:
    try:
        logger.info(f.result())
    except Exception as e:                      # noqa: BLE001 (qualquer erro de worker deve ser registrado)
        FALHAS.append(repr(e))
        logger.error("JOB FALHOU: %r", e)


def _rodar(cond, tipo: str, sem: int, pasta: Path, sufixo: str = "") -> str:
    if csv_valido(cond, tipo, sem, pasta, sufixo):
        return f"pulado {cond.nome}{sufixo} {tipo} s{sem} (CSV válido)"
    executar(cond, tipo, sem, pasta, sufixo=sufixo)
    return f"ok {cond.nome}{sufixo} {tipo} s{sem}"


# ---------------------------------------------------------------- jobs (funções de topo: picklable)
def job_baseline(tipo: str, dir_r4: str) -> str:
    pasta = pasta_calib(Path(dir_r4))
    return _rodar(condicao_r4("fedavg_iid", tipo, Path(dir_r4), rodadas=R_BASE_R4), tipo, SEMENTE_CALIB, pasta)


def arq_calibracao(dir_r4: Path, nome: str, tarefa: str, hp=None) -> Path:
    return pasta_calib(Path(dir_r4)) / f"calibracao_R_{nome}{_sufixo_hp(hp)}_{tarefa}_m{NIVEL_R2}.json"


def job_calibrar(nome: str, tarefa: str, hp, dir_r4: str) -> dict:
    arq = arq_calibracao(Path(dir_r4), nome, tarefa, hp)
    if arq.exists():                                          # só reaproveita se o alvo é o mesmo (senão recalibra)
        antigo = json.loads(arq.read_text())
        if antigo.get("alvo_sha256") == hashlib.sha256(arq_alvo(Path(dir_r4), tarefa).read_bytes()).hexdigest():
            return antigo
    cond = condicao_r4(nome, tarefa, Path(dir_r4), hp=hp, rodadas=R_BASE_R4)
    return calibrar_R(cond, tarefa, Path(dir_r4), arq)


def job_selecao_run(nome: str, tarefa: str, hp, rodadas: int, dir_r4: str) -> str:
    pasta = pasta_calib(Path(dir_r4)) / "selecao_hp"
    return _rodar(condicao_r4(nome, tarefa, Path(dir_r4), hp=hp, rodadas=rodadas), tarefa, SEMENTE_CALIB, pasta, _sufixo_hp(hp))


def job_rodar(nome: str, tipo: str, sem: int, dir_r4: str) -> str:
    return _rodar(condicao_r4(nome, tipo, Path(dir_r4)), tipo, sem, Path(dir_r4))


def gravar_selecao(dir_r4: Path, nome: str, tarefa: str, hp, cal: dict, notas: dict | None = None, grade=None) -> dict:
    res = {"condicao": nome, "tarefa": tarefa, "hp": hp, "rodadas": cal["rodadas"], "ok": bool(cal["ok"]),
           "erro_rel": cal["erro_rel"], "notas": notas or {}, "borda": bool(grade and len(grade) >= 3 and isinstance(hp, (int, float)) and hp in (grade[0], grade[-1]))}
    arq_selecao(dir_r4, nome, tarefa).write_text(json.dumps(res, indent=1))
    return res


def escolher_hp(nome: str, tarefa: str, dir_r4: Path) -> dict:
    """Entre os candidatos com calibração ok: maior `acc_cont` média nas 5 últimas tarefas (semente 1000, com o R
    calibrado do candidato); empate: o primeiro da grade. Sem nenhum ok: o de menor |erro|, marcado ok=false."""
    grade = CATALOGO[next(p for p in CATALOGO if nome in CATALOGO[p])][nome][0]
    cals = {hp: json.loads(arq_calibracao(dir_r4, nome, tarefa, hp).read_text()) for hp in grade}
    notas = {}
    for hp, c in cals.items():
        if c["ok"]:
            df = pd.read_csv(pasta_calib(dir_r4) / "selecao_hp" / f"{nome}{_sufixo_hp(hp)}_m{NIVEL_R2}_{tarefa}_{SEMENTE_CALIB}.csv")
            notas[hp] = float(df["acc_cont"].tail(5).mean())
    if notas:
        melhor = max(grade, key=lambda hp: (notas.get(hp, -1.0), -grade.index(hp)))
    else:
        melhor = min(grade, key=lambda hp: (abs(cals[hp]["erro_rel"]), grade.index(hp)))
    res = gravar_selecao(dir_r4, nome, tarefa, melhor, cals[melhor], {f"{k:g}": v for k, v in notas.items()}, grade)
    res["candidatos"] = {f"{hp:g}": {"R": c["rodadas"], "erro_rel": c["erro_rel"], "ok": c["ok"]} for hp, c in cals.items()}
    arq_selecao(dir_r4, nome, tarefa).write_text(json.dumps(res, indent=1))
    logger.info("hiperparâmetro %s %s: %g (R=%d, ok=%s) notas=%s", nome, tarefa, melhor, res["rodadas"], res["ok"], res["notas"])
    return res


def job_selecao_final(nome: str, tarefa: str, dir_r4: str) -> dict:
    """Condições sem grade (e as combinações): calibra R com o hiperparâmetro derivado e grava a seleção."""
    d = Path(dir_r4)
    hp = None
    if nome in COMBINACOES:
        from src.condicoes_r4 import hp_de
        hp = hp_de(nome, tarefa, d)
    cal = job_calibrar(nome, tarefa, hp if nome in COMBINACOES else None, dir_r4)
    return gravar_selecao(d, nome, tarefa, hp, cal)


# ---------------------------------------------------------------- custo
def _fator_R(nome: str, R: int) -> float:
    return R / R_BASE_R4


def _custo_exec(nome: str, R: int = 75) -> float:
    """Segundos de CPU de uma execução de T=30 (a de DiLoCo cresce com R·H; as demais com R)."""
    return CUSTO_S[nome] * _fator_R(nome, R)


def estimar(parte: str) -> dict:
    """Estimativa de CPU (não medida): execuções confirmatórias + calibração de R + seleção de hiperparâmetros."""
    conds = list(CATALOGO[parte])
    grade = {n: CATALOGO[parte][n][0] for n in conds}

    def total(R: int) -> dict:
        conf = sum(_custo_exec(n, R) if n != "fedavg_iid" else CUSTO_S[n] for n in conds) * 2 * len(SEMENTES_PARTE[parte])
        cal = sel = 0.0
        for n in conds:
            if n == "fedavg_iid":
                continue
            ncand = len(grade[n]) if grade[n] else 1
            # calibração: 16 redes novas x 200 rodadas; 10 clientes x 15 passos (FL), H passos (DiLoCo), 15 passos (central)
            unidade = {"sem_media_k1": 0.0116, "central_aprendizado": 0.0077, "central_l2": 0.0077}.get(n, 0.116)
            if "_k" in n and n.rsplit("_k", 1)[1].isdigit():              # rodada 3: custo por rodada ~ K x S
                K = int(n.rsplit("_k", 1)[1])
                unidade = 0.0116 * K * (-(-3 * (1000 // K) // 20) / 15 if "epocas" in n else 1.0)
            if n.startswith("diloco"):
                unidade *= {"diloco_h15": 1, "diloco_h50": 50 / 15, "diloco_h150": 10}[n]
            cal += ncand * 2 * 16 * 200 * unidade
            if grade[n]:
                sel += ncand * 2 * 0.6 * _custo_exec(n, R)       # corrida de T=30 na semente 1000, sem diagnósticos
        base = 2 * (CUSTO_S["fedavg_iid"] * 0.6)                  # fedavg_iid na semente 1000 (alvo)
        return {"confirmatorio": conf, "calibracao": cal, "selecao": sel + base}

    lo, hi = total(R_ESTIMADO[0]), total(R_ESTIMADO[1])
    soma = lambda d: sum(d.values()) / 3600
    return {"parte": parte, "condicoes": len(conds), "execucoes_confirmatorias": len(plano_confirmatorio(parte)),
            "R_supostos": list(R_ESTIMADO), "cpu_horas": [soma(lo), soma(hi)],
            "detalhe_cpu_h": {k: [lo[k] / 3600, hi[k] / 3600] for k in lo},
            "parede_20_proc_h": [soma(lo) / 20 * 1.5, soma(hi) / 20 * 2.5]}


# ---------------------------------------------------------------- completo
def _etapa(ex, dir_r4: Path, parte: str, nomes: list[str]) -> None:
    """Calibra R e escolhe hiperparâmetros (semente 1000) das condições `nomes`, nas duas tarefas."""
    cat = CATALOGO[parte]
    pedidos = []                                              # (nome, tarefa, hp, futuro de calibração)
    for nome in nomes:
        for tarefa in ("A", "B"):
            if cat[nome][0]:
                for hp in cat[nome][0]:
                    pedidos.append((nome, tarefa, hp, ex.submit(job_calibrar, nome, tarefa, hp, str(dir_r4))))
            else:
                pedidos.append((nome, tarefa, None, ex.submit(job_selecao_final, nome, tarefa, str(dir_r4))))
    corridas = []
    for nome, tarefa, hp, f in pedidos:
        try:
            cal = f.result()
        except Exception as e:                                # noqa: BLE001
            FALHAS.append(repr(e))
            logger.error("CALIBRAÇÃO FALHOU %s %s %s: %r", nome, tarefa, hp, e)
            continue
        logger.info("calibração %s %s hp=%s: R=%d erro_rel=%+.3f ok=%s", nome, tarefa, hp, cal["rodadas"], cal["erro_rel"], cal["ok"])
        if hp is not None and cal["ok"]:
            corridas.append(ex.submit(job_selecao_run, nome, tarefa, hp, cal["rodadas"], str(dir_r4)))
    for f in corridas:
        _resultado(f)
    for nome in nomes:
        for tarefa in ("A", "B"):
            if cat[nome][0]:
                try:
                    escolher_hp(nome, tarefa, dir_r4)
                except Exception as e:                        # noqa: BLE001 (falta de arquivo de um candidato)
                    FALHAS.append(repr(e))
                    logger.error("ESCOLHA DE HIPERPARÂMETRO FALHOU %s %s: %r", nome, tarefa, e)


def fase_selecao(ex, dir_r4: Path, parte: str) -> dict:
    t0 = time.time()
    pasta_calib(dir_r4).mkdir(parents=True, exist_ok=True)
    for f in [ex.submit(job_baseline, tipo, str(dir_r4)) for tipo in ("A", "B")]:
        _resultado(f)
    for tipo in ("A", "B"):                                   # fedavg_iid: R = 75 por definição (é o alvo)
        arq_selecao(dir_r4, "fedavg_iid", tipo).write_text(json.dumps(
            {"condicao": "fedavg_iid", "tarefa": tipo, "hp": None, "rodadas": R_BASE_R4, "ok": True, "erro_rel": 0.0,
             "notas": {}, "borda": False}, indent=1))
    nomes = [n for n in CATALOGO[parte] if n != "fedavg_iid"]
    dependentes = [n for n in nomes if n in COMBINACOES or n in DEPENDE]
    _etapa(ex, dir_r4, parte, [n for n in nomes if n not in dependentes])
    if dependentes:
        _etapa(ex, dir_r4, parte, dependentes)
    return {"selecao_e_calibracao_s": time.time() - t0}


def fase_completo(ex, dir_r4: Path, parte: str, sem_teste_curto: bool = False) -> dict:
    res = dir_r4 / "calibracao" / "teste_curto" / "RESULTADO.txt"
    if not sem_teste_curto and not (res.exists() and res.read_text().startswith("OK")):
        raise SystemExit("o teste curto não deu OK (ou não rodou): rode `--fase teste_curto` e leia RESULTADO.txt. "
                         "Para ignorar, --sem-teste-curto.")
    logger.info("RODADA4 parte=%s fase=completo iniciada", parte)     # o verificador exige esta linha
    tempos = fase_selecao(ex, dir_r4, parte)
    t0 = time.time()
    plano = sorted(plano_confirmatorio(parte), key=lambda j: _custo_exec(j[0]), reverse=True)
    for f in [ex.submit(job_rodar, c, tipo, sem, str(dir_r4)) for c, tipo, sem in plano]:
        _resultado(f)
    tempos["confirmatorio_s"] = time.time() - t0
    return tempos


# ---------------------------------------------------------------- teste curto
# R pequeno por técnica (o teste cobre o caminho de código, não o nível de aprendizado)
TC_R = {"diloco_h50": 4, "diloco_h150": 2}
TC_R_PADRAO = 12
TC_HP = {"fedprox": 0.1, "fed_l2": 1e-3, "fed_l2init": 1e-2, "fed_redo_local": 0.5, "fed_sp": 0.8, "fed_redo_fed": 0.5,
         "fed_sp_rodada": 0.99, "central_l2": 1e-3,
         "fed_l2_sp": {"fed_l2": 1e-3, "fed_sp": 0.8}, "fed_l2init_sp": {"fed_l2init": 1e-2, "fed_sp": 0.8},
         "fed_redo_ambos": {"fed_redo_local": 0.5, "fed_redo_fed": 0.5}, "fed_redo_fed_voto": {"tau": 0.5, "q": 0.5}}
# sanidades de equivalência (T=3, R=3): comparadas com o fedavg_iid, que é o primeiro
TC_SANIDADE = {"fedprox_mu0": {"tipo": "fedprox", "mu": 0.0}, "gossip_completo": {"tipo": "gossip", "topologia": "completo"},
               "diloco_lr1": {"tipo": "diloco", "H": 15, "lr_ext": 1.0, "mom": 0.0},
               "fedavg_wd0": {"tipo": "fedavg", "wd": 0.0, "l2init": 0.0}}


def jobs_teste_curto(parte: str) -> list[dict]:
    J = lambda nome, T, R, suf, hp=None, tec=None: dict(nome=nome, tarefa="A", T=T, rodadas=R, sufixo=suf, hp=hp, tec=tec)
    jobs = [J("fedavg_iid", 3, 3, "_san_base")]
    if parte != "3":
        jobs += [J("fedavg_iid", 3, 3, f"_san_{k}", tec=v) for k, v in TC_SANIDADE.items()]
    if parte == "3":                                       # sanidades próprias: K=10 explícito e copias_k10 = FedAvg com 'total'
        jobs += [J("fedavg_iid", 3, 3, "_san_part_k10", tec={"tipo": "fedavg", "K": 10, "modo": "iid"}),
                 dict(J("fedavg_iid", 3, 3, "_san_fl_total"), especial="fl_total"),
                 J("fedavg_iid", 3, 3, "_san_copias_k10", tec={"tipo": "fedavg", "K": 10, "modo": "copias"})]
    for nome in CATALOGO[parte]:
        jobs.append(J(nome, 12, TC_R.get(nome, TC_R_PADRAO), "_tc", hp=TC_HP.get(nome)))
    return jobs


def _cond_tc(j: dict, dir_r4: Path):
    if j.get("especial") == "fl_total":                    # FedAvg da rodada 2 com os mesmos 1000 pontos em 10 clientes
        from src.condicoes import condicao_base
        return replace(condicao_base("media_sem_particao", NIVEL_R2), T=j["T"], rodadas=j["rodadas"], rodada=4, sonda_versao=2)
    cond = condicao_r4(j["nome"], j["tarefa"], dir_r4, hp=j["hp"], rodadas=j["rodadas"], T=j["T"])
    return replace(cond, tec=json.dumps(j["tec"], sort_keys=True)) if j["tec"] else cond


def _csv_tc(pasta: Path, j: dict, dir_r4: Path) -> pd.DataFrame:
    return pd.read_csv(pasta / f"{_base_arquivo(_cond_tc(j, dir_r4), j['tarefa'], SEMENTE_CALIB, j['sufixo'])}.csv")


def rodar_tc(j: dict, pasta: str, dir_r4: str) -> str:
    return _rodar(_cond_tc(j, Path(dir_r4)), j["tarefa"], SEMENTE_CALIB, Path(pasta), j["sufixo"])


def avaliar_teste_curto(dir_r4: Path, parte: str) -> tuple[list[str], list[str]]:
    from src.config import tarefas_com_nova
    pasta = pasta_calib(dir_r4) / "teste_curto"
    pare, info = [], []
    jobs = jobs_teste_curto(parte)
    df = {(j["nome"], j["sufixo"]): _csv_tc(pasta, j, dir_r4) for j in jobs}
    # 1) equivalências exatas: o que deve reproduzir o fedavg_iid reproduz (as colunas de resultado, bit a bit)
    base = df[("fedavg_iid", "_san_base")]
    cols = ["acc_cont", "loss_cont", "norma", "rank_ef", "acc_nova", "loss_nova", "gap", "cos_medio", "encolhimento"]
    sanidades = list(TC_SANIDADE) if parte != "3" else ["part_k10"]
    for k in sanidades:
        d = df[("fedavg_iid", f"_san_{k}")]
        igual = np.allclose(base[cols].to_numpy(float), d[cols].to_numpy(float), rtol=0, atol=1e-6 if k == "diloco_lr1" else 0,
                            equal_nan=True)
        if not igual:
            pare.append(f"sanidade {k}: NÃO reproduz o fedavg_iid (colunas {cols})")
    if parte == "3":                                       # copias_k10 (fedx) = FedAvg com a partição 'total' (fl), sem a coluna de encolhimento
        d = df[("fedavg_iid", "_san_copias_k10")]
        ref = df[("fedavg_iid", "_san_fl_total")]
        cs = [c for c in cols if c != "encolhimento"]
        if not np.allclose(ref[cs].to_numpy(float), d[cs].to_numpy(float), rtol=0, atol=0, equal_nan=True):
            pare.append("sanidade copias_k10: NÃO reproduz o FedAvg com a partição 'total' (colunas %s)" % cs)
    info.append("sanidades vs fedavg_iid: " + ("reproduzem" if not any(p.startswith("sanidade") for p in pare) else "FALHARAM")
                + (" (rodada 3: K=10 explícito e copias_k10 = FedAvg 'total')" if parte == "3" else
                   " (fedprox μ=0, gossip completo, DiLoCo lr=1 sem momento, FedAvg sem remédio)"))
    # 2) valores finitos e rede nova nas tarefas certas
    ref = set(tarefas_com_nova(12))
    for (nome, suf), d in df.items():
        if suf != "_tc":
            continue
        if d[["acc_cont", "loss_cont", "norma"]].isna().any().any() or not np.isfinite(d[["loss_cont", "norma"]].to_numpy(float)).all():
            pare.append(f"{nome}: NaN/inf em acc_cont, loss_cont ou norma")
        if set(d.index[d["acc_nova"].notna()]) != ref or set(d.index[d["loss_nova"].notna()]) != ref:
            pare.append(f"{nome}: redes novas fora das tarefas esperadas")
        info.append(f"{nome} (R={TC_R.get(nome, TC_R_PADRAO)}): acc_cont 3 últimas={d['acc_cont'].tail(3).mean():.3f} "
                    f"acc_nova={d['acc_nova'].dropna().mean():.3f} loss_nova={d['loss_nova'].dropna().mean():.3f}")
    # 3) o fedavg_iid aprende; cada remédio/técnica muda o resultado (senão não está agindo)
    b = df[("fedavg_iid", "_tc")]
    if float(b["acc_cont"].tail(3).mean()) < LIMITE_APRENDE:
        pare.append(f"fedavg_iid (R pequeno) não aprende: acc_cont das 3 últimas tarefas = {b['acc_cont'].tail(3).mean():.3f}")
    for nome in CATALOGO[parte]:
        if nome in ("fedavg_iid", "sem_media_k1", "central_aprendizado", "central_l2"):
            continue
        d = df[(nome, "_tc")]
        if TC_R.get(nome, TC_R_PADRAO) == TC_R_PADRAO and b[["acc_cont", "loss_cont", "norma"]].equals(d[["acc_cont", "loss_cont", "norma"]]):
            pare.append(f"{nome}: resultado idêntico ao do fedavg_iid; a técnica/remédio não está agindo")
    for nome in CATALOGO[parte]:                           # ReDo: o teste curto usa tau = 0,5; tem que reiniciar algo
        d = df[(nome, "_tc")]
        if "redo" in nome and nome != "fed_redo_ambos":
            n = int(d[["redo_servidor", "redo_cliente"]].sum().sum())
            info.append(f"{nome}: neurônios reiniciados no teste curto = {n}")
            if n == 0:
                pare.append(f"{nome}: o ReDo não reiniciou nenhum neurônio (tau = 0,5): o remédio não é acionado")
    # 4) calibração em miniatura (uma condição com hiperparâmetro), com o alvo = o fedavg_iid T=12 deste teste
    alvo = pasta / f"{_base_arquivo(_cond_tc(next(j for j in jobs if j['nome'] == 'fedavg_iid' and j['sufixo'] == '_tc'), dir_r4), 'A', SEMENTE_CALIB, '_tc')}.csv"
    nome_cal = {"3": "particao_iid_k5", "4a": "fedprox", "4b": "fed_l2"}[parte]
    cond = condicao_r4(nome_cal, "A", dir_r4, hp=TC_HP.get(nome_cal), rodadas=3, T=12)
    c = calibrar_R(cond, "A", dir_r4, pasta / "calibracao_R_tc.json", grade=(3, 4, 5), T=12, alvo_csv=alvo)
    if not (np.isfinite(c["erro_rel"]) and c["rodadas"] in (3, 4, 5) and len(c["tentativas"]) == 3):
        pare.append(f"calibração em miniatura malformada: {c['rodadas']=} {c['erro_rel']=}")
    info.append(f"calibração em miniatura ({nome_cal}, T=12, R de 3 a 5, alvo = fedavg_iid deste teste): R={c['rodadas']} "
                f"erro_rel={c['erro_rel']:+.3f} (só testa o código)")
    # 5) o alvo real existe? (o completo o produz; aqui só se avisa)
    return pare, info


def fase_teste_curto(ex, dir_r4: Path, parte: str) -> int:
    pasta = pasta_calib(dir_r4) / "teste_curto"
    if pasta.exists():                          # o teste curto sempre recomeça: não reaproveita CSV de código antigo
        shutil.rmtree(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    for f in [ex.submit(rodar_tc, j, str(pasta), str(dir_r4)) for j in jobs_teste_curto(parte)]:
        _resultado(f)
    try:
        pare, info = avaliar_teste_curto(dir_r4, parte)
    except Exception as e:                      # noqa: BLE001 (CSV faltando por falha de job)
        pare, info = [f"não foi possível avaliar o teste curto: {e!r}"], []
    for i in info:
        logger.info("TESTE CURTO: %s", i)
    for p in pare:
        logger.warning("PARE: %s", p)
    linhas = ["PARE" if pare else "OK"] + [f"PARE: {p}" for p in pare] + [f"INFO: {i}" for i in info]
    (pasta / "RESULTADO.txt").write_text("\n".join(linhas) + "\n")
    logger.info("TESTE CURTO: %s", "PARE (NÃO solte o completo)" if pare else "OK, pode soltar o completo")
    return 3 if pare else 0


def _preparar_processo() -> None:
    os.environ["PLAST_DISPOSITIVO"] = "cpu"
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", choices=["3", "4a", "4b"], required=True)
    ap.add_argument("--fase", choices=["plano", "teste_curto", "completo"], required=True)
    ap.add_argument("--saida", default=None)
    ap.add_argument("--processos", type=int, default=3, help="cada processo usa ~0,3 GB")
    ap.add_argument("--sem-teste-curto", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.fase == "plano":
        print(json.dumps(estimar(a.parte), indent=1))
        return
    os.environ["PLAST_DISPOSITIVO"] = "cpu"
    dir_r4 = Path(a.saida or DIRS[a.parte])
    dir_r4.mkdir(exist_ok=True)
    t0, codigo = time.time(), 0
    with ProcessPoolExecutor(a.processos, initializer=_preparar_processo) as ex:
        if a.fase == "teste_curto":
            codigo = fase_teste_curto(ex, dir_r4, a.parte)
            tempos = {}
        else:
            tempos = fase_completo(ex, dir_r4, a.parte, a.sem_teste_curto)
    tempos["total_s"] = time.time() - t0
    # com jobs falhos o arquivo de tempo leva outro nome: a trava do `longo` (tempo_completo.json existe) não bloqueia a retomada
    (dir_r4 / (f"tempo_{a.fase}_com_falhas.json" if FALHAS else f"tempo_{a.fase}.json")).write_text(json.dumps(tempos, indent=1))
    logger.info("tempos: %s", tempos)
    if FALHAS:
        logger.error("%d JOB(S) FALHARAM: %s (rode a mesma fase de novo para retomar)", len(FALHAS), FALHAS)
        codigo = codigo or 1
    raise SystemExit(codigo)


if __name__ == "__main__":
    main()
