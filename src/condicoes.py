"""As 7 condições e o laço que percorre a sequência de tarefas."""
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config
from src.diagnosticos import medir, vazio
from src.config import (
    BATCH, N, NIVEL_R2, PASSOS_CLIENTE, REAPRENDE_ANTIGAS, REAPRENDE_EM, REAPRENDE_FRACAO,
    RESET_CENTRAL, RODADAS_BASE, S_LOCAL, SEMENTE_CALIB, SONDA_CADA, SP_RUIDO, T_PADRAO,
    T_SEM_PARTICAO, TAUS, VERSAO_DADOS, tarefas_com_nova,
)
from src.dados import Sequencia, particionar, particionar_dirichlet
from src.esquecimento import linha_da_matriz, metricas_esquecimento
from src.federado import treinar_fedavg
from src.media_tempo import clonar_estado, treinar_ht, treinar_lookahead
from src.metricas import acuracia, metricas_modelo, norma_parametros, perda_ce, vetor_parametros
from src.modelo import MLP, criar_mlp
from src.reaprendizado import reaprender
from src.sementes import gerador, semente
from src.sondas import SONDA2_COLUNAS, sonda_alvos_aleatorios, sonda_v2
from src.treino import shrink_and_perturb, treinar_passos

logger = logging.getLogger(__name__)

CONDICOES = (
    "fedavg_iid", "sem_media_k1", "media_sem_particao", "central_passos",
    "central_aprendizado", "central_L2", "central_SP",
)
SUBPASTA_CALIB = "calibracao"   # tudo da semente 1000 mora em <results>/calibracao/


@dataclass(frozen=True)
class Condicao:
    nome: str
    familia: str                  # "fl", "central", "ht" (Hare & Tortoise) ou "lookahead"
    particao: str | None = None   # "iid", "um", "total" ou "dirichlet" (só FL e médias no tempo)
    T: int = T_PADRAO
    nivel: int = 1                # eixo de orçamento m
    rodadas: int | None = None   # rodadas por tarefa = 25·m (só FL; None nas centralizadas)
    s_local: int | None = None   # passos locais por rodada (só FL; None nas centralizadas)
    passos: int | None = None     # passos por tarefa (só central)
    wd: float = 0.0               # L2 (só central_L2)
    sp_lambda: float | None = None  # shrink-and-perturb (só central_SP)
    calibracao_ok: bool = True    # False = calibração fora de ±0,03 (registrado, segue)
    versao_dados: int = VERSAO_DADOS
    # --- rodada 2 ---
    mu: float | None = None       # central_HT e central_lookahead
    alfa: float | None = None     # fedavg_dirichlet
    sonda_versao: int = 1         # 1 = sonda de 300 passos (rodada 1); 2 = 20 passos, perdas por alvo
    geral: bool = False           # amostragem local para clientes de qualquer tamanho (Dirichlet)
    rodada: int = 1


def condicao_base(nome: str, nivel: int = 1) -> Condicao:
    """Condições que não dependem de nada calibrado."""
    comum = dict(nivel=nivel, rodadas=RODADAS_BASE * nivel, s_local=S_LOCAL)
    if nome == "fedavg_iid":
        return Condicao(nome, "fl", "iid", **comum)
    if nome == "sem_media_k1":
        return Condicao(nome, "fl", "um", **comum)
    if nome == "media_sem_particao":
        return Condicao(nome, "fl", "total", T=T_SEM_PARTICAO, **comum)
    if nome == "central_passos":
        return Condicao(nome, "central", passos=PASSOS_CLIENTE * nivel, nivel=nivel)
    raise ValueError(f"condição sem versão base: {nome}")


def pasta_calib(dir_res: Path) -> Path:
    return Path(dir_res) / SUBPASTA_CALIB


def _json(caminho: Path) -> dict:
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe: rode a calibração/seleção antes")
    return json.loads(caminho.read_text())


def carregar_condicao(nome: str, tarefa: str, nivel: int, dir_res: Path) -> Condicao:
    """Monta a condição; as três últimas leem valores CONGELADOS na semente 1000.

    Calibração fora de ±0,03 não interrompe: a condição usa o valor mais próximo e
    leva calibracao_ok=False nos parâmetros gravados (Desvios v3, item b).
    """
    if nome in ("fedavg_iid", "sem_media_k1", "media_sem_particao", "central_passos"):
        return condicao_base(nome, nivel)
    pasta = pasta_calib(dir_res)
    calib = _json(pasta / f"calibracao_{tarefa}_m{nivel}.json")
    comum = dict(passos=calib["passos"], nivel=nivel, calibracao_ok=bool(calib["ok"]))
    if nome == "central_aprendizado":
        return Condicao(nome, "central", **comum)
    if nome == "central_L2":
        sel = _json(pasta / f"selecao_L2_{tarefa}_m{nivel}.json")
        return Condicao(nome, "central", wd=sel["lambda"], **comum)
    if nome == "central_SP":
        sel = _json(pasta / f"selecao_SP_{tarefa}_m{nivel}.json")
        return Condicao(nome, "central", sp_lambda=sel["lambda"], **comum)
    raise ValueError(f"condição desconhecida: {nome}")


def condicao_r2(nome: str, tarefa: str, dir_r2: Path, *, T: int = T_PADRAO, alfa: float | None = None,
                mu: float | None = None, dir_r1: Path = Path("results")) -> Condicao:
    """Condições da rodada 2 (m* = 3, R = 75 rodadas, sonda de 20 passos).

    central_aprendizado: passos congelados da rodada 1 (A e B) ou calibrados na semente 1000 (B_fixa).
    central_HT e central_lookahead: mu escolhido na semente 1000 (`selecao_<método>_<tarefa>_m3.json`),
    um por tipo de tarefa; na B_fixa vale o mu da tarefa B. `mu` explícito (testes, sanidade) o substitui.
    """
    comum = dict(nivel=NIVEL_R2, rodadas=RODADAS_BASE * NIVEL_R2, s_local=S_LOCAL, T=T, sonda_versao=2, rodada=2)
    pasta = pasta_calib(dir_r2)
    tarefa_mu = "B" if tarefa == "B_fixa" else tarefa
    if nome == "fedavg_iid":
        return Condicao(nome, "fl", "iid", **comum)
    if nome == "sem_media_k1":
        return Condicao(nome, "fl", "um", **comum)
    if nome == "media_sem_particao":
        return Condicao(nome, "fl", "total", **comum)
    if nome == "fedavg_dirichlet":
        assert alfa is not None, "fedavg_dirichlet precisa de alfa"
        return Condicao(nome, "fl", "dirichlet", alfa=float(alfa), geral=True, **comum)
    if nome in ("central_HT", "central_lookahead"):
        if mu is None:
            metodo = "HT" if nome == "central_HT" else "lookahead"
            mu = _json(pasta / f"selecao_{metodo}_{tarefa_mu}_m{NIVEL_R2}.json")["mu"]
        return Condicao(nome, "ht" if nome == "central_HT" else "lookahead", "um", mu=float(mu), **comum)
    if nome == "central_aprendizado":
        arq = (pasta / f"calibracao_B_fixa_m{NIVEL_R2}.json" if tarefa == "B_fixa"
               else Path(dir_r1) / SUBPASTA_CALIB / f"calibracao_{tarefa}_m{NIVEL_R2}.json")
        calib = _json(arq)
        return Condicao(nome, "central", passos=calib["passos"], calibracao_ok=bool(calib["ok"]),
                        **{k: v for k, v in comum.items() if k not in ("rodadas", "s_local")})
    raise ValueError(f"condição desconhecida na rodada 2: {nome}")


def _base_arquivo(cond: Condicao, tipo: str, sem: int, sufixo: str = "") -> str:
    return f"{cond.nome}{sufixo}_m{cond.nivel}_{tipo}_{sem}"


def csv_valido(cond: Condicao, tipo: str, sem: int, saida: Path, sufixo: str = "") -> bool:
    """True se já existe um CSV completo gerado com exatamente os parâmetros de `cond`
    (inclui nível, rodadas, passos locais e versão dos dados)."""
    base = Path(saida) / _base_arquivo(cond, tipo, sem, sufixo)
    csv, params = Path(f"{base}.csv"), Path(f"{base}.params.json")
    if not (csv.exists() and params.exists()):
        return False
    if json.loads(params.read_text()) != json.loads(json.dumps(asdict(cond))):
        return False
    colunas = pd.read_csv(csv, nrows=0).columns
    if sem != SEMENTE_CALIB and "fim_hess_max" not in colunas:    # CSV de antes do v3.1: refazer
        return False
    if "loss_cont" not in colunas:                                  # CSV de antes da rodada 2: refazer
        return False
    return len(pd.read_csv(csv)) == cond.T


def treinar_na_tarefa(cond: Condicao, modelo, X, y, particoes, g, extra: dict | None = None) -> list[float]:
    """Protocolo de treino de UMA tarefa (igual para a rede contínua e a nova).

    `extra` guarda o estado que persiste entre tarefas fora do `modelo` (a tartaruga do central_HT)."""
    if cond.familia == "fl":
        return treinar_fedavg(modelo, X, y, particoes, g, cond.rodadas, geral=cond.geral)
    if cond.familia == "lookahead":
        return treinar_lookahead(modelo, X, y, particoes, g, cond.rodadas, cond.mu)
    if cond.familia == "ht":
        return treinar_ht(modelo, extra, X, y, particoes, g, cond.rodadas, cond.mu)
    todos = torch.arange(N, device=X.device)
    treinar_passos(modelo, X, y, todos, cond.passos, BATCH, RESET_CENTRAL, g, wd=cond.wd)
    return []


def _sp(cond: Condicao, modelo, sem: int, t: int, quem: int) -> None:
    """Shrink-and-perturb no início da tarefa (quem: 0 = contínua, 1 = nova)."""
    if cond.sp_lambda is not None:
        shrink_and_perturb(modelo, cond.sp_lambda, SP_RUIDO, gerador(sem, "sp_ruido", t, quem))


def executar(cond: Condicao, tipo: str, sem: int, saida: Path, sufixo: str = "",
             diagnosticos: bool | None = None) -> pd.DataFrame:
    """Percorre as T tarefas de uma condição, nível, tipo de tarefa e semente.

    Salva saida/<condicao>_m<nivel>_<tarefa>_<semente>.csv (uma linha por tarefa) e,
    nas condições com mais de 1 cliente, ..._rodadas.csv com o cosseno por rodada.
    A semente de calibração (1000) só pode gravar dentro de .../calibracao/, e as
    outras sementes nunca ali: assim a análise confirmatória não a enxerga.

    diagnosticos (Desvios v3.1): liga as métricas de Klein et al. no início e no fim de cada
    tarefa, no contínuo e em cada rede nova. Padrão: ligado, menos na semente de calibração.
    Nunca altera o contínuo (tudo numa cópia, com geradores próprios).
    """
    if diagnosticos is None:
        diagnosticos = sem != SEMENTE_CALIB
    saida = Path(saida)
    if (sem == SEMENTE_CALIB) != (SUBPASTA_CALIB in saida.parts):
        raise ValueError(f"semente {sem}: saída {saida} inconsistente com a regra da pasta '{SUBPASTA_CALIB}'")
    saida.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    seq = Sequencia(tipo, sem, cond.T)
    esquece = tipo in ("B_disjunta", "B_fixa")   # esquecimento só faz sentido com X_t distintos
    com_nova = set(tarefas_com_nova(cond.T, janela_hpart=cond.T <= T_PADRAO))   # T=100: sem a janela de H_part
    matriz = np.full((cond.T, cond.T), np.nan)
    dirichlet = cond.particao == "dirichlet"
    particoes = (particionar(cond.particao, sem)
                 if cond.familia in ("fl", "ht", "lookahead") and not dirichlet else None)
    continua = criar_mlp(semente(sem, "init_continua"))
    extra = {"tartaruga": clonar_estado(continua)} if cond.familia == "ht" else None
    linhas, rodadas, reaprende = [], [], []
    v_cont = None                               # autovetor da Hessiana (só usado no corte (b))
    for t in range(cond.T):
        X, y = seq.dados(t)
        if dirichlet:                           # partição refeita a cada tarefa; vale para a contínua e a nova
            particoes = particionar_dirichlet(sem, t, y, cond.alfa)
        if t >= 1:
            _sp(cond, continua, sem, t, 0)
        if cond.familia == "ht":                # a lebre volta para a tartaruga ANTES de medir o início da tarefa
            continua.load_state_dict(extra["tartaruga"])
        theta0 = vetor_parametros(continua)
        completo = (not config.DIAG_SO_REFERENCIAS) or t in com_nova     # corte (c)
        d_ini = d_fim = {}
        if diagnosticos:                        # início da tarefa: depois do SP, antes do 1º passo
            d_ini, v_cont = medir(continua, X, y, sem, t, 0, 0, "ini_", v_cont, completo)
        cos = treinar_na_tarefa(cond, continua, X, y, particoes,
                                gerador(sem, "embaralha_continua", t), extra)
        linha = {"tarefa": t, "tipo": tipo, "cond": cond.nome, "nivel": cond.nivel, "semente": sem,
                 "maj_frac": seq.maj_frac(t),
                 "acc_cont": acuracia(continua, X, y),
                 "loss_cont": perda_ce(continua, X, y),
                 "norma": norma_parametros(continua),
                 "distancia": (vetor_parametros(continua) - theta0).norm().item()}
        linha.update(metricas_modelo(continua, X, TAUS))
        if diagnosticos:                        # fim da tarefa
            d_fim, v_cont = medir(continua, X, y, sem, t, 1, 0, "fim_", v_cont, completo)
            linha.update(d_ini)
            linha.update(d_fim)
        if esquece:
            matriz[t] = linha_da_matriz(continua, seq, t, cond.T)
            linha.update(metricas_esquecimento(matriz, t))
        linha["cos_medio"] = float(pd.Series(cos).mean()) if cos else float("nan")
        # rodada 2: colunas que dizem como a condição foi montada (NaN quando não se aplicam)
        linha["mu"] = cond.mu if cond.mu is not None else float("nan")
        linha["alfa"] = cond.alfa if cond.alfa is not None else float("nan")
        tam = [len(p) for p in particoes] if dirichlet else None
        linha["part_min"] = min(tam) if tam else float("nan")
        linha["part_max"] = max(tam) if tam else float("nan")
        linha["part_vazios"] = sum(1 for n in tam if n == 0) if tam else float("nan")
        if cond.familia == "ht":                # a lebre é a rede medida; a tartaruga fica à parte
            tart = MLP().to(X.device)
            tart.load_state_dict(extra["tartaruga"])
            linha["acc_tartaruga"] = acuracia(tart, X, y)
        else:
            linha["acc_tartaruga"] = float("nan")
        rodadas += [{"tarefa": t, "rodada": r, "cos": c} for r, c in enumerate(cos)]
        # rede nova de referência: mesmo protocolo, semente própria
        nova_acc, nova_loss = float("nan"), float("nan")
        nova_diag = (vazio("nova_ini_") | vazio("nova_fim_")) if diagnosticos else {}
        if t in com_nova:
            nova = criar_mlp(semente(sem, "init_nova", t))
            extra_nova = {"tartaruga": clonar_estado(nova)} if cond.familia == "ht" else None
            _sp(cond, nova, sem, t, 1)
            v_nova = None
            if diagnosticos:                    # rede nova, início: depois do SP, antes do 1º passo
                d, v_nova = medir(nova, X, y, sem, t, 0, 1, "nova_ini_", None, completo)
                nova_diag.update(d)
            treinar_na_tarefa(cond, nova, X, y, particoes, gerador(sem, "embaralha_nova", t), extra_nova)
            nova_acc, nova_loss = acuracia(nova, X, y), perda_ce(nova, X, y)
            if diagnosticos:                    # rede nova, fim
                d, _ = medir(nova, X, y, sem, t, 1, 1, "nova_fim_", v_nova, completo)
                nova_diag.update(d)
        linha.update(nova_diag)
        linha["acc_nova"] = nova_acc
        linha["loss_nova"] = nova_loss
        linha["gap"] = linha["acc_cont"] - nova_acc      # sempre pareado: mesma tarefa
        sonda = {"sonda_perda_copia": float("nan"), "sonda_perda_nova": float("nan"),
                 "sonda_razao": float("nan")}
        if cond.sonda_versao == 2:
            sonda.update({c: float("nan") for c in SONDA2_COLUNAS})
        if t % SONDA_CADA == SONDA_CADA - 1:
            sonda = (sonda | sonda_v2(continua, X, sem, t)) if cond.sonda_versao == 2 \
                else sonda_alvos_aleatorios(continua, X, sem, t)
        linha.update(sonda)
        if tipo == "B_fixa" and t in REAPRENDE_EM:      # reaprendizado das tarefas antigas (régua comum)
            for j in REAPRENDE_ANTIGAS:
                if j < t:
                    reaprende.append(reaprender(continua, seq, sem, t, j, float(matriz[j][j]), REAPRENDE_FRACAO))
        linhas.append(linha)
        logger.info("%s m%d %s s%d tarefa %d/%d acc=%.3f nova=%.3f (%.0fs)", cond.nome, cond.nivel,
                    tipo, sem, t + 1, cond.T, linha["acc_cont"], nova_acc, time.time() - inicio)
    df = pd.DataFrame(linhas)
    base = _base_arquivo(cond, tipo, sem, sufixo)
    df.to_csv(saida / f"{base}.csv", index=False)
    if esquece:
        pd.DataFrame(matriz, columns=[f"j{j}" for j in range(cond.T)]).rename_axis("i").to_csv(
            saida / f"{base}_matriz.csv")
    if reaprende:
        pd.DataFrame(reaprende).to_csv(saida / f"{base}_reaprendizado.csv", index=False)
    if particoes is not None and len(particoes) > 1 and rodadas and not dirichlet:
        pd.DataFrame(rodadas).to_csv(saida / f"{base}_rodadas.csv", index=False)
    # parâmetros usados (nível, rodadas, passos, L2, SP, versão dos dados): conferidos ao retomar
    (saida / f"{base}.params.json").write_text(json.dumps(asdict(cond), sort_keys=True))
    return df
