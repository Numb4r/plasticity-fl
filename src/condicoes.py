"""As 7 condições e o laço que percorre a sequência de tarefas."""
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import (
    BATCH, CLIENTES, FRESCA_CADA, N, PASSOS_CLIENTE, RESET_CENTRAL, SONDA_CADA,
    SP_RUIDO, T_PADRAO, T_SEM_PARTICAO, TAUS,
)
from src.dados import Sequencia, particionar
from src.esquecimento import linha_da_matriz, metricas_esquecimento
from src.federado import treinar_fedavg
from src.metricas import acuracia, metricas_modelo, norma_parametros, vetor_parametros
from src.modelo import criar_mlp
from src.sementes import gerador, semente
from src.sondas import sonda_alvos_aleatorios
from src.treino import shrink_and_perturb, treinar_passos

logger = logging.getLogger(__name__)

CONDICOES = (
    "fedavg_iid", "sem_media_k1", "media_sem_particao", "central_passos",
    "central_aprendizado", "central_L2", "central_SP",
)


@dataclass(frozen=True)
class Condicao:
    nome: str
    familia: str                  # "fl" ou "central"
    particao: str | None = None   # "iid", "um" ou "total" (só FL)
    T: int = T_PADRAO
    passos: int | None = None     # passos por tarefa (só central)
    wd: float = 0.0               # L2 (só central_L2)
    sp_lambda: float | None = None  # shrink-and-perturb (só central_SP)


def condicao_base(nome: str) -> Condicao:
    """Condições que não dependem de nada calibrado."""
    if nome == "fedavg_iid":
        return Condicao(nome, "fl", "iid")
    if nome == "sem_media_k1":
        return Condicao(nome, "fl", "um")
    if nome == "media_sem_particao":
        return Condicao(nome, "fl", "total", T=T_SEM_PARTICAO)
    if nome == "central_passos":
        return Condicao(nome, "central", passos=PASSOS_CLIENTE)
    raise ValueError(f"condição sem versão base: {nome}")


def _json(dir_res: Path, arquivo: str) -> dict:
    caminho = Path(dir_res) / arquivo
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe: rode a calibração/seleção antes")
    return json.loads(caminho.read_text())


def carregar_condicao(nome: str, tarefa: str, dir_res: Path) -> Condicao:
    """Monta a condição; as três últimas leem valores CONGELADOS na semente 0."""
    if nome in ("fedavg_iid", "sem_media_k1", "media_sem_particao", "central_passos"):
        return condicao_base(nome)
    calib = _json(dir_res, f"calibracao_{tarefa}.json")
    if not calib["ok"]:
        raise RuntimeError(f"calibração {tarefa} fora de ±0,03: registrar em Desvios antes de seguir")
    passos = calib["passos"]
    if nome == "central_aprendizado":
        return Condicao(nome, "central", passos=passos)
    if nome == "central_L2":
        return Condicao(nome, "central", passos=passos,
                        wd=_json(dir_res, f"selecao_L2_{tarefa}.json")["lambda"])
    if nome == "central_SP":
        return Condicao(nome, "central", passos=passos,
                        sp_lambda=_json(dir_res, f"selecao_SP_{tarefa}.json")["lambda"])
    raise ValueError(f"condição desconhecida: {nome}")


def csv_valido(cond: Condicao, tipo: str, sem: int, saida: Path) -> bool:
    """True se já existe um CSV completo gerado com exatamente os parâmetros de `cond`."""
    base = Path(saida) / f"{cond.nome}_{tipo}_{sem}"
    csv, params = base.with_suffix(".csv"), Path(f"{base}.params.json")
    if not (csv.exists() and params.exists()):
        return False
    if json.loads(params.read_text()) != json.loads(json.dumps(asdict(cond))):
        return False
    return len(pd.read_csv(csv)) == cond.T


def treinar_na_tarefa(cond: Condicao, modelo, X, y, sem: int, particoes, g) -> list[float]:
    """Protocolo de treino de UMA tarefa (igual para a rede contínua e a nova)."""
    if cond.familia == "fl":
        return treinar_fedavg(modelo, X, y, particoes, g)
    todos = torch.arange(N, device=X.device)
    treinar_passos(modelo, X, y, todos, cond.passos, BATCH, RESET_CENTRAL, g, wd=cond.wd)
    return []


def _sp(cond: Condicao, modelo, sem: int, t: int, quem: int) -> None:
    """Shrink-and-perturb no início da tarefa (quem: 0 = contínua, 1 = nova)."""
    if cond.sp_lambda is not None:
        shrink_and_perturb(modelo, cond.sp_lambda, SP_RUIDO, gerador(sem, "sp_ruido", t, quem))


def executar(cond: Condicao, tipo: str, sem: int, saida: Path, sufixo: str = "") -> pd.DataFrame:
    """Percorre as T tarefas de uma condição, tipo de tarefa e semente.

    Salva saida/<condicao>_<tarefa>_<semente>.csv (uma linha por tarefa) e, nas
    condições com mais de 1 cliente, ..._rodadas.csv com o cosseno por rodada.
    """
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    seq = Sequencia(tipo, sem)
    esquece = tipo == "B_disjunta"          # esquecimento só faz sentido com X_t distintos
    matriz = np.full((cond.T, cond.T), np.nan)
    particoes = particionar(cond.particao, sem) if cond.familia == "fl" else None
    continua = criar_mlp(semente(sem, "init_continua"))
    linhas, rodadas = [], []
    for t in range(cond.T):
        X, y = seq.dados(t)
        if t >= 1:
            _sp(cond, continua, sem, t, 0)
        theta0 = vetor_parametros(continua)
        cos = treinar_na_tarefa(cond, continua, X, y, sem, particoes,
                                gerador(sem, "embaralha_continua", t))
        linha = {"tarefa": t, "tipo": tipo, "cond": cond.nome, "semente": sem,
                 "acc_cont": acuracia(continua, X, y),
                 "norma": norma_parametros(continua),
                 "distancia": (vetor_parametros(continua) - theta0).norm().item()}
        linha.update(metricas_modelo(continua, X, TAUS))
        if esquece:
            matriz[t] = linha_da_matriz(continua, seq, t, cond.T)
            linha.update(metricas_esquecimento(matriz, t))
        linha["cos_medio"] = float(pd.Series(cos).mean()) if cos else float("nan")
        rodadas += [{"tarefa": t, "rodada": r, "cos": c} for r, c in enumerate(cos)]
        # rede nova de referência: mesmo protocolo, semente própria
        nova_acc = float("nan")
        if t % FRESCA_CADA == FRESCA_CADA - 1:
            nova = criar_mlp(semente(sem, "init_nova", t))
            _sp(cond, nova, sem, t, 1)
            treinar_na_tarefa(cond, nova, X, y, sem, particoes, gerador(sem, "embaralha_nova", t))
            nova_acc = acuracia(nova, X, y)
        linha["acc_nova"] = nova_acc
        linha["gap"] = linha["acc_cont"] - nova_acc
        sonda = {"sonda_perda_copia": float("nan"), "sonda_perda_nova": float("nan"),
                 "sonda_razao": float("nan")}
        if t % SONDA_CADA == SONDA_CADA - 1:
            sonda = sonda_alvos_aleatorios(continua, X, sem, t)
        linha.update(sonda)
        linhas.append(linha)
        logger.info("%s %s s%d tarefa %d/%d acc=%.3f nova=%.3f (%.0fs)", cond.nome, tipo, sem,
                    t + 1, cond.T, linha["acc_cont"], nova_acc, time.time() - inicio)
    df = pd.DataFrame(linhas)
    base = f"{cond.nome}{sufixo}_{tipo}_{sem}"
    df.to_csv(saida / f"{base}.csv", index=False)
    if esquece:
        pd.DataFrame(matriz, columns=[f"j{j}" for j in range(cond.T)]).rename_axis("i").to_csv(
            saida / f"{base}_matriz.csv")
    if particoes is not None and len(particoes) > 1:
        pd.DataFrame(rodadas).to_csv(saida / f"{base}_rodadas.csv", index=False)
    # parâmetros usados (passos, L2, SP, T): permite conferir ao retomar o pipeline
    (saida / f"{base}.params.json").write_text(json.dumps(asdict(cond), sort_keys=True))
    return df
