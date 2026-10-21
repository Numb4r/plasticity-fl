"""Dados sintéticos. A e B: X fixo, rótulos por tarefa (A: aleatórios; B: professor).
B_disjunta: um X_t próprio por tarefa (ver Sequencia). B_fixa (rodada 2): X_t próprio, professor fixo."""
import numpy as np
import torch

from src.config import CLASSES, CLIENTES, D, N, REF_CENTRO_PONTOS, T_PADRAO, dispositivo
from src.modelo import criar_mlp
from src.sementes import gerador, semente


def gerar_X(sem: int) -> torch.Tensor:
    """X ~ N(0,1), N x D, fixo durante todo o experimento de uma semente."""
    return torch.randn(N, D, generator=gerador(sem, "dados")).to(dispositivo())


def rotulos_tarefa(tipo: str, sem: int, t: int, X: torch.Tensor) -> torch.Tensor:
    """Rótulos da tarefa t.

    A: sorteados de novo a cada tarefa, sem estrutura (memorização).
    B: uma MLP professora recém-inicializada (semente nova a cada tarefa) rotula
       cada ponto pelo argmax das suas 10 saídas, depois de centralizar cada saída
       pela média da classe (há estrutura a aprender e as classes ficam balanceadas).
    """
    if tipo == "A":
        y = torch.randint(0, CLASSES, (N,), generator=gerador(sem, "rotulos", t))
        return y.to(dispositivo())
    if tipo in ("B", "B_disjunta"):
        professor = criar_mlp(semente(sem, "professor", t))
        with torch.no_grad():
            logits = professor(X)
            # Desvios v3 (e): centraliza cada logit pela média da sua classe sobre os
            # pontos da tarefa; sem isso uma classe domina (maioria > 0,3 em 83-97% das tarefas).
            return (logits - logits.mean(dim=0, keepdim=True)).argmax(dim=1)
    raise ValueError(f"tipo de tarefa desconhecido: {tipo}")


def particionar(modo: str, sem: int) -> list[torch.Tensor]:
    """Índices de cada cliente.

    iid:   10 clientes, 100 pontos cada, partição aleatória.
    um:    1 cliente com todos os pontos.
    total: 10 clientes, todos com os mesmos 1000 pontos.
    """
    todos = torch.arange(N)
    if modo == "iid":
        perm = torch.randperm(N, generator=gerador(sem, "particao"))
        partes = list(perm.chunk(CLIENTES))
    elif modo == "um":
        partes = [todos]
    elif modo == "total":
        partes = [todos.clone() for _ in range(CLIENTES)]
    else:
        raise ValueError(f"partição desconhecida: {modo}")
    return [p.to(dispositivo()) for p in partes]


NORMA_MU = 3.0   # norma do deslocamento mu_t de cada tarefa em B_disjunta


def mu_tarefa(sem: int, t: int) -> torch.Tensor:
    """Centro mu_t da tarefa t em B_disjunta: direção uniforme, norma exatamente 3."""
    v = torch.randn(D, generator=gerador(sem, "media_tarefa", t))
    return NORMA_MU * v / v.norm()


def particionar_dirichlet(sem: int, t: int, y: torch.Tensor, alfa: float, clientes: int = CLIENTES) -> list[torch.Tensor]:
    """Partição não IID por classe (rodada 2): para cada classe, proporções ~ Dirichlet(alfa)
    entre os clientes; contagem por classe = diferenças de round(cumsum(p) * n_c); os pontos
    da classe, embaralhados, vão para os clientes nessa ordem. Refeita a cada tarefa (os rótulos
    mudam). alfa grande ~ IID; alfa pequeno = cada cliente com poucas classes.
    Devolve `clientes` tensores de índices (alguns podem ficar vazios)."""
    y_cpu = y.detach().cpu().numpy()
    partes: list[list[int]] = [[] for _ in range(clientes)]
    for c in range(CLASSES):
        idx_c = np.nonzero(y_cpu == c)[0]
        n_c = len(idx_c)
        if n_c == 0:
            continue
        rng = np.random.default_rng(semente(sem, "particao_dirichlet", t, c))
        p = rng.dirichlet(alfa * np.ones(clientes))
        cortes = np.round(np.cumsum(p) * n_c).astype(int)
        cortes[-1] = n_c
        contagens = np.diff(np.concatenate([[0], cortes]))
        idx_c = idx_c[rng.permutation(n_c)]
        inicio = 0
        for k, cnt in enumerate(contagens):
            partes[k] += idx_c[inicio:inicio + cnt].tolist()
            inicio += cnt
    return [torch.tensor(sorted(p), dtype=torch.long, device=dispositivo()) for p in partes]


class Sequencia:
    """Entradas e rótulos de cada tarefa de uma semente.

    A e B: X é o mesmo em todas as tarefas; só os rótulos mudam.
    B_disjunta: cada tarefa tem o seu X_t ~ N(mu_t, I), com 1000 pontos, e o
    professor da tarefa rotula X_t. Os dados são guardados para a avaliação de
    esquecimento reusar (X_j, y_j) das tarefas anteriores.
    """

    def __init__(self, tipo: str, sem: int, T: int = T_PADRAO) -> None:
        self.tipo, self.sem, self.T = tipo, sem, T
        self._X_fixo = None if tipo in ("B_disjunta", "B_fixa") else gerar_X(sem)
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if tipo == "B_fixa":
            # Rodada 2: professor FIXO (o mesmo em todas as tarefas da semente) e só a entrada
            # desloca. O deslocamento de centralização dos logits é calculado UMA vez, numa amostra
            # de referência (REF_CENTRO_PONTOS // T pontos de cada N(mu_t, I)), para a função não
            # mudar de tarefa para tarefa (a centralização por tarefa a mudaria).
            self._professor = criar_mlp(semente(sem, "professor", 0))
            n_por = REF_CENTRO_PONTOS // T
            ref = [(torch.randn(n_por, D, generator=gerador(sem, "ref_centro", t)) + mu_tarefa(sem, t))
                   for t in range(T)]
            with torch.no_grad():
                self._centro = self._professor(torch.cat(ref).to(dispositivo())).mean(dim=0, keepdim=True)

    def rotular(self, X: torch.Tensor) -> torch.Tensor:
        """B_fixa: a MESMA função em qualquer entrada (professor fixo e centro fixo)."""
        with torch.no_grad():
            return (self._professor(X) - self._centro).argmax(dim=1)

    def maj_frac(self, t: int) -> float:
        """Fração da classe majoritária dos rótulos da tarefa t."""
        return torch.bincount(self.dados(t)[1], minlength=CLASSES).max().item() / N

    def dados(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        if t not in self._cache:
            if self.tipo in ("B_disjunta", "B_fixa"):
                ruido = torch.randn(N, D, generator=gerador(self.sem, "dados_tarefa", t))
                X = (ruido + mu_tarefa(self.sem, t)).to(dispositivo())
            else:
                X = self._X_fixo
            y = self.rotular(X) if self.tipo == "B_fixa" else rotulos_tarefa(self.tipo, self.sem, t, X)
            self._cache[t] = (X, y)
        return self._cache[t]
