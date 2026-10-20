"""Dados sintéticos. A e B: X fixo, rótulos por tarefa (A: aleatórios; B: professor).
B_disjunta: um X_t próprio por tarefa (ver Sequencia)."""
import torch

from src.config import CLASSES, CLIENTES, D, N, dispositivo
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


class Sequencia:
    """Entradas e rótulos de cada tarefa de uma semente.

    A e B: X é o mesmo em todas as tarefas; só os rótulos mudam.
    B_disjunta: cada tarefa tem o seu X_t ~ N(mu_t, I), com 1000 pontos, e o
    professor da tarefa rotula X_t. Os dados são guardados para a avaliação de
    esquecimento reusar (X_j, y_j) das tarefas anteriores.
    """

    def __init__(self, tipo: str, sem: int) -> None:
        self.tipo, self.sem = tipo, sem
        self._X_fixo = None if tipo == "B_disjunta" else gerar_X(sem)
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def maj_frac(self, t: int) -> float:
        """Fração da classe majoritária dos rótulos da tarefa t."""
        return torch.bincount(self.dados(t)[1], minlength=CLASSES).max().item() / N

    def dados(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        if t not in self._cache:
            if self.tipo == "B_disjunta":
                ruido = torch.randn(N, D, generator=gerador(self.sem, "dados_tarefa", t))
                X = (ruido + mu_tarefa(self.sem, t)).to(dispositivo())
            else:
                X = self._X_fixo
            self._cache[t] = (X, rotulos_tarefa(self.tipo, self.sem, t, X))
        return self._cache[t]
