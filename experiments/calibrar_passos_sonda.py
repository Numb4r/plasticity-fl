"""Quantos passos a sonda de alvos aleatórios precisa para a rede NOVA ficar entre 10% e 30% da
perda inicial? (Proposta para a próxima rodada; exploratório, definido em PREREGISTRO.md, C.)

20 alvos com sementes fora das de avaliação (>= 2000); rede nova com o protocolo da sonda
(Adam, lr 1e-3, lotes de 100, MSE); perda em todos os 1000 pontos a cada passo. Não toca em results/.
Uso: python -m experiments.calibrar_passos_sonda
"""
import numpy as np
import torch
import torch.nn.functional as F

from src import config
from src.dados import gerar_X
from src.modelo import criar_mlp

PASSOS = config.SONDA_PASSOS


def curva(sem_x: int, sem_alvo: int, sem_nova: int) -> np.ndarray:
    X = gerar_X(sem_x)
    with torch.no_grad():
        alvo = criar_mlp(sem_alvo)(X)
    rede = criar_mlp(sem_nova)
    opt = torch.optim.Adam(rede.parameters(), lr=config.LR)
    g = torch.Generator().manual_seed(sem_nova)
    perdas = []
    for _ in range(PASSOS + 1):
        with torch.no_grad():
            perdas.append(F.mse_loss(rede(X), alvo).item())
        idx = torch.randint(0, len(X), (config.SONDA_BATCH,), generator=g).to(X.device)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(rede(X[idx]), alvo[idx]).backward()
        opt.step()
    return np.array(perdas)


def primeiro(razao: np.ndarray, lim: float) -> float:
    ok = np.nonzero(razao <= lim)[0]
    return float(ok[0]) if len(ok) else float("nan")


def main() -> None:
    torch.set_num_threads(1)
    curvas = np.stack([curva(2000 + k // 5, 3000 + k, 4000 + k) for k in range(20)])
    razao = curvas / curvas[:, :1]
    s30 = np.array([primeiro(r, 0.30) for r in razao])
    s10 = np.array([primeiro(r, 0.10) for r in razao])
    s20 = np.array([primeiro(r, 0.20) for r in razao])
    f = lambda v: f"mediana {np.nanmedian(v):.0f} [mín {np.nanmin(v):.0f}, máx {np.nanmax(v):.0f}] (n={int(np.sum(~np.isnan(v)))})"
    linhas = ["# Calibração exploratória dos passos da sonda (proposta para a próxima rodada)", "",
              "20 alvos (sementes >= 2000, fora das de avaliação), rede nova, protocolo da sonda "
              f"(Adam lr {config.LR}, lotes de {config.SONDA_BATCH}, MSE), perda nos {config.N} pontos a cada passo.", "",
              f"- perda inicial (passo 0): mediana {np.median(curvas[:, 0]):.3g} [mín {curvas[:, 0].min():.3g}, máx {curvas[:, 0].max():.3g}]",
              f"- perda no passo {PASSOS}: mediana {np.median(curvas[:, -1]):.3g}; razão para a inicial: "
              f"mediana {np.median(razao[:, -1]):.2g} [mín {razao[:, -1].min():.2g}, máx {razao[:, -1].max():.2g}]",
              f"- primeiro passo com perda <= 30% da inicial: {f(s30)}",
              f"- primeiro passo com perda <= 20% da inicial: {f(s20)}",
              f"- primeiro passo com perda <= 10% da inicial: {f(s10)}", ""]
    ent = (np.nanmedian(s30), np.nanmedian(s10))
    linhas += [f"**Leitura:** a rede nova já está entre 30% e 10% da perda inicial entre os passos ~{ent[0]:.0f} e ~{ent[1]:.0f}; "
               f"os {PASSOS} passos atuais levam a perda a {np.median(razao[:, -1]) * 100:.2g}% da inicial, ou seja, a sonda **satura** "
               "(o denominador deixa de discriminar). **Proposta:** usar um número de passos dentro dessa faixa, "
               f"por exemplo {int(round(np.sqrt(ent[0] * ent[1]) / 5) * 5)} (média geométrica, arredondada a 5), e validar "
               "com redes treinadas que a cópia também não satura nem fica longe do alvo nesse número de passos."]
    open("analise_exploratoria_sonda_passos.md", "w").write("\n".join(linhas) + "\n")
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
