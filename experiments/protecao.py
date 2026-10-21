"""Proteção de results/ contra apagar ou sobrescrever por engano (incidente de 2026-10-20).

Depois de uma execução completa, arquivos ficam 0444 e pastas 0555: nem `rm` nem uma gravação
de CSV conseguem alterar o que já está lá (exceto root). Para rodar de novo, use outra pasta
(--saida) ou, com autorização do autor, --reabrir-resultados.
"""
import os
import stat
from pathlib import Path


def proteger(raiz: Path) -> int:
    """Torna somente leitura tudo sob `raiz`. Devolve o número de arquivos protegidos."""
    raiz, n = Path(raiz), 0
    for dirpath, dirs, files in os.walk(raiz):
        for f in files:
            os.chmod(Path(dirpath) / f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            n += 1
    for dirpath, dirs, files in os.walk(raiz, topdown=False):      # pastas de baixo para cima
        os.chmod(dirpath, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return n


def reabrir(raiz: Path) -> None:
    """Desfaz `proteger` (pastas 0755, arquivos 0644). Decisão do autor, não de um agente."""
    for dirpath, dirs, files in os.walk(raiz):
        os.chmod(dirpath, 0o755)
        for f in files:
            os.chmod(Path(dirpath) / f, 0o644)


def protegido(raiz: Path) -> bool:
    """True se `raiz` existe e não aceita gravação."""
    raiz = Path(raiz)
    return raiz.exists() and not os.access(raiz, os.W_OK)
