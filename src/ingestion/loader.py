"""
Módulo de ingestão do log bruto do HDFS.

Lê o arquivo de log linha a linha (streaming), sem carregar o arquivo
inteiro em memória, retornando um gerador de LineId + RawContent.
"""

from pathlib import Path
from typing import Iterator, Dict, Optional
import pandas as pd


def read_raw_log(log_path: str | Path) -> Iterator[Dict[str, object]]:
    """
    Lê o arquivo de log linha a linha em modo streaming.

    Parameters
    ----------
    log_path : str | Path
        Caminho para o log bruto (ex: data/raw/HDFS.log).

    Yields
    ------
    dict
        {"LineId": int, "RawContent": str}
    """
    log_path = Path(log_path)

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_id, line in enumerate(f, start=1):
            content = line.rstrip("\n")
            if not content:
                continue
            yield {"LineId": line_id, "RawContent": content}


def load_raw_log_as_dataframe(
    log_path: str | Path, limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Carrega o log bruto como DataFrame.

    ATENÇÃO: só usar para amostras/testes. Para o HDFS.log completo
    (~11M linhas), prefira iterar com read_raw_log() diretamente.
    """
    rows = []
    for i, row in enumerate(read_raw_log(log_path)):
        if limit is not None and i >= limit:
            break
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    log_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/HDFS.log"

    count = 0
    for _ in read_raw_log(log_file):
        count += 1

    print(f"Total de linhas lidas em {log_file}: {count}")