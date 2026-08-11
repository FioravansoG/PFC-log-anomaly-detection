"""
Módulo de ingestão dos rótulos de anomalia do HDFS.

Lê o anomaly_label.csv e retorna o mapeamento BlockId -> Label.
"""

from pathlib import Path
import pandas as pd


def load_labels(label_path: str | Path) -> pd.DataFrame:
    """
    Lê o anomaly_label.csv.

    Returns
    -------
    pandas.DataFrame
        Colunas: BlockId, Label ("Normal" / "Anomaly").
    """
    label_path = Path(label_path)
    df = pd.read_csv(label_path)
    df.columns = [c.strip() for c in df.columns]

    expected_cols = {"BlockId", "Label"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Colunas esperadas {expected_cols} não encontradas. "
            f"Colunas presentes: {list(df.columns)}"
        )

    return df[["BlockId", "Label"]]


def load_labels_as_dict(label_path: str | Path) -> dict:
    """Mapeamento BlockId -> Label como dict (lookup O(1) na Fase 4)."""
    df = load_labels(label_path)
    return dict(zip(df["BlockId"], df["Label"]))


if __name__ == "__main__":
    import sys

    label_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/anomaly_label.csv"

    df = load_labels(label_file)
    print(f"Total de BlockIds únicos: {df['BlockId'].nunique()}")
    print(df["Label"].value_counts())