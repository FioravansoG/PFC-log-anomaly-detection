"""
Agrupamento das linhas parseadas por BlockId (Fase 4).

Para cada BlockId, constrói a sequência ordenada de EventId (na ordem
de LineId, que preserva a ordem temporal original do log) e anexa o
rótulo Normal/Anomaly via merge com anomaly_label.csv.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.ingestion.label_loader import load_labels


def build_block_sequences(parsed_csv_path, labels_csv_path, output_path):
    start = time.time()

    print("Carregando linhas parseadas (apenas LineId, BlockId, EventId)...")
    df = pd.read_csv(
        parsed_csv_path,
        usecols=["LineId", "BlockId", "EventId"],
        dtype={"LineId": "int64", "BlockId": "string", "EventId": "int32"},
    )

    # Linhas sem BlockId (ex: eventos administrativos não associados a
    # nenhum bloco específico) não participam do agrupamento por bloco.
    n_total = len(df)
    df = df.dropna(subset=["BlockId"])
    df = df[df["BlockId"] != ""]
    n_sem_block = n_total - len(df)
    print(f"Linhas totais: {n_total} | sem BlockId: {n_sem_block}")

    # Garante ordem cronológica dentro de cada bloco antes de agrupar.
    df = df.sort_values(["BlockId", "LineId"])

    print("Agrupando por BlockId...")
    grouped = (
        df.groupby("BlockId")["EventId"]
        .apply(lambda s: " ".join(s.astype(str)))
        .reset_index()
        .rename(columns={"EventId": "EventSequence"})
    )
    grouped["SequenceLength"] = grouped["EventSequence"].str.split().str.len()

    print("Anexando rótulos (anomaly_label.csv)...")
    labels_df = load_labels(labels_csv_path)
    result = grouped.merge(labels_df, on="BlockId", how="left")

    n_sem_label = result["Label"].isna().sum()
    if n_sem_label > 0:
        print(f"AVISO: {n_sem_label} blocos sem rótulo correspondente em anomaly_label.csv.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    elapsed = time.time() - start
    print(f"\nConcluído em {elapsed:.1f}s.")
    print(f"Total de blocos: {len(result)}")
    print(result["Label"].value_counts())
    print(f"Salvo em: {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", default="data/processed/hdfs_parsed.csv")
    parser.add_argument("--labels", default="data/raw/anomaly_label.csv")
    parser.add_argument("--output", default="data/processed/blocks_sequences.csv")
    args = parser.parse_args()

    build_block_sequences(args.parsed, args.labels, args.output)