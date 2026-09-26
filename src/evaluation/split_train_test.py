"""
Split treino/teste estratificado por Label (Fase 5).

Executado UMA ÚNICA VEZ. Gera duas listas de BlockId (treino e teste)
que devem ser reutilizadas por todas as abordagens (Isolation Forest,
Random Forest, LLM) para garantir comparação justa sobre o mesmo
conjunto de teste.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2


def split_train_test(blocks_csv_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_block_ids.csv"
    test_path = output_dir / "test_block_ids.csv"

    if train_path.exists() or test_path.exists():
        raise FileExistsError(
            f"Split já existe em {output_dir}. O split deve ser gerado uma "
            f"única vez. Apague os arquivos manualmente se realmente "
            f"precisar refazer o split (isso invalida qualquer resultado "
            f"já obtido com o split anterior)."
        )

    df = pd.read_csv(blocks_csv_path, usecols=["BlockId", "Label"])

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["Label"],
        random_state=RANDOM_STATE,
    )

    train_df[["BlockId", "Label"]].to_csv(train_path, index=False)
    test_df[["BlockId", "Label"]].to_csv(test_path, index=False)

    print(f"Split concluído (random_state={RANDOM_STATE}, test_size={TEST_SIZE})")
    print(f"\nTreino: {len(train_df)} blocos")
    print(train_df["Label"].value_counts())
    print(f"  Proporção Anomaly: {(train_df['Label'] == 'Anomaly').mean():.4%}")

    print(f"\nTeste: {len(test_df)} blocos")
    print(test_df["Label"].value_counts())
    print(f"  Proporção Anomaly: {(test_df['Label'] == 'Anomaly').mean():.4%}")

    print(f"\nSalvos em:\n  {train_path}\n  {test_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="data/processed/blocks_sequences.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    split_train_test(args.blocks, args.output_dir)