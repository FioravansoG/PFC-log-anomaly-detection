"""
Construção da matriz bloco x evento (contagem), a partir das sequências
geradas na Fase 4 (Fase 6, características para os modelos de ML).

O vocabulário de colunas (EventIds) é definido EXCLUSIVAMENTE a partir
do conjunto de treino, para evitar vazamento de informação do teste
para a etapa de modelagem.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def _sequences_as_tokens(series: pd.Series) -> pd.Series:
    """EventSequence já vem como string 'E1 E3 E1 ...' -> apenas garante str."""
    return series.astype(str)


def build_event_count_matrix(
    blocks_csv_path,
    train_ids_path,
    test_ids_path,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = pd.read_csv(
        blocks_csv_path, usecols=["BlockId", "EventSequence", "Label"]
    )
    train_ids = pd.read_csv(train_ids_path, usecols=["BlockId"])
    test_ids = pd.read_csv(test_ids_path, usecols=["BlockId"])

    train_blocks = blocks.merge(train_ids, on="BlockId", how="inner")
    test_blocks = blocks.merge(test_ids, on="BlockId", how="inner")

    print(f"Blocos de treino: {len(train_blocks)} | Blocos de teste: {len(test_blocks)}")

    # CountVectorizer com tokenização simples por espaço: cada EventId é um
    # "token" (ex: "1 3 1 1 5" -> contagem de ocorrência de cada EventId).
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
    )

    X_train = vectorizer.fit_transform(_sequences_as_tokens(train_blocks["EventSequence"]))
    X_test = vectorizer.transform(_sequences_as_tokens(test_blocks["EventSequence"]))

    print(f"Vocabulário (EventIds distintos no treino): {len(vectorizer.vocabulary_)}")

    feature_names = [f"E{tok}" for tok in vectorizer.get_feature_names_out()]

    train_matrix = pd.DataFrame(X_train.toarray(), columns=feature_names)
    train_matrix.insert(0, "BlockId", train_blocks["BlockId"].values)
    train_matrix["Label"] = train_blocks["Label"].values

    test_matrix = pd.DataFrame(X_test.toarray(), columns=feature_names)
    test_matrix.insert(0, "BlockId", test_blocks["BlockId"].values)
    test_matrix["Label"] = test_blocks["Label"].values

    train_out = output_dir / "features_train.csv"
    test_out = output_dir / "features_test.csv"
    train_matrix.to_csv(train_out, index=False)
    test_matrix.to_csv(test_out, index=False)

    print(f"\nSalvos em:\n  {train_out}\n  {test_out}")
    return train_matrix, test_matrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="data/processed/blocks_sequences.csv")
    parser.add_argument("--train-ids", default="data/processed/train_block_ids.csv")
    parser.add_argument("--test-ids", default="data/processed/test_block_ids.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    build_event_count_matrix(args.blocks, args.train_ids, args.test_ids, args.output_dir)