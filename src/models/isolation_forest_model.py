"""
Modelo 1 — Isolation Forest (abordagem não supervisionada).

Treinado exclusivamente sobre blocos NORMAIS do conjunto de treino,
sem qualquer rótulo fornecido durante o ajuste (natureza não
supervisionada). O threshold de decisão é o padrão do scikit-learn
(decision_function = 0), sem uso de `contamination` baseado na
proporção real de anomalias, para não introduzir informação
supervisionada indireta no ajuste do modelo.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.ensemble import IsolationForest


RANDOM_STATE = 42


def load_features(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("BlockId", "Label")]
    X = df[feature_cols].values
    y = df["Label"].values
    return df["BlockId"].values, X, y, feature_cols


def train_isolation_forest(train_features_path):
    block_ids, X_train, y_train, feature_cols = load_features(train_features_path)

    # Treina SOMENTE com blocos normais do treino — o modelo aprende o
    # perfil de comportamento normal e nunca vê exemplos anômalos.
    normal_mask = y_train == "Normal"
    X_train_normal = X_train[normal_mask]

    print(f"Treinando Isolation Forest com {X_train_normal.shape[0]} blocos normais "
          f"(de {X_train.shape[0]} blocos totais no treino).")

    model = IsolationForest(random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_normal)

    return model, feature_cols


def predict_isolation_forest(model, feature_cols, features_path):
    block_ids, X, y_true, _ = load_features(features_path)

    start = time.time()
    raw_pred = model.predict(X)  # 1 = normal, -1 = anômalo (default do sklearn)
    scores = model.decision_function(X)  # quanto menor, mais anômalo
    elapsed = time.time() - start

    inference_time_per_block = elapsed / len(X)

    classification = pd.Series(raw_pred).map({1: "Normal", -1: "Anomaly"}).values

    result = pd.DataFrame({
        "BlockId": block_ids,
        "Score": scores,
        "Classification": classification,
        "Label": y_true,
        "InferenceTimeSec": inference_time_per_block,
    })

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/features_train.csv")
    parser.add_argument("--test", default="data/processed/features_test.csv")
    parser.add_argument("--output", default="data/processed/predictions_isolation_forest.csv")
    args = parser.parse_args()

    model, feature_cols = train_isolation_forest(args.train)
    result = predict_isolation_forest(model, feature_cols, args.test)

    result.to_csv(args.output, index=False)

    print(f"\nPredições salvas em: {args.output}")
    print("\nDistribuição das classificações no teste:")
    print(result["Classification"].value_counts())
    print("\nMatriz de confusão simples (contagem):")
    print(pd.crosstab(result["Label"], result["Classification"]))