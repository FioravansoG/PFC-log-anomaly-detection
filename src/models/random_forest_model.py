"""
Modelo 2 — Random Forest (abordagem supervisionada).

Treinado com o conjunto de treino completo (blocos Normal + Anomaly),
com rótulos fornecidos durante o ajuste. Usa class_weight="balanced"
para compensar o desbalanceamento natural das classes (~2,9% de
anomalias), evitando que o modelo favorece trivialmente a classe
majoritária.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42


def load_features(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("BlockId", "Label")]
    X = df[feature_cols].values
    y = df["Label"].values
    return df["BlockId"].values, X, y, feature_cols


def train_random_forest(train_features_path):
    block_ids, X_train, y_train, feature_cols = load_features(train_features_path)

    print(f"Treinando Random Forest com {X_train.shape[0]} blocos "
          f"({(y_train == 'Anomaly').sum()} Anomaly, {(y_train == 'Normal').sum()} Normal).")

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model, feature_cols


def predict_random_forest(model, feature_cols, features_path):
    block_ids, X, y_true, _ = load_features(features_path)

    start = time.time()
    classification = model.predict(X)
    proba = model.predict_proba(X)
    elapsed = time.time() - start

    inference_time_per_block = elapsed / len(X)

    # Probabilidade estimada de a amostra ser Anomaly (coluna correspondente
    # à classe "Anomaly" no predict_proba).
    anomaly_col_idx = list(model.classes_).index("Anomaly")
    anomaly_proba = proba[:, anomaly_col_idx]

    result = pd.DataFrame({
        "BlockId": block_ids,
        "AnomalyProbability": anomaly_proba,
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
    parser.add_argument("--output", default="data/processed/predictions_random_forest.csv")
    args = parser.parse_args()

    model, feature_cols = train_random_forest(args.train)
    result = predict_random_forest(model, feature_cols, args.test)

    result.to_csv(args.output, index=False)

    print(f"\nPredições salvas em: {args.output}")
    print("\nDistribuição das classificações no teste:")
    print(result["Classification"].value_counts())
    print("\nMatriz de confusão simples (contagem):")
    print(pd.crosstab(result["Label"], result["Classification"]))

    # Importância das features — útil para discussão posterior (quais
    # EventIds mais contribuem para a decisão do modelo).
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 features mais importantes:")
    print(importances.head(10))