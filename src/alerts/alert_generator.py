"""
Geração de alertas a partir das predições de qualquer uma das três
abordagens (Fase 8).

Cada abordagem produz um "score" com semântica própria (scores de IF,
RF e confidence de LLM não são equivalentes). Este módulo aplica os 
thresholds de severidade de forma independente para cada abordagem — 
não tenta unificar as escalas.

Estrutura do alerta:
{
  "block_id": "...",
  "classification": "ANOMALY",
  "score": 0.87,
  "model": "isolation_forest" | "random_forest" | "llm",
  "events": ["E1", "E3", ...],
  "severity": "LOW" | "MEDIUM" | "HIGH"
}
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd


# Thresholds de severidade — aplicados sobre um score normalizado em [0, 1]
# específico de cada modelo (ver normalize_score por abordagem).
SEVERITY_THRESHOLDS = [
    (0.80, "HIGH"),
    (0.60, "MEDIUM"),
    (0.40, "LOW"),
]


def severity_from_score(score: float) -> str:
    if score is None:
        return "UNKNOWN"
    for threshold, label in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return label
    return "NORMAL"


def normalize_isolation_forest_score(raw_score: float) -> float:
    """
    O decision_function do Isolation Forest retorna valores tipicamente
    entre -0.5 e 0.5 (quanto MENOR, mais anômalo). Normaliza para [0, 1]
    onde 1 = mais anômalo, via inversão de sinal + clip.

    Esta normalização é local a este módulo, apenas para fins de
    severidade do alerta — não deve ser interpretada como probabilidade,
    nem comparada a scores de outras abordagens.
    """
    normalized = 0.5 - raw_score  # inverte: score baixo (anômalo) -> valor alto
    return max(0.0, min(1.0, normalized))


def normalize_random_forest_score(anomaly_probability: float) -> float:
    """RF já produz uma probabilidade em [0, 1] para a classe Anomaly."""
    return max(0.0, min(1.0, anomaly_probability))


def normalize_llm_score(confidence) -> float | None:
    """
    Confidence declarada pela LLM, esperada em [0, 1] por construção do
    prompt. Reforça-se: não é uma probabilidade calibrada.

    A LLM demonstrou inconsistência de formatação na prática: parte das
    respostas retorna confidence como fração decimal (ex: 0.9) e parte
    como porcentagem com símbolo (ex: "90%") — mesmo com prompt idêntico.
    Esta função normaliza ambos os formatos para a escala [0, 1].
    """
    if confidence is None:
        return None

    if isinstance(confidence, str):
        confidence = confidence.strip()
        if confidence.endswith("%"):
            try:
                value = float(confidence.rstrip("%")) / 100.0
            except ValueError:
                return None
        else:
            try:
                value = float(confidence)
            except ValueError:
                return None
    else:
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return None

    return max(0.0, min(1.0, value))


NORMALIZERS = {
    "isolation_forest": normalize_isolation_forest_score,
    "random_forest": normalize_random_forest_score,
    "llm": normalize_llm_score,
}

SCORE_COLUMNS = {
    "isolation_forest": "Score",
    "random_forest": "AnomalyProbability",
    "llm": "Confidence",
}


def generate_alerts(predictions_path, model_name, blocks_sequences_path, output_path):
    if model_name not in NORMALIZERS:
        raise ValueError(f"model_name deve ser um de {list(NORMALIZERS.keys())}")

    preds = pd.read_csv(predictions_path)
    blocks = pd.read_csv(blocks_sequences_path, usecols=["BlockId", "EventSequence"])

    df = preds.merge(blocks, on="BlockId", how="left")

    normalizer = NORMALIZERS[model_name]
    score_col = SCORE_COLUMNS[model_name]

    alerts = []
    for _, row in df.iterrows():
        if row["Classification"] != "Anomaly":
            continue  # alertas só são gerados para classificação ANOMALY

        raw_score = row.get(score_col)
        norm_score = normalizer(raw_score) if pd.notna(raw_score) else None
        severity = severity_from_score(norm_score)

        events = str(row.get("EventSequence", "")).split()
        events_labeled = [f"E{e}" for e in events]

        alert = {
            "block_id": row["BlockId"],
            "classification": "ANOMALY",
            "score": round(norm_score, 4) if norm_score is not None else None,
            "model": model_name,
            "events": events_labeled,
            "severity": severity,
        }
        alerts.append(alert)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)

    print(f"Alertas gerados para '{model_name}': {len(alerts)} de {len(df)} blocos.")
    if alerts:
        severities = pd.Series([a["severity"] for a in alerts]).value_counts()
        print(severities)
    print(f"Salvo em: {output_path}")

    return alerts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["isolation_forest", "random_forest", "llm"]
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--blocks", default="data/processed/blocks_sequences.csv"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    generate_alerts(args.predictions, args.model, args.blocks, args.output)