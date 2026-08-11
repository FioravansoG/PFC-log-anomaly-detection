"""
Avaliação comparativa das três abordagens (Fase 9).

Calcula Precision, Recall, F1-score, FPR e tempo médio de inferência
para cada abordagem, usando exclusivamente a classificação final
(NORMAL/ANOMALY) contra o ground truth do HDFS — nunca comparando
scores/confidence/probabilidades diretamente entre modelos, conforme
orientação do professor.

Gera duas tabelas:
  1. IF e RF sobre o conjunto de teste COMPLETO (mais robusto).
  2. IF, RF e LLM sobre a AMOSTRA de 1.000 blocos (comparação justa
     entre as três abordagens, usando o mesmo conjunto de BlockIds).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


POSITIVE_LABEL = "Anomaly"


def compute_metrics(df: pd.DataFrame, label_col="Label", pred_col="Classification") -> dict:
    """
    Calcula Precision, Recall, F1 e FPR, tratando 'Anomaly' como classe
    positiva. Linhas com predição nula (ex: falha de parsing da LLM)
    são reportadas separadamente e excluídas do cálculo de métricas.
    """
    n_total = len(df)
    valid = df.dropna(subset=[pred_col])
    n_invalid = n_total - len(valid)

    y_true = valid[label_col]
    y_pred = valid[pred_col]

    precision = precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=["Normal", "Anomaly"])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    avg_inference_time = None
    if "InferenceTimeSec" in valid.columns:
        avg_inference_time = valid["InferenceTimeSec"].mean()

    return {
        "N_Total": n_total,
        "N_Invalido": n_invalid,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TempoMedioInferenciaSeg": avg_inference_time,
    }


def evaluate_full_test(if_preds_path, rf_preds_path):
    if_df = pd.read_csv(if_preds_path)
    rf_df = pd.read_csv(rf_preds_path)

    rows = {
        "Isolation Forest (nao supervisionado)": compute_metrics(if_df),
        "Random Forest (supervisionado)": compute_metrics(rf_df),
    }
    return pd.DataFrame(rows).T


def evaluate_on_sample(if_preds_path, rf_preds_path, llm_preds_path, sample_ids_path):
    sample_ids = pd.read_csv(sample_ids_path, usecols=["BlockId"])

    if_df = pd.read_csv(if_preds_path).merge(sample_ids, on="BlockId", how="inner")
    rf_df = pd.read_csv(rf_preds_path).merge(sample_ids, on="BlockId", how="inner")
    llm_df = pd.read_csv(llm_preds_path)  # já é a amostra completa

    assert len(if_df) == len(sample_ids), (
        f"Esperado {len(sample_ids)} blocos para IF na amostra, obtido {len(if_df)}."
    )
    assert len(rf_df) == len(sample_ids), (
        f"Esperado {len(sample_ids)} blocos para RF na amostra, obtido {len(rf_df)}."
    )

    rows = {
        "Isolation Forest (nao supervisionado)": compute_metrics(if_df),
        "Random Forest (supervisionado)": compute_metrics(rf_df),
        "LLM (Ollama / qwen2.5-coder:7b)": compute_metrics(llm_df),
    }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--if-preds", default="data/processed/predictions_isolation_forest.csv")
    parser.add_argument("--rf-preds", default="data/processed/predictions_random_forest.csv")
    parser.add_argument("--llm-preds", default="data/processed/predictions_llm.csv")
    parser.add_argument("--sample-ids", default="data/processed/llm_eval_sample_block_ids.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("MÉTRICAS — CONJUNTO DE TESTE COMPLETO (115.013 blocos)")
    print("Apenas Isolation Forest e Random Forest (LLM não rodou sobre o teste completo)")
    print("=" * 70)
    full_results = evaluate_full_test(args.if_preds, args.rf_preds)
    print(full_results.to_string())
    full_results.to_csv(output_dir / "metricas_teste_completo.csv")

    print("\n" + "=" * 70)
    print("MÉTRICAS — AMOSTRA COMUM (1.000 blocos)")
    print("Comparação justa entre as três abordagens sobre o mesmo conjunto de BlockIds")
    print("=" * 70)
    sample_results = evaluate_on_sample(
        args.if_preds, args.rf_preds, args.llm_preds, args.sample_ids
    )
    print(sample_results.to_string())
    sample_results.to_csv(output_dir / "metricas_amostra_comum.csv")

    print(f"\nSalvos em:\n  {output_dir / 'metricas_teste_completo.csv'}\n  {output_dir / 'metricas_amostra_comum.csv'}")