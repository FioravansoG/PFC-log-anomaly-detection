"""
Orquestração do pipeline completo de detecção de anomalias em logs
(HDFS), do dado bruto às métricas finais.

Uso:
    python main.py                  # roda o pipeline completo, incluindo a LLM (~2h+)
    python main.py --skip-llm       # roda tudo exceto o Modelo 3 (LLM)
    python main.py --only-metrics   # apenas recalcula métricas e alertas
                                       a partir de predições já existentes

Cada fase é implementada em seu próprio módulo sob src/; este script
apenas documenta e executa a ordem correta, conforme detalhado em
METODOLOGIA.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.parsing.drain_parser import run_parsing
from src.parsing.grouping import build_block_sequences
from src.evaluation.split_train_test import split_train_test
from src.features.vectorizer import build_event_count_matrix
from src.models.isolation_forest_model import train_isolation_forest, predict_isolation_forest
from src.models.random_forest_model import train_random_forest, predict_random_forest
from src.evaluation.sample_test_for_llm import sample_test_set
from src.features.text_representation import build_text_representations
from src.models.llm_classifier import run_llm_classification
from src.alerts.alert_generator import generate_alerts
from src.evaluation.metrics import evaluate_full_test, evaluate_on_sample


DATA_RAW = Path("data/raw")
DATA_PROC = Path("data/processed")


def step(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_pipeline(skip_llm: bool, only_metrics: bool):
    if only_metrics:
        step("Recalculando alertas e métricas a partir de predições existentes")
        for model, preds_file in [
            ("isolation_forest", "predictions_isolation_forest.csv"),
            ("random_forest", "predictions_random_forest.csv"),
            ("llm", "predictions_llm.csv"),
        ]:
            preds_path = DATA_PROC / preds_file
            if preds_path.exists():
                generate_alerts(
                    preds_path, model, DATA_PROC / "blocks_sequences.csv",
                    DATA_PROC / f"alerts_{model}.json",
                )
            else:
                print(f"Aviso: {preds_path} não encontrado, pulando alertas de '{model}'.")

        full = evaluate_full_test(
            DATA_PROC / "predictions_isolation_forest.csv",
            DATA_PROC / "predictions_random_forest.csv",
        )
        print(full.to_string())

        if (DATA_PROC / "predictions_llm.csv").exists():
            sample = evaluate_on_sample(
                DATA_PROC / "predictions_isolation_forest.csv",
                DATA_PROC / "predictions_random_forest.csv",
                DATA_PROC / "predictions_llm.csv",
                DATA_PROC / "llm_eval_sample_block_ids.csv",
            )
            print(sample.to_string())
        return

    # Fase 3 — Parsing
    parsed_path = DATA_PROC / "hdfs_parsed.csv"
    if not parsed_path.exists():
        step("Fase 3 — Parsing (Drain3)")
        run_parsing(DATA_RAW / "HDFS.log", parsed_path)
    else:
        print(f"[Fase 3] {parsed_path} já existe, pulando.")

    # Fase 4 — Agrupamento por BlockId
    blocks_path = DATA_PROC / "blocks_sequences.csv"
    if not blocks_path.exists():
        step("Fase 4 — Agrupamento por BlockId")
        build_block_sequences(parsed_path, DATA_RAW / "anomaly_label.csv", blocks_path)
    else:
        print(f"[Fase 4] {blocks_path} já existe, pulando.")

    # Fase 5 — Split treino/teste (uma única vez)
    train_ids_path = DATA_PROC / "train_block_ids.csv"
    test_ids_path = DATA_PROC / "test_block_ids.csv"
    if not (train_ids_path.exists() and test_ids_path.exists()):
        step("Fase 5 — Split treino/teste")
        split_train_test(blocks_path, DATA_PROC)
    else:
        print(f"[Fase 5] Split já existe, pulando.")

    # Fase 6 — Features (matriz bloco x evento)
    features_train_path = DATA_PROC / "features_train.csv"
    features_test_path = DATA_PROC / "features_test.csv"
    if not (features_train_path.exists() and features_test_path.exists()):
        step("Fase 6 — Extração de características (matriz bloco x evento)")
        build_event_count_matrix(blocks_path, train_ids_path, test_ids_path, DATA_PROC)
    else:
        print(f"[Fase 6] Matriz de features já existe, pulando.")

    # Fase 7 — Modelo 1: Isolation Forest
    if_preds_path = DATA_PROC / "predictions_isolation_forest.csv"
    if not if_preds_path.exists():
        step("Fase 7 — Modelo 1: Isolation Forest")
        model, cols = train_isolation_forest(features_train_path)
        result = predict_isolation_forest(model, cols, features_test_path)
        result.to_csv(if_preds_path, index=False)
    else:
        print(f"[Fase 7] {if_preds_path} já existe, pulando.")

    # Fase 7 — Modelo 2: Random Forest
    rf_preds_path = DATA_PROC / "predictions_random_forest.csv"
    if not rf_preds_path.exists():
        step("Fase 7 — Modelo 2: Random Forest")
        model, cols = train_random_forest(features_train_path)
        result = predict_random_forest(model, cols, features_test_path)
        result.to_csv(rf_preds_path, index=False)
    else:
        print(f"[Fase 7] {rf_preds_path} já existe, pulando.")

    # Fase 7 — Modelo 3: LLM (opcional, ~2h)
    llm_sample_ids_path = DATA_PROC / "llm_eval_sample_block_ids.csv"
    llm_sequences_path = DATA_PROC / "text_sequences_llm_sample.csv"
    llm_preds_path = DATA_PROC / "predictions_llm.csv"

    if skip_llm:
        print("[Fase 7] Modelo 3 (LLM) pulado via --skip-llm.")
    else:
        if not llm_sample_ids_path.exists():
            step("Fase 7 — Amostragem para avaliação da LLM")
            sample_test_set(test_ids_path, llm_sample_ids_path)
        else:
            print(f"[Fase 7] {llm_sample_ids_path} já existe, pulando.")

        if not llm_sequences_path.exists():
            step("Fase 7 — Representação textual para a LLM")
            build_text_representations(
                blocks_path, DATA_PROC / "templates_gerados.csv",
                llm_sample_ids_path, llm_sequences_path,
            )
        else:
            print(f"[Fase 7] {llm_sequences_path} já existe, pulando.")

        step("Fase 7 — Modelo 3: LLM (Ollama) — pode levar ~2h")
        run_llm_classification(llm_sequences_path, llm_preds_path)

    # Fase 8 — Alertas
    step("Fase 8 — Geração de alertas")
    generate_alerts(if_preds_path, "isolation_forest", blocks_path, DATA_PROC / "alerts_isolation_forest.json")
    generate_alerts(rf_preds_path, "random_forest", blocks_path, DATA_PROC / "alerts_random_forest.json")
    if llm_preds_path.exists():
        generate_alerts(llm_preds_path, "llm", blocks_path, DATA_PROC / "alerts_llm.json")

    # Fase 9 — Métricas comparativas
    step("Fase 9 — Métricas comparativas")
    full = evaluate_full_test(if_preds_path, rf_preds_path)
    print("\nTeste completo (IF e RF):")
    print(full.to_string())
    full.to_csv(DATA_PROC / "metricas_teste_completo.csv")

    if llm_preds_path.exists():
        sample = evaluate_on_sample(if_preds_path, rf_preds_path, llm_preds_path, llm_sample_ids_path)
        print("\nAmostra comum (1.000 blocos, três abordagens):")
        print(sample.to_string())
        sample.to_csv(DATA_PROC / "metricas_amostra_comum.csv")

    step("Pipeline concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-llm", action="store_true", help="Pula o Modelo 3 (LLM), que leva ~2h.")
    parser.add_argument("--only-metrics", action="store_true", help="Apenas recalcula alertas e métricas a partir de predições já existentes.")
    args = parser.parse_args()

    run_pipeline(skip_llm=args.skip_llm, only_metrics=args.only_metrics)