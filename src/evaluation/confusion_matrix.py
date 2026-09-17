"""
Matrizes de confusão para as abordagens de detecção de anomalias
(Fase 9, complementando metrics.py).

Gera, para um conjunto de predições (uma ou mais abordagens), a
matriz de confusão anotada de cada uma, lado a lado em uma única
figura -- substituindo os gráficos de barras usados anteriormente
para apresentar os resultados sobre o conjunto de teste completo e
sobre a amostra comum.

Reaproveita a mesma convenção de rótulos de metrics.py: "Normal" /
"Anomaly", com "Anomaly" como classe positiva. Funciona tanto para o
HDFS quanto para a base Apache, sem nenhuma alteração -- só muda quais
arquivos de predição são passados.

DESTINO: src/evaluation/confusion_matrix.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


LABELS = ["Normal", "Anomaly"]


def plot_confusion_matrix(y_true, y_pred, titulo, ax):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, ax=ax, cbar=False,
    )
    ax.set_title(titulo)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    return cm


def generate_confusion_matrices(
    predicoes: dict,
    output_path,
    label_col: str = "Label",
    pred_col: str = "Classification",
    sample_ids_path=None,
):
    """
    predicoes: {"Nome exibido no titulo": "caminho/do/csv/de/predicoes.csv"}

    sample_ids_path: se fornecido (ex: llm_eval_sample_block_ids.csv),
    filtra cada CSV de predições para conter só os BlockIds da amostra
    antes de calcular a matriz -- necessário para comparar com a LLM,
    que só roda sobre a amostra de 1.000 blocos no HDFS.
    """
    sample_ids = None
    if sample_ids_path:
        sample_ids = pd.read_csv(sample_ids_path, usecols=["BlockId"])

    n = len(predicoes)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (nome, caminho) in zip(axes, predicoes.items()):
        df = pd.read_csv(caminho)
        if sample_ids is not None:
            df = df.merge(sample_ids, on="BlockId", how="inner")
        df = df.dropna(subset=[pred_col])
        plot_confusion_matrix(df[label_col], df[pred_col], nome, ax)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Figura salva em: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["hdfs", "hdfs-sample", "apache", "apache-llm"], required=True
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sample_ids_path = None

    if args.dataset == "hdfs":
        predicoes = {
            "Isolation Forest (HDFS, teste completo)": "data/processed/predictions_isolation_forest.csv",
            "Random Forest (HDFS, teste completo)": "data/processed/predictions_random_forest.csv",
        }
        output = args.output or "data/processed/matrizes_confusao_hdfs_completo.png"

    elif args.dataset == "hdfs-sample":
        predicoes = {
            "Isolation Forest (amostra 1000)": "data/processed/predictions_isolation_forest.csv",
            "Random Forest (amostra 1000)": "data/processed/predictions_random_forest.csv",
            "LLM (amostra 1000)": "data/processed/predictions_llm.csv",
        }
        sample_ids_path = "data/processed/llm_eval_sample_block_ids.csv"
        output = args.output or "data/processed/matrizes_confusao_hdfs_amostra.png"

    elif args.dataset == "apache":
        predicoes = {
            "Isolation Forest (Apache)": "data/processed/apache_predictions_isolation_forest.csv",
            "Random Forest (Apache)": "data/processed/apache_predictions_random_forest.csv",
        }
        output = args.output or "data/processed/matrizes_confusao_apache.png"

    else:  # apache-llm
        predicoes = {
            "Isolation Forest (Apache)": "data/processed/apache_predictions_isolation_forest.csv",
            "Random Forest (Apache)": "data/processed/apache_predictions_random_forest.csv",
            "LLM (Apache)": "data/processed/apache_predictions_llm_v2.csv",
        }
        output = args.output or "data/processed/matrizes_confusao_apache_com_llm.png"

    generate_confusion_matrices(predicoes, output, sample_ids_path=sample_ids_path)