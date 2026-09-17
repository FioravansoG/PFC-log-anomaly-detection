"""
Construção da matriz de características por linha do access_log do
Apache (adaptado de vectorizer.py + split_train_test.py do HDFS).

Ao contrário do HDFS, aqui não existe uma sequência de eventos por
bloco -- cada linha é uma única ocorrência de um evento (EventId), e o
"vetor de contagem de eventos" degenera para uma codificação one-hot
do EventId daquela linha, combinada com os atributos auxiliares
extraídos no parsing (código de status, tamanho da resposta, etc.).

O vocabulário de EventIds (colunas one-hot) é definido EXCLUSIVAMENTE
a partir do conjunto de treino, pelo mesmo motivo do HDFS: evitar
vazamento de informação do teste para a etapa de modelagem.

Simplificação assumida (documentar no relatório/METODOLOGIA.md): o
split treino/teste é feito aqui mesmo, dentro deste script, em vez de
persistido antecipadamente em um módulo separado como split_train_test.py
faz para o HDFS -- como a base é bem menor (8.530 linhas), não há o
mesmo risco prático de custo de reprocessamento que motivou separar
essa etapa no HDFS.

DESTINO: src/features/apache_vectorizer.py

Saída: apache_features_train.csv / apache_features_test.csv, com o
MESMO formato (colunas BlockId, <features>, Label) que
src/models/isolation_forest_model.py, random_forest_model.py e
src/evaluation/metrics.py já esperam -- portanto esses três arquivos
são reaproveitados SEM NENHUMA ALTERAÇÃO, só apontando --train/--test/
--output para os novos CSVs.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.ingestion.apache_label_loader import build_apache_labels_df


RANDOM_STATE = 42
TEST_SIZE = 0.2

AUX_FEATURE_COLS = [
    "StatusCode", "ResponseSize", "UrlLength", "NQueryParams",
    "UserAgentLength", "MethodGet", "MethodPost", "ParsedOk",
]


def build_apache_feature_matrix(parsed_csv_path, labels_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = pd.read_csv(parsed_csv_path)
    n_total_lines = len(parsed)

    labels_df = build_apache_labels_df(labels_path, n_total_lines)
    df = parsed.merge(labels_df, left_on="LineId", right_on="BlockId", how="inner")

    print(f"Total de linhas: {len(df)} | Anômalas: {(df['Label'] == 'Anomaly').sum()} "
          f"({(df['Label'] == 'Anomaly').mean():.1%})")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["Label"], random_state=RANDOM_STATE
    )
    print(f"Treino: {len(train_df)} | Teste: {len(test_df)} "
          f"(random_state={RANDOM_STATE}, test_size={TEST_SIZE})")

    # one-hot do EventId -- vocabulário definido só a partir do treino
    train_event_dummies = pd.get_dummies(train_df["EventId"], prefix="E")
    known_events = train_event_dummies.columns

    test_event_dummies = pd.get_dummies(test_df["EventId"], prefix="E")
    test_event_dummies = test_event_dummies.reindex(columns=known_events, fill_value=0)

    def montar_saida(base_df, event_dummies):
        saida = pd.concat(
            [base_df[AUX_FEATURE_COLS].reset_index(drop=True),
             event_dummies.reset_index(drop=True)],
            axis=1,
        )
        saida.insert(0, "BlockId", base_df["LineId"].values)
        saida["Label"] = base_df["Label"].values
        return saida

    train_matrix = montar_saida(train_df, train_event_dummies)
    test_matrix = montar_saida(test_df, test_event_dummies)

    train_out = output_dir / "apache_features_train.csv"
    test_out = output_dir / "apache_features_test.csv"
    train_matrix.to_csv(train_out, index=False)
    test_matrix.to_csv(test_out, index=False)

    print(f"\nSalvos em:\n  {train_out}\n  {test_out}")
    return train_matrix, test_matrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", default="data/processed/apache_parsed.csv")
    parser.add_argument(
        "--labels",
        default="data/raw/apache/intranet.smith.russellmitchell.com-access.log.2.labels",
    )
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    build_apache_feature_matrix(args.parsed, args.labels, args.output_dir)