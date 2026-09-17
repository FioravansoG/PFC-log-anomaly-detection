"""
Catalogação de tipos de anomalia no HDFS via importância de features
do Random Forest (Fase 9, complementando metrics.py e
resample_robustness.py).

O HDFS/Loghub não fornece rótulo de TIPO de anomalia (só Normal/
Anomaly) -- diferente do AIT-LDS (Apache), que já vem com rótulos
nomeados (service_scan, webshell_upload, etc.). Para catalogar mesmo
assim, este módulo usa a importância de features do Random Forest
(já treinado sobre a matriz de contagem de eventos) para identificar
quais EventIds mais discriminam blocos anômalos, e então verifica, em
cada bloco anômalo, quais desses eventos-chave estão presentes.

IMPORTANTE (documentar no relatório): as categorias resultantes são
categorias de FALHA OPERACIONAL do sistema HDFS (ex.: erro ao
excluir um bloco, timeout de replicação), não tipos de ataque
cibernético -- a base não documenta ataques, documenta anomalias
operacionais. Isso contrasta com a catalogação já feita para a base
Apache/AIT-LDS, cujos rótulos SÃO nomeados por etapa de ataque.

Reaproveita train_random_forest() de src/models/random_forest_model.py
sem nenhuma alteração.

DESTINO: src/evaluation/catalogo_anomalias.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.models.random_forest_model import train_random_forest


def load_event_templates(templates_csv_path) -> dict:
    df = pd.read_csv(templates_csv_path, usecols=["EventId", "EventTemplate"])
    return dict(zip(df["EventId"], df["EventTemplate"]))


def get_top_events_by_importance(train_features_path, top_n=15):
    """
    Treina o RF (reaproveitando train_random_forest sem alteração) e
    retorna os top_n EventIds mais importantes, na ordem de
    importância decrescente.
    """
    model, feature_cols = train_random_forest(train_features_path)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    top_features = importances.head(top_n)

    # feature_cols vêm no formato "E<EventId>" (ex: "E22") -- extrai o
    # EventId numérico de volta.
    registros = []
    for rank, (feature_name, importance) in enumerate(top_features.items(), start=1):
        event_id = int(feature_name[1:])  # remove o prefixo "E"
        registros.append({"Rank": rank, "EventId": event_id, "Importance": importance})

    return pd.DataFrame(registros)


def catalog_anomalies(train_features_path, templates_csv_path, blocks_csv_path,
                       top_n=15, output_path=None):
    top_events_df = get_top_events_by_importance(train_features_path, top_n)
    template_map = load_event_templates(templates_csv_path)
    top_events_df["EventTemplate"] = top_events_df["EventId"].map(template_map)

    blocks = pd.read_csv(blocks_csv_path, usecols=["BlockId", "EventSequence", "Label"])
    anomalous_blocks = blocks[blocks["Label"] == "Anomaly"].copy()
    normal_blocks = blocks[blocks["Label"] == "Normal"].copy()
    n_anomalous_total = len(anomalous_blocks)
    n_normal_total = len(normal_blocks)

    print(f"Total de blocos anômalos no HDFS: {n_anomalous_total}")
    print(f"Total de blocos normais no HDFS: {n_normal_total}")

    anomalous_blocks["EventSet"] = anomalous_blocks["EventSequence"].apply(
        lambda seq: set(int(tok) for tok in str(seq).split())
    )
    normal_blocks["EventSet"] = normal_blocks["EventSequence"].apply(
        lambda seq: set(int(tok) for tok in str(seq).split())
    )

    contagens = []
    blocos_cobertos = set()
    for _, row in top_events_df.iterrows():
        eid = row["EventId"]

        mask_anomalo = anomalous_blocks["EventSet"].apply(lambda s: eid in s)
        n_blocos_anomalos = mask_anomalo.sum()
        blocos_cobertos.update(anomalous_blocks.loc[mask_anomalo, "BlockId"])

        mask_normal = normal_blocks["EventSet"].apply(lambda s: eid in s)
        n_blocos_normais = mask_normal.sum()

        pct_anomalo = n_blocos_anomalos / n_anomalous_total if n_anomalous_total else 0.0
        pct_normal = n_blocos_normais / n_normal_total if n_normal_total else 0.0

        contagens.append({
            "Rank": row["Rank"],
            "EventId": eid,
            "EventTemplate": row["EventTemplate"],
            "Importance": row["Importance"],
            "N_Blocos_Anomalos": n_blocos_anomalos,
            "Pct_Blocos_Anomalos": pct_anomalo,
            "Pct_Blocos_Normais": pct_normal,
            "Especificidade": pct_anomalo - pct_normal,
        })

    resultado = pd.DataFrame(contagens)

    n_cobertos = len(blocos_cobertos)
    print(f"\nTop {top_n} eventos por importância no Random Forest:")
    print(resultado.to_string(index=False))
    print(f"\nCobertura combinada (união dos top {top_n} eventos): "
          f"{n_cobertos} de {n_anomalous_total} blocos anômalos "
          f"({100 * n_cobertos / n_anomalous_total:.1f}%) "
          f"-- atenção: inflada por eventos rotineiros presentes em quase "
          f"todo bloco; ver ordenação por Especificidade abaixo.")

    print(f"\nMesmos eventos, ordenados por ESPECIFICIDADE "
          f"(Pct_Blocos_Anomalos - Pct_Blocos_Normais) -- quanto mais "
          f"próximo de 1, mais exclusivo de blocos anômalos:")
    print(resultado.sort_values("Especificidade", ascending=False).to_string(index=False))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resultado.to_csv(output_path, index=False)
        print(f"\nSalvo em: {output_path}")

    return resultado


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default="data/processed/features_train.csv")
    parser.add_argument("--templates", default="data/processed/templates_gerados.csv")
    parser.add_argument("--blocks", default="data/processed/blocks_sequences.csv")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--output", default="data/processed/catalogo_anomalias_hdfs.csv")
    args = parser.parse_args()

    catalog_anomalies(
        args.train_features, args.templates, args.blocks,
        top_n=args.top_n, output_path=args.output,
    )