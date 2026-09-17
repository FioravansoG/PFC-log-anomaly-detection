"""
Módulo de ingestão dos rótulos de ataque do AIT-LDS (access_log do Apache).

Lê o arquivo de rótulos em JSON-lines (formato do AIT-LDS: um objeto por
linha rotulada, referenciando o número da linha no log original) e
retorna o mapeamento LineId -> Label ("Normal" / "Anomaly"), análogo ao
load_labels_as_dict() do HDFS (src/ingestion/label_loader.py), mas
adaptado ao esquema de rótulo por linha em vez de por BlockId.

Linhas do access_log sem entrada correspondente no arquivo de rótulos
são consideradas Normal, conforme documentação oficial do AIT-LDS.

DESTINO: src/ingestion/apache_label_loader.py
"""

import json
from pathlib import Path

import pandas as pd


def load_apache_attack_lines(label_path: str | Path) -> dict:
    """
    Lê o arquivo de rótulos JSON-lines do AIT-LDS.

    Returns
    -------
    dict
        {LineId (int): set(labels de ataque)} -- apenas para as linhas
        que aparecem no arquivo de rótulos (todas anômalas).
    """
    label_path = Path(label_path)
    tags_por_linha: dict = {}

    with label_path.open(encoding="utf-8", errors="replace") as f:
        for linha_json in f:
            linha_json = linha_json.strip()
            if not linha_json:
                continue
            obj = json.loads(linha_json)
            n = obj["line"]
            tags_por_linha.setdefault(n, set()).update(obj["labels"])

    return tags_por_linha


def build_apache_labels_df(label_path: str | Path, n_total_lines: int) -> pd.DataFrame:
    """
    Constrói o DataFrame de rótulos para TODAS as linhas do access_log
    (1..n_total_lines), análogo ao anomaly_label.csv do HDFS.

    Colunas: BlockId, Label, AttackTags.

    NOTA DE COMPATIBILIDADE: a coluna é chamada "BlockId" (e não
    "LineId") propositalmente, para que os módulos genéricos já
    existentes (src/models/isolation_forest_model.py,
    random_forest_model.py e src/evaluation/metrics.py) funcionem sem
    nenhuma alteração -- eles esperam uma coluna "BlockId" como
    identificador da unidade de classificação, seja ela um bloco
    (HDFS) ou uma linha de requisição HTTP (Apache).
    """
    tags_por_linha = load_apache_attack_lines(label_path)

    rows = []
    for line_id in range(1, n_total_lines + 1):
        tags = tags_por_linha.get(line_id, set())
        rows.append({
            "BlockId": line_id,
            "Label": "Anomaly" if tags else "Normal",
            "AttackTags": ";".join(sorted(tags)),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        default="data/raw/apache/intranet.smith.russellmitchell.com-access.log.2.labels",
    )
    parser.add_argument("--n-lines", type=int, required=True,
                         help="Total de linhas do access_log correspondente")
    args = parser.parse_args()

    df = build_apache_labels_df(args.labels, args.n_lines)
    print(f"Total de linhas: {len(df)}")
    print(df["Label"].value_counts())
    print(f"Proporção Anomaly: {(df['Label'] == 'Anomaly').mean():.4%}")