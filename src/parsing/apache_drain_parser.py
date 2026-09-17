"""
Parsing do access_log do Apache (AIT-LDS v2.0) usando Drain3.

Ao contrário do HDFS, aqui cada linha do log já é a unidade de análise
(não há agrupamento por bloco): uma linha = uma requisição HTTP. O
Drain3 roda sobre o método + path + protocolo de cada requisição (a
parte com estrutura fixa+variável mais próxima do "Content" usado no
parsing do HDFS), com uma instância NOVA do TemplateMiner -- não
reaproveita o state persistido do parsing do HDFS.

Extrai também, diretamente da linha, os atributos auxiliares (código
de status, tamanho da resposta, tamanho da URL, nº de parâmetros de
query, tamanho do user-agent, método HTTP) que alimentam a etapa de
extração de características (src/features/apache_vectorizer.py).

DESTINO: src/parsing/apache_drain_parser.py

read_raw_log() é reaproveitado sem qualquer alteração a partir de
src/ingestion/loader.py -- já é genérico o bastante (só lê LineId +
RawContent), não é específico do formato do HDFS.
"""

import csv
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ingestion.loader import read_raw_log

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction


COMBINED_LOG_PATTERN = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
    r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>[^"]+)" '
    r'(?P<status>\d+) (?P<size>\S+) '
    r'"(?P<referer>[^"]*)" "(?P<useragent>[^"]*)"'
)


def build_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False

    config.masking_instructions = [
        MaskingInstruction(r"\?[^\s\"]*", "QUERYSTRING"),
        MaskingInstruction(r"\b\d+\b", "NUM"),
    ]

    return TemplateMiner(config=config)


def parse_line(raw_content: str):
    """Extrai os campos do Apache Combined Log Format de uma linha bruta."""
    match = COMBINED_LOG_PATTERN.match(raw_content)
    if not match:
        return None
    return match.groupdict()


def run_parsing(log_path, output_path):
    miner = build_template_miner()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_unmatched = 0
    start = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "LineId", "EventId", "EventTemplate", "Path", "StatusCode", "ResponseSize",
            "UrlLength", "NQueryParams", "UserAgentLength", "MethodGet",
            "MethodPost", "ParsedOk",
        ])

        for row in read_raw_log(log_path):
            parsed = parse_line(row["RawContent"])

            if parsed is None:
                # linha fora do formato esperado -- registrada, não descartada
                writer.writerow([row["LineId"], -1, "", "", 0, 0, 0, 0, 0, 0, 0, 0])
                n_unmatched += 1
                continue

            content = f'{parsed["method"]} {parsed["path"]} {parsed["protocol"]}'
            result = miner.add_log_message(content)

            path = parsed["path"]
            size_str = parsed["size"].replace("-", "0")

            writer.writerow([
                row["LineId"],
                result["cluster_id"],
                result["template_mined"],
                path,
                int(parsed["status"]),
                int(size_str) if size_str.isdigit() else 0,
                len(path),
                path.count("&") + (1 if "?" in path else 0),
                len(parsed["useragent"]),
                1 if parsed["method"] == "GET" else 0,
                1 if parsed["method"] == "POST" else 0,
                1,
            ])

            n_processed += 1

    elapsed = time.time() - start
    print(f"Concluído: {n_processed} linhas processadas, {n_unmatched} sem match do padrão de linha.")
    print(f"Templates distintos (clusters Drain3): {len(miner.drain.clusters)}")
    print(f"Tempo total: {elapsed:.1f}s")

    templates_output = output_path.parent / "apache_templates_gerados.csv"
    with templates_output.open("w", newline="", encoding="utf-8") as tf:
        writer = csv.writer(tf)
        writer.writerow(["EventId", "EventTemplate", "Occurrences"])
        for cluster in miner.drain.clusters:
            writer.writerow([cluster.cluster_id, cluster.get_template(), cluster.size])
    print(f"Templates salvos em: {templates_output}")

    return miner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="data/raw/apache/intranet.smith.russellmitchell.com-access.log.2",
    )
    parser.add_argument("--output", default="data/processed/apache_parsed.csv")
    args = parser.parse_args()

    run_parsing(args.log, args.output)