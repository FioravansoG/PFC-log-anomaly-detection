"""
Parsing dos logs do HDFS usando Drain3.

Separa o cabeçalho (Date, Time, Pid, Level, Component) do conteúdo da
mensagem — só o conteúdo é alimentado ao Drain3, pois o cabeçalho varia
livremente entre linhas e não deve interferir na mineração de templates.
Extrai o BlockId do conteúdo via regex para uso na Fase 4 (agrupamento).
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


LOG_LINE_PATTERN = re.compile(
    r"^(?P<Date>\d{6})\s+"
    r"(?P<Time>\d{6})\s+"
    r"(?P<Pid>\d+)\s+"
    r"(?P<Level>\w+)\s+"
    r"(?P<Component>[^:]+):\s+"
    r"(?P<Content>.*)$"
)

BLOCK_ID_PATTERN = re.compile(r"(blk_-?\d+)")


def build_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False

    config.masking_instructions = [
        MaskingInstruction(r"blk_-?\d+", "BLOCK_ID"),
        MaskingInstruction(
            r"/?(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?"
            r"(?:[,\s]+(?=\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))?)+",
            "IP",
        ),
        MaskingInstruction(r"java\.[\w.]+(?:Exception|Error)(:.*)?$", "EXCEPTION_DETAIL"),
        MaskingInstruction(r"\b\d+\b", "NUM"),
    ]

    return TemplateMiner(config=config)


def parse_line(raw_content: str):
    """Separa cabeçalho + conteúdo de uma linha bruta do HDFS.log."""
    match = LOG_LINE_PATTERN.match(raw_content)
    return match.groupdict() if match else None


def extract_block_id(content: str):
    match = BLOCK_ID_PATTERN.search(content)
    return match.group(1) if match else None


def run_parsing(log_path, output_path, limit=None):
    miner = build_template_miner()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_unmatched = 0
    start = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["LineId", "BlockId", "EventId", "EventTemplate", "Date", "Time"])

        for row in read_raw_log(log_path):
            if limit is not None and n_processed >= limit:
                break

            parsed = parse_line(row["RawContent"])
            if parsed is None:
                n_unmatched += 1
                continue

            block_id = extract_block_id(parsed["Content"])
            result = miner.add_log_message(parsed["Content"])

            writer.writerow([
                row["LineId"],
                block_id or "",
                result["cluster_id"],
                result["template_mined"],
                parsed["Date"],
                parsed["Time"],
            ])

            n_processed += 1
            if n_processed % 500_000 == 0:
                elapsed = time.time() - start
                print(f"{n_processed} linhas processadas ({elapsed:.1f}s)...")

    elapsed = time.time() - start
    print(f"\nConcluído: {n_processed} linhas processadas, {n_unmatched} sem match do padrão de linha.")
    print(f"Templates distintos (clusters Drain3): {len(miner.drain.clusters)}")
    print(f"Tempo total: {elapsed:.1f}s")

    templates_output = output_path.parent / "templates_gerados.csv"
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
    parser.add_argument("--log", default="data/raw/HDFS.log")
    parser.add_argument("--output", default="data/processed/hdfs_parsed.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limitar nº de linhas (teste rápido)")
    args = parser.parse_args()

    run_parsing(args.log, args.output, limit=args.limit)