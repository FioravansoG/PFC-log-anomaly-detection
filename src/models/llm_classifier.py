"""
Modelo 3 — Classificação via LLM local (Ollama), seguindo o formato de
prompt especificado pelo orientador: retorno exclusivo de um JSON com
classification, confidence e explanation.

Executa sobre a AMOSTRA estratificada do teste (não o teste completo),
por inviabilidade computacional de rodar LLM local sobre 115 mil blocos.

Para a avaliação quantitativa (Fase 9), usa-se apenas o campo
`classification`. O campo `confidence` é reportado como produto
adicional do modelo, mas não é comparável ao Score do Isolation Forest
nem à AnomalyProbability do Random Forest — cada valor tem semântica
própria (ver orientações do professor).
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:7b"

# Temperatura progressiva em caso de falha de parsing na tentativa
# anterior — mitiga o problema conhecido de temperature=0.0 causar
# falhas idênticas repetidas em retries.
TEMPERATURE_SCHEDULE = [0.0, 0.3, 0.5]
MAX_RETRIES = len(TEMPERATURE_SCHEDULE)

PROMPT_TEMPLATE = """Você é um classificador de anomalias em logs de sistemas computacionais, especializado em logs do HDFS (Hadoop Distributed File System).

Contexto sobre o funcionamento normal do HDFS:
- Blocos são replicados entre múltiplos DataNodes por padrão, e réplicas excedentes são periodicamente removidas por processos de rebalanceamento (eventos de "delete"/"invalidSet" são rotina, não indicam ataque).
- Exceções de rede pontuais (timeout, conexão interrompida) durante transferência de blocos são comuns em operação normal e nem sempre indicam falha real.
- Múltiplas ocorrências de um mesmo tipo de evento para o mesmo bloco frequentemente refletem replicação normal entre vários nós, não repetição suspeita.

Analise a sequência de eventos abaixo e determine se ela representa comportamento NORMAL ou ANÔMALO, considerando esse contexto de domínio.

Retorne exclusivamente um objeto JSON contendo:
classification, confidence e explanation.

Sequência:
{sequence}
"""


def call_ollama(prompt: str, temperature: float) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "seed": 42},
        },
        timeout=120,
    )
    if response.status_code != 200:
        print(f"\n--- ERRO OLLAMA (status {response.status_code}) ---")
        print(f"Corpo da resposta: {response.text}")
        print("--- fim do erro ---\n")
    response.raise_for_status()
    return response.json()["response"]


def extract_json(raw_text: str) -> dict | None:
    """Extrai o primeiro objeto JSON válido da resposta do modelo."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    # Normalização de chave (variabilidade de capitalização do modelo)
    normalized = {k.strip().lower(): v for k, v in data.items()}
    if "classification" not in normalized:
        return None

    return normalized


def normalize_classification(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().upper()
    if v in ("NORMAL",):
        return "Normal"
    if v in ("ANOMALO", "ANÔMALO", "ANOMALY"):
        return "Anomaly"
    return None


def classify_sequence(sequence_text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(sequence=sequence_text)

    for attempt, temperature in enumerate(TEMPERATURE_SCHEDULE, start=1):
        raw = call_ollama(prompt, temperature)
        parsed = extract_json(raw)

        if parsed is not None:
            classification = normalize_classification(parsed.get("classification"))
            if classification is not None:
                return {
                    "Classification": classification,
                    "Confidence": parsed.get("confidence"),
                    "Explanation": parsed.get("explanation"),
                    "Attempts": attempt,
                    "ParseFailed": False,
                }

    # Todas as tentativas falharam em produzir um JSON válido e utilizável.
    return {
        "Classification": None,
        "Confidence": None,
        "Explanation": None,
        "Attempts": MAX_RETRIES,
        "ParseFailed": True,
    }


def run_llm_classification(text_sequences_path, output_path, resume=True):
    df = pd.read_csv(text_sequences_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "BlockId", "Classification", "Confidence", "Explanation",
        "Attempts", "ParseFailed", "Label", "InferenceTimeSec",
    ]

    # Retomada: se já existe um CSV parcial, pula os BlockIds já processados.
    already_done = set()
    file_mode = "w"
    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        already_done = set(existing["BlockId"])
        file_mode = "a"
        print(f"Retomando: {len(already_done)} blocos já processados anteriormente.")

    remaining = df[~df["BlockId"].isin(already_done)]
    print(f"Blocos a processar nesta execução: {len(remaining)} de {len(df)}.")

    start = time.time()

    with output_path.open(file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()

        processed_this_run = 0
        for i, row in remaining.iterrows():
            block_start = time.time()
            outcome = classify_sequence(row["TextSequence"])
            inference_time = time.time() - block_start

            record = {
                "BlockId": row["BlockId"],
                "Classification": outcome["Classification"],
                "Confidence": outcome["Confidence"],
                "Explanation": outcome["Explanation"],
                "Attempts": outcome["Attempts"],
                "ParseFailed": outcome["ParseFailed"],
                "Label": row["Label"],
                "InferenceTimeSec": inference_time,
            }

            writer.writerow(record)
            f.flush()

            processed_this_run += 1
            if processed_this_run % 50 == 0:
                elapsed = time.time() - start
                total_done = len(already_done) + processed_this_run
                print(f"{processed_this_run}/{len(remaining)} nesta execução "
                      f"({total_done}/{len(df)} no total) "
                      f"({elapsed:.1f}s)...")

    elapsed = time.time() - start
    result_df = pd.read_csv(output_path)
    n_failed = result_df["ParseFailed"].sum()

    print(f"\nConcluído nesta execução em {elapsed:.1f}s.")
    print(f"Total acumulado no arquivo: {len(result_df)} blocos.")
    print(f"Falhas de parsing (sem classificação): {n_failed}")
    print(f"\nSalvo em: {output_path}")

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="data/processed/text_sequences_llm_sample.csv")
    parser.add_argument("--output", default="data/processed/predictions_llm.csv")
    args = parser.parse_args()

    run_llm_classification(args.sequences, args.output)