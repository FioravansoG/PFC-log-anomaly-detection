"""
Modelo 3 — Classificação via LLM local (Ollama), adaptado para o
access_log do Apache (AIT-LDS).

Versão 2: usa o CAMINHO REAL da URL (coluna "Path" do apache_parsed.csv,
adicionada após a correção do parser) em vez do EventTemplate mascarado
pelo Drain3. A primeira execução usou o template mascarado e obteve
recall=0% -- causa provável: o mascaramento colapsou quase todas as
URLs em poucos templates genéricos, eliminando o único sinal semântico
real disponível (qual caminho foi acessado).

IMPORTANTE (rigor metodológico): o PROMPT_TEMPLATE abaixo contém
conhecimento de domínio GENÉRICO sobre segurança web -- o mesmo nível
de generalidade do prompt usado no HDFS (que descreve o comportamento
normal do protocolo, não o que causou as anomalias observadas NESTE
teste específico). Nenhuma pista foi adicionada com base no
conhecimento privilegiado dos rótulos do AIT-LDS (ex.: não menciona
"wpscan", "dirb" ou "wpdiscuz" especificamente) -- isso constituiria
vazamento de informação do gabarito para o classificador, inflando
artificialmente seu desempenho de um jeito que Isolation Forest e
Random Forest não têm (eles só veem estatística bruta, nunca a cadeia
de ataque em si).

Reaproveita call_ollama(), extract_json() e normalize_classification()
de src/models/llm_classifier.py sem nenhuma alteração.

DESTINO: src/models/apache_llm_classifier.py (substitui a versão anterior)
"""

import csv
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.llm_classifier import call_ollama, extract_json, normalize_classification
from src.ingestion.apache_label_loader import build_apache_labels_df


RANDOM_STATE = 42
TEST_SIZE = 0.2

TEMPERATURE_SCHEDULE = [0.0, 0.3, 0.5]
MAX_RETRIES = len(TEMPERATURE_SCHEDULE)

# Conhecimento de dominio GENERICO sobre seguranca web -- nao contem
# nenhuma pista especifica do ataque real deste dataset.
PROMPT_TEMPLATE = """Você é um classificador de anomalias em requisições HTTP de um servidor Apache que hospeda uma aplicação web, especializado em identificar tentativas de ataque e reconhecimento.

Contexto geral sobre o funcionamento normal de um servidor web:
- A maior parte do tráfego legítimo é navegação comum e carregamento de recursos estáticos (imagens, scripts, folhas de estilo, fontes).
- Nem toda requisição incomum é maliciosa: erros HTTP pontuais podem ocorrer em uso legítimo (link quebrado, recurso removido).
- Indícios genéricos de comportamento malicioso incluem: varredura sistemática de caminhos/arquivos não referenciados no site, tentativas de acessar arquivos administrativos, de configuração ou de controle de versão, parâmetros com sintaxe de injeção de código ou comando, upload de arquivos executáveis, e sequências de requisições malformadas em alta frequência.

Analise a requisição HTTP abaixo (caminho da URL e metadados da resposta) e determine se ela representa comportamento NORMAL ou ANÔMALO, considerando esse contexto geral.

Retorne exclusivamente um objeto JSON contendo:
classification, confidence e explanation.

Requisição:
{sequence}
"""


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

    return {
        "Classification": None,
        "Confidence": None,
        "Explanation": None,
        "Attempts": MAX_RETRIES,
        "ParseFailed": True,
    }


def _get_test_split(parsed_csv_path, labels_path):
    """Recria o mesmo split treino/teste do apache_vectorizer.py (mesma
    seed e test_size) e retorna só a parte de teste."""
    parsed = pd.read_csv(parsed_csv_path)
    n_total_lines = len(parsed)

    labels_df = build_apache_labels_df(labels_path, n_total_lines)
    df = parsed.merge(labels_df, left_on="LineId", right_on="BlockId", how="inner")

    _, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["Label"], random_state=RANDOM_STATE
    )
    return test_df


def _linha_para_texto(row) -> str:
    return (
        f"Caminho: {row['Path']} | "
        f"Status HTTP: {row['StatusCode']} | "
        f"Tamanho da resposta: {row['ResponseSize']} bytes | "
        f"Tamanho do User-Agent: {row['UserAgentLength']} caracteres"
    )


def build_test_text_sequences(parsed_csv_path, labels_path, output_path, sample_size=None):
    """
    sample_size: se fornecido, aplica uma amostragem estratificada
    adicional sobre o conjunto de teste (mesmo espírito do
    sample_test_for_llm.py do HDFS) -- útil para uma rodada de
    validação rápida antes de comprometer o tempo da execução completa.
    """
    test_df = _get_test_split(parsed_csv_path, labels_path)

    if sample_size is not None and sample_size < len(test_df):
        test_df, _ = train_test_split(
            test_df, train_size=sample_size, stratify=test_df["Label"],
            random_state=RANDOM_STATE,
        )
        print(f"Amostra estratificada: {len(test_df)} de {len(test_df) + len(_)} linhas de teste.")

    test_df = test_df.copy()
    test_df["TextSequence"] = test_df.apply(_linha_para_texto, axis=1)

    result = test_df[["LineId", "TextSequence", "Label"]].rename(columns={"LineId": "BlockId"})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Representação textual gerada para {len(result)} linhas.")
    print(result["Label"].value_counts())
    print(f"Salvo em: {output_path}")
    return result


def run_llm_classification(text_sequences_path, output_path, resume=True):
    df = pd.read_csv(text_sequences_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "BlockId", "Classification", "Confidence", "Explanation",
        "Attempts", "ParseFailed", "Label", "InferenceTimeSec",
    ]

    already_done = set()
    file_mode = "w"
    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        already_done = set(existing["BlockId"])
        file_mode = "a"
        print(f"Retomando: {len(already_done)} linhas já processadas anteriormente.")

    remaining = df[~df["BlockId"].isin(already_done)]
    print(f"Linhas a processar nesta execução: {len(remaining)} de {len(df)}.")

    start = time.time()

    with output_path.open(file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()

        processed_this_run = 0
        for i, row in remaining.iterrows():
            line_start = time.time()
            outcome = classify_sequence(row["TextSequence"])
            inference_time = time.time() - line_start

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
                      f"({total_done}/{len(df)} no total) ({elapsed:.1f}s)...")

    elapsed = time.time() - start
    result_df = pd.read_csv(output_path)
    n_failed = result_df["ParseFailed"].sum()

    print(f"\nConcluído nesta execução em {elapsed:.1f}s.")
    print(f"Total acumulado no arquivo: {len(result_df)} linhas.")
    print(f"Falhas de parsing (sem classificação): {n_failed}")
    print(f"\nSalvo em: {output_path}")

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", default="data/processed/apache_parsed.csv")
    parser.add_argument(
        "--labels",
        default="data/raw/apache/intranet.smith.russellmitchell.com-access.log.2.labels",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Se fornecido, roda só numa amostra estratificada desse tamanho em vez do teste completo.",
    )
    parser.add_argument("--sequences-output", default="data/processed/apache_text_sequences_test_v2.csv")
    parser.add_argument("--predictions-output", default="data/processed/apache_predictions_llm_v2.csv")
    parser.add_argument(
        "--skip-build-sequences", action="store_true",
        help="Pula a geração da representação textual (já existe de uma execução anterior)",
    )
    args = parser.parse_args()

    if not args.skip_build_sequences:
        build_test_text_sequences(
            args.parsed, args.labels, args.sequences_output, sample_size=args.sample_size
        )

    run_llm_classification(args.sequences_output, args.predictions_output)