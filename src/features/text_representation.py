"""
Construção da representação textual das sequências de eventos, para
consumo pela LLM (Fase 6, parte 2).

Para cada bloco, gera uma string numerada com os templates de evento
na ordem de ocorrência, ex:
    Event 1: Receiving block <BLOCK_ID> src: <IP> dest: <IP>
    Event 2: PacketResponder <NUM> for block <BLOCK_ID> <*>
    ...

Diferente da matriz de contagem (vectorizer.py), aqui o vocabulário
não precisa ser restrito ao treino: a LLM recebe o texto do template
diretamente (não um índice numérico aprendido), então não há risco de
vazamento de dimensão de feature entre treino e teste.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd


def load_event_id_to_template(templates_csv_path) -> dict:
    df = pd.read_csv(templates_csv_path, usecols=["EventId", "EventTemplate"])
    return dict(zip(df["EventId"], df["EventTemplate"]))


def sequence_to_text(event_sequence: str, event_map: dict) -> str:
    """'1 3 1' -> 'Event 1: <template E1>\nEvent 2: <template E3>\nEvent 3: <template E1>'"""
    event_ids = [int(tok) for tok in event_sequence.split()]
    lines = []
    for i, eid in enumerate(event_ids, start=1):
        template = event_map.get(eid, "<TEMPLATE DESCONHECIDO>")
        lines.append(f"Event {i}: {template}")
    return "\n".join(lines)


def build_text_representations(
    blocks_csv_path,
    templates_csv_path,
    block_ids_path,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    event_map = load_event_id_to_template(templates_csv_path)

    blocks = pd.read_csv(
        blocks_csv_path, usecols=["BlockId", "EventSequence", "Label"]
    )
    subset_ids = pd.read_csv(block_ids_path, usecols=["BlockId"])
    subset = blocks.merge(subset_ids, on="BlockId", how="inner")

    print(f"Gerando representação textual para {len(subset)} blocos...")

    subset = subset.copy()
    subset["TextSequence"] = subset["EventSequence"].apply(
        lambda seq: sequence_to_text(seq, event_map)
    )

    result = subset[["BlockId", "TextSequence", "Label"]]
    result.to_csv(output_path, index=False)

    print(f"Salvo em: {output_path}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="data/processed/blocks_sequences.csv")
    parser.add_argument("--templates", default="data/processed/templates_gerados.csv")
    parser.add_argument(
        "--block-ids",
        default="data/processed/test_block_ids.csv",
        help="Por padrão gera só para o conjunto de TESTE, que é o que a LLM vai classificar.",
    )
    parser.add_argument("--output", default="data/processed/text_sequences_test.csv")
    args = parser.parse_args()

    build_text_representations(args.blocks, args.templates, args.block_ids, args.output)