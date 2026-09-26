"""
Amostragem estratificada de um subconjunto do conjunto de teste, para
uso na avaliação da abordagem baseada em LLM (Fase 7, Modelo 3).

Motivo: rodar a LLM sobre os 115.013 blocos do conjunto de teste
completo é computacionalmente inviável em ambiente local (latência de
alguns segundos por chamada). Esta amostra reduz a escala mantendo a
proporção Normal/Anomaly do teste original, preservando comparação
justa: o mesmo subconjunto de BlockIds deve ser usado ao avaliar as
três abordagens na comparação final (Fase 9), não apenas a LLM.

Executado UMA ÚNICA VEZ, no mesmo espírito do split_train_test.py.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
SAMPLE_SIZE = 1000


def sample_test_set(test_ids_path, output_path, sample_size=SAMPLE_SIZE):
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(
            f"Amostra já existe em {output_path}. Deve ser gerada uma única "
            f"vez. Apague o arquivo manualmente se realmente precisar "
            f"refazer a amostragem (isso invalida qualquer resultado já "
            f"obtido com a LLM sobre a amostra anterior)."
        )

    test_df = pd.read_csv(test_ids_path, usecols=["BlockId", "Label"])

    sample_df, _ = train_test_split(
        test_df,
        train_size=sample_size,
        stratify=test_df["Label"],
        random_state=RANDOM_STATE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)

    print(f"Amostra estratificada gerada: {len(sample_df)} blocos "
          f"(de {len(test_df)} no teste completo).")
    print(sample_df["Label"].value_counts())
    print(f"  Proporção Anomaly: {(sample_df['Label'] == 'Anomaly').mean():.4%}")
    print(f"\nSalvo em: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-ids", default="data/processed/test_block_ids.csv")
    parser.add_argument("--output", default="data/processed/llm_eval_sample_block_ids.csv")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    sample_test_set(args.test_ids, args.output, args.sample_size)