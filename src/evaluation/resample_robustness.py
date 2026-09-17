"""
Robustez estatística da amostra de 1.000 blocos usada na comparação de
três abordagens (Fase 9, complementando metrics.py).

Motivação: a Tabela 3 do relatório reporta 100/100/100/0% para o
Random Forest sobre UMA amostra específica de 1.000 blocos, enquanto a
Tabela 2 (teste completo, 115.013 blocos) reporta 99,56/99,97/99,76/
0,013% -- a diferença é pequena, mas um resultado "perfeito" isolado
levanta a suspeita de que o modelo estaria ajustado à base de forma
não genuína.

Este módulo reamostra repetidamente (por padrão, 30 vezes, cada uma
com uma seed diferente) o MESMO conjunto de predições já geradas sobre
o teste completo -- não há novo treino de modelo nenhum aqui, é pura
reamostragem estatística das predições já existentes -- e reporta a
média, o desvio-padrão e os extremos das métricas em cada sorteio.

O resultado esperado é uma resposta direta e quantitativa à pergunta
da banca: "o 100% foi um efeito de amostra pequena com poucos erros
absolutos (16 em 115.013), ou o modelo está genuinamente instável?".

DESTINO: src/evaluation/resample_robustness.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import compute_metrics


N_RESAMPLES = 30
SAMPLE_SIZE = 1000


def resample_robustness(preds_path, n_resamples=N_RESAMPLES, sample_size=SAMPLE_SIZE):
    df_completo = pd.read_csv(preds_path)

    resultados = []
    for seed in range(n_resamples):
        amostra, _ = train_test_split(
            df_completo, train_size=sample_size,
            stratify=df_completo["Label"], random_state=seed,
        )
        m = compute_metrics(amostra)
        m["Seed"] = seed
        resultados.append(m)

    resultados_df = pd.DataFrame(resultados)
    return resultados_df


def summarize(resultados_df, nome_abordagem, metricas_completo=None):
    print(f"\n{'=' * 70}")
    print(f"ROBUSTEZ ESTATÍSTICA — {nome_abordagem}")
    print(f"({len(resultados_df)} reamostragens de {SAMPLE_SIZE} blocos, seeds 0-{len(resultados_df) - 1})")
    print("=" * 70)

    resumo = resultados_df[["Precision", "Recall", "F1", "FPR"]].agg(
        ["mean", "std", "min", "max"]
    )
    print(resumo.to_string())

    n_perfeitos = (resultados_df["F1"] == 1.0).sum()
    print(f"\nReamostragens com F1 = 100% exato: {n_perfeitos} de {len(resultados_df)} "
          f"({100 * n_perfeitos / len(resultados_df):.1f}%)")

    if metricas_completo is not None:
        print(f"\nPara referência, métricas sobre o TESTE COMPLETO "
              f"({metricas_completo['N_Total']} blocos):")
        for k in ["Precision", "Recall", "F1", "FPR"]:
            print(f"  {k}: {metricas_completo[k]:.4f}")

    return resumo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-preds", default="data/processed/predictions_random_forest.csv")
    parser.add_argument("--if-preds", default="data/processed/predictions_isolation_forest.csv")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    rf_completo_df = pd.read_csv(args.rf_preds)
    rf_metricas_completo = compute_metrics(rf_completo_df)

    rf_resultados = resample_robustness(args.rf_preds, args.n_resamples, args.sample_size)
    summarize(rf_resultados, "Random Forest (HDFS)", rf_metricas_completo)
    rf_out = output_dir / "robustez_amostragem_rf.csv"
    rf_resultados.to_csv(rf_out, index=False)

    # Bonus: mesma análise para o Isolation Forest, sem custo adicional
    # de execução -- útil para contrastar com a consistência de FPR já
    # observada entre HDFS e Apache para essa abordagem.
    if_completo_df = pd.read_csv(args.if_preds)
    if_metricas_completo = compute_metrics(if_completo_df)

    if_resultados = resample_robustness(args.if_preds, args.n_resamples, args.sample_size)
    summarize(if_resultados, "Isolation Forest (HDFS)", if_metricas_completo)
    if_out = output_dir / "robustez_amostragem_if.csv"
    if_resultados.to_csv(if_out, index=False)

    print(f"\nSalvos em:\n  {rf_out}\n  {if_out}")