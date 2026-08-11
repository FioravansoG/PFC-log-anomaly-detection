"""
Dashboard — Log Anomaly Detector (Fase 10).

Interface Streamlit para visualização comparativa das três abordagens
(Isolation Forest, Random Forest, LLM), seguindo o layout especificado
pelo orientador: métricas agregadas, tabela de alertas, drill-down por
bloco com seleção de modelo.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.alerts.alert_generator import (
    NORMALIZERS,
    SCORE_COLUMNS,
    severity_from_score,
)
from src.evaluation.metrics import compute_metrics


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

MODEL_CONFIG = {
    "Isolation Forest (não supervisionado)": {
        "key": "isolation_forest",
        "predictions": DATA_DIR / "predictions_isolation_forest.csv",
    },
    "Random Forest (supervisionado)": {
        "key": "random_forest",
        "predictions": DATA_DIR / "predictions_random_forest.csv",
    },
    "LLM (Ollama)": {
        "key": "llm",
        "predictions": DATA_DIR / "predictions_llm.csv",
    },
}


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_block_sequences() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "blocks_sequences.csv", usecols=["BlockId", "EventSequence"])


@st.cache_data
def load_templates() -> dict:
    df = pd.read_csv(DATA_DIR / "templates_gerados.csv", usecols=["EventId", "EventTemplate"])
    return dict(zip(df["EventId"], df["EventTemplate"]))


def build_alerts_table(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Gera a tabela de alertas ao vivo a partir das predições, aplicando
    a mesma normalização de score usada em alert_generator.py."""
    normalizer = NORMALIZERS[model_key]
    score_col = SCORE_COLUMNS[model_key]

    anomalies = df[df["Classification"] == "Anomaly"].copy()
    if anomalies.empty:
        return anomalies

    def _norm(raw):
        if pd.isna(raw):
            return None
        return normalizer(raw)

    anomalies["ScoreNormalizado"] = anomalies[score_col].apply(_norm)
    anomalies["Risco"] = anomalies["ScoreNormalizado"].apply(severity_from_score)
    return anomalies


def render_event_chain(event_sequence: str, templates: dict, max_events: int = 15) -> str:
    event_ids = [int(tok) for tok in str(event_sequence).split()]
    labels = [f"E{eid}" for eid in event_ids]
    chain = " → ".join(labels[:max_events])
    if len(labels) > max_events:
        chain += f" → ... (+{len(labels) - max_events} eventos)"
    return chain


st.set_page_config(page_title="Log Anomaly Detector", layout="wide")

st.title("LOG ANOMALY DETECTOR")
st.caption(
    "Comparação de abordagens de IA para detecção de anomalias em logs do HDFS "
    "(não supervisionada, supervisionada e baseada em LLM)."
)

model_label = st.selectbox("Modelo", list(MODEL_CONFIG.keys()))
model_conf = MODEL_CONFIG[model_label]
model_key = model_conf["key"]
pred_path = model_conf["predictions"]

if not pred_path.exists():
    st.warning(f"Ainda não há predições para este modelo em `{pred_path.name}`.")
    st.stop()

df = load_predictions(pred_path)
sequences = load_block_sequences()
templates = load_templates()

metrics = compute_metrics(df)

st.subheader("Métricas gerais")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Logs analisados", f"{metrics['N_Total']:,}".replace(",", "."))
col2.metric("Anomalias detectadas", f"{int((df['Classification'] == 'Anomaly').sum()):,}".replace(",", "."))
col3.metric("Precision", f"{metrics['Precision']:.2f}")
col4.metric("Recall", f"{metrics['Recall']:.2f}")
col5.metric("F1-score", f"{metrics['F1']:.2f}")

col6, col7 = st.columns(2)
col6.metric("FPR", f"{metrics['FPR']:.2%}")
if metrics["TempoMedioInferenciaSeg"] is not None:
    col7.metric("Tempo médio de inferência", f"{metrics['TempoMedioInferenciaSeg']:.4f}s")

if metrics["N_Invalido"] > 0:
    st.caption(f"⚠️ {metrics['N_Invalido']} predições inválidas (ex: falha de parsing) excluídas das métricas.")

st.divider()

st.subheader("Alertas")
alerts_df = build_alerts_table(df, model_key)

if alerts_df.empty:
    st.info("Nenhum alerta gerado para este modelo com os dados disponíveis.")
else:
    risco_filter = st.multiselect(
        "Filtrar por risco", options=["HIGH", "MEDIUM", "LOW", "UNKNOWN"],
        default=["HIGH", "MEDIUM", "LOW", "UNKNOWN"],
    )
    filtered = alerts_df[alerts_df["Risco"].isin(risco_filter)]
    filtered_sorted = filtered.sort_values("ScoreNormalizado", ascending=False)

    display_table = filtered_sorted[["BlockId", "ScoreNormalizado", "Risco", "Classification"]].rename(
        columns={
            "BlockId": "Block ID",
            "ScoreNormalizado": "Score",
            "Classification": "Resultado",
        }
    )
    st.dataframe(display_table, use_container_width=True, height=300)

    st.divider()
    st.subheader("Detalhe do alerta")

    selected_block = st.selectbox(
        "Selecione um Block ID para inspecionar",
        options=filtered_sorted["BlockId"].tolist(),
    )

    if selected_block:
        row = df[df["BlockId"] == selected_block].iloc[0]
        seq_row = sequences[sequences["BlockId"] == selected_block]

        st.markdown(f"**Block ID:** `{selected_block}`")

        if not seq_row.empty:
            chain = render_event_chain(seq_row.iloc[0]["EventSequence"], templates)
            st.markdown(f"**Eventos:** {chain}")

        detail_col1, detail_col2, detail_col3 = st.columns(3)
        detail_col1.markdown(f"**Modelo:**\n\n{model_label}")

        score_col = SCORE_COLUMNS[model_key]
        raw_score = row.get(score_col)
        detail_col2.markdown(f"**Score bruto ({score_col}):**\n\n{raw_score}")

        detail_col3.markdown(f"**Classificação:**\n\n{row['Classification']}")

        detail_col4, detail_col5 = st.columns(2)
        detail_col4.markdown(f"**Ground truth:**\n\n{row['Label']}")
        if "InferenceTimeSec" in row and pd.notna(row["InferenceTimeSec"]):
            detail_col5.markdown(f"**Tempo de inferência:**\n\n{row['InferenceTimeSec']:.4f}s")

        if "Explanation" in row and pd.notna(row.get("Explanation")):
            st.markdown("**Explicação (LLM):**")
            st.write(row["Explanation"])

        acertou = row["Classification"] == row["Label"]
        if acertou:
            st.success("Classificação corresponde ao ground truth.")
        else:
            st.error("Classificação DIVERGE do ground truth.")