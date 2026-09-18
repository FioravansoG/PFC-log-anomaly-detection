"""
Dashboard — Log Anomaly Detector.

Interface Streamlit para visualização comparativa das três abordagens
(Isolation Forest, Random Forest, LLM) sobre as duas bases avaliadas
neste trabalho (HDFS e Apache/AIT-LDS), atendendo a RF13 (seleção de
abordagem e inspeção de alertas/métricas) e RNF4 (interpretabilidade),
além de reunir os achados discutidos no Capítulo 7: matrizes de
confusão, robustez estatística e catalogação de anomalias.

DESTINO: dashboard/app.py (substitui a versão anterior, HDFS-only)
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.evaluation.metrics import compute_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO DAS DUAS BASES
# ----------------------------------------------------------------------------
DATASETS = {
    "HDFS": {
        "unidade": "Bloco",
        "abordagens": {
            "Isolation Forest (não supervisionado)": {
                "preds": DATA_DIR / "predictions_isolation_forest.csv",
                "escopo": "Teste completo (115.013 blocos)",
            },
            "Random Forest (supervisionado)": {
                "preds": DATA_DIR / "predictions_random_forest.csv",
                "escopo": "Teste completo (115.013 blocos)",
            },
            "LLM (qwen2.5-coder:7b)": {
                "preds": DATA_DIR / "predictions_llm.csv",
                "escopo": "Amostra estratificada (1.000 blocos) — teste completo é computacionalmente inviável para a LLM",
            },
        },
        "confusao_completo": DATA_DIR / "matrizes_confusao_hdfs_completo.png",
        "confusao_amostra": DATA_DIR / "matrizes_confusao_hdfs_amostra.png",
        "sequences_file": DATA_DIR / "blocks_sequences.csv",
        "templates_file": DATA_DIR / "templates_gerados.csv",
        "catalogo_file": DATA_DIR / "catalogo_anomalias_hdfs.csv",
        "robustez_files": {
            "Random Forest (supervisionado)": DATA_DIR / "robustez_amostragem_rf.csv",
            "Isolation Forest (não supervisionado)": DATA_DIR / "robustez_amostragem_if.csv",
        },
    },
    "Apache / AIT-LDS": {
        "unidade": "Linha (requisição HTTP)",
        "abordagens": {
            "Isolation Forest (não supervisionado)": {
                "preds": DATA_DIR / "apache_predictions_isolation_forest.csv",
                "escopo": "Teste completo (1.706 linhas)",
            },
            "Random Forest (supervisionado)": {
                "preds": DATA_DIR / "apache_predictions_random_forest.csv",
                "escopo": "Teste completo (1.706 linhas)",
            },
            "LLM (qwen2.5-coder:7b, caminho real)": {
                "preds": DATA_DIR / "apache_predictions_llm_v2.csv",
                "escopo": "Teste completo (1.706 linhas) — viável por ser uma base pequena",
            },
        },
        "confusao_completo": DATA_DIR / "matrizes_confusao_apache_com_llm.png",
        "confusao_amostra": None,
        "parsed_file": DATA_DIR / "apache_parsed.csv",
        "tags_file": DATA_DIR / "apache_line_tags.csv",
    },
}


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def metrics_row(nome_abordagem: str, df: pd.DataFrame) -> dict:
    m = compute_metrics(df)
    return {
        "Abordagem": nome_abordagem,
        "Precisão": m["Precision"],
        "Revocação": m["Recall"],
        "F1-score": m["F1"],
        "FPR": m["FPR"],
        "Tempo médio (s)": m.get("TempoMedioInferenciaSeg"),
    }


def fmt_pct(x):
    return f"{100 * x:.2f}%" if pd.notna(x) else "—"


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Log Anomaly Detector", layout="wide")
st.title("LOG ANOMALY DETECTOR")
st.caption(
    "Comparação de três paradigmas de IA (não supervisionado, supervisionado, "
    "baseado em LLM) para detecção de anomalias em logs, validada sobre duas bases "
    "de naturezas distintas: HDFS (Loghub) e Apache/AIT-LDS."
)

dataset_nome = st.sidebar.selectbox("Base de dados", list(DATASETS.keys()))
dataset_cfg = DATASETS[dataset_nome]

abordagem_nome = st.sidebar.radio("Abordagem", list(dataset_cfg["abordagens"].items()).__len__() and list(dataset_cfg["abordagens"].keys()))
abordagem_cfg = dataset_cfg["abordagens"][abordagem_nome]

df_pred = load_csv(abordagem_cfg["preds"])

st.sidebar.markdown(f"**Unidade de análise:** {dataset_cfg['unidade']}")
st.sidebar.markdown(f"**Escopo:** {abordagem_cfg['escopo']}")

tabs = st.tabs([
    "Resumo", "Matriz de Confusão", "Alertas",
    "Robustez Estatística", "Catalogação de Anomalias", "Comparação entre Bases",
])

# ----------------------------------------------------------------------------
# ABA 1 — RESUMO
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader(f"Resumo — {dataset_nome} / {abordagem_nome}")

    if df_pred is None:
        st.warning(f"Arquivo de predições não encontrado: `{abordagem_cfg['preds']}`. "
                   "Rode o pipeline correspondente antes de visualizar esta aba.")
    else:
        m = compute_metrics(df_pred)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precisão", fmt_pct(m["Precision"]))
        col2.metric("Revocação", fmt_pct(m["Recall"]))
        col3.metric("F1-score", fmt_pct(m["F1"]))
        col4.metric("FPR", fmt_pct(m["FPR"]))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Verdadeiros Positivos", int(m["TP"]))
        col6.metric("Falsos Positivos", int(m["FP"]))
        col7.metric("Verdadeiros Negativos", int(m["TN"]))
        col8.metric("Falsos Negativos", int(m["FN"]))

        if m.get("TempoMedioInferenciaSeg") is not None:
            st.caption(f"Tempo médio de inferência: {m['TempoMedioInferenciaSeg']:.6f} s por "
                       f"{dataset_cfg['unidade'].lower()}")

        st.markdown("**Todas as abordagens desta base:**")
        linhas = []
        for nome, cfg in dataset_cfg["abordagens"].items():
            df_i = load_csv(cfg["preds"])
            if df_i is not None:
                linhas.append(metrics_row(nome, df_i))
        if linhas:
            resumo_df = pd.DataFrame(linhas).set_index("Abordagem")
            resumo_fmt = resumo_df.copy()
            for c in ["Precisão", "Revocação", "F1-score", "FPR"]:
                resumo_fmt[c] = resumo_fmt[c].apply(fmt_pct)
            st.dataframe(resumo_fmt, use_container_width=True)

# ----------------------------------------------------------------------------
# ABA 2 — MATRIZ DE CONFUSÃO
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader(f"Matriz de Confusão — {dataset_nome} / {abordagem_nome}")

    if df_pred is None:
        st.warning("Sem predições carregadas para esta abordagem.")
    else:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        plot_confusion_matrix(df_pred["Label"], df_pred["Classification"], abordagem_nome, ax)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("**Figuras consolidadas (todas as abordagens desta base, já geradas para o relatório):**")

    colA, colB = st.columns(2)
    if dataset_cfg["confusao_completo"] and dataset_cfg["confusao_completo"].exists():
        colA.image(str(dataset_cfg["confusao_completo"]), caption="Conjunto de teste completo")
    else:
        colA.info("Figura do teste completo ainda não gerada.")

    if dataset_cfg.get("confusao_amostra") and dataset_cfg["confusao_amostra"].exists():
        colB.image(str(dataset_cfg["confusao_amostra"]), caption="Amostra comum (três abordagens)")

# ----------------------------------------------------------------------------
# ABA 3 — ALERTAS
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader(f"Alertas — {dataset_nome} / {abordagem_nome}")

    if df_pred is None:
        st.warning("Sem predições carregadas para esta abordagem.")
    else:
        alerts_df = df_pred[df_pred["Classification"] == "Anomaly"].copy()
        st.caption(f"{len(alerts_df)} alertas gerados de {len(df_pred)} {dataset_cfg['unidade'].lower()}s analisados.")

        if alerts_df.empty:
            st.info("Nenhum alerta gerado.")
        else:
            alerts_df["Acertou"] = alerts_df["Classification"] == alerts_df["Label"]
            display_cols = [c for c in ["BlockId", "Classification", "Label", "Acertou",
                                         "Confidence", "InferenceTimeSec"] if c in alerts_df.columns]
            st.dataframe(
                alerts_df[display_cols].sort_values("Acertou"),
                use_container_width=True, height=320,
            )

            id_col = "BlockId" if "BlockId" in alerts_df.columns else alerts_df.columns[0]
            selected_id = st.selectbox(f"Selecione um {dataset_cfg['unidade']} para inspecionar",
                                        options=alerts_df[id_col].tolist())

            row = alerts_df[alerts_df[id_col] == selected_id].iloc[0]
            st.markdown(f"**{dataset_cfg['unidade']} selecionado:** `{selected_id}`")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Classificação:**\n\n{row['Classification']}")
            c2.markdown(f"**Ground truth:**\n\n{row['Label']}")
            if "InferenceTimeSec" in row and pd.notna(row.get("InferenceTimeSec")):
                c3.markdown(f"**Tempo de inferência:**\n\n{row['InferenceTimeSec']:.4f}s")

            if "Explanation" in row and pd.notna(row.get("Explanation")):
                st.markdown("**Explicação (LLM):**")
                st.write(row["Explanation"])

            # Detalhe específico por base: sequência de eventos (HDFS) vs. requisição HTTP (Apache)
            if dataset_nome == "HDFS":
                sequences = load_csv(dataset_cfg["sequences_file"])
                templates = load_csv(dataset_cfg["templates_file"])
                if sequences is not None and templates is not None:
                    seq_row = sequences[sequences["BlockId"] == selected_id]
                    if not seq_row.empty:
                        event_ids = [int(t) for t in str(seq_row.iloc[0]["EventSequence"]).split()]
                        chain = " → ".join(f"E{e}" for e in event_ids[:20])
                        if len(event_ids) > 20:
                            chain += f" → ... (+{len(event_ids) - 20} eventos)"
                        st.markdown(f"**Sequência de eventos:** {chain}")
            else:
                parsed = load_csv(dataset_cfg["parsed_file"])
                if parsed is not None:
                    line_row = parsed[parsed["LineId"] == selected_id]
                    if not line_row.empty and "Path" in line_row.columns:
                        st.markdown(f"**Caminho da requisição:** `{line_row.iloc[0]['Path']}`")
                        st.markdown(f"**Código de status HTTP:** {line_row.iloc[0].get('StatusCode', '—')}")

# ----------------------------------------------------------------------------
# ABA 4 — ROBUSTEZ ESTATÍSTICA (hoje só existe para o HDFS)
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Robustez Estatística da Amostra Comum")

    if dataset_nome != "HDFS":
        st.info("A análise de robustez por reamostragem foi realizada apenas para o HDFS, "
                "já que a base Apache/AIT-LDS foi avaliada sobre seu teste completo em todas "
                "as abordagens, sem necessidade de amostragem.")
    else:
        for nome, path in dataset_cfg["robustez_files"].items():
            df_rob = load_csv(path)
            st.markdown(f"**{nome}**")
            if df_rob is None:
                st.info(f"Arquivo `{path.name}` não encontrado — rode "
                       "`python -m src.evaluation.resample_robustness` primeiro.")
                continue
            resumo = df_rob[["Precision", "Recall", "F1", "FPR"]].agg(["mean", "std", "min", "max"])
            st.dataframe(resumo.map(fmt_pct), use_container_width=True)
            n_perfeitos = (df_rob["F1"] == 1.0).sum()
            st.caption(f"{n_perfeitos} de {len(df_rob)} reamostragens com F1 = 100% exato "
                      f"({100 * n_perfeitos / len(df_rob):.1f}%)")
            st.bar_chart(df_rob.set_index("Seed")[["F1"]])

# ----------------------------------------------------------------------------
# ABA 5 — CATALOGAÇÃO DE ANOMALIAS
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Catalogação de Anomalias")

    if dataset_nome == "HDFS":
        st.caption(
            "O HDFS não possui rótulo de tipo de anomalia — a categorização abaixo é "
            "inferida pela especificidade estatística de cada EventId (frequência em "
            "blocos anômalos menos frequência em blocos normais), e representa "
            "**categorias de falha operacional**, não tipos de ataque."
        )
        cat_df = load_csv(dataset_cfg["catalogo_file"])
        if cat_df is None:
            st.info("Arquivo `catalogo_anomalias_hdfs.csv` não encontrado — rode "
                   "`python -m src.evaluation.catalogo_anomalias` primeiro.")
        else:
            st.dataframe(cat_df, use_container_width=True, height=420)
    else:
        st.caption(
            "A base Apache/AIT-LDS fornece rótulos nomeados por etapa de ataque real, "
            "permitindo uma catalogação direta, sem inferência estatística."
        )
        tags_df = load_csv(dataset_cfg["tags_file"])
        if tags_df is None:
            st.info(f"Arquivo `{dataset_cfg['tags_file'].name}` não encontrado — rode "
                   "`python -m src.features.apache_vectorizer` primeiro.")
        else:
            contagem = {}
            for tags in tags_df["AttackTags"].dropna():
                for tag in str(tags).split(";"):
                    tag = tag.strip()
                    if tag:
                        contagem[tag] = contagem.get(tag, 0) + 1
            if contagem:
                tag_df = pd.DataFrame(
                    sorted(contagem.items(), key=lambda x: -x[1]),
                    columns=["Categoria de Ataque", "Nº de Linhas"],
                )
                st.dataframe(tag_df, use_container_width=True)
                st.bar_chart(tag_df.set_index("Categoria de Ataque"))
            else:
                st.info("Nenhuma tag de ataque encontrada no arquivo processado.")

# ----------------------------------------------------------------------------
# ABA 6 — COMPARAÇÃO ENTRE BASES (independente do seletor de base)
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Comparação entre HDFS e Apache/AIT-LDS")
    st.caption("Esta aba independe da base selecionada na barra lateral — reúne os "
              "resultados de ambas para comparação direta.")

    linhas = []
    for ds_nome, ds_cfg in DATASETS.items():
        for ab_nome, ab_cfg in ds_cfg["abordagens"].items():
            df_i = load_csv(ab_cfg["preds"])
            if df_i is not None:
                row = metrics_row(ab_nome, df_i)
                row["Base"] = ds_nome
                row["Escopo"] = ab_cfg["escopo"]
                linhas.append(row)

    if not linhas:
        st.warning("Nenhum arquivo de predição encontrado em nenhuma das duas bases.")
    else:
        comp_df = pd.DataFrame(linhas)[["Base", "Abordagem", "Precisão", "Revocação", "F1-score", "FPR", "Escopo"]]
        comp_fmt = comp_df.copy()
        for c in ["Precisão", "Revocação", "F1-score", "FPR"]:
            comp_fmt[c] = comp_fmt[c].apply(fmt_pct)
        st.dataframe(comp_fmt, use_container_width=True, height=280)

        st.markdown("""
**Achados-chave desta comparação (Seção 7.4 do relatório):**
- O *Random Forest* mantém desempenho quase perfeito em ambas as bases, apesar da proporção de anomalias estar invertida entre elas (2,93% no HDFS vs. 90,2% no Apache) — evidência direta contra a hipótese de sobreajuste ao HDFS.
- O *Isolation Forest* apresenta FPR notavelmente semelhante nas duas bases (~35-38%), sugerindo que essa taxa é uma propriedade do limiar de decisão padrão do algoritmo, não da base em si.
- A LLM apresenta vieses de erro opostos: superalerta no HDFS (FPR alto), subalerta no Apache (recall baixo) — indicando ausência de um viés sistemático único e transferível entre domínios.
        """)