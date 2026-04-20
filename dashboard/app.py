import streamlit as st
import pandas as pd
import os
import glob

st.set_page_config(page_title="LLM Eval Harness", layout="wide")
st.title("LLM Reasoning Evaluation Dashboard")
st.caption("Benchmarking GPT-4o-mini, Claude Haiku, and Mistral across domains")

# ── Load Data ──────────────────────────────────────────────────────────────────

summary_files = glob.glob("results/*_scores_summary.csv")
raw_files = glob.glob("results/*_scores_raw.csv")

if not summary_files:
    st.warning("No results found. Run the eval pipeline first.")
    st.stop()

# let user pick a run if multiple exist
selected_summary = st.selectbox("Select eval run:", sorted(summary_files, reverse=True))
selected_raw = selected_summary.replace("_scores_summary.csv", "_scores_raw.csv")

summary_df = pd.read_csv(selected_summary)
raw_df = pd.read_csv(selected_raw) if os.path.exists(selected_raw) else None

# ── Filters ────────────────────────────────────────────────────────────────────

st.sidebar.header("Filters")
all_models = summary_df["model"].unique().tolist()
all_domains = summary_df["domain"].unique().tolist()

selected_models = st.sidebar.multiselect("Models", all_models, default=all_models)
selected_domains = st.sidebar.multiselect("Domains", all_domains, default=all_domains)

filtered = summary_df[
    summary_df["model"].isin(selected_models) &
    summary_df["domain"].isin(selected_domains)
]

# ── Top Metrics ────────────────────────────────────────────────────────────────

st.subheader("Overall Averages")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Factual Accuracy", f"{filtered['avg_factual_accuracy'].mean():.2f}")
col2.metric("Calibration", f"{filtered['avg_calibration'].mean():.2f}")
col3.metric("Hallucination", 
    f"{filtered['avg_hallucination'].mean():.2f}" 
    if filtered['avg_hallucination'].notna().any() else "N/A")
col4.metric("CoT Coherence", 
    f"{filtered['avg_cot'].mean():.2f}" 
    if filtered['avg_cot'].notna().any() else "N/A")
col5.metric("Avg Latency", f"{filtered['avg_latency'].mean():.2f}s")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────

st.subheader("Factual Accuracy by Model & Domain")
st.bar_chart(
    filtered.pivot_table(
        index="domain", columns="model", values="avg_factual_accuracy"
    )
)

st.subheader("Calibration Score by Model & Domain")
st.bar_chart(
    filtered.pivot_table(
        index="domain", columns="model", values="avg_calibration"
    )
)

st.subheader("Latency by Model & Domain")
st.bar_chart(
    filtered.pivot_table(
        index="domain", columns="model", values="avg_latency"
    )
)

if filtered["avg_hallucination"].notna().any():
    st.subheader("Hallucination Rate by Model & Domain")
    st.bar_chart(
        filtered.pivot_table(
            index="domain", columns="model", values="avg_hallucination"
        )
    )

if filtered["avg_cot"].notna().any():
    st.subheader("CoT Coherence by Model & Domain")
    st.bar_chart(
        filtered.pivot_table(
            index="domain", columns="model", values="avg_cot"
        )
    )

st.divider()

# ── Raw Results Table ──────────────────────────────────────────────────────────

st.subheader("Raw Prompt-Level Results")
if raw_df is not None:
    display_cols = [
        "prompt_id", "domain", "difficulty", "model",
        "factual_accuracy", "calibration_score",
        "hallucination_score", "cot_score", "latency"
    ]
    st.dataframe(
        raw_df[display_cols].sort_values(["domain", "model"]),
        use_container_width=True
    )
else:
    st.info("No raw scores file found for this run.")

st.divider()
st.subheader("Full Summary Table")
st.dataframe(filtered, use_container_width=True)