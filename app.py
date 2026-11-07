from pathlib import Path

import pandas as pd
import streamlit as st

from claimcheck import ClaimChecker, load_dataset

st.set_page_config(page_title="SciFact Claim Checker", layout="wide")
st.title("Scientific Claim Checker")
st.write("Search a small biomedical abstract collection, inspect sentence-level evidence, and abstain when the model is unsure.")
st.warning("Research demo only. It cannot verify arbitrary claims and must not be used for medical decisions.")


@st.cache_resource(show_spinner="Downloading SciFact and loading models on first run...")
def load_checker():
    corpus, _ = load_dataset()
    return ClaimChecker(corpus)


def default_threshold(method: str) -> float:
    report_path = Path("reports/ablation.csv")
    if report_path.exists():
        try:
            report = pd.read_csv(report_path)
            row = report.loc[report["retrieval"] == method]
            if not row.empty:
                return float(row.iloc[0]["abstain_threshold"])
        except (OSError, ValueError, KeyError):
            pass
    return 0.55


claim = st.text_area("Scientific claim", placeholder="Example: Vitamin D supplementation reduces the risk of respiratory infections.", height=110)
left, mid, right = st.columns(3)
with left:
    method = st.selectbox("Retrieval method", ["Hybrid", "BM25", "Dense"])
with mid:
    top_k = st.slider("Abstracts to check", 3, 20, 10)
with right:
    threshold = st.slider(
        "Abstain threshold", 0.30, 0.90,
        default_threshold(method), 0.05, key=f"threshold-{method}",
    )

if st.button("Find evidence", type="primary", disabled=not claim.strip()):
    try:
        checker = load_checker()
        with st.spinner("Retrieving abstracts and comparing evidence sentences..."):
            result = checker.verify(claim.strip(), method, top_k, threshold)
        labels = {
            "SUPPORTS": "Evidence found that supports the claim",
            "CONTRADICTS": "Evidence found that contradicts the claim",
            "MIXED_EVIDENCE": "Mixed evidence found",
            "NOT_ENOUGH_EVIDENCE": "Not enough evidence in this collection",
        }
        st.subheader(labels[result["verdict"]])
        score_a, score_b = st.columns(2)
        score_a.metric("Strongest support score", f"{result['support']:.2f}")
        score_b.metric("Strongest contradiction score", f"{result['contradict']:.2f}")
        st.caption("Scores are model outputs, not calibrated probabilities. The threshold controls abstention; changing it can change the verdict.")
        st.subheader("Evidence sentences")
        if not result["evidence"]:
            st.info("No abstract sentences were available for comparison.")
        for item in result["evidence"]:
            with st.container(border=True):
                st.markdown(f"**{item['title']}** · SciFact document `{item['doc_id']}`")
                st.write(item["sentence"])
                st.caption(f"Entailment {item['entailment']:.2f} · Contradiction {item['contradiction']:.2f}")
        with st.expander("Retrieved abstracts"):
            for doc in result["documents"]:
                st.markdown(f"**{doc.get('title', 'Untitled')}** · `{doc['doc_id']}`")
    except Exception as exc:
        st.error(f"Could not run the checker: {exc}")

with st.expander("What this demo can and cannot do"):
    st.write("The collection contains a limited set of biomedical abstracts. A missing result means only that this collection and model did not find strong evidence. It does not mean a claim is false or that no evidence exists elsewhere.")
    st.write("The model may miss nuance, negation, study quality, or conflicts between papers. Read the source paper and consult qualified experts for health questions.")
