from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from claimcheck import ClaimChecker, load_dataset, verdict_for_scores

LABELS = ["SUPPORTS", "CONTRADICTS", "MIXED_EVIDENCE", "NOT_ENOUGH_EVIDENCE"]


def expected_label(row: dict) -> str:
    evidence = row.get("evidence", {})
    evidence_labels = {
        entry.get("label", "").upper()
        for rationales in evidence.values()
        for entry in rationales
    } if isinstance(evidence, dict) else set()
    if "SUPPORT" in evidence_labels and "CONTRADICT" in evidence_labels:
        return "MIXED_EVIDENCE"
    if "SUPPORT" in evidence_labels:
        return "SUPPORTS"
    if "CONTRADICT" in evidence_labels:
        return "CONTRADICTS"
    value = str(row.get("evidence_label", "NOT_ENOUGH_INFO")).upper()
    if value == "SUPPORT":
        return "SUPPORTS"
    if value in {"CONTRADICT", "REFUTES"}:
        return "CONTRADICTS"
    return "NOT_ENOUGH_EVIDENCE"


def score_claim(checker: ClaimChecker, row: dict, method: str, top_k: int) -> tuple[float, float, bool]:
    result = checker.verify(row["claim"], method, top_k, threshold=0.0)
    support, contradict = result["support"], result["contradict"]
    retrieved_ids = {str(doc["doc_id"]) for doc in result["documents"]}
    gold_ids = set(map(str, row.get("evidence", {}).keys()))
    found_gold = bool(gold_ids and retrieved_ids.intersection(gold_ids))
    return support, contradict, found_gold


def tune_threshold(raw: list[tuple[float, float]], labels: list[str]) -> tuple[float, float]:
    best_threshold, best_f1 = 0.55, -1.0
    for threshold in [i / 100 for i in range(30, 86, 5)]:
        preds = [verdict_for_scores(support, contradict, threshold) for support, contradict in raw]
        score = f1_score(labels, preds, labels=LABELS, average="macro", zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = threshold, score
    return best_threshold, best_f1


def main():
    parser = argparse.ArgumentParser(description="Compare SciFact retrieval methods and tune abstention on dev data.")
    parser.add_argument("--limit", type=int, default=0, help="Optional claim limit; 0 uses the full dev split.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    corpus, dev = load_dataset()
    if args.limit:
        dev = dev[:args.limit]
    checker = ClaimChecker(corpus)
    truths = [expected_label(row) for row in dev]
    report = [{
        "retrieval": "None (always abstain)",
        "dev_claims": len(dev),
        "top_k": args.top_k,
        "abstain_threshold": None,
        "evidence_recall_at_k": None,
        "macro_f1_at_default_threshold": None,
        "tuned_macro_f1": f1_score(
            truths, ["NOT_ENOUGH_EVIDENCE"] * len(truths),
            labels=LABELS, average="macro", zero_division=0,
        ),
    }]
    for method in ["BM25", "Dense", "Hybrid"]:
        raw, evidence_hits = [], []
        for row in dev:
            support, contradict, found_gold = score_claim(checker, row, method, args.top_k)
            raw.append((support, contradict))
            if row.get("evidence"):
                evidence_hits.append(found_gold)
        threshold, tuned_f1 = tune_threshold(raw, truths)
        default_preds = [verdict_for_scores(support, contradict, 0.55) for support, contradict in raw]
        report.append({
            "retrieval": method, "dev_claims": len(dev), "top_k": args.top_k,
            "abstain_threshold": threshold,
            "evidence_recall_at_k": sum(evidence_hits) / len(evidence_hits) if evidence_hits else 0.0,
            "macro_f1_at_default_threshold": f1_score(truths, default_preds, labels=LABELS, average="macro", zero_division=0),
            "tuned_macro_f1": tuned_f1,
        })
    out = Path("reports")
    out.mkdir(exist_ok=True)
    path = out / "ablation.csv"
    pd.DataFrame(report).to_csv(path, index=False)
    print(pd.DataFrame(report).to_string(index=False))
    print(f"\nSaved {path}. Threshold tuning and the reported development score use the same split.")


if __name__ == "__main__":
    main()
