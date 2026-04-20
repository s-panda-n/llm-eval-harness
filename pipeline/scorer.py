import json
import pandas as pd
import os
from evaluators.latency import evaluate_latency
from evaluators.factual_accuracy import evaluate_factual_accuracy
from evaluators.hallucination import evaluate_hallucination
from evaluators.cot_coherence import evaluate_cot_coherence
from evaluators.calibration import evaluate_calibration


def score_results(results_filepath: str) -> str:
    """
    Runs all evaluators on a results file and produces two CSVs:
    - scores_raw.csv: one row per prompt per model
    - scores_summary.csv: aggregated by model and domain
    """
    with open(results_filepath) as f:
        results = json.load(f)

    rows = []

    for r in results:
        lat = evaluate_latency(r)
        acc = evaluate_factual_accuracy(r)
        hal = evaluate_hallucination(r)
        cot = evaluate_cot_coherence(r)
        cal = evaluate_calibration(r, acc["factual_accuracy_score"])

        rows.append({
            "prompt_id": r["prompt_id"],
            "domain": r["domain"],
            "difficulty": r["difficulty"],
            "model": r["model"],
            "latency": lat["latency"],
            "factual_accuracy": acc["factual_accuracy_score"],
            "match_type": acc["match_type"],
            "hallucination_score": hal["hallucination_score"],
            "hallucination_reasoning": hal["reasoning"],
            "cot_score": cot["cot_score"],
            "cot_reasoning": cot["reasoning"],
            "expressed_confidence": cal["expressed_confidence"],
            "calibration_score": cal["calibration_score"],
            "calibration_note": cal["calibration_note"]
        })

    raw_df = pd.DataFrame(rows)

    # summary: average scores per model + domain
    summary_df = raw_df.groupby(["model", "domain"]).agg(
        avg_latency=("latency", "mean"),
        avg_factual_accuracy=("factual_accuracy", "mean"),
        avg_hallucination=("hallucination_score", "mean"),
        avg_cot=("cot_score", "mean"),
        avg_calibration=("calibration_score", "mean"),
        prompt_count=("prompt_id", "count")
    ).reset_index()

    summary_df = summary_df.round(3)

    # save both
    base = os.path.splitext(results_filepath)[0]
    raw_path = base + "_scores_raw.csv"
    summary_path = base + "_scores_summary.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Raw scores saved to:     {raw_path}")
    print(f"Summary scores saved to: {summary_path}")
    print("\n--- Summary ---")
    print(summary_df.to_string(index=False))

    return summary_path