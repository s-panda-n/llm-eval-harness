import json
from evaluators.latency import evaluate_latency
from evaluators.factual_accuracy import evaluate_factual_accuracy
from evaluators.hallucination import evaluate_hallucination
from evaluators.cot_coherence import evaluate_cot_coherence
from evaluators.calibration import evaluate_calibration

with open("results/run_20260420_161257.json") as f:
    results = json.load(f)

for r in results:
    lat = evaluate_latency(r)
    acc = evaluate_factual_accuracy(r)
    hal = evaluate_hallucination(r)
    cot = evaluate_cot_coherence(r)
    cal = evaluate_calibration(r, acc["factual_accuracy_score"])

    print(f"\n{r['prompt_id']} | {r['model']}")
    print(f"  Latency:       {lat['latency']}s")
    print(f"  Accuracy:      {acc['factual_accuracy_score']} ({acc['match_type']})")
    print(f"  Hallucination: {hal['hallucination_score']} — {hal['reasoning']}")
    print(f"  CoT Score:     {cot['cot_score']} — {cot['reasoning']}")
    print(f"  Calibration:   {cal['calibration_score']} — {cal['calibration_note']}")
    print(f"  (expressed confidence: {cal['expressed_confidence']})")