from rapidfuzz import fuzz

def evaluate_factual_accuracy(result: dict) -> dict:
    """
    Scores response against ground truth using fuzzy matching.
    Returns a score between 0 and 1.
    """
    if result["error"] or not result["response"]:
        return {
            "prompt_id": result["prompt_id"],
            "model": result["model"],
            "domain": result["domain"],
            "factual_accuracy_score": 0.0,
            "match_type": "error"
        }

    response = result["response"].lower().strip()
    ground_truth = result["ground_truth"].lower().strip()

    # exact match
    if ground_truth in response:
        score = 1.0
        match_type = "exact"
    else:
        # fuzzy match — partial ratio handles cases where answer is buried in longer response
        ratio = fuzz.partial_ratio(ground_truth, response) / 100.0
        score = round(ratio, 3)
        match_type = "fuzzy"

    return {
        "prompt_id": result["prompt_id"],
        "model": result["model"],
        "domain": result["domain"],
        "factual_accuracy_score": score,
        "match_type": match_type
    }