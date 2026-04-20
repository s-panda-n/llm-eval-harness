def extract_confidence(response: str) -> float:
    """
    Infers confidence level from language cues in the response.
    Returns a float between 0.0 (very uncertain) and 1.0 (very certain).
    """
    response_lower = response.lower()

    high_confidence = [
        "definitely", "certainly", "clearly", "obviously", "without a doubt",
        "i am certain", "i'm certain", "the answer is", "absolutely", "undoubtedly"
    ]
    medium_confidence = [
        "i think", "i believe", "likely", "probably", "seems", "appears",
        "should be", "most likely", "generally"
    ]
    low_confidence = [
        "i'm not sure", "i am not sure", "possibly", "maybe", "might",
        "could be", "uncertain", "not certain", "i'm unsure", "unclear"
    ]

    # count cues
    high_hits = sum(1 for phrase in high_confidence if phrase in response_lower)
    medium_hits = sum(1 for phrase in medium_confidence if phrase in response_lower)
    low_hits = sum(1 for phrase in low_confidence if phrase in response_lower)

    # if no cues detected, assume medium-high confidence (models rarely hedge by default)
    if high_hits == 0 and medium_hits == 0 and low_hits == 0:
        return 0.75

    total = high_hits + medium_hits + low_hits
    weighted = (high_hits * 1.0 + medium_hits * 0.5 + low_hits * 0.1) / total
    return round(weighted, 3)


def evaluate_calibration(result: dict, factual_accuracy_score: float) -> dict:
    """
    Calibration score measures how well expressed confidence matches actual accuracy.
    A well-calibrated model is certain when correct and uncertain when wrong.
    Score of 1.0 = perfect calibration, 0.0 = completely miscalibrated.
    """
    if result["error"] or not result["response"]:
        return {
            "prompt_id": result["prompt_id"],
            "model": result["model"],
            "domain": result["domain"],
            "expressed_confidence": None,
            "factual_accuracy_score": factual_accuracy_score,
            "calibration_score": 0.0,
            "calibration_note": "no response generated"
        }

    expressed_confidence = extract_confidence(result["response"])

    # calibration = 1 - absolute gap between confidence and accuracy
    # e.g. confident + correct = good, confident + wrong = bad
    calibration_score = round(1.0 - abs(expressed_confidence - factual_accuracy_score), 3)

    # interpret the result
    if expressed_confidence > 0.7 and factual_accuracy_score < 0.4:
        note = "overconfident — high certainty but low accuracy"
    elif expressed_confidence < 0.4 and factual_accuracy_score > 0.7:
        note = "underconfident — hedging despite being correct"
    else:
        note = "well calibrated"

    return {
        "prompt_id": result["prompt_id"],
        "model": result["model"],
        "domain": result["domain"],
        "expressed_confidence": expressed_confidence,
        "factual_accuracy_score": factual_accuracy_score,
        "calibration_score": calibration_score,
        "calibration_note": note
    }