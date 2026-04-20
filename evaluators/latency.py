def evaluate_latency(result: dict) -> dict:
    """Latency is already captured by the runner."""
    return {
        "prompt_id": result["prompt_id"],
        "model": result["model"],
        "domain": result["domain"],
        "latency": result["latency"]
    }