import litellm
import time
import json
import os
from datetime import datetime
from config import MODELS, MAX_TOKENS, TEMPERATURE
from prompts.loader import load_prompts

litellm.drop_params = True  # avoids errors from unsupported params across models

def run_single(prompt_obj, model: str) -> dict:
    """Run a single prompt against a single model."""
    try:
        start = time.time()
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt_obj.question}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        latency = time.time() - start
        return {
            "prompt_id": prompt_obj.id,
            "domain": prompt_obj.domain,
            "difficulty": prompt_obj.difficulty,
            "question": prompt_obj.question,
            "ground_truth": prompt_obj.ground_truth,
            "model": model,
            "response": response.choices[0].message.content,
            "latency": round(latency, 3),
            "tokens_used": response.usage.total_tokens,
            "error": None
        }
    except Exception as e:
        return {
            "prompt_id": prompt_obj.id,
            "domain": prompt_obj.domain,
            "difficulty": prompt_obj.difficulty,
            "question": prompt_obj.question,
            "ground_truth": prompt_obj.ground_truth,
            "model": model,
            "response": None,
            "latency": None,
            "tokens_used": None,
            "error": str(e)
        }

def run_eval(domains: list = None, models: list = None) -> str:
    """Run full eval across all prompts and models. Saves results to disk."""
    prompts = load_prompts(domains)
    models_to_run = models if models else MODELS
    results = []

    total = len(prompts) * len(models_to_run)
    count = 0

    for prompt_obj in prompts:
        for model in models_to_run:
            count += 1
            print(f"[{count}/{total}] {model} | {prompt_obj.id}")
            result = run_single(prompt_obj, model)
            results.append(result)

    # save to results/
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/run_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {filepath}")
    return filepath