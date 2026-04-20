import litellm
import os
from dotenv import load_dotenv
load_dotenv()

JUDGE_MODEL = "claude-haiku-4-5"

JUDGE_PROMPT = """You are an expert fact-checker. Given a question and a model's response, determine if the response contains any hallucinations — statements that are fabricated, unsupported, or factually incorrect.

Question: {question}
Model Response: {response}

Respond in this exact format:
HALLUCINATION_SCORE: <float between 0.0 and 1.0>
REASONING: <one sentence explanation>

Where 0.0 means no hallucinations detected and 1.0 means the response is entirely hallucinated."""

def evaluate_hallucination(result: dict) -> dict:
    if result["error"] or not result["response"]:
        return {
            "prompt_id": result["prompt_id"],
            "model": result["model"],
            "domain": result["domain"],
            "hallucination_score": 1.0,
            "reasoning": "no response generated"
        }

    prompt = JUDGE_PROMPT.format(
        question=result["question"],
        response=result["response"]
    )

    try:
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0
        )
        raw = response.choices[0].message.content.strip()

        # parse out score and reasoning
        score = 0.0
        reasoning = ""
        for line in raw.split("\n"):
            if line.startswith("HALLUCINATION_SCORE:"):
                score = float(line.split(":")[1].strip())
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return {
            "prompt_id": result["prompt_id"],
            "model": result["model"],
            "domain": result["domain"],
            "hallucination_score": score,
            "reasoning": reasoning
        }

    except Exception as e:
        return {
            "prompt_id": result["prompt_id"],
            "model": result["model"],
            "domain": result["domain"],
            "hallucination_score": None,
            "reasoning": f"judge error: {str(e)}"
        }