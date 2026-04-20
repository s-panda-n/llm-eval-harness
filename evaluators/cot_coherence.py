import litellm
from dotenv import load_dotenv
load_dotenv()

JUDGE_MODEL = "claude-haiku-4-5"

JUDGE_PROMPT = """You are an expert at evaluating logical reasoning. Given a question and a model's response, score the coherence and quality of the chain-of-thought reasoning.

Question: {question}
Model Response: {response}

Evaluate on these criteria:
- Are reasoning steps logical and sequential?
- Does the conclusion follow from the steps?
- Are there any logical leaps or contradictions?

Respond in this exact format:
COT_SCORE: <float between 0.0 and 1.0>
REASONING: <one sentence explanation>

Where 0.0 means no coherent reasoning and 1.0 means perfect step-by-step reasoning."""

def evaluate_cot_coherence(result: dict) -> dict:
    if result["error"] or not result["response"]:
        return {
            "prompt_id": result["prompt_id"],
            "model": result["model"],
            "domain": result["domain"],
            "cot_score": 0.0,
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

        score = 0.0
        reasoning = ""
        for line in raw.split("\n"):
            if line.startswith("COT_SCORE:"):
                score = float(line.split(":")[1].strip())
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return {
            "prompt_id": result["prompt_id"],
            "model": result["model"],
            "domain": result["domain"],
            "cot_score": score,
            "reasoning": reasoning
        }

    except Exception as e:
        return {
            "prompt_id": result["prompt_id"],
            "model": result["model"],
            "domain": result["domain"],
            "cot_score": None,
            "reasoning": f"judge error: {str(e)}"
        }