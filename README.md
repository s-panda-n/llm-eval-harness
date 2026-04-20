# LLM Eval Harness: Benchmarking Reasoning Degradation Under Domain Shift

## Overview
This project benchmarks how well frontier LLMs maintain reasoning accuracy when prompts shift from general to domain-specific contexts. The core hypothesis is that models degrade — in accuracy, calibration, and reasoning coherence — when faced with specialized terminology and nuance in medical, legal, and financial domains.

## Models Evaluated
| Model | Provider | Role |
|---|---|---|
| GPT-4o-mini | OpenAI | Primary eval model |
| Claude Haiku | Anthropic | Primary eval model + judge LLM |
| Mistral 7B | Ollama (local) | Primary eval model |

## Evaluation Dimensions
| Dimension | Method | Description |
|---|---|---|
| Factual Accuracy | Fuzzy string match vs ground truth | How often the model answers correctly |
| Hallucination Rate | Judge LLM (Claude Haiku) | Whether the response contains fabricated or unsupported claims |
| Confidence Calibration | Linguistic cue analysis | Whether expressed certainty matches actual accuracy |
| CoT Coherence | Judge LLM (Claude Haiku) | Whether the reasoning chain is logical and internally consistent |
| Latency | Wall-clock time | Response time per prompt |

## Dataset
200+ curated prompts across 4 domains sourced from established NLP benchmarks:

| Domain | Source | Size | Ground Truth Type |
|---|---|---|---|
| General | ARC-Challenge (AI2) | 50 | Multiple choice answer |
| Medical | PubMedQA | 50 | yes / no / maybe |
| Legal | LegalBench (contract_qa) | 50 | yes / no |
| Financial | FinDER | 50 | Free-form answer |

## Architecture
```
prompts/          # curated prompt JSONs by domain
evaluators/       # one file per evaluation dimension
pipeline/
  runner.py       # LiteLLM inference loop across models
  scorer.py       # aggregates evaluator outputs into CSVs
dashboard/
  app.py          # Streamlit visualization
results/          # raw responses + scores (gitignored)
```

## Stack
- **LiteLLM** — unified interface across OpenAI, Anthropic, and Ollama
- **Python** — custom scoring pipeline, no eval framework dependencies
- **Streamlit** — interactive results dashboard
- **HuggingFace Datasets** — prompt sourcing

## How to Run

### 1. Setup
```bash
git clone https://github.com/s-panda-n/llm-eval-harness.git
cd llm-eval-harness
python -m venv proj2
source proj2/bin/activate
pip install -r requirements.txt
```

### 2. Add API Keys
```bash
# .env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```
For Mistral, install [Ollama](https://ollama.com) and run `ollama pull mistral`.

### 3. Run the Eval
```bash
python test_runner.py       # runs all models across all domains
python test_scorer.py       # scores all evaluation dimensions
streamlit run dashboard/app.py  # launch dashboard
```

## Methodology Limitations

**Fuzzy matching for factual accuracy is imperfect.** The scorer uses rapidfuzz partial ratio matching against ground truth strings. This penalizes correct answers phrased differently from the reference (e.g. "$0.05" vs "5 cents"). A judge LLM would be more robust but adds cost per prompt.

**Judge LLM introduces circular evaluation.** Claude Haiku scores hallucination and CoT coherence for all three models including itself. This creates a potential bias where Claude may implicitly favor its own reasoning style. A stronger design would use a separate, independent judge model or human annotation for a validation subset.

**Ground truth quality varies by domain.** PubMedQA and LegalBench have clean, verifiable ground truths. FinDER answers are free-form financial analysis which are harder to score objectively — factual accuracy scores for the financial domain should be interpreted with more caution.

**Sample size is modest.** 50 prompts per domain is sufficient to observe trends but not to draw statistically significant conclusions. A rigorous study would require larger samples and multiple eval runs to account for model temperature variance.

**No prompt standardization across domains.** Medical prompts are research questions, legal prompts are clause classification tasks, and financial prompts are analyst-style queries. Differences in score across domains may reflect prompt format differences as much as genuine domain shift effects.

## What a Stronger Design Would Look Like
- Use a held-out third model (e.g. GPT-4o) as judge to avoid circularity
- Human annotation on a 10% validation subset to calibrate judge LLM accuracy
- Standardize prompt format across domains to isolate domain shift from task format effects
- Run multiple eval passes with different random seeds and report confidence intervals
- Expand to 200+ prompts per domain for statistical power

## Results
*Full results available after running the pipeline. See the dashboard for interactive visualizations.*

## License
MIT
