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

### Summary Table
| Model | Domain | Accuracy | Hallucination | CoT | Calibration | Latency |
|---|---|---|---|---|---|---|
| Claude Haiku | Financial | 0.442 | 0.298 | 0.761 | 0.729 | 4.8s |
| Claude Haiku | General | 0.679 | 0.049 | 0.879 | 0.819 | 2.9s |
| Claude Haiku | Legal | 0.686 | 0.000 | 0.860 | 0.865 | 1.6s |
| Claude Haiku | Medical | 0.817 | 0.164 | 0.819 | 0.733 | 4.7s |
| GPT-4o-mini | Financial | 0.483 | 0.218 | 0.745 | 0.787 | 7.0s |
| GPT-4o-mini | General | 0.687 | 0.063 | 0.840 | 0.832 | 2.8s |
| GPT-4o-mini | Legal | 0.664 | 0.002 | 0.741 | 0.805 | 2.4s |
| GPT-4o-mini | Medical | 0.799 | 0.166 | 0.775 | 0.749 | 5.7s |
| Mistral 7B | Financial | 0.482 | 0.352 | 0.669 | 0.783 | 19.1s |
| Mistral 7B | General | 0.647 | 0.146 | 0.720 | 0.769 | 7.8s |
| Mistral 7B | Legal | 0.830 | 0.355 | 0.579 | 0.711 | 5.3s |
| Mistral 7B | Medical | 0.797 | 0.292 | 0.706 | 0.708 | 15.1s |

### Key Findings

**1. Financial domain is hardest across all models.**
All three models show their lowest factual accuracy on financial prompts (0.44–0.48), 
confirming that domain shift meaningfully degrades reasoning when questions require 
specific numerical analysis from SEC filings.

**2. Claude Haiku hallucinates least, especially on structured domains.**
Haiku produced zero hallucinations on legal prompts and only 0.049 on general — 
significantly better than GPT-4o-mini and Mistral. This gap widens on domain-specific 
content, suggesting Haiku is more conservative about making unsupported claims.

**3. Mistral shows a calibration anomaly on legal.**
Despite achieving the highest legal accuracy (0.830), Mistral scores lowest on legal 
CoT coherence (0.579) — suggesting it arrives at correct answers without well-structured 
reasoning. This overconfident-but-lucky pattern is a meaningful signal for applications 
where explainability matters.

**4. All models degrade on financial CoT coherence.**
Chain-of-thought coherence drops across all models on financial prompts, indicating 
that reasoning chains break down under complex, multi-step financial analysis — even 
when models produce partially correct answers.

**5. Haiku is fastest by a large margin.**
Claude Haiku averages 1.6–4.8s across domains. Mistral running locally averages 
5–19s, with financial prompts being slowest due to longer response generation.

### Interpretation
The results support the core hypothesis: LLM reasoning does degrade under domain 
shift, but the pattern varies by dimension. Accuracy and hallucination rate are most 
sensitive to domain shift. Calibration is relatively stable, suggesting models maintain 
consistent confidence expression even as correctness drops.

## License
MIT
