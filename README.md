# llm-eval-harness
Benchmarking LLM Reasoning Degradation Under Domain Shift

### Evaluates how well frontier LLMs (GPT-4o, Claude Sonnet, Mistral) maintain reasoning accuracy when prompts shift from general to domain-specific contexts (medical, legal, financial). Implements 5 evaluation dimensions — factual accuracy, hallucination rate, confidence calibration, chain-of-thought coherence, and latency — across 200+ curated prompts per domain. Built with LiteLLM for model abstraction, a custom scoring pipeline in Python, and a Streamlit dashboard for result visualization.
