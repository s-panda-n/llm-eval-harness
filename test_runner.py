from pipeline.runner import run_eval

# test with just general domain and one model first
filepath = run_eval(
    domains=["general"],
    models=["ollama/mistral"]
)