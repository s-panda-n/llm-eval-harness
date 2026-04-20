from dotenv import load_dotenv
load_dotenv()

MODELS = [
    "gpt-4o-mini",
    "claude-haiku-4-5",
    "ollama/mistral"
]

DOMAINS = ["general", "medical", "legal", "financial"]

MAX_TOKENS = 500
TEMPERATURE = 0.0   # deterministic outputs, important for eval