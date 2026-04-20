from dataclasses import dataclass

@dataclass
class Prompt:
    id: str                  # e.g. "med_001"
    domain: str              # "medical", "legal", "financial", "general"
    question: str            # the actual prompt text
    ground_truth: str        # correct answer, used for factual accuracy scoring
    difficulty: str          # "easy", "medium", "hard"