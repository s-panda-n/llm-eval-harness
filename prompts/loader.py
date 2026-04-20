import json
import os
from prompts.prompt_schema import Prompt

def load_prompts(domains: list = None) -> list:
    """Load prompts from JSON files for given domains."""
    all_prompts = []
    base_dir = os.path.dirname(__file__)

    available_domains = ["general", "medical", "legal", "financial"]
    domains_to_load = domains if domains else available_domains

    for domain in domains_to_load:
        filepath = os.path.join(base_dir, domain, f"{domain}.json")
        if not os.path.exists(filepath):
            print(f"Warning: No prompt file found for domain '{domain}'")
            continue

        with open(filepath, "r") as f:
            raw = json.load(f)

        for item in raw:
            all_prompts.append(Prompt(**item))

    print(f"Loaded {len(all_prompts)} prompts across {len(domains_to_load)} domains")
    return all_prompts