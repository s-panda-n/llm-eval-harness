from datasets import load_dataset
import json
import os

def save_prompts(prompts: list, domain: str):
    os.makedirs(f"prompts/{domain}", exist_ok=True)
    filepath = f"prompts/{domain}/{domain}.json"
    with open(filepath, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} prompts to {filepath}")


def harvest_medical(n: int = 50):
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    prompts = []
    count = 0
    for i, item in enumerate(dataset["train"]):
        if count >= n:
            break
        try:
            prompts.append({
                "id": f"med_{str(count+1).zfill(3)}",
                "domain": "medical",
                "question": item["question"],
                "ground_truth": item["final_decision"],  # "yes", "no", "maybe"
                "difficulty": "medium"
            })
            count += 1
        except Exception as e:
            print(f"Skipping item {i}: {e}")
    save_prompts(prompts, "medical")


def harvest_general(n: int = 50):
    dataset = load_dataset("ai2_arc", "ARC-Challenge")
    prompts = []
    count = 0
    for i, item in enumerate(dataset["train"]):
        if count >= n:
            break
        try:
            # get the correct answer text from choices
            choices = item["choices"]
            answer_key = item["answerKey"]
            answer_index = choices["label"].index(answer_key)
            answer_text = choices["text"][answer_index]

            prompts.append({
                "id": f"gen_{str(count+1).zfill(3)}",
                "domain": "general",
                "question": item["question"],
                "ground_truth": answer_text,
                "difficulty": "hard"
            })
            count += 1
        except Exception as e:
            print(f"Skipping item {i}: {e}")
    save_prompts(prompts, "general")


def harvest_financial(n: int = 50):
    dataset = load_dataset("Linq-AI-Research/FinDER")
    prompts = []
    count = 0
    first = dataset["train"][0]
    print(f"Financial fields: {list(first.keys())}")

    for i, item in enumerate(dataset["train"]):
        if count >= n:
            break
        try:
            question = item.get("question", "") or item.get("text", "")
            answer = item.get("answer", "") or item.get("reasoning", "")
            if len(question) < 20 or len(str(answer)) < 5:
                continue
            prompts.append({
                "id": f"fin_{str(count+1).zfill(3)}",
                "domain": "financial",
                "question": question,
                "ground_truth": str(answer)[:300],
                "difficulty": "medium"
            })
            count += 1
        except Exception as e:
            print(f"Skipping item {i}: {e}")
    save_prompts(prompts, "financial")


def harvest_legal(n: int = 50):
    # contract_qa only has 8 train examples, use test split too
    train = load_dataset("nguha/legalbench", "contract_qa", split="train")
    test = load_dataset("nguha/legalbench", "contract_qa", split="test")

    prompts = []
    count = 0

    for item in list(train) + list(test):
        if count >= n:
            break
        try:
            question = item.get("question", "")
            answer = item.get("answer", "")
            if len(question) < 20 or len(str(answer)) < 2:
                continue
            prompts.append({
                "id": f"leg_{str(count+1).zfill(3)}",
                "domain": "legal",
                "question": question,
                "ground_truth": str(answer)[:300],
                "difficulty": "hard"
            })
            count += 1
        except Exception as e:
            print(f"Skipping item {i}: {e}")
    save_prompts(prompts, "legal")


if __name__ == "__main__":
    print("Harvesting medical prompts...")
    harvest_medical(50)

    print("\nHarvesting general prompts...")
    harvest_general(50)

    print("\nHarvesting financial prompts...")
    harvest_financial(50)

    print("\nHarvesting legal prompts...")
    harvest_legal(50)

    print("\nDone. Run test_loader.py to verify.")