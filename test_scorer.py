from pipeline.scorer import score_results

score_results("results/run_20260422_194001.json")
# test_scorer.py
# import json
# from pipeline.scorer import score_results

# # filter to just mistral before scoring
# with open("results/run_20260422_194001.json") as f:
#     all_results = json.load(f)

# mistral_only = [r for r in all_results if r["model"] == "ollama/mistral"]

# with open("results/mistral_only.json", "w") as f:
#     json.dump(mistral_only, f)

# score_results("results/mistral_only.json")