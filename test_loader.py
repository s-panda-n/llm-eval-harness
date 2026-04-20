from prompts.loader import load_prompts

prompts = load_prompts()
for p in prompts:
    print(f"[{p.domain}] {p.id}: {p.question[:60]}...")