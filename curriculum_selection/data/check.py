import json

taco_path = "taco/taco_cleaned.jsonl"
taco_questions = set()

with open(taco_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        question = data.get("question", "").strip()
        if question:
            taco_questions.add(question)

print(f"[TACO] Loaded {len(taco_questions)} unique questions.")

bench_path = "livecodebench.json"
with open(bench_path, 'r', encoding='utf-8') as f:
    codegen_data = json.load(f)

overlap = []
for item in codegen_data:
    q = item.get("question_content", "").strip()
    if q in taco_questions:
        overlap.append(item)

print(f"Found {len(overlap)} overlapping questions.")

