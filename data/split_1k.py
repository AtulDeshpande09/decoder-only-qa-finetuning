import json
import random

SEED = 42
path = "dataset_1k.jsonl"
INPUT_FILE = "dataset_semantic_filtered.jsonl"

dataset = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

random.seed(SEED)
random.shuffle(dataset)


data = dataset[:1000]

print(f"Dataset size {len(data)}")

with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done!!!")