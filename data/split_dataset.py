import json
import random
import os
from datetime import datetime

# ======================
# CONFIG
# ======================
INPUT_FILE = "dataset_semantic_filtered.jsonl"
OUTPUT_DIR = "data"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
LOG_FILE = "preprocessing_log.txt"


# ======================
# LOG FUNCTION
# ======================
def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


# ======================
# LOAD JSONL DATA
# ======================
dataset = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

original_size = len(dataset)

write_log("Dataset splitting started.")
write_log(f"Total samples: {original_size}")
write_log(f"Seed used: {SEED}")


# ======================
# SHUFFLE (reproducible)
# ======================
random.seed(SEED)
random.shuffle(dataset)


# ======================
# SPLIT
# ======================
train_end = int(TRAIN_RATIO * original_size)
val_end = train_end + int(VAL_RATIO * original_size)

train_data = dataset[:train_end]
val_data = dataset[train_end:val_end]
test_data = dataset[val_end:]


# ======================
# SAVE JSONL
# ======================
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


save_jsonl(train_data, f"{OUTPUT_DIR}/train.jsonl")
save_jsonl(val_data, f"{OUTPUT_DIR}/val.jsonl")
save_jsonl(test_data, f"{OUTPUT_DIR}/test.jsonl")


# ======================
# LOG RESULTS
# ======================
write_log(f"Train size: {len(train_data)}")
write_log(f"Validation size: {len(val_data)}")
write_log(f"Test size: {len(test_data)}")
write_log("Dataset splitting completed.\n")

print("Dataset split successfully.")
