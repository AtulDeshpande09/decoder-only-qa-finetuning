import json
from logger import write_log

INPUT_FILE = "dataset.jsonl"
OUTPUT_FILE = "dataset_no_duplicates.jsonl"


def normalize(text):
    return text.strip().lower()


def remove_exact_duplicates(input_file, output_file):
    seen = set()
    filtered_count = 0
    original_count = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            original_count += 1
            item = json.loads(line)

            q = normalize(item["question"])

            if q not in seen:
                seen.add(q)
                fout.write(json.dumps(item) + "\n")
                filtered_count += 1

    return original_count, filtered_count


# ======================
# RUN
# ======================
write_log("Exact duplicate removal started.")

original_size, filtered_size = remove_exact_duplicates(
    INPUT_FILE, OUTPUT_FILE
)

removed = original_size - filtered_size

write_log(f"Original size: {original_size}")
write_log(f"Exact duplicates removed: {removed}")
write_log(f"New dataset size: {filtered_size}")
write_log("Exact duplicate removal completed.\n")
