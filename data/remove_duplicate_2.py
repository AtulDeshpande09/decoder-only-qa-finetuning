import json
import numpy as np
from sentence_transformers import SentenceTransformer
from logger import write_log

INPUT_FILE = "dataset_no_duplicates.jsonl"
OUTPUT_FILE = "dataset_semantic_filtered.jsonl"

SIM_THRESHOLD = 0.90

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_filter(input_file, output_file):
    embeddings = []
    original_count = 0
    filtered_count = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            original_count += 1
            item = json.loads(line)

            question = item["question"]
            emb = model.encode(question, normalize_embeddings=True)

            keep = True
            for prev_emb in embeddings:
                sim = cosine_similarity(emb, prev_emb)
                if sim > SIM_THRESHOLD:
                    keep = False
                    break

            if keep:
                embeddings.append(emb)
                fout.write(json.dumps(item) + "\n")
                filtered_count += 1

    return original_count, filtered_count


# ======================
# RUN
# ======================
write_log("Semantic filtering started.")
write_log(f"Similarity threshold: {SIM_THRESHOLD}")

original_size, filtered_size = semantic_filter(
    INPUT_FILE, OUTPUT_FILE
)

removed = original_size - filtered_size

write_log(f"Original dataset size: {original_size}")
write_log(f"Semantic duplicates removed: {removed}")
write_log(f"Final dataset size: {filtered_size}")
write_log("Semantic filtering completed.\n")
