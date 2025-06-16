# software_mention_pipeline/embed_software_library.py

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Paths
ONTO_PATH_ASCL = Path("data/ascl_software.json")
ONTO_PATH_ONTO = Path("data/ontosoft_software.json")
FILTERED_LABELS_PATH = Path("data/filtered_labels.json")
OUTPUT_EMBEDDINGS_PATH = Path("data/software_embeddings.json")
SKIPPED_LOG_PATH = Path("data/skipped_labels.log")

# Embedding model (NASA SMD fine-tuned)
MODEL_NAME = "nasa-impact/nasa-smd-ibm-st-v2"
model = SentenceTransformer(MODEL_NAME)


def load_filtered_labels():
    with open(FILTERED_LABELS_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))


def load_software_entries(filtered_labels):
    entries = []
    skipped = []

    print("🔎 Loading software entries...")

    # --- ASCL ---
    with open(ONTO_PATH_ASCL, "r", encoding="utf-8") as f:
        for entry in json.load(f):
            full_name = entry.get("name", "")
            description = entry.get("description")

            if not full_name or ":" not in full_name:
                continue

            label = full_name.split(":")[0].strip()

            if label not in filtered_labels:
                continue

            if not description:
                skipped.append(f"ASCL: {label} → missing description")
                continue

            text = f"{full_name.strip()}: {description.strip()}"
            entries.append({
                "label": label,
                "source": "ASCL",
                "text": text
            })

    # --- OntoSoft ---
    with open(ONTO_PATH_ONTO, "r", encoding="utf-8") as f:
        for entry in json.load(f):
            label = entry.get("label", "")
            description = entry.get("description")

            if label not in filtered_labels:
                continue

            if not description:
                skipped.append(f"OntoSoft: {label} → missing description")
                continue

            text = f"{label.strip()} - {description.strip()}"
            entries.append({
                "label": label,
                "source": "OntoSoft",
                "text": text
            })

    # Save skipped logs
    with open(SKIPPED_LOG_PATH, "w", encoding="utf-8") as log_f:
        log_f.write("\n".join(skipped))

    print(f"✅ Loaded {len(entries)} valid software entries.")
    print(f"⚠️ Skipped {len(skipped)} entries. Logged to {SKIPPED_LOG_PATH}")
    return entries


def generate_embeddings(entries):
    print("🧠 Generating embeddings with nasa-smd-ibm-st-v2...")
    for entry in tqdm(entries, desc="Embedding entries"):
        embedding = model.encode(entry["text"])
        entry["embedding"] = embedding.tolist()
    return entries


if __name__ == "__main__":
    filtered_labels = load_filtered_labels()
    entries = load_software_entries(filtered_labels)
    embedded = generate_embeddings(entries)

    with open(OUTPUT_EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(embedded, f, indent=2)

    print(f"✅ Software embeddings saved to {OUTPUT_EMBEDDINGS_PATH}")
