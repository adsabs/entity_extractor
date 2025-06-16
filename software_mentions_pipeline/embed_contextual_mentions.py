# software_mention_pipeline/embed_contextual_mentions.py

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CONTEXTS_PATH = Path("data/contextual_matches.jsonl")
OUTPUT_PATH = Path("data/context_embeddings.json")
MODEL_NAME = "nasa-impact/nasa-smd-ibm-st-v2"

model = SentenceTransformer(MODEL_NAME)


def load_contextual_entries():
    entries = []
    with open(CONTEXTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            bibcode = doc.get("bibcode")
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            labels = doc.get("labels", [])
            contexts = doc.get("contexts", [])

            if not bibcode or not labels or not contexts:
                continue

            # one context per label assumed (if multiple, align index-wise)
            for label, context in zip(labels, contexts):
                full_text = f"{title}\n{abstract}\n{context}"
                entries.append({
                    "bibcode": bibcode,
                    "label": label,
                    "text": full_text
                })
    return entries


def generate_embeddings(entries):
    print(f"🧠 Generating embeddings using model: {MODEL_NAME}")
    for entry in tqdm(entries, desc="🔧 Encoding context entries"):
        embedding = model.encode(entry["text"])
        entry["embedding"] = embedding.tolist()
    return entries


if __name__ == "__main__":
    contextual_entries = load_contextual_entries()
    embedded = generate_embeddings(contextual_entries)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(embedded, f, indent=2)

    print(f"✅ Saved {len(embedded)} context embeddings to {OUTPUT_PATH}")
