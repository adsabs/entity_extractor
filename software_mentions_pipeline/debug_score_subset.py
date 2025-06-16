# software_mention_pipeline/debug_score_subset.py

import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import torch

SOFTWARE_EMBEDDINGS_PATH = Path("data/software_embeddings.json")
CONTEXT_EMBEDDINGS_PATH = Path("data/context_embeddings.json")
FILTERED_LABELS_PATH = Path("data/filtered_labels.json")
MAX_ENTRIES = 10

RECOGNIZED_SOFTWARE_TAGS = {"SOFTWARE", "TOOL", "Application_Mention"}
SOFTWARE_KEYWORDS = {"software", "model", "algorithm", "code", "we used", "we ran", "tool", "implementation"}

ner_pipe = pipeline(
    "ner",
    model="oeg/software_benchmark_multidomain",
    tokenizer="oeg/software_benchmark_multidomain",
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

def run_ner_on_text(text, label):
    try:
        sentences = sent_tokenize(text)
        relevant = [s for s in sentences if label.lower() in s.lower()]
        for sent in relevant:
            ner_results = ner_pipe(sent[:512])
            for ent in ner_results:
                if ent["entity_group"] in RECOGNIZED_SOFTWARE_TAGS:
                    return True
        return False
    except Exception as e:
        print(f"❌ NER error: {e}")
        return False

def compute_similarity(vec1, vec2):
    return float(cosine_similarity([vec1], [vec2])[0][0])

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    software_embeddings = load_jsonl(SOFTWARE_EMBEDDINGS_PATH)
    software_dict = {entry["label"].lower(): np.array(entry["embedding"]) for entry in software_embeddings}

    context_entries = load_jsonl(CONTEXT_EMBEDDINGS_PATH)[:MAX_ENTRIES]

    with open(FILTERED_LABELS_PATH, "r", encoding="utf-8") as f:
        valid_labels = set(label.lower() for label in json.load(f))

    print(f"\n📦 Loaded {len(software_dict)} software embeddings and {len(context_entries)} context entries.")
    print(f"\n🔍 Inspecting first {MAX_ENTRIES} entries...\n")

    for i, entry in enumerate(context_entries):
        label = entry["label"].lower()
        text = entry["text"]
        bibcode = entry.get("bibcode", "N/A")
        print(f"🔎 [{i}] Bibcode: {bibcode}")
        print(f"   🏷️ Label: {label}")

        if label not in valid_labels:
            print(f"   ⚠️ Label not in filtered_labels.json — skipping similarity")
            continue

        if "embedding" not in entry:
            print(f"   ❌ Context embedding missing.")
            continue

        if label not in software_dict:
            print(f"   ❌ No software embedding found for label.")
            continue

        sim_score = compute_similarity(software_dict[label], np.array(entry["embedding"]))
        ner_tag = run_ner_on_text(text, label)
        keywords = sorted([kw for kw in SOFTWARE_KEYWORDS if kw in text.lower()])

        print(f"   🔑 Keywords found: {keywords}")
        print(f"   🧠 NER tag match: {ner_tag}")
        print(f"   📈 Embedding similarity: {sim_score:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
