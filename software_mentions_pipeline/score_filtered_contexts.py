# software_mention_pipeline/score_filtered_contexts.py

import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# === File paths ===
CONTEXT_EMBEDDINGS_PATH = Path("data/context_embeddings.json")
SOFTWARE_EMBEDDINGS_PATH = Path("data/software_embeddings.json")
FILTERED_LABELS_PATH = Path("data/filtered_labels.json")
OUTPUT_PATH = Path("data/scored_contexts.jsonl")

# === Heuristic setup ===
SOFTWARE_KEYWORDS = {
    "software", "model", "algorithm", "code", "we used",
    "we ran", "tool", "implementation"
}
RECOGNIZED_SOFTWARE_TAGS = {"SOFTWARE", "Software", "tool", "software", "application_mention", "TOOL", "Tool", "Application_Mention"}

# === Load NER models ===
device = 0 if torch.cuda.is_available() else -1
ner_pipes = {
    "oeg": pipeline(
        "ner",
        model="oeg/software_benchmark_multidomain",
        tokenizer="oeg/software_benchmark_multidomain",
        aggregation_strategy="simple",
        device=device
    ),
    "ibm": pipeline(
        "ner",
        model="adsabs/nasa-smd-ibm-v0.1_NER_DEAL",
        tokenizer="adsabs/nasa-smd-ibm-v0.1_NER_DEAL",
        aggregation_strategy="simple",
        device=device
    )
}


def load_embeddings():
    with open(SOFTWARE_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        software_embeddings = {
            entry["label"]: np.array(entry["embedding"]) for entry in json.load(f)
        }

    with open(CONTEXT_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        context_entries = json.load(f)

    with open(FILTERED_LABELS_PATH, "r", encoding="utf-8") as f:
        valid_labels = set(json.load(f))  # keep original casing
    return software_embeddings, context_entries, valid_labels


def compute_similarity(vec1, vec2):
    return float(cosine_similarity([vec1], [vec2])[0][0])


def run_ner_on_text(text, label, model_key):
    try:
        ner_model = ner_pipes[model_key]
        sentences = sent_tokenize(text)
        relevant = [s for s in sentences if label in s]
        for sent in relevant:
            ner_results = ner_model(sent[:512])
            for ent in ner_results:
                if ent["entity_group"] in RECOGNIZED_SOFTWARE_TAGS:
                    return True
        return False
    except Exception as e:
        print(f"⚠️ NER-{model_key} error: {e}")
        return False


def keyword_hits(text):
    return sorted([kw for kw in SOFTWARE_KEYWORDS if kw in text.lower()])


def main():
    software_embeddings, context_entries, valid_labels = load_embeddings()
    print(f"✅ Loaded {len(software_embeddings)} software embeddings and {len(context_entries)} context entries.")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for entry in tqdm(context_entries, desc="📊 Scoring context entries"):
            label = entry["label"]  # preserve casing
            if label not in valid_labels:
                continue

            context_text = entry["text"]
            context_vec = np.array(entry["embedding"])

            heuristics = keyword_hits(context_text)
            ner_tag_oeg = run_ner_on_text(context_text, label, "oeg")
            ner_tag_ibm = run_ner_on_text(context_text, label, "ibm")
            ner_tag_any = ner_tag_oeg or ner_tag_ibm

            sim_score = None
            if label in software_embeddings:
                sim_score = compute_similarity(software_embeddings[label], context_vec)
            else:
                print(f"⚠️ No software embedding found for label: {label}")

            result = {
                "bibcode": entry["bibcode"],
                "label": label,
                "heuristic_keywords": heuristics,
                "ner_tag_oeg": ner_tag_oeg,
                "ner_tag_ibm": ner_tag_ibm,
                "ner_tag_any": ner_tag_any,
                "embedding_similarity": sim_score
            }
            json.dump(result, out_f)
            out_f.write("\n")

    print(f"✅ Scoring complete. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
