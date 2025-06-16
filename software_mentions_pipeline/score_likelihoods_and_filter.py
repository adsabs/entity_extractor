# software_mention_pipeline/score_likelihoods_and_filter.py

import json
from pathlib import Path
from tqdm import tqdm

SCORED_CONTEXTS_PATH = Path("data/scored_contexts.jsonl")
ALL_OUTPUT_PATH = Path("data/all_scored_mentions.jsonl")
LIKELY_OUTPUT_PATH = Path("data/likely_mentions.jsonl")

# === Threshold Tuning Weights ===
WEIGHT_NER = 0.5
WEIGHT_EMBED = 0.3
WEIGHT_KEYWORDS = 0.2

# === Likelihood Categories ===
LIKELIHOOD_THRESHOLDS = {
    "very likely": 0.75,
    "somewhat likely": 0.45,
    "unlikely": 0.0
}


def calculate_score(entry):
    score = 0.0
    score += WEIGHT_NER * (1.0 if entry.get("ner_tag") else 0.0)
    embed = entry.get("embedding_similarity")
    score += WEIGHT_EMBED * (embed if embed is not None else 0.0)
    keyword_count = len(entry.get("heuristic_keywords", []))
    score += WEIGHT_KEYWORDS * min(keyword_count / 3.0, 1.0)
    return round(score, 3)


def assign_likelihood(score):
    for label, threshold in LIKELIHOOD_THRESHOLDS.items():
        if score >= threshold:
            return label
    return "unlikely"


def score_and_filter():
    with open(SCORED_CONTEXTS_PATH, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    all_scored = []
    likely_mentions = []

    for entry in tqdm(entries, desc="🧮 Scoring and classifying mentions"):
        score = calculate_score(entry)
        likelihood = assign_likelihood(score)

        result = {
            "bibcode": entry["bibcode"],
            "label": entry["label"],
            "likelihood": likelihood,
            "score": score
        }
        all_scored.append(result)

        if likelihood in {"somewhat likely", "very likely"}:
            likely_mentions.append(result)

    with open(ALL_OUTPUT_PATH, "w", encoding="utf-8") as out_all:
        for r in all_scored:
            json.dump(r, out_all)
            out_all.write("\n")

    with open(LIKELY_OUTPUT_PATH, "w", encoding="utf-8") as out_likely:
        for r in likely_mentions:
            json.dump(r, out_likely)
            out_likely.write("\n")

    print(f"✅ Wrote {len(all_scored)} total mentions to {ALL_OUTPUT_PATH}")
    print(f"✅ Wrote {len(likely_mentions)} likely mentions to {LIKELY_OUTPUT_PATH}")


if __name__ == "__main__":
    score_and_filter()
