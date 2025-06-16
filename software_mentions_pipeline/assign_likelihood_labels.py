# software_mention_pipeline/assign_likelihood_labels.py
import json
from pathlib import Path
from tqdm import tqdm

INPUT_PATH = Path("data/scored_contexts.jsonl")
OUTPUT_CONTEXTS_PATH = Path("data/labeled_contexts.jsonl")
OUTPUT_SUMMARY_PATH = Path("data/labeled_entities.jsonl")

# Thresholds for scoring
THRESHOLDS = {
    "unlikely": 0.3,
    "somewhat likely": 0.6,
    "very likely": 1.0
}


def determine_likelihood(similarity, ner_tag, heuristic_keywords):
    if similarity is None:
        return "unlikely"

    if similarity >= THRESHOLDS["somewhat likely"] and (ner_tag or heuristic_keywords):
        return "very likely"
    elif similarity >= THRESHOLDS["unlikely"]:
        return "somewhat likely"
    else:
        return "unlikely"


def assign_labels():
    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_CONTEXTS_PATH, "w", encoding="utf-8") as f_out_context, \
         open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f_out_summary:

        for line in tqdm(f_in, desc="📋 Assigning likelihood labels"):
            record = json.loads(line)
            sim = record.get("embedding_similarity")
            ner = record.get("ner_tag")
            heuristics = record.get("heuristic_keywords", [])

            likelihood = determine_likelihood(sim, ner, heuristics)
            record["likelihood"] = likelihood

            # Write enriched context record
            json.dump(record, f_out_context)
            f_out_context.write("\n")

            # Write summary record for deduped ID matching
            json.dump({
                "bibcode": record["bibcode"],
                "label": record["label"],
                "likelihood": likelihood
            }, f_out_summary)
            f_out_summary.write("\n")

    print(f"✅ Likelihood assignment complete. Saved to {OUTPUT_CONTEXTS_PATH} and {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    assign_labels()
