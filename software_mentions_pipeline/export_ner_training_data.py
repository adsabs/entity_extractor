# software_mention_pipeline/export_ner_training_data.py
import json
from pathlib import Path
from tqdm import tqdm

INPUT_PATH = Path("data/labeled_contexts.jsonl")
OUTPUT_PATH = Path("data/ner_training_data.jsonl")

SIMILARITY_THRESHOLD = 0.6


def extract_ner_training_examples():
    training_examples = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f_in:
        for line in tqdm(f_in, desc="✂️ Exporting NER training data"):
            record = json.loads(line)
            similarity = record.get("embedding_similarity", 0)
            label = record.get("label")
            text = record.get("context", "")

            if similarity < SIMILARITY_THRESHOLD:
                continue

            # Find all occurrences of the label in the context (case-sensitive exact match)
            spans = []
            start = 0
            while True:
                index = text.find(label, start)
                if index == -1:
                    break
                spans.append({
                    "start": index,
                    "end": index + len(label),
                    "label": "SOFTWARE"
                })
                start = index + len(label)

            if spans:
                training_examples.append({
                    "text": text,
                    "entities": spans
                })

    # Save output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for example in training_examples:
            json.dump(example, f_out)
            f_out.write("\n")

    print(f"✅ Exported {len(training_examples)} NER training examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_ner_training_examples()