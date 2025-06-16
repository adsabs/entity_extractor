# software_mention_pipeline/batch_filter.py
import json
from pathlib import Path
from tqdm import tqdm
import re

CORPUS_PATH = Path("data/corpus.jsonl")
LABELS_PATH = Path("data/filtered_labels.json")
OUTPUT_PATH = Path("data/matched_bibcodes.jsonl")


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))


def load_corpus():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def exact_match_pass(label_set):
    total_lines = count_lines(CORPUS_PATH)

    # Separate single-word and phrase labels
    token_labels = {label for label in label_set if " " not in label}
    phrase_labels = {label for label in label_set if " " in label}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as outf:
        for doc in tqdm(load_corpus(), total=total_lines, desc="🔍 Strict exact matches"):
            title = " ".join(doc.get("title", [])) if isinstance(doc.get("title", []), list) else doc.get("title", "")
            abstract = doc.get("abstract", "")
            body = doc.get("body", "")
            bibcode = doc.get("bibcode", "")
            if not bibcode:
                continue

            full_text = f"{title} {abstract} {body}"
            tokens = set(re.findall(r"\b[\w\-]+\b", full_text))

            found_labels = set()
            for label in token_labels:
                if label in tokens:
                    found_labels.add(label)
            for label in phrase_labels:
                if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", full_text):
                    found_labels.add(label)

            for label in found_labels:
                json.dump({
                    "bibcode": bibcode,
                    "title": title,
                    "label": label
                }, outf)
                outf.write("\n")


if __name__ == "__main__":
    label_set = load_labels()
    exact_match_pass(label_set)
    print(f"✅ Exact match pass complete. Results saved to {OUTPUT_PATH}")
