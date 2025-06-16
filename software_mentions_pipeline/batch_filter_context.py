# software_mention_pipeline/batch_filter_context.py
import json
from pathlib import Path
from tqdm import tqdm
import re

CORPUS_PATH = Path("data/corpus.jsonl")
LABELS_PATH = Path("data/filtered_labels.json")
OUTPUT_PATH = Path("data/contextual_matches.jsonl")
WINDOW_WORDS = 100


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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


def extract_context(text, label):
    words = re.findall(r"\w+|\W+", text)
    lowered_words = [w.lower() for w in words]
    label_lower = label.lower()
    match_indices = [i for i, word in enumerate(lowered_words) if word == label_lower]
    contexts = []
    for i in match_indices:
        start = max(0, i - WINDOW_WORDS)
        end = min(len(words), i + WINDOW_WORDS + 1)
        context = ''.join(words[start:end]).replace("\n", " ")
        contexts.append(context.strip())
    return contexts


def contextual_exact_match_pass(label_list):
    total_lines = count_lines(CORPUS_PATH)

    # Separate single-word and phrase labels
    token_labels = {label for label in label_list if " " not in label}
    phrase_labels = {label for label in label_list if " " in label}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as outf:
        for doc in tqdm(load_corpus(), total=total_lines, desc="🧠 Extracting contextual matches"):
            title = " ".join(doc.get("title", [])) if isinstance(doc.get("title", []), list) else doc.get("title", "")
            abstract = doc.get("abstract", "")
            body = doc.get("body", "")
            bibcode = doc.get("bibcode", "")
            if not bibcode:
                continue

            full_text = f"{title} {abstract} {body}"
            tokens = set(re.findall(r"\b[\w\-]+\b", full_text))

            found_labels = []
            found_contexts = []

            for label in token_labels:
                if label in tokens:
                    found_labels.append(label)
                    found_contexts.extend(extract_context(full_text, label))

            for label in phrase_labels:
                if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", full_text):
                    found_labels.append(label)
                    found_contexts.extend(extract_context(full_text, label))

            if found_labels:
                json.dump({
                    "bibcode": bibcode,
                    "title": title,
                    "abstract": abstract,
                    "labels": found_labels,
                    "contexts": found_contexts
                }, outf)
                outf.write("\n")


if __name__ == "__main__":
    label_list = load_labels()
    contextual_exact_match_pass(label_list)
    print(f"✅ Contextual match pass complete. Results saved to {OUTPUT_PATH}")