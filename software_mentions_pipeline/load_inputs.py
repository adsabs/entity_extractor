# software_mention_pipeline/load_inputs.py
import json
import sqlite3
from pathlib import Path

CORPUS_PATH = Path("data/corpus.jsonl")
ONTOSOFT_PATH = Path("data/ontosoft.json")
ASCL_PATH = Path("data/ascl.json")
DB_PATH = Path("data/match_candidates.db")
LABELS_PATH = Path("data/labels.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY,
            bibcode TEXT,
            title TEXT,
            abstract TEXT,
            body TEXT,
            label TEXT,
            match_type TEXT,
            context TEXT,
            final_classification TEXT,
            UNIQUE(bibcode, label, context)
        )
    """)
    conn.commit()
    return conn


def load_corpus():
    for doc in load_jsonl(CORPUS_PATH):
        yield {
            "bibcode": doc.get("bibcode"),
            "title": " ".join(doc.get("title", [])),
            "abstract": doc.get("abstract", ""),
            "body": doc.get("body", "")
        }


def load_software_labels():
    all_labels = set()
    for sw in load_json(ONTOSOFT_PATH):
        if "label" in sw:
            all_labels.add(sw["label"].strip())
    for sw in load_json(ASCL_PATH):
        if "name" in sw:
            raw_name = sw["name"]
            first_word = raw_name.split(":")[0].strip()
            all_labels.add(first_word)
    return sorted(all_labels)


def save_labels(labels):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


if __name__ == "__main__":
    print("✅ Loading software labels...")
    labels = load_software_labels()
    save_labels(labels)
    print(f"✅ Saved {len(labels)} labels to {LABELS_PATH}")

    print("🧱 Initializing database...")
    conn = init_db()
    print("📦 Done.")
