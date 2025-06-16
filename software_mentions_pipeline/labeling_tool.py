# software_mention_pipeline/labeling_tool.py
import sqlite3

DB_PATH = "data/match_candidates.db"

LABELS = {
    "l": "likely_match",
    "p": "possible",
    "f": "false_match",
    "s": "skip"
}


def label_records():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT rowid, bibcode, label, match_type, context FROM candidates WHERE final_classification IS NULL")
    rows = cur.fetchall()

    print(f"🧪 Found {len(rows)} unlabeled candidates")
    for rowid, bibcode, label, match_type, context in rows:
        print("\n-----------------------------")
        print(f"Bibcode: {bibcode}")
        print(f"Match Type: {match_type}")
        print(f"Label: {label}")
        print(f"Context:\n{context}")
        print("-----------------------------")

        while True:
            choice = input("Label (l=likely, p=possible, f=false, s=skip): ").strip().lower()
            if choice in LABELS:
                if choice != "s":
                    cur.execute("UPDATE candidates SET final_classification = ? WHERE rowid = ?", (LABELS[choice], rowid))
                    conn.commit()
                break

    conn.close()
    print("✅ Labeling complete.")


if __name__ == "__main__":
    label_records()