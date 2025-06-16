```
software_mention_pipeline/
├── load_inputs.py               # Load corpus + software metadata, init DB
├── batch_filter.py              # Stage 1: extract exact/fuzzy matches
├── feature_builder.py           # Stage 2–3: embeddings + heuristic features
├── run_ner_models.py            # Stage 4: NER labeling from Indus, etc.
├── score_and_rank.py            # Stage 5: XGBoost classifier + DB write
├── data/
│   ├── corpus.jsonl             # Input paper corpus (3 GB)
│   ├── ontosoft.json            # OntoSoft software metadata
│   ├── ascl.json                # ASCL metadata
│   └── match_candidates.db      # SQLite DB with all results
```

---

### ⚙️ Setup
```bash
pip install -r requirements.txt
```
Add to `requirements.txt`:
```txt
transformers
sentence-transformers
xgboost
scikit-learn
pandas
joblib
```

---

### 🚀 Pipeline Execution
Run each step in sequence:
```bash
python load_inputs.py
python batch_filter.py
python feature_builder.py
python run_ner_models.py
python score_and_rank.py
```

---

### 🧪 Notes
- `score_and_rank.py` requires a column `final_classification` with labels to train
- Add these manually or use `labeling_tool.py` (see below)
- You can re-run classification after model training to populate predictions

---

### 🛠 SQLite Schema (`candidates` table)
| Column              | Description                         |
|---------------------|-------------------------------------|
| bibcode             | Paper ID                            |
| title, abstract     | Metadata from the paper             |
| label               | Matched software label              |
| match_type          | "exact" or "fuzzy"                  |
| context             | ~50-word snippet around match       |
| context_score       | Embedding sim: context ↔ label desc |
| abstract_score      | Embedding sim: abstract ↔ desc      |
| has_heuristics      | 1 if context contains software terms|
| ner_indus           | NER result from Indus               |
| ner_multidomain     | NER result from multidomain model   |
| final_classification| Manual or predicted match label     |

---

### 🖍 `labeling_tool.py` (planned)
This module will:
- Load candidates from DB where `final_classification IS NULL`
- Display context and label, and accept human input (`likely`, `uncertain`, `false`)
- Write decisions back to the DB

---