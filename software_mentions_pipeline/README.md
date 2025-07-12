# Software Mentions Extraction Pipeline

A multi-stage pipeline for identifying and scoring software mentions in academic papers using NLP and machine learning techniques.

## Overview

This pipeline extracts software mentions from a corpus of academic papers by combining:
- Exact string matching against software registries
- Contextual embeddings using sentence transformers
- Heuristic keyword detection
- Named Entity Recognition (NER) models

## Pipeline Architecture

```
software_mentions_pipeline/
├── load_inputs.py                 # Stage 1: Load corpus + metadata, init DB
├── batch_filter.py                # Stage 2: Extract exact/fuzzy matches  
├── batch_filter_context.py        # Stage 3: Extract matches with context windows
├── embed_contextual_mentions.py   # Stage 4: Generate embeddings for contexts
├── embed_software_library.py      # Stage 5: Generate embeddings for software registry
├── score_filtered_contexts.py     # Stage 6: Score context similarity + NER + keywords
├── assign_likelihood_labels.py    # Stage 7A: Assign likelihood labels (rule-based)
├── score_likelihoods_and_filter.py # Stage 7B: Assign likelihood scores (weighted)
├── export_ner_training_data.py    # Stage 8: Export NER training data
├── labeling_tool.py               # Manual annotation interface
├── test_indus_ner_tags.py         # Test NER model functionality
└── data/
    ├── corpus.jsonl               # Input paper corpus (3 GB)
    ├── ontosoft.json              # OntoSoft software metadata
    ├── ascl.json                  # ASCL metadata
    ├── labels.json                # Extracted software labels
    ├── filtered_labels.json       # Manually curated labels
    └── match_candidates.db        # SQLite DB with all results
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data directory:**
   - Place `corpus.jsonl` (academic papers) in `data/`
   - Place `ontosoft.json` and `ascl.json` (software registries) in `data/`

## Pipeline Execution

Run stages in sequence:

```bash
# Stage 1: Initialize database and extract software labels
python load_inputs.py

# Manual step: Curate labels.json → filtered_labels.json
# (Remove astronomy acronyms, units, common words)

# Stage 2-3: Extract exact matches
python batch_filter.py
python batch_filter_context.py

# Stage 4-5: Generate embeddings
python embed_contextual_mentions.py
python embed_software_library.py

# Stage 6: Score contexts using multiple signals
python score_filtered_contexts.py

# Stage 7: Assign likelihood labels (choose one approach)
python assign_likelihood_labels.py
# OR
python score_likelihoods_and_filter.py

# Stage 8: Export training data for NER models (optional)
python export_ner_training_data.py
```

## Key Components

### Matching Strategy
- **Exact matching**: Token-level for single words, regex boundary matching for phrases
- **Context extraction**: ±100 words around each match
- **Deduplication**: Unique (bibcode, label, context) combinations

### Scoring Methods
1. **Embedding similarity**: Cosine similarity between context and software description vectors
2. **Keyword heuristics**: Presence of software-related terms (software, model, algorithm, etc.)
3. **NER classification**: Two models:
   - `oeg/software_benchmark_multidomain`
   - `adsabs/nasa-smd-ibm-v0.1_NER_DEAL` (INDUS)

### Likelihood Classification

**Rule-based (assign_likelihood_labels.py):**
- `very_likely`: similarity ≥ 0.6 AND (NER hit OR keywords)
- `somewhat_likely`: similarity ≥ 0.3
- `unlikely`: everything else

**Weighted scoring (score_likelihoods_and_filter.py):**
- Composite score: 0.5×NER + 0.3×similarity + 0.2×keywords_ratio
- Thresholds: 0.75 / 0.45 / 0.0 → very / somewhat / unlikely

## Manual Annotation

Use the labeling tool for manual annotation and quality assurance:

```bash
python labeling_tool.py
```

This provides a CLI interface to review and label candidates stored in the SQLite database.

## Testing

Test NER model functionality:
```bash
python test_indus_ner_tags.py
```

## Data Formats

### Input Files
- **corpus.jsonl**: One JSON object per line with `bibcode`, `title`, `abstract`, `body`
- **ontosoft.json/ascl.json**: Software registry metadata with descriptions

### Output Files
- **labeled_contexts.jsonl**: Full context records with likelihood labels
- **labeled_entities.jsonl**: Deduplicated (bibcode, label) pairs
- **ner_training_data.jsonl**: SpaCy-format training data

### SQLite Schema (`candidates` table)
| Column | Description |
|--------|-------------|
| bibcode | Paper identifier |
| title, abstract, body | Paper metadata |
| label | Matched software name |
| context | ~200-word snippet around match |
| embedding_similarity | Cosine similarity score |
| ner_tag_* | NER model predictions |
| heuristic_keywords | Matched keyword indicators |
| likelihood | Final classification label |

## Models Used

- **Embeddings**: `nasa-impact/nasa-smd-ibm-st-v2` (SentenceTransformer)
- **NER**: 
  - `oeg/software_benchmark_multidomain`
  - `adsabs/nasa-smd-ibm-v0.1_NER_DEAL`

## Requirements

- Python 3.9+
- See `requirements.txt` for complete dependency list
- ~4GB disk space for data files
- GPU recommended for faster embedding generation
