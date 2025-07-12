# Agent Configuration

## Build/Test Commands

### Setup
- `pip install -r software_mentions_pipeline/requirements.txt` - Install dependencies
- `python software_mentions_pipeline/test_indus_ner_tags.py` - Test NER model functionality

### Pipeline Execution (run in sequence)
- `python software_mentions_pipeline/load_inputs.py` - Initialize database and load corpus/metadata
- `python software_mentions_pipeline/batch_filter.py` - Stage 1: Extract exact/fuzzy matches
- `python software_mentions_pipeline/embed_contextual_mentions.py` - Generate embeddings for contexts
- `python software_mentions_pipeline/score_filtered_contexts.py` - Score context similarity
- `python software_mentions_pipeline/labeling_tool.py` - Manual labeling interface
- `python software_mentions_pipeline/assign_likelihood_labels.py` - Assign likelihood scores

## Architecture

Entity extraction pipeline for scientific literature using NLP and ML:
- **software_mentions_pipeline/**: Main pipeline for extracting software mentions from academic papers
- **data/**: Contains corpus.jsonl (3GB papers), metadata (OntoSoft, ASCL), SQLite database
- **Database**: SQLite with `candidates` table storing matches, contexts, embeddings, classifications
- **Models**: INDUS NER (adsabs/nasa-smd-ibm-v0.1_NER_DEAL), sentence-transformers for embeddings

## Code Style

### Python (software_mentions_pipeline)
- Use pathlib.Path for file paths
- Constants in UPPERCASE at module level (e.g., `CORPUS_PATH`, `DB_PATH`)
- f-strings for string formatting
- tqdm for progress bars on long operations
- sqlite3 for database operations with proper connection handling
- Use transformers/sentence-transformers for ML models
- JSON/JSONL for data serialization
