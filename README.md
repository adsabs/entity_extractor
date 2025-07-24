# Agent Configuration

🔧 **Development Environment Setup for Entity Extractor**

Always activate the local Python environment before running any code or installing dependencies.

## Project Bootstrap Checklist
Note: If running this on scixmuse the virtual environment has already been created and so you just need to cd to entity_extractor, type source venv/bin/activate, then cd to streamlit_dashboard and type streamlit run app.py
```bash
# 1. Environment setup
python -m venv venv && source venv/bin/activate
pip install --upgrade pip setuptools wheel

# 2. Install dependencies
pip install -r software_mentions_pipeline/requirements.txt

# 3. Install development tools (optional but recommended)
pip install black isort flake8 mypy pytest pre-commit

# 4. Setup pre-commit hooks (optional)
pre-commit install

# 5. Test installation
python software_mentions_pipeline/test_indus_ner_tags.py
```

## Build/Test Commands

### Environment Management
- `source venv/bin/activate` - Activate virtual environment (ALWAYS FIRST)
- `pip install -r software_mentions_pipeline/requirements.txt` - Install dependencies
- `pip list | grep -E "(transformers|sentence-transformers|streamlit)"` - Verify key packages

### Testing & Validation
- `python software_mentions_pipeline/test_indus_ner_tags.py` - Test NER model functionality
- `pytest software_mentions_pipeline/tests/` - Run unit tests (if available)
- `python -c "import transformers; print('✓ Transformers OK')"` - Quick dependency check

### Pipeline Execution (Sequential Stages)
1. `python software_mentions_pipeline/load_inputs.py` - Initialize database and load corpus/metadata
2. `python software_mentions_pipeline/batch_filter.py` - Stage 1: Extract exact/fuzzy matches  
3. `python software_mentions_pipeline/embed_contextual_mentions.py` - Generate embeddings for contexts
4. `python software_mentions_pipeline/score_filtered_contexts.py` - Score context similarity
5. `python software_mentions_pipeline/labeling_tool.py` - Manual labeling interface
6. `python software_mentions_pipeline/assign_likelihood_labels.py` - Assign likelihood scores

### Dashboard & Interactive Tools
- `streamlit run streamlit_dashboard/app.py` - Launch streamlit dashboard

## Architecture

Entity extraction pipeline for scientific literature using NLP and ML:
- **software_mentions_pipeline/**: Main pipeline for extracting software mentions from academic papers
- **data/**: Contains corpus.jsonl (3GB papers), metadata (OntoSoft, ASCL), SQLite database
- **Database**: SQLite with `candidates` table storing matches, contexts, embeddings, classifications
- **Models**: INDUS NER (adsabs/nasa-smd-ibm-v0.1_NER_DEAL), sentence-transformers for embeddings

## Coding Standards & Best Practices

### Python Code Style
- **Formatting**: Black 23.7+ with line length 88, isort profile=black
- **Type hints**: Use throughout, mypy strict mode (opt-in per module)
- **Docstrings**: Google style for all public functions and classes
- **File paths**: Always use `pathlib.Path` instead of string concatenation
- **Constants**: UPPERCASE at module level (e.g., `CORPUS_PATH`, `DB_PATH`)
- **String formatting**: f-strings preferred over `.format()` or `%`

### Scientific Computing Conventions
- **Progress bars**: Use `tqdm` for long-running operations (>10 seconds)
- **Database**: SQLite with proper connection handling and WAL mode
- **ML Models**: HuggingFace transformers with explicit device management
- **Data serialization**: JSONL for streaming, JSON for small configs
- **Memory management**: Use generators for large datasets, chunked processing

### Environment Variables
Document key environment variables in code:
- `SOFTEXTRACT_DB_PATH` - Override default database location
- `HF_HOME` - HuggingFace cache directory
- `CUDA_VISIBLE_DEVICES` - GPU selection for embedding generation
- `STREAMLIT_SERVER_PORT` - Dashboard port configuration

### Error Handling
- Use specific exception types, not bare `except:`
- Log errors with context using Python logging module
- Provide actionable error messages with suggested fixes
- Handle GPU/CPU fallback gracefully in ML components

### Testing Philosophy
- **Unit tests**: pytest for individual functions and components
- **Integration tests**: End-to-end pipeline validation with sample data  
- **Performance tests**: Benchmark critical paths (embedding, filtering)
- **Smoke tests**: Quick validation that all imports and basic functionality work

### Git Workflow
- **Branch naming**: `feature/description`, `bugfix/issue-123`, `docs/section-name`
- **Commit messages**: Conventional commits format with scope
- **Pre-commit hooks**: Run black, isort, flake8 before commits
- **PR checklist**: Tests pass, documentation updated, performance unchanged
