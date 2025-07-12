# Entity Extractor

An intelligent entity extraction system for academic literature, specifically designed to identify and classify software mentions in scientific papers from the Science Explorer (SciX, https://scixplorer.org).

## Overview

This repository contains a comprehensive pipeline for extracting software mentions from academic papers using advanced NLP techniques including embeddings, named entity recognition, and heuristic matching.

## Project Structure

```
entity_extractor/
├── software_mentions_pipeline/    # Main extraction pipeline
│   ├── load_inputs.py            # Data initialization
│   ├── batch_filter*.py          # Exact matching stages
│   ├── embed_*.py                # Embedding generation
│   ├── score_*.py                # Context scoring
│   ├── assign_likelihood_labels.py # Classification
│   ├── labeling_tool.py          # Manual annotation
│   └── data/                     # Data files and outputs
├── AGENT.md                      # Development configuration
└── scix_entity_pipeline_diagram.md # Architecture documentation
```

## Quick Start

### Prerequisites
- Python 3.9+
- 4GB+ disk space for data files
- GPU recommended for faster processing

### Installation
```bash
cd software_mentions_pipeline
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Initialize and run the complete pipeline
python load_inputs.py
python batch_filter.py
python embed_contextual_mentions.py
python score_filtered_contexts.py
python assign_likelihood_labels.py
```

See [`software_mentions_pipeline/README.md`](software_mentions_pipeline/README.md) for detailed instructions.

## What it Does

The pipeline processes academic papers to:

1. **Extract software names** from OntoSoft and ASCL registries
2. **Find exact matches** in paper text (titles, abstracts, body)
3. **Generate embeddings** for context understanding using NASA's SentenceTransformer model
4. **Score matches** using multiple signals:
   - Semantic similarity between context and software description
   - Keyword heuristics (software, model, algorithm, etc.)
   - Named Entity Recognition from specialized models
5. **Classify likelihood** of true software mentions
6. **Export results** for further analysis or model training

## Key Features

- **Multi-modal scoring**: Combines embeddings, NER, and keyword heuristics
- **Modular pipeline**: Each stage can be run independently
- **Manual annotation tools**: Built-in labeling interface for quality assurance
- **Multiple output formats**: JSON, SQLite, NER training data
- **Robust matching**: Handles both exact and fuzzy matching strategies

## Data Sources

- **Paper corpus**: ADS academic papers (corpus.jsonl)
- **Software registries**: 
  - [OntoSoft](https://ontosoft.org/) - Community software registry
  - [ASCL](https://ascl.net/) - Astrophysics Source Code Library

## Models Used

- **Embeddings**: `nasa-impact/nasa-smd-ibm-st-v2`
- **NER Models**:
  - `oeg/software_benchmark_multidomain`
  - `adsabs/nasa-smd-ibm-v0.1_NER_DEAL` (INDUS)

## Development

This project includes configuration for AI development assistance. See [`AGENT.md`](AGENT.md) for:
- Build and test commands
- Architecture overview
- Code style guidelines

## Repository

**GitHub**: https://github.com/adsabs/entity_extractor

## License

Part of the ADS (Astrophysics Data System) project by the Smithsonian Astrophysical Observatory.
