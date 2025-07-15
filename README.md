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
├── streamlit_dashboard/          # Interactive web dashboard
│   ├── app.py                    # Main Streamlit application
│   ├── components/               # UI components
│   ├── core_pipeline/            # Pipeline wrappers
│   ├── sample_data/              # Sample data for testing
│   └── README.md                 # Dashboard documentation
├── AGENT.md                      # Development configuration
└── scix_entity_pipeline_diagram.md # Architecture documentation
```

## Quick Start

### Option 1: Interactive Dashboard (Recommended)

**For rapid testing and experimentation:**

```bash
# Install dependencies
pip install streamlit pandas numpy transformers sentence-transformers tqdm

# Launch the dashboard
cd streamlit_dashboard
python launch_app.py
```

Open your browser to `http://localhost:8501` for an interactive interface with:
- Entity input and ontology selection
- Real-time pipeline execution
- Interactive results with filtering and export
- Parameter tuning and visualization

### Option 2: Command Line Pipeline

**For production/batch processing:**

```bash
# Install dependencies
cd software_mentions_pipeline
pip install -r requirements.txt

# Run the complete pipeline
python load_inputs.py
python batch_filter.py
python embed_contextual_mentions.py
python score_filtered_contexts.py
python assign_likelihood_labels.py
```

### Prerequisites
- Python 3.9+
- 4GB+ disk space for data files
- GPU recommended for faster processing

See [`software_mentions_pipeline/README.md`](software_mentions_pipeline/README.md) for detailed CLI instructions.

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

### Core Pipeline
- **Multi-modal scoring**: Combines embeddings, NER, and keyword heuristics
- **Modular pipeline**: Each stage can be run independently
- **Manual annotation tools**: Built-in labeling interface for quality assurance
- **Multiple output formats**: JSON, SQLite, NER training data
- **Robust matching**: Handles both exact and fuzzy matching strategies

### Interactive Dashboard
- **Real-time experimentation**: Test different parameters instantly
- **Entity management**: Custom entities or ontology selection (OntoSoft, ASCL)
- **Parameter tuning**: Adjust context windows, thresholds, and matching types
- **Visual analytics**: Interactive results with filtering and statistics
- **Export functionality**: Download results in multiple formats
- **Sample data included**: Ready-to-use dataset for testing

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

## Dashboard Usage

The Streamlit dashboard provides an intuitive interface for entity extraction:

1. **Entity Input**: 
   - Enter custom entities manually
   - Select from OntoSoft or ASCL ontologies
   - Search and filter available entities

2. **Configuration**:
   - Set corpus sample size (100-5000 documents)
   - Choose document sections (title, abstract, body)
   - Adjust context window size and matching types

3. **Execution**:
   - Run complete pipeline with progress tracking
   - Step-by-step execution for granular control
   - Real-time status updates and metrics

4. **Results**:
   - Interactive table with filtering by entity, match type, and section
   - Export options (CSV, JSON, raw data)
   - Statistics and visualizations

## Development

### Dashboard Development
The Streamlit dashboard provides a modern interface for rapid prototyping and experimentation:
- **Modular architecture**: Separate UI components and pipeline wrappers
- **Caching system**: Efficient data loading and processing
- **Testing framework**: Unit tests and sample data for validation
- **Future-ready**: Architecture supports LLM integration and model comparison

See [`streamlit_dashboard/README.md`](streamlit_dashboard/README.md) for detailed dashboard documentation.

### General Development
This project includes configuration for AI development assistance. See [`AGENT.md`](AGENT.md) for:
- Build and test commands
- Architecture overview
- Code style guidelines

## Repository

**GitHub**: https://github.com/adsabs/entity_extractor

## License

Part of the ADS (Astrophysics Data System) project by the Smithsonian Astrophysical Observatory.
