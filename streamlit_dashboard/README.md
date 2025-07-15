# Entity Extractor Streamlit Dashboard

A Streamlit dashboard for rapid testing and experimentation with entity disambiguation in academic literature.

## Features

- **Interactive Entity Input**: Enter custom entities or select from ontologies (OntoSoft, ASCL)
- **Flexible Pipeline Control**: Run complete pipeline or step-by-step execution
- **Parameter Tuning**: Adjust matching types, context windows, and scoring thresholds
- **Real-time Results**: View matches, statistics, and detailed contexts
- **Export Options**: Download results as CSV or JSON

## Quick Start

### Prerequisites

```bash
pip install streamlit pandas numpy transformers sentence-transformers tqdm
```

### Run the Dashboard

```bash
# Option 1: Direct launch
python launch_app.py

# Option 2: Manual streamlit command
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Directory Structure

```
streamlit_dashboard/
├── app.py                      # Main Streamlit application
├── launch_app.py              # Launcher script
├── config.py                  # Configuration settings
├── components/                # UI components
│   ├── input_controls.py     # Sidebar controls
│   └── result_tables.py      # Results display
├── core_pipeline/            # Pipeline wrappers
│   ├── load_inputs.py        # Data loading
│   ├── batch_filter.py       # Exact matching
│   ├── embed.py              # Embedding generation
│   ├── score.py              # Context scoring
│   ├── likelihood.py         # Classification
│   └── utils.py              # Utility functions
├── sample_data/              # Sample data for testing
│   ├── corpus.jsonl          # Sample papers
│   ├── ontosoft.json         # OntoSoft entities
│   └── ascl.json             # ASCL entities
└── tests/                    # Test files
```

## Usage

### 1. Entity Input

**Custom Entities:**
- Select "Custom" from the ontology dropdown
- Enter entities one per line in the text area

**Ontology Selection:**
- Choose "OntoSoft" or "ASCL" from the dropdown
- Search and select entities from the available list

### 2. Configuration

**Corpus Settings:**
- Sample Size: Number of documents to process (100-5000)
- Include Sections: Choose title, abstract, and/or body
- Context Window: Characters around match for context

**Matching Options:**
- Exact matching: Case-sensitive word/phrase matching
- Case-insensitive: Ignore case differences
- Fuzzy matching: (Coming soon) Approximate matching

### 3. Pipeline Execution

**Complete Pipeline:**
- Click "🚀 Run Complete Pipeline" to process all steps
- View results in the main table

**Step-by-Step:**
- Use individual step buttons for granular control
- Inspect intermediate results at each stage

### 4. Results

**Interactive Table:**
- Filter by entity, match type, or document section
- Click rows to view detailed context
- Export results as CSV or JSON

**Statistics:**
- View match counts, entity distribution
- Analyze match types and section distribution

## Configuration Options

### Advanced Settings

- **NER Models**: Select which Named Entity Recognition models to use
- **Heuristic Keywords**: Customize keywords for context scoring
- **Similarity Threshold**: Minimum similarity for embeddings
- **Likelihood Thresholds**: Adjust classification boundaries

### Future Features

- **LLM Integration**: Use language models for disambiguation
- **Fuzzy Matching**: Approximate string matching
- **Batch Processing**: Process multiple ontologies simultaneously
- **Model Comparison**: Compare different NER models side-by-side

## Testing

Run the basic functionality test:

```bash
python test_basic.py
```

## Development

The dashboard is designed with a modular architecture:

- **Core Pipeline**: UI-agnostic wrappers around original pipeline scripts
- **Components**: Reusable UI components
- **Configuration**: Centralized settings management
- **Caching**: Streamlit caching for performance

To add new features:

1. Add core functionality to `core_pipeline/`
2. Create UI components in `components/`
3. Update `app.py` to wire everything together
4. Add configuration options to `config.py`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Check data file paths in `config.py`
3. **Performance Issues**: Reduce sample size or use caching
4. **Model Loading**: Ensure transformers models are accessible

### Sample Data

The dashboard includes sample data for testing:
- 5 sample academic papers
- 10 OntoSoft entities
- 10 ASCL entities

For full functionality, point to the actual data files in `../software_mentions_pipeline/data/`

## License

Part of the ADS (Astrophysics Data System) entity_extractor project.
