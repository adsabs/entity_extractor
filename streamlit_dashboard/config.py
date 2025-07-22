# streamlit_dashboard/config.py
"""Configuration settings for the Streamlit dashboard."""
from pathlib import Path

# Data paths
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR.parent / "software_mentions_pipeline" / "data"
SAMPLE_DATA_DIR = CURRENT_DIR / "sample_data"

# Use sample data if original data doesn't exist
if (DATA_DIR / "corpus.jsonl").exists():
    CORPUS_PATH = DATA_DIR / "corpus.jsonl"
else:
    CORPUS_PATH = SAMPLE_DATA_DIR / "corpus.jsonl"

# ASCL ontology path
ASCL_PATH = CURRENT_DIR.parent / "ontologies" / "ASCL" / "ascl.json"

DB_PATH = DATA_DIR / "match_candidates.db"
LABELS_PATH = DATA_DIR / "labels.json"

# Default model settings
DEFAULT_EMBED_MODEL = "nasa-impact/nasa-smd-ibm-st-v2"
DEFAULT_NER_MODELS = [
    "oeg/software_benchmark_multidomain",
    "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"
]

# Default thresholds and parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_CONTEXT_WINDOW = 100
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_BATCH_SIZE = 32

# Default heuristic keywords
DEFAULT_HEURISTIC_KEYWORDS = [
    "software", "tool", "package", "library", "framework", 
    "code", "algorithm", "model", "program", "application",
    "system", "platform", "suite", "toolkit", "engine"
]

# Likelihood thresholds
DEFAULT_LIKELIHOOD_THRESHOLDS = {
    "very_likely": 0.8,
    "likely": 0.6,
    "unlikely": 0.4
}

# UI settings
MAX_CORPUS_SAMPLE = 5000
MIN_CORPUS_SAMPLE = 100
