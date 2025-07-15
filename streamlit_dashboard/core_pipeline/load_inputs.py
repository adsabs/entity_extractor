# streamlit_dashboard/core_pipeline/load_inputs.py
"""Load inputs wrapper for Streamlit dashboard."""

import json
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
try:
    from ..config import *
    from .utils import load_json, load_jsonl, get_db_connection
except ImportError:
    # Fallback for testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import *
    sys.path.append(str(Path(__file__).parent))
    from utils import load_json, load_jsonl, get_db_connection


@st.cache_data
def load_ontologies() -> Dict[str, List[str]]:
    """Load available ontologies and return entity lists."""
    ontologies = {}
    
    # Load OntoSoft
    try:
        ontosoft_data = load_json(ONTOSOFT_PATH)
        ontologies["OntoSoft"] = [item.get("name", "") for item in ontosoft_data if item.get("name")]
        st.success(f"Loaded {len(ontologies['OntoSoft'])} OntoSoft entities")
    except FileNotFoundError:
        if "sample_data" in str(ONTOSOFT_PATH):
            st.info("Using sample OntoSoft data")
            ontologies["OntoSoft"] = []
        else:
            st.warning(f"OntoSoft file not found at {ONTOSOFT_PATH}")
            ontologies["OntoSoft"] = []
    
    # Load ASCL
    try:
        ascl_data = load_json(ASCL_PATH)
        ontologies["ASCL"] = [item.get("name", "") for item in ascl_data if item.get("name")]
        st.success(f"Loaded {len(ontologies['ASCL'])} ASCL entities")
    except FileNotFoundError:
        if "sample_data" in str(ASCL_PATH):
            st.info("Using sample ASCL data")
            ontologies["ASCL"] = []
        else:
            st.warning(f"ASCL file not found at {ASCL_PATH}")
            ontologies["ASCL"] = []
    
    return ontologies


@st.cache_data
def load_corpus_sample(sample_size: int = DEFAULT_SAMPLE_SIZE, 
                      sections: List[str] = ["title", "abstract", "body"]) -> pd.DataFrame:
    """Load a sample of the corpus for processing."""
    if not CORPUS_PATH.exists():
        st.error(f"Corpus file not found at {CORPUS_PATH}")
        return pd.DataFrame()
    
    documents = []
    count = 0
    
    for doc in load_jsonl(CORPUS_PATH):
        if count >= sample_size:
            break
            
        # Extract and clean sections
        title = " ".join(doc.get("title", [])) if isinstance(doc.get("title", []), list) else doc.get("title", "")
        abstract = doc.get("abstract", "")
        body = doc.get("body", "")
        
        # Combine requested sections
        full_text = ""
        if "title" in sections:
            full_text += title + " "
        if "abstract" in sections:
            full_text += abstract + " "
        if "body" in sections:
            full_text += body + " "
        
        documents.append({
            "bibcode": doc.get("bibcode", ""),
            "title": title,
            "abstract": abstract,
            "body": body,
            "full_text": full_text.strip()
        })
        count += 1
    
    return pd.DataFrame(documents)


def init_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize the SQLite database for candidates."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    
    # Create candidates table
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
            embedding TEXT,
            similarity_score REAL,
            ner_score REAL,
            heuristic_score REAL,
            composite_score REAL,
            likelihood_label TEXT,
            final_classification TEXT,
            UNIQUE(bibcode, label, context)
        )
    """)
    
    conn.commit()
    return conn


def get_entity_list(ontology: str, custom_entities: Optional[List[str]] = None) -> List[str]:
    """Get entity list from ontology or custom input."""
    if custom_entities:
        return custom_entities
    
    ontologies = load_ontologies()
    return ontologies.get(ontology, [])


@st.cache_data
def validate_entities(entities: List[str]) -> Dict[str, Any]:
    """Validate entity list and return statistics."""
    if not entities:
        return {"valid": False, "error": "No entities provided"}
    
    # Basic validation
    filtered_entities = [e.strip() for e in entities if e.strip()]
    
    stats = {
        "valid": True,
        "total_count": len(entities),
        "filtered_count": len(filtered_entities),
        "duplicates": len(entities) - len(set(entities)),
        "entities": filtered_entities
    }
    
    return stats
