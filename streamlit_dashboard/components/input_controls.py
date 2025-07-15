# streamlit_dashboard/components/input_controls.py
"""Input controls for the Streamlit dashboard."""

import streamlit as st
from typing import List, Dict, Any
try:
    from ..core_pipeline import load_ontologies, get_entity_list, validate_entities
    from ..config import *
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core_pipeline import load_ontologies, get_entity_list, validate_entities
    from config import *


def render_sidebar_controls() -> Dict[str, Any]:
    """Render sidebar controls and return configuration."""
    
    st.sidebar.header("🔧 Configuration")
    
    # Entity Input Section
    st.sidebar.subheader("1. Entity Input")
    
    # Ontology selection
    ontologies = load_ontologies()
    ontology_options = ["Custom"] + list(ontologies.keys())
    
    selected_ontology = st.sidebar.selectbox(
        "Select Ontology:",
        ontology_options,
        index=0
    )
    
    entities = []
    
    if selected_ontology == "Custom":
        # Custom entity input
        custom_entities_text = st.sidebar.text_area(
            "Enter entities (one per line):",
            placeholder="software\nPython\nTensorFlow\nmatplotlib",
            height=100
        )
        
        if custom_entities_text:
            entities = [e.strip() for e in custom_entities_text.split('\n') if e.strip()]
    
    else:
        # Ontology-based selection
        available_entities = ontologies.get(selected_ontology, [])
        
        if available_entities:
            # Search/filter entities
            search_term = st.sidebar.text_input(
                "Search entities:",
                placeholder="Type to filter..."
            )
            
            if search_term:
                filtered_entities = [e for e in available_entities if search_term.lower() in e.lower()]
            else:
                filtered_entities = available_entities[:50]  # Limit to first 50 for performance
            
            # Multi-select for entities
            entities = st.sidebar.multiselect(
                f"Select from {selected_ontology} ({len(available_entities)} total):",
                filtered_entities,
                default=filtered_entities[:3] if len(filtered_entities) >= 3 else filtered_entities
            )
        else:
            st.sidebar.warning(f"No entities found in {selected_ontology}")
    
    # Validate entities
    if entities:
        validation = validate_entities(entities)
        if validation["valid"]:
            st.sidebar.success(f"✅ {len(entities)} entities ready")
        else:
            st.sidebar.error(f"❌ {validation['error']}")
    else:
        st.sidebar.info("No entities selected")
    
    # Corpus Configuration Section
    st.sidebar.subheader("2. Corpus Configuration")
    
    # Sample size
    sample_size = st.sidebar.slider(
        "Sample Size:",
        min_value=MIN_CORPUS_SAMPLE,
        max_value=MAX_CORPUS_SAMPLE,
        value=DEFAULT_SAMPLE_SIZE,
        step=100,
        help="Number of documents to process"
    )
    
    # Corpus sections
    corpus_sections = st.sidebar.multiselect(
        "Include Sections:",
        ["title", "abstract", "body"],
        default=["title", "abstract"],
        help="Which sections of papers to search"
    )
    
    # Context window
    context_window = st.sidebar.slider(
        "Context Window Size:",
        min_value=50,
        max_value=500,
        value=DEFAULT_CONTEXT_WINDOW,
        step=25,
        help="Number of characters around match to include as context"
    )
    
    # Matching Configuration Section
    st.sidebar.subheader("3. Matching Configuration")
    
    # Match types
    match_types = st.sidebar.multiselect(
        "Match Types:",
        ["exact", "case_insensitive", "fuzzy"],
        default=["exact", "case_insensitive"],
        help="Types of matching to perform"
    )
    
    # Advanced settings in expander
    with st.sidebar.expander("Advanced Settings"):
        
        # NER Models (for future use)
        ner_models = st.multiselect(
            "NER Models:",
            DEFAULT_NER_MODELS,
            default=DEFAULT_NER_MODELS,
            help="Named Entity Recognition models to use"
        )
        
        # Heuristic keywords
        heuristic_keywords_text = st.text_area(
            "Heuristic Keywords (comma-separated):",
            value=", ".join(DEFAULT_HEURISTIC_KEYWORDS),
            height=100
        )
        heuristic_keywords = [k.strip() for k in heuristic_keywords_text.split(',') if k.strip()]
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "Similarity Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_SIMILARITY_THRESHOLD,
            step=0.05,
            help="Minimum similarity score for matches"
        )
        
        # Likelihood thresholds
        st.write("**Likelihood Thresholds:**")
        very_likely_threshold = st.slider(
            "Very Likely:",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_LIKELIHOOD_THRESHOLDS["very_likely"],
            step=0.05
        )
        
        likely_threshold = st.slider(
            "Likely:",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_LIKELIHOOD_THRESHOLDS["likely"],
            step=0.05
        )
        
        unlikely_threshold = st.slider(
            "Unlikely:",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_LIKELIHOOD_THRESHOLDS["unlikely"],
            step=0.05
        )
    
    # Return configuration
    return {
        "ontology": selected_ontology,
        "entities": entities,
        "sample_size": sample_size,
        "corpus_sections": corpus_sections,
        "context_window": context_window,
        "match_types": match_types,
        "ner_models": ner_models,
        "heuristic_keywords": heuristic_keywords,
        "similarity_threshold": similarity_threshold,
        "likelihood_thresholds": {
            "very_likely": very_likely_threshold,
            "likely": likely_threshold,
            "unlikely": unlikely_threshold
        }
    }


def render_entity_input_summary(entities: List[str]) -> None:
    """Render a summary of selected entities."""
    if not entities:
        st.info("No entities selected")
        return
    
    st.write(f"**Selected Entities ({len(entities)}):**")
    
    if len(entities) <= 10:
        for i, entity in enumerate(entities, 1):
            st.write(f"{i}. {entity}")
    else:
        # Show first 5 and last 5
        for i, entity in enumerate(entities[:5], 1):
            st.write(f"{i}. {entity}")
        st.write(f"... and {len(entities) - 10} more ...")
        for i, entity in enumerate(entities[-5:], len(entities) - 4):
            st.write(f"{i}. {entity}")
