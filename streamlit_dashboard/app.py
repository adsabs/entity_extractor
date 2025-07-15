# streamlit_dashboard/app.py
"""Main Streamlit dashboard for entity extractor."""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add the parent directory to the path so we can import from software_mentions_pipeline
sys.path.append(str(Path(__file__).parent.parent))

from core_pipeline import (
    load_ontologies, 
    load_corpus_sample, 
    init_database,
    find_exact_matches,
    get_entity_list,
    validate_entities
)
from components.input_controls import render_sidebar_controls
from components.result_tables import render_results_table
from config import *

# Page configuration
st.set_page_config(
    page_title="Entity Extractor Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<div class="main-header">🔍 Entity Extractor Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Rapid testing and experimentation for entity disambiguation in academic literature**")
    
    # Initialize session state
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = {}
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    
    # Sidebar controls
    controls = render_sidebar_controls()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pipeline Execution")
        
        # Display current configuration
        with st.expander("Current Configuration", expanded=False):
            st.json({
                "entities": controls["entities"][:5] if len(controls["entities"]) > 5 else controls["entities"],
                "entity_count": len(controls["entities"]),
                "ontology": controls["ontology"],
                "corpus_sections": controls["corpus_sections"],
                "sample_size": controls["sample_size"],
                "context_window": controls["context_window"],
                "match_types": controls["match_types"]
            })
        
        # Pipeline execution buttons
        if st.button("🚀 Run Complete Pipeline", type="primary", use_container_width=True):
            run_complete_pipeline(controls)
        
        st.markdown("---")
        
        # Step-by-step execution
        st.subheader("Step-by-Step Execution")
        
        col_step1, col_step2, col_step3 = st.columns(3)
        
        with col_step1:
            if st.button("1️⃣ Load Data", use_container_width=True):
                run_step_1(controls)
        
        with col_step2:
            if st.button("2️⃣ Find Matches", use_container_width=True):
                run_step_2(controls)
        
        with col_step3:
            if st.button("3️⃣ More Steps", use_container_width=True, disabled=True):
                st.info("Coming soon: Embeddings, Scoring, Classification")
    
    with col2:
        st.subheader("Quick Stats")
        
        # Display metrics if we have results
        if "match_stats" in st.session_state.pipeline_results:
            stats = st.session_state.pipeline_results["match_stats"]
            
            st.metric("Total Matches", stats["total_matches"])
            st.metric("Unique Entities", stats["unique_entities"])
            st.metric("Unique Documents", stats["unique_documents"])
            
            # Match type breakdown
            if stats["matches_by_type"]:
                st.subheader("Match Types")
                for match_type, count in stats["matches_by_type"].items():
                    st.metric(match_type.replace("_", " ").title(), count)
    
    # Results display
    st.markdown("---")
    st.subheader("Results")
    
    if "candidates" in st.session_state.pipeline_results:
        df_candidates = st.session_state.pipeline_results["candidates"]
        render_results_table(df_candidates)
    else:
        st.info("👆 Run the pipeline above to see results")


def run_complete_pipeline(controls):
    """Run the complete pipeline."""
    with st.spinner("Running complete pipeline..."):
        
        # Step 1: Load data
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading corpus sample...")
        df_corpus = load_corpus_sample(
            sample_size=controls["sample_size"],
            sections=controls["corpus_sections"]
        )
        progress_bar.progress(0.3)
        
        # Step 2: Find matches
        status_text.text("Finding exact matches...")
        df_candidates = find_exact_matches(
            df_corpus=df_corpus,
            entities=controls["entities"],
            match_types=controls["match_types"],
            context_window=controls["context_window"]
        )
        progress_bar.progress(0.6)
        
        # Step 3: Calculate statistics
        status_text.text("Calculating statistics...")
        from core_pipeline.batch_filter import get_match_statistics
        match_stats = get_match_statistics(df_candidates)
        progress_bar.progress(1.0)
        
        # Store results
        st.session_state.pipeline_results = {
            "corpus": df_corpus,
            "candidates": df_candidates,
            "match_stats": match_stats
        }
        
        status_text.text("Pipeline complete!")
        st.success(f"Found {len(df_candidates)} matches across {match_stats['unique_documents']} documents")


def run_step_1(controls):
    """Run step 1: Load data."""
    with st.spinner("Loading corpus sample..."):
        df_corpus = load_corpus_sample(
            sample_size=controls["sample_size"],
            sections=controls["corpus_sections"]
        )
        
        st.session_state.pipeline_results["corpus"] = df_corpus
        st.session_state.current_step = 1
        
        st.success(f"Loaded {len(df_corpus)} documents")


def run_step_2(controls):
    """Run step 2: Find matches."""
    if "corpus" not in st.session_state.pipeline_results:
        st.error("Please run Step 1 first to load the corpus")
        return
    
    with st.spinner("Finding exact matches..."):
        df_corpus = st.session_state.pipeline_results["corpus"]
        
        df_candidates = find_exact_matches(
            df_corpus=df_corpus,
            entities=controls["entities"],
            match_types=controls["match_types"],
            context_window=controls["context_window"]
        )
        
        from core_pipeline.batch_filter import get_match_statistics
        match_stats = get_match_statistics(df_candidates)
        
        st.session_state.pipeline_results["candidates"] = df_candidates
        st.session_state.pipeline_results["match_stats"] = match_stats
        st.session_state.current_step = 2
        
        st.success(f"Found {len(df_candidates)} matches")


if __name__ == "__main__":
    main()
