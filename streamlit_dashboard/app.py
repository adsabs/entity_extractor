# streamlit_dashboard/app.py
"""Main Streamlit dashboard for entity extractor with integrated search functionality."""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import gzip
from typing import List, Dict, Any

# Add the parent directory to the path so we can import from software_mentions_pipeline
sys.path.append(str(Path(__file__).parent.parent))

from components.result_tables import render_results_table
from components.autocomplete import render_autocomplete_search
from components.autocomplete_simple import render_simple_autocomplete_search

# Import repair functions directly to avoid module path issues
import re
from typing import List, Dict, Union

def repair_target_mentions(
    entities: List[Dict],
    context: str,
    target_terms: Union[str, List[str]],
    target_label: str = "SOFTWARE",
    min_model_conf: float = 0.30,
) -> List[Dict]:
    """
    Ensures that every literal occurrence of `target_terms` in `context`
    is returned as a single complete entity span.
    """
    if isinstance(target_terms, str):
        target_terms = [target_terms]

    # Build regex pattern for target terms
    escaped = [re.escape(t) for t in target_terms]
    pattern = r'\b(?:' + "|".join(escaped) + r')\b'
    regex = re.compile(pattern, flags=re.I)

    # Map existing entities by (start, end) for quick lookup
    out: List[Dict] = [e.copy() for e in entities]
    by_span = {(e["start"], e["end"]): e for e in out}

    # Walk through every literal occurrence
    for m in regex.finditer(context):
        s, e = m.start(), m.end()
        span_key = (s, e)

        # Does any entity fully cover the span already?
        full_covered = [k for k in by_span if k[0] <= s and k[1] >= e]
        if full_covered:
            continue  # already good

        # Do we have partial overlaps we need to merge/extend?
        partial = [k for k in by_span if not (k[1] <= s or k[0] >= e)]
        if partial:
            # Merge all partial entities into one full span
            best_score = max(by_span[k].get("score", 0.0) for k in partial)
            # Remove partial pieces
            for k in partial:
                out.remove(by_span[k])
                by_span.pop(k, None)
            # Insert repaired span
            repaired = {
                "text": context[s:e],
                "label": target_label,
                "score": round(max(best_score, min_model_conf), 4),
                "start": s,
                "end": e,
            }
            out.append(repaired)
            by_span[span_key] = repaired
            continue

        # Span missing completely -> inject synthetic entity
        injected = {
            "text": context[s:e],
            "label": target_label,
            "score": 1.0,
            "start": s,
            "end": e,
        }
        out.append(injected)
        by_span[span_key] = injected

    # Return entities sorted by position
    out.sort(key=lambda x: x["start"])
    return out

def run_ner_with_target(context: str, pipeline_obj, target_term: Union[str, List[str]]) -> List[Dict]:
    """
    Run NER with target-aware repair to fix fragmented mentions of known terms.
    """
    try:
        # Run standard NER
        raw_entities = pipeline_obj(context)
        
        # Convert to standard format
        clean_entities = [
            {
                "text": ent["word"],
                "label": ent["entity_group"],
                "score": ent["score"],
                "start": ent["start"],
                "end": ent["end"],
            } for ent in raw_entities
        ]
        
        # Apply target-aware repair
        return repair_target_mentions(clean_entities, context, target_term)
        
    except Exception as e:
        st.error(f"Error in run_ner_with_target: {e}")
        return []

def load_hf_ner(model_id: str, prefer_gpu: bool = True):
    """
    Load a HuggingFace token-classification model with automatic GPU fallback.
    
    Args:
        model_id: HuggingFace model identifier
        prefer_gpu: Whether to try GPU first (falls back to CPU on failure)
        
    Returns:
        Transformers pipeline for NER
        
    Raises:
        RuntimeError: If model loading fails on both GPU and CPU
    """
    from transformers import pipeline
    import torch
    import logging
    
    gpu_available = torch.cuda.is_available() and prefer_gpu
    device_id = 0 if gpu_available else -1

    try:
        return pipeline(
            "ner",
            model=model_id,
            tokenizer=model_id,
            aggregation_strategy="simple",
            device=device_id
        )
    except RuntimeError as err:
        if gpu_available and "CUDA" in str(err):
            logging.warning(
                f"⚠️  GPU inference failed for {model_id}: {err}. "
                "Retrying on CPU."
            )
            return pipeline(
                "ner",
                model=model_id,
                tokenizer=model_id,
                aggregation_strategy="simple",
                device=-1
            )
        raise  # some other runtime error → bubble up

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

def load_delta_changes():
    """Load any existing curation changes from delta file."""
    delta_path = Path("curation_delta.csv")
    
    if not delta_path.exists():
        return pd.DataFrame(columns=['row_id', 'bibcode_label', 'curator', 'timestamp'])
    
    try:
        return pd.read_csv(delta_path)
    except Exception as e:
        st.warning(f"Could not load delta file: {e}")
        return pd.DataFrame(columns=['row_id', 'bibcode_label', 'curator', 'timestamp'])


@st.cache_data
def load_software_mentions_data():
    """Load the software mentions data from compressed CSV with row_id and delta merging."""
    csv_path = Path(__file__).parent.parent / "optimized_extractor/results/exports/software_mentions_all_with_labels.csv.gzip"
    
    if not csv_path.exists():
        st.error(f"❌ Data file not found at {csv_path}")
        st.info("Please make sure the extraction pipeline has been run.")
        return None
    
    try:
        with st.spinner("Loading software mentions data (612k+ records)..."):
            # Load main dataset
            df = pd.read_csv(csv_path, compression='gzip')
            
            # Add row_id as unique identifier for each row
            df = df.reset_index(drop=True)
            df['row_id'] = df.index
            
            # Load any curation changes from delta file
            delta_df = load_delta_changes()
            
            # Merge delta changes with main dataset
            if not delta_df.empty:
                # Create a mapping of row_id to updated bibcode_label
                label_updates = dict(zip(delta_df['row_id'], delta_df['bibcode_label']))
                
                # Apply updates to main dataframe
                for row_id, new_label in label_updates.items():
                    if row_id < len(df):
                        df.loc[row_id, 'bibcode_label'] = new_label
                
                st.info(f"✅ Applied {len(label_updates)} curation changes from delta file")
            
            # Add software_key column for exact matching
            df['software_key'] = (
                df['term_name']
                .fillna('')
                .apply(lambda x: str(x).split(':')[0].strip().lower())
            )
            
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


def search_term(df: pd.DataFrame, search_query: str) -> pd.DataFrame:
    """Search for a specific term in the data."""
    if df is None:
        return pd.DataFrame()
    
    query = search_query.strip().lower()
    if not query:
        return df
    
    # 1) Exact match against canonical software key (highest priority)
    exact_results = df[df['software_key'] == query]
    if len(exact_results) > 0:
        return exact_results
    
    # 2) Fallback: fuzzy contains search on term_name (previous behavior)
    fuzzy_results = df[df['term_name'].str.contains(search_query, case=False, na=False)]
    if len(fuzzy_results) > 0:
        return fuzzy_results
    
    # 3) Last resort: contains search on software_key  
    fuzzy_key_results = df[df['software_key'].str.contains(query, na=False)]
    return fuzzy_key_results


def get_available_ner_models():
    """Get list of available NER models."""
    return [
        "adsabs/nasa-smd-ibm-v0.1_NER_DEAL",        # INDUS NER - primary model for astronomy/astrophysics
        "oeg/software_benchmark_multidomain"        # OEG software NER - specialized for software entity extraction
    ]


@st.cache_resource
def load_ner_model(model_name: str):
    """Load and cache Hugging Face transformer NER models."""
    try:
        # Supported transformer models
        hf_models = {
            "adsabs/nasa-smd-ibm-v0.1_NER_DEAL",
            "oeg/software_benchmark_multidomain"
        }
        
        if model_name in hf_models:
            ner_pipeline = load_hf_ner(model_name, prefer_gpu=True)
            return {"pipeline": ner_pipeline, "status": "success"}
        else:
            return {"pipeline": None, "status": "error", "error": f"Model {model_name} not supported"}
    except Exception as e:
        return {"pipeline": None, "status": "error", "error": str(e)}


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50):
    """Split text into smaller chunks for processing."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
        if end >= len(words):
            break
    return chunks


def run_ner_on_context(context: str, model_name: str):
    """Run NER model on given context."""
    try:
        model_info = load_ner_model(model_name)
        
        if model_info["status"] == "error":
            return {"entities": [], "status": "error", "error": model_info["error"]}
        
        pipeline_obj = model_info["pipeline"]
        
        # Process transformer models that need chunking
        if model_name in {
            "adsabs/nasa-smd-ibm-v0.1_NER_DEAL",
            "oeg/software_benchmark_multidomain"
        }:
            entities = []
            chunks = chunk_text(context)
            
            for chunk in chunks:
                try:
                    chunk_entities = pipeline_obj(chunk)
                    for entity in chunk_entities:
                        entities.append({
                            "text": entity["word"],
                            "label": entity["entity_group"], 
                            "score": round(entity["score"], 4),
                            "start": entity.get("start", 0),
                            "end": entity.get("end", len(entity["word"]))
                        })
                except Exception as e:
                    import logging
                    logging.error(f"NER failure on chunk: {e}")
                    continue
        else:
            entities = []
        
        return {"entities": entities, "status": "success"}
        
    except Exception as e:
        return {"entities": [], "status": "error", "error": str(e)}


def run_ner_on_context_with_target(context: str, model_name: str, target_term: str):
    """Run NER model on given context with target-aware repair."""
    try:
        model_info = load_ner_model(model_name)
        
        if model_info["status"] == "error":
            return {"entities": [], "status": "error", "error": model_info["error"]}
        
        pipeline_obj = model_info["pipeline"]
        
        # Use target-aware NER for better entity detection
        entities = run_ner_with_target(context, pipeline_obj, target_term)
        
        return {"entities": entities, "status": "success"}
        
    except Exception as e:
        return {"entities": [], "status": "error", "error": str(e)}


def render_ner_results(ner_results: Dict[str, Any], context: str):
    """Render NER results with highlighted entities."""
    if ner_results["status"] == "error":
        st.error(f"❌ NER Error: {ner_results['error']}")
        return
    
    entities = ner_results["entities"]
    if not entities:
        st.info("No entities detected by this model")
        return
    
    st.success(f"✅ Found {len(entities)} entities")
    
    # Show entity summary with colors
    entity_types = {}
    for entity in entities:
        label = entity["label"]
        if label not in entity_types:
            entity_types[label] = []
        entity_types[label].append(entity["text"])
    
    # Display entity type summary
    st.write("**Entity Types Found:**")
    for label, texts in entity_types.items():
        unique_texts = list(set(texts))
        st.write(f"• **{label}**: {', '.join(unique_texts[:5])}{'...' if len(unique_texts) > 5 else ''}")
    
    # Show detailed entity table
    st.write("**Detailed Results:**")
    entity_df = pd.DataFrame(entities)
    if 'score' in entity_df.columns:
        entity_df = entity_df.sort_values('score', ascending=False)
    st.dataframe(entity_df, use_container_width=True)
    
    # Create highlighted text (simplified approach)
    st.write("**Highlighted Context:**")
    highlighted_text = context
    
    # Sort entities by start position (reverse order to maintain positions)
    sorted_entities = sorted(entities, key=lambda x: x.get("start", 0), reverse=True)
    
    for entity in sorted_entities:
        text = entity["text"]
        label = entity["label"]
        score = entity.get("score", 1.0)
        
        # Simple highlighting by replacing text
        if text in highlighted_text:
            replacement = f"**[{text}]** `({label})`"
            highlighted_text = highlighted_text.replace(text, replacement, 1)
    
    st.markdown(highlighted_text)


def initialize_session_state():
    """Initialize session state for curation tracking."""
    if 'pending_changes' not in st.session_state:
        st.session_state.pending_changes = {}
    if 'curator_name' not in st.session_state:
        st.session_state.curator_name = ""


def add_pending_change(row_id: int, new_label: str, curator: str):
    """Add a label change to the pending changes queue."""
    st.session_state.pending_changes[row_id] = {
        'bibcode_label': new_label,
        'curator': curator,
        'timestamp': pd.Timestamp.now().isoformat()
    }


def commit_pending_changes():
    """Save all pending changes to the delta file."""
    if not st.session_state.pending_changes:
        return 0
    
    delta_path = Path("curation_delta.csv")
    
    # Convert pending changes to DataFrame
    changes_data = []
    for row_id, change_info in st.session_state.pending_changes.items():
        changes_data.append({
            'row_id': row_id,
            'bibcode_label': change_info['bibcode_label'],
            'curator': change_info['curator'],
            'timestamp': change_info['timestamp']
        })
    
    new_df = pd.DataFrame(changes_data)
    
    # Append to existing delta file or create new one
    if delta_path.exists():
        existing_df = pd.read_csv(delta_path)
        # Remove any existing entries for the same row_ids to avoid duplicates
        existing_df = existing_df[~existing_df['row_id'].isin(new_df['row_id'])]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Save to file
    combined_df.to_csv(delta_path, index=False)
    
    # Clear pending changes
    change_count = len(st.session_state.pending_changes)
    st.session_state.pending_changes = {}
    
    return change_count


def render_label_curation_widget(row_id: int, current_label: str, context_key: str):
    """Render label curation widget for a specific context."""
    label_options = ['positive', 'negative', 'unknown']
    
    # Check if there's a pending change for this row
    pending_label = st.session_state.pending_changes.get(row_id, {}).get('bibcode_label', current_label)
    
    # Show current and pending status
    status_text = f"Current: **{current_label}**"
    if row_id in st.session_state.pending_changes:
        status_text += f" → Pending: **{pending_label}** ⏳"
    
    st.markdown(status_text)
    
    # Label selection
    current_idx = label_options.index(pending_label) if pending_label in label_options else 0
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_label = st.selectbox(
            "Update label:",
            label_options,
            index=current_idx,
            key=f"label_select_{context_key}_{row_id}",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("Queue Change", key=f"queue_change_{context_key}_{row_id}"):
            if new_label != current_label and st.session_state.curator_name.strip():
                add_pending_change(row_id, new_label, st.session_state.curator_name.strip())
                st.success(f"Queued: {current_label} → {new_label}")
                st.rerun()
            elif not st.session_state.curator_name.strip():
                st.error("Please enter curator name in sidebar")
            else:
                st.info("No change needed")


def render_curation_sidebar():
    """Render the curation controls in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Curation Controls")
    
    # Curator name input
    st.session_state.curator_name = st.sidebar.text_input(
        "Curator Name:",
        value=st.session_state.curator_name,
        help="Enter your name for attribution"
    )
    
    # Show pending changes count
    pending_count = len(st.session_state.pending_changes)
    if pending_count > 0:
        st.sidebar.warning(f"⏳ {pending_count} pending changes")
        
        # Show pending changes details
        with st.sidebar.expander("View Pending Changes", expanded=False):
            for row_id, change in st.session_state.pending_changes.items():
                st.write(f"Row {row_id}: → **{change['bibcode_label']}**")
        
        # Commit button
        if st.sidebar.button("💾 Commit All Changes", type="primary"):
            if st.session_state.curator_name.strip():
                committed_count = commit_pending_changes()
                st.sidebar.success(f"✅ Committed {committed_count} changes!")
                # Force refresh of cached data
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.error("Please enter curator name first")
        
        # Clear button
        if st.sidebar.button("🗑️ Clear All Pending"):
            st.session_state.pending_changes = {}
            st.sidebar.success("Cleared pending changes")
            st.rerun()
    else:
        st.sidebar.info("No pending changes")


def main():
    """Main dashboard function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🔍 Software Mentions Search & NER Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Search software mentions and run NER models on extracted contexts**")
    
    # Load data
    df = load_software_mentions_data()
    
    if df is None:
        return
    
    # Sidebar for search and NER configuration
    st.sidebar.header("🔧 Configuration")
    
    # Search section with autocomplete
    st.sidebar.subheader("1. Entity Search")
    
    # Use built-in search as default
    search_query = render_simple_autocomplete_search()
    
    # NER Model selection
    st.sidebar.subheader("2. NER Models")
    available_models = get_available_ner_models()
    selected_models = st.sidebar.multiselect(
        "Select NER models to run:",
        available_models,
        default=[available_models[0]],
        help="Choose which NER models to apply to contexts"
    )
    
    # Render curation controls
    render_curation_sidebar()
    
    # Main content
    if search_query:
        # Search for mentions
        with st.spinner(f"Searching for '{search_query}'..."):
            results = search_term(df, search_query)
        
        if len(results) == 0:
            st.warning(f"❌ No results found for '{search_query}'")
            return
        
        # Display search results summary
        st.subheader(f"🔍 Search Results for '{search_query}'")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Mentions", len(results))
        with col2:
            st.metric("Unique Software", results['term_name'].nunique())
        with col3:
            st.metric("Unique Papers", results['bibcode'].nunique())
        with col4:
            st.metric("Total Matches", results['match_count'].sum())
        
        # Group by term_name for display
        grouped = results.groupby('term_name')
        
        for term_name, group in grouped:
            with st.expander(f"📦 {term_name} ({len(group)} mentions)", expanded=True):
                
                # Show mention details
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    # Quick stats
                    st.write("**Statistics:**")
                    st.write(f"🔄 Total mentions: {len(group)}")
                    st.write(f"📚 Unique papers: {group['bibcode'].nunique()}")
                    st.write(f"🎯 Total matches: {group['match_count'].sum()}")
                    
                    # Location distribution
                    location_counts = group['match_location'].value_counts()
                    st.write("**Locations:**")
                    for location, count in location_counts.items():
                        st.write(f"• {location}: {count}")
                
                with col1:
                    # Context analysis with NER
                    st.write("**Context Analysis with NER:**")
                    
                    # Filter contexts with text
                    contexts_with_text = group[group['context'].notna() & (group['context'] != '')]
                    
                    if len(contexts_with_text) > 0:
                        # Categorize contexts based on bibcode_label
                        positive_contexts = contexts_with_text[contexts_with_text['bibcode_label'] == 'positive']
                        negative_contexts = contexts_with_text[
                            (contexts_with_text['bibcode_label'] == 'negative') | 
                            (contexts_with_text['bibcode_label'] == 'unknown')
                        ]
                        uncurated_contexts = contexts_with_text[
                            ~contexts_with_text['bibcode_label'].isin(['positive', 'negative', 'unknown'])
                        ]
                        
                        # Create tabs for different context categories
                        tab1, tab2, tab3 = st.tabs([
                            f"✅ Positive ({len(positive_contexts)})",
                            f"📝 Uncurated ({len(uncurated_contexts)})",
                            f"🚫 Negative ({len(negative_contexts)})"
                        ])
                        
                        with tab1:
                            if len(positive_contexts) > 0:
                                context_options = []
                                for idx, (_, row) in enumerate(positive_contexts.iterrows()):
                                    preview = str(row['context'])[:50] + "..." if len(str(row['context'])) > 50 else str(row['context'])
                                    context_options.append(f"{idx+1}. {row['bibcode']} - {preview}")
                                
                                selected_idx = st.selectbox(
                                    "Select positive context:",
                                    range(len(context_options)),
                                    format_func=lambda x: context_options[x],
                                    key=f"positive_context_{term_name}"
                                )
                                
                                selected_row = positive_contexts.iloc[selected_idx]
                                selected_context = selected_row['context']
                                selected_row_id = selected_row['row_id']
                                
                                # Show original context
                                st.write("**Original Context:**")
                                st.text_area("Context", value=selected_context, height=100, disabled=True, key=f"orig_pos_{term_name}_{selected_row_id}", label_visibility="collapsed")
                                
                                # Add curation widget
                                st.markdown("**Label Curation:**")
                                render_label_curation_widget(selected_row_id, selected_row['bibcode_label'], f"positive_{term_name}")
                                
                                # Run NER models
                                if selected_models:
                                    st.write("**NER Analysis:**")
                                    for model_name in selected_models:
                                        with st.expander(f"🤖 {model_name}", expanded=True):
                                            if st.button(f"Run {model_name}", key=f"ner_pos_{term_name}_{model_name}_{selected_row_id}"):
                                                with st.spinner(f"Running {model_name}..."):
                                                # Extract clean term name (remove any suffix after colon)
                                                     clean_term = term_name.split(':')[0] if ':' in term_name else term_name
                                                     ner_results = run_ner_on_context_with_target(selected_context, model_name, clean_term)
                                                render_ner_results(ner_results, selected_context)
                                else:
                                    st.info("Select NER models in the sidebar to analyze contexts")
                            else:
                                st.info("No positive contexts available")
                        
                        with tab2:
                            if len(uncurated_contexts) > 0:
                                context_options = []
                                for idx, (_, row) in enumerate(uncurated_contexts.iterrows()):
                                    preview = str(row['context'])[:50] + "..." if len(str(row['context'])) > 50 else str(row['context'])
                                    context_options.append(f"{idx+1}. {row['bibcode']} - {preview}")
                                
                                selected_idx = st.selectbox(
                                    "Select uncurated context:",
                                    range(len(context_options)),
                                    format_func=lambda x: context_options[x],
                                    key=f"uncurated_context_{term_name}"
                                )
                                
                                selected_row = uncurated_contexts.iloc[selected_idx]
                                selected_context = selected_row['context']
                                selected_row_id = selected_row['row_id']
                                
                                # Show original context
                                st.write("**Original Context:**")
                                st.text_area("Context", value=selected_context, height=100, disabled=True, key=f"orig_unc_{term_name}_{selected_row_id}", label_visibility="collapsed")
                                
                                # Add curation widget
                                st.markdown("**Label Curation:**")
                                render_label_curation_widget(selected_row_id, selected_row['bibcode_label'], f"uncurated_{term_name}")
                                
                                # Run NER models
                                if selected_models:
                                    st.write("**NER Analysis:**")
                                    for model_name in selected_models:
                                        with st.expander(f"🤖 {model_name}", expanded=True):
                                            if st.button(f"Run {model_name}", key=f"ner_unc_{term_name}_{model_name}_{selected_row_id}"):
                                                with st.spinner(f"Running {model_name}..."):
                                                # Extract clean term name (remove any suffix after colon)
                                                     clean_term = term_name.split(':')[0] if ':' in term_name else term_name
                                                     ner_results = run_ner_on_context_with_target(selected_context, model_name, clean_term)
                                                render_ner_results(ner_results, selected_context)
                                else:
                                    st.info("Select NER models in the sidebar to analyze contexts")
                            else:
                                st.info("No uncurated contexts available")
                        
                        with tab3:
                            if len(negative_contexts) > 0:
                                context_options = []
                                for idx, (_, row) in enumerate(negative_contexts.iterrows()):
                                    preview = str(row['context'])[:50] + "..." if len(str(row['context'])) > 50 else str(row['context'])
                                    context_options.append(f"{idx+1}. {row['bibcode']} - {preview}")
                                
                                selected_idx = st.selectbox(
                                    "Select negative context:",
                                    range(len(context_options)),
                                    format_func=lambda x: context_options[x],
                                    key=f"negative_context_{term_name}"
                                )
                                
                                selected_row = negative_contexts.iloc[selected_idx]
                                selected_context = selected_row['context']
                                selected_row_id = selected_row['row_id']
                                
                                # Show original context
                                st.write("**Original Context:**")
                                st.text_area("Context", value=selected_context, height=100, disabled=True, key=f"orig_neg_{term_name}_{selected_row_id}", label_visibility="collapsed")
                                
                                # Add curation widget
                                st.markdown("**Label Curation:**")
                                render_label_curation_widget(selected_row_id, selected_row['bibcode_label'], f"negative_{term_name}")
                                
                                # Run NER models
                                if selected_models:
                                    st.write("**NER Analysis:**")
                                    for model_name in selected_models:
                                        with st.expander(f"🤖 {model_name}", expanded=True):
                                            if st.button(f"Run {model_name}", key=f"ner_neg_{term_name}_{model_name}_{selected_row_id}"):
                                                with st.spinner(f"Running {model_name}..."):
                                                # Extract clean term name (remove any suffix after colon)
                                                     clean_term = term_name.split(':')[0] if ':' in term_name else term_name
                                                     ner_results = run_ner_on_context_with_target(selected_context, model_name, clean_term)
                                                render_ner_results(ner_results, selected_context)
                                else:
                                    st.info("Select NER models in the sidebar to analyze contexts")
                            else:
                                st.info("No negative contexts available")

                    else:
                        st.info("No contexts available for this software")
                
                # Show detailed mentions table
                st.write("**Detailed Mentions:**")
                display_df = group[['bibcode', 'title', 'match_location', 'match_count', 'in_title', 'in_abstract']].copy()
                
                # Truncate long titles
                display_df['title'] = display_df['title'].apply(
                    lambda x: str(x)[:80] + "..." if pd.notna(x) and len(str(x)) > 80 else str(x)
                )
                
                st.dataframe(display_df, use_container_width=True)
    
    else:
        # Show welcome message and data overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Unique Software", f"{df['term_name'].nunique():,}")
        with col3:
            st.metric("Unique Papers", f"{df['bibcode'].nunique():,}")
        
        st.info("👆 Enter a software name in the sidebar to search for mentions and analyze contexts with NER models.")





if __name__ == "__main__":
    main()
