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
    csv_path = Path("../optimized_extractor/results/exports/software_mentions_all_with_labels.csv.gzip")
    
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
            
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


def search_term(df: pd.DataFrame, search_query: str) -> pd.DataFrame:
    """Search for a specific term in the data."""
    if df is None:
        return pd.DataFrame()
    
    # Search by term name (case-insensitive)
    results = df[df['term_name'].str.contains(search_query, case=False, na=False)]
    
    if len(results) == 0:
        # Also search by extracted software name (before colon)
        df_copy = df.copy()
        df_copy['software_name'] = df_copy['term_name'].apply(
            lambda x: x.split(':')[0].strip() if ':' in str(x) else str(x)
        )
        results = df_copy[df_copy['software_name'].str.contains(search_query, case=False, na=False)]
    
    return results


def get_available_ner_models():
    """Get list of available NER models."""
    models = []
    
    # Always include INDUS NER (it's installed via transformers)
    models.append("adsabs/nasa-smd-ibm-v0.1_NER_DEAL")  # INDUS NER - primary model for astronomy/astrophysics
    
    # Check if spaCy is available
    try:
        import spacy
        models.append("en_core_web_sm")  # spaCy English - general purpose NER
    except ImportError:
        pass
    
    return models


@st.cache_resource
def load_ner_model(model_name: str):
    """Load and cache NER models."""
    try:
        if model_name == "adsabs/nasa-smd-ibm-v0.1_NER_DEAL":
            from transformers import pipeline
            import torch
            ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            return {"pipeline": ner_pipeline, "status": "success"}
        elif model_name == "en_core_web_sm":
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return {"pipeline": nlp, "status": "success"}
        else:
            return {"pipeline": None, "status": "error", "error": f"Model {model_name} not implemented"}
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
        
        if model_name == "adsabs/nasa-smd-ibm-v0.1_NER_DEAL":
            # Process with INDUS NER using chunking
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
                    continue
                    
        elif model_name == "en_core_web_sm":
            # Process with spaCy
            doc = pipeline_obj(context)
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "score": 1.0,  # spaCy doesn't provide confidence scores by default
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        else:
            entities = []
        
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


def main():
    """Main dashboard function."""
    
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
                        
                        selected_context = None
                        selected_row_id = None
                        
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
                                selected_context = positive_contexts.iloc[selected_idx]['context']
                                selected_row_id = positive_contexts.iloc[selected_idx]['row_id']
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
                                selected_context = uncurated_contexts.iloc[selected_idx]['context']
                                selected_row_id = uncurated_contexts.iloc[selected_idx]['row_id']
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
                                selected_context = negative_contexts.iloc[selected_idx]['context']
                                selected_row_id = negative_contexts.iloc[selected_idx]['row_id']
                            else:
                                st.info("No negative contexts available")
                        
                        # Show original context (only if one is selected)
                        if selected_context:
                            st.write("**Original Context:**")
                            st.text_area("Context", value=selected_context, height=100, disabled=True, key=f"orig_{term_name}_{selected_row_id}", label_visibility="collapsed")
                        
                        # Run NER models (only if context is selected)
                        if selected_context and selected_models:
                            st.write("**NER Analysis:**")
                            
                            for model_name in selected_models:
                                with st.expander(f"🤖 {model_name}", expanded=True):
                                    if st.button(f"Run {model_name}", key=f"ner_{term_name}_{model_name}_{selected_row_id}"):
                                        with st.spinner(f"Running {model_name}..."):
                                            ner_results = run_ner_on_context(selected_context, model_name)
                                        render_ner_results(ner_results, selected_context)
                        elif selected_context and not selected_models:
                            st.info("Select NER models to analyze contexts")
                        elif not selected_context:
                            st.info("Select a context from the tabs above to run NER analysis")
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
