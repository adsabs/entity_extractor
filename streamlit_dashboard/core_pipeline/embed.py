# streamlit_dashboard/core_pipeline/embed.py
"""Embedding wrapper for Streamlit dashboard."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
try:
    from ..config import DEFAULT_EMBED_MODEL, DEFAULT_BATCH_SIZE
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DEFAULT_EMBED_MODEL, DEFAULT_BATCH_SIZE


@st.cache_resource
def load_embedding_model(model_name: str = DEFAULT_EMBED_MODEL) -> SentenceTransformer:
    """Load and cache the embedding model."""
    return SentenceTransformer(model_name)


@st.cache_data
def embed_contextual_mentions(df_contexts: pd.DataFrame,
                             model_name: str = DEFAULT_EMBED_MODEL,
                             batch_size: int = DEFAULT_BATCH_SIZE) -> pd.DataFrame:
    """Generate embeddings for contextual mentions."""
    
    if df_contexts.empty:
        return df_contexts
    
    # Load model
    model = load_embedding_model(model_name)
    
    # Extract contexts
    contexts = df_contexts["context"].tolist()
    
    # Generate embeddings
    with st.spinner(f"Generating embeddings for {len(contexts)} contexts..."):
        embeddings = model.encode(
            contexts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    # Add embeddings to dataframe
    df_result = df_contexts.copy()
    df_result["embedding"] = [emb.tolist() for emb in embeddings]
    
    return df_result


@st.cache_data
def embed_entity_descriptions(entities: List[str],
                             descriptions: List[str],
                             model_name: str = DEFAULT_EMBED_MODEL,
                             batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, List[float]]:
    """Generate embeddings for entity descriptions."""
    
    if not entities or not descriptions:
        return {}
    
    # Load model
    model = load_embedding_model(model_name)
    
    # Generate embeddings
    with st.spinner(f"Generating embeddings for {len(descriptions)} entity descriptions..."):
        embeddings = model.encode(
            descriptions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    # Create entity -> embedding mapping
    entity_embeddings = {}
    for entity, embedding in zip(entities, embeddings):
        entity_embeddings[entity] = embedding.tolist()
    
    return entity_embeddings


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    
    # Convert to numpy arrays
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def add_similarity_scores(df_candidates: pd.DataFrame,
                         entity_embeddings: Dict[str, List[float]]) -> pd.DataFrame:
    """Add similarity scores to candidates dataframe."""
    
    if df_candidates.empty or not entity_embeddings:
        return df_candidates
    
    df_result = df_candidates.copy()
    similarity_scores = []
    
    for _, row in df_result.iterrows():
        entity = row["label"]
        context_embedding = row.get("embedding", [])
        
        if entity in entity_embeddings and context_embedding:
            similarity = calculate_similarity(
                context_embedding, 
                entity_embeddings[entity]
            )
            similarity_scores.append(similarity)
        else:
            similarity_scores.append(0.0)
    
    df_result["similarity_score"] = similarity_scores
    return df_result
