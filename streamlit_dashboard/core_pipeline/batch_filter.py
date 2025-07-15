# streamlit_dashboard/core_pipeline/batch_filter.py
"""Batch filter wrapper for Streamlit dashboard."""

import re
import pandas as pd
from typing import List, Dict, Any, Set
import streamlit as st
from tqdm import tqdm
try:
    from .utils import extract_context_window, normalize_text
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from utils import extract_context_window, normalize_text


@st.cache_data
def find_exact_matches(df_corpus: pd.DataFrame, 
                      entities: List[str],
                      match_types: List[str] = ["exact", "case_insensitive"],
                      context_window: int = 100) -> pd.DataFrame:
    """Find exact matches for entities in the corpus."""
    
    if df_corpus.empty or not entities:
        return pd.DataFrame()
    
    # Prepare entity sets for different match types
    entity_set = set(entities)
    entity_set_lower = {e.lower() for e in entities}
    
    # Separate single-word and phrase entities
    token_entities = {e for e in entity_set if " " not in e}
    phrase_entities = {e for e in entity_set if " " in e}
    
    candidates = []
    
    # Process each document
    for idx, row in tqdm(df_corpus.iterrows(), total=len(df_corpus), desc="🔍 Finding exact matches"):
        bibcode = row["bibcode"]
        title = row["title"]
        abstract = row["abstract"]
        body = row["body"]
        full_text = row["full_text"]
        
        if not bibcode:
            continue
        
        # Find matches in different sections
        for section_name, section_text in [("title", title), ("abstract", abstract), ("body", body)]:
            if not section_text:
                continue
                
            # Token-based matching for single words
            if "exact" in match_types:
                tokens = set(re.findall(r"\b[\w\-]+\b", section_text))
                found_tokens = tokens.intersection(token_entities)
                
                for token in found_tokens:
                    # Find actual position in text for context
                    match = re.search(r"\b" + re.escape(token) + r"\b", section_text)
                    if match:
                        context = extract_context_window(
                            section_text, 
                            match.start(), 
                            match.end(), 
                            context_window
                        )
                        
                        candidates.append({
                            "bibcode": bibcode,
                            "title": title,
                            "abstract": abstract,
                            "body": body,
                            "label": token,
                            "match_type": "exact_token",
                            "context": context,
                            "section": section_name,
                            "match_start": match.start(),
                            "match_end": match.end()
                        })
            
            # Case-insensitive matching
            if "case_insensitive" in match_types:
                tokens_lower = set(re.findall(r"\b[\w\-]+\b", section_text.lower()))
                found_tokens_lower = tokens_lower.intersection(entity_set_lower)
                
                for token_lower in found_tokens_lower:
                    # Find original entity
                    original_entity = next((e for e in entities if e.lower() == token_lower), token_lower)
                    
                    # Find actual position in text for context
                    match = re.search(r"\b" + re.escape(token_lower) + r"\b", section_text.lower())
                    if match:
                        context = extract_context_window(
                            section_text, 
                            match.start(), 
                            match.end(), 
                            context_window
                        )
                        
                        candidates.append({
                            "bibcode": bibcode,
                            "title": title,
                            "abstract": abstract,
                            "body": body,
                            "label": original_entity,
                            "match_type": "case_insensitive",
                            "context": context,
                            "section": section_name,
                            "match_start": match.start(),
                            "match_end": match.end()
                        })
            
            # Phrase matching
            for phrase in phrase_entities:
                if "exact" in match_types:
                    match = re.search(r"\b" + re.escape(phrase) + r"\b", section_text)
                    if match:
                        context = extract_context_window(
                            section_text, 
                            match.start(), 
                            match.end(), 
                            context_window
                        )
                        
                        candidates.append({
                            "bibcode": bibcode,
                            "title": title,
                            "abstract": abstract,
                            "body": body,
                            "label": phrase,
                            "match_type": "exact_phrase",
                            "context": context,
                            "section": section_name,
                            "match_start": match.start(),
                            "match_end": match.end()
                        })
                
                if "case_insensitive" in match_types:
                    match = re.search(r"\b" + re.escape(phrase) + r"\b", section_text, re.IGNORECASE)
                    if match:
                        context = extract_context_window(
                            section_text, 
                            match.start(), 
                            match.end(), 
                            context_window
                        )
                        
                        candidates.append({
                            "bibcode": bibcode,
                            "title": title,
                            "abstract": abstract,
                            "body": body,
                            "label": phrase,
                            "match_type": "case_insensitive_phrase",
                            "context": context,
                            "section": section_name,
                            "match_start": match.start(),
                            "match_end": match.end()
                        })
    
    # Convert to DataFrame and remove duplicates
    df_candidates = pd.DataFrame(candidates)
    
    if not df_candidates.empty:
        # Remove duplicates based on bibcode, label, and context
        df_candidates = df_candidates.drop_duplicates(
            subset=["bibcode", "label", "context"]
        ).reset_index(drop=True)
    
    return df_candidates


def get_match_statistics(df_candidates: pd.DataFrame) -> Dict[str, Any]:
    """Get statistics about the matches found."""
    if df_candidates.empty:
        return {
            "total_matches": 0,
            "unique_entities": 0,
            "unique_documents": 0,
            "matches_by_type": {},
            "matches_by_section": {}
        }
    
    stats = {
        "total_matches": len(df_candidates),
        "unique_entities": df_candidates["label"].nunique(),
        "unique_documents": df_candidates["bibcode"].nunique(),
        "matches_by_type": df_candidates["match_type"].value_counts().to_dict(),
        "matches_by_section": df_candidates["section"].value_counts().to_dict()
    }
    
    return stats
