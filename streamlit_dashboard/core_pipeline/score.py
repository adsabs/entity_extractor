# streamlit_dashboard/core_pipeline/score.py
"""Scoring wrapper for Streamlit dashboard."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st
import re
try:
    from ..config import DEFAULT_HEURISTIC_KEYWORDS
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DEFAULT_HEURISTIC_KEYWORDS


def score_filtered_contexts(df_candidates: pd.DataFrame,
                           heuristic_keywords: List[str] = DEFAULT_HEURISTIC_KEYWORDS,
                           ner_models: Optional[List[str]] = None) -> pd.DataFrame:
    """Score filtered contexts using multiple signals."""
    
    if df_candidates.empty:
        return df_candidates
    
    df_result = df_candidates.copy()
    
    # Calculate heuristic scores
    df_result = add_heuristic_scores(df_result, heuristic_keywords)
    
    # Add NER scores if models are provided
    if ner_models:
        df_result = add_ner_scores(df_result, ner_models)
    else:
        df_result["ner_score"] = 0.0
    
    # Calculate composite score
    df_result = calculate_composite_score(df_result)
    
    return df_result


def add_heuristic_scores(df_candidates: pd.DataFrame,
                        heuristic_keywords: List[str]) -> pd.DataFrame:
    """Add heuristic scores based on keyword co-occurrence."""
    
    df_result = df_candidates.copy()
    heuristic_scores = []
    
    for _, row in df_result.iterrows():
        context = row["context"].lower()
        
        # Count keyword occurrences
        keyword_count = 0
        for keyword in heuristic_keywords:
            if keyword.lower() in context:
                keyword_count += 1
        
        # Calculate score (normalize by number of keywords)
        if heuristic_keywords:
            score = keyword_count / len(heuristic_keywords)
        else:
            score = 0.0
        
        heuristic_scores.append(score)
    
    df_result["heuristic_score"] = heuristic_scores
    return df_result


def add_ner_scores(df_candidates: pd.DataFrame,
                  ner_models: List[str]) -> pd.DataFrame:
    """Add NER scores using specified models."""
    
    # Placeholder implementation
    # In a real implementation, this would load NER models and score contexts
    df_result = df_candidates.copy()
    
    # For now, return random scores for demonstration
    np.random.seed(42)  # For reproducibility
    ner_scores = np.random.uniform(0, 1, len(df_result))
    
    df_result["ner_score"] = ner_scores
    return df_result


def calculate_composite_score(df_candidates: pd.DataFrame,
                             weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Calculate composite score from multiple signals."""
    
    if weights is None:
        weights = {
            "similarity_score": 0.5,
            "heuristic_score": 0.3,
            "ner_score": 0.2
        }
    
    df_result = df_candidates.copy()
    composite_scores = []
    
    for _, row in df_result.iterrows():
        score = 0.0
        total_weight = 0.0
        
        # Similarity score
        if "similarity_score" in row and pd.notna(row["similarity_score"]):
            score += row["similarity_score"] * weights.get("similarity_score", 0)
            total_weight += weights.get("similarity_score", 0)
        
        # Heuristic score
        if "heuristic_score" in row and pd.notna(row["heuristic_score"]):
            score += row["heuristic_score"] * weights.get("heuristic_score", 0)
            total_weight += weights.get("heuristic_score", 0)
        
        # NER score
        if "ner_score" in row and pd.notna(row["ner_score"]):
            score += row["ner_score"] * weights.get("ner_score", 0)
            total_weight += weights.get("ner_score", 0)
        
        # Normalize by total weight
        if total_weight > 0:
            score = score / total_weight
        
        composite_scores.append(score)
    
    df_result["composite_score"] = composite_scores
    return df_result


def get_scoring_statistics(df_candidates: pd.DataFrame) -> Dict[str, Any]:
    """Get statistics about the scoring results."""
    
    if df_candidates.empty:
        return {}
    
    stats = {}
    
    # Score columns to analyze
    score_columns = ["similarity_score", "heuristic_score", "ner_score", "composite_score"]
    
    for col in score_columns:
        if col in df_candidates.columns:
            scores = df_candidates[col].dropna()
            if len(scores) > 0:
                stats[col] = {
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                    "median": float(scores.median())
                }
    
    return stats


def filter_by_score_threshold(df_candidates: pd.DataFrame,
                             score_column: str = "composite_score",
                             threshold: float = 0.5) -> pd.DataFrame:
    """Filter candidates by score threshold."""
    
    if df_candidates.empty or score_column not in df_candidates.columns:
        return df_candidates
    
    return df_candidates[df_candidates[score_column] >= threshold].copy()
