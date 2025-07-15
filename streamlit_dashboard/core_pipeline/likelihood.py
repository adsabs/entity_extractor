# streamlit_dashboard/core_pipeline/likelihood.py
"""Likelihood assignment wrapper for Streamlit dashboard."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st
try:
    from ..config import DEFAULT_LIKELIHOOD_THRESHOLDS
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DEFAULT_LIKELIHOOD_THRESHOLDS


def assign_likelihood_labels(df_candidates: pd.DataFrame,
                           thresholds: Dict[str, float] = DEFAULT_LIKELIHOOD_THRESHOLDS,
                           score_column: str = "composite_score") -> pd.DataFrame:
    """Assign likelihood labels based on composite scores."""
    
    if df_candidates.empty or score_column not in df_candidates.columns:
        return df_candidates
    
    df_result = df_candidates.copy()
    likelihood_labels = []
    
    for _, row in df_result.iterrows():
        score = row[score_column]
        
        if pd.isna(score):
            label = "unknown"
        elif score >= thresholds["very_likely"]:
            label = "very_likely"
        elif score >= thresholds["likely"]:
            label = "likely"
        elif score >= thresholds["unlikely"]:
            label = "unlikely"
        else:
            label = "very_unlikely"
        
        likelihood_labels.append(label)
    
    df_result["likelihood_label"] = likelihood_labels
    return df_result


def get_likelihood_statistics(df_candidates: pd.DataFrame) -> Dict[str, Any]:
    """Get statistics about likelihood label distribution."""
    
    if df_candidates.empty or "likelihood_label" not in df_candidates.columns:
        return {}
    
    label_counts = df_candidates["likelihood_label"].value_counts()
    total = len(df_candidates)
    
    stats = {
        "total_candidates": total,
        "label_counts": label_counts.to_dict(),
        "label_percentages": {
            label: (count / total) * 100 
            for label, count in label_counts.items()
        }
    }
    
    return stats


def filter_by_likelihood(df_candidates: pd.DataFrame,
                        min_likelihood: str = "likely") -> pd.DataFrame:
    """Filter candidates by minimum likelihood level."""
    
    if df_candidates.empty or "likelihood_label" not in df_candidates.columns:
        return df_candidates
    
    # Define likelihood order
    likelihood_order = {
        "very_unlikely": 0,
        "unlikely": 1,
        "likely": 2,
        "very_likely": 3,
        "unknown": -1
    }
    
    min_level = likelihood_order.get(min_likelihood, 0)
    
    # Filter candidates
    filtered_df = df_candidates[
        df_candidates["likelihood_label"].apply(
            lambda x: likelihood_order.get(x, -1) >= min_level
        )
    ].copy()
    
    return filtered_df


def calibrate_thresholds(df_candidates: pd.DataFrame,
                        target_percentages: Dict[str, float],
                        score_column: str = "composite_score") -> Dict[str, float]:
    """Calibrate thresholds based on target percentages."""
    
    if df_candidates.empty or score_column not in df_candidates.columns:
        return DEFAULT_LIKELIHOOD_THRESHOLDS
    
    scores = df_candidates[score_column].dropna().sort_values(ascending=False)
    
    if len(scores) == 0:
        return DEFAULT_LIKELIHOOD_THRESHOLDS
    
    # Calculate percentiles
    very_likely_percentile = target_percentages.get("very_likely", 0.1)
    likely_percentile = target_percentages.get("likely", 0.3)
    unlikely_percentile = target_percentages.get("unlikely", 0.6)
    
    # Get threshold values
    very_likely_threshold = scores.quantile(very_likely_percentile)
    likely_threshold = scores.quantile(very_likely_percentile + likely_percentile)
    unlikely_threshold = scores.quantile(very_likely_percentile + likely_percentile + unlikely_percentile)
    
    calibrated_thresholds = {
        "very_likely": float(very_likely_threshold),
        "likely": float(likely_threshold),
        "unlikely": float(unlikely_threshold)
    }
    
    return calibrated_thresholds


def export_for_training(df_candidates: pd.DataFrame,
                       min_likelihood: str = "likely") -> pd.DataFrame:
    """Export candidates in format suitable for training."""
    
    # Filter by minimum likelihood
    filtered_df = filter_by_likelihood(df_candidates, min_likelihood)
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Select relevant columns for training
    training_columns = [
        "bibcode",
        "label",
        "context",
        "likelihood_label",
        "composite_score",
        "similarity_score",
        "heuristic_score",
        "ner_score"
    ]
    
    # Only include columns that exist
    available_columns = [col for col in training_columns if col in filtered_df.columns]
    
    training_df = filtered_df[available_columns].copy()
    
    # Add binary classification labels
    training_df["is_positive"] = training_df["likelihood_label"].isin(
        ["likely", "very_likely"]
    )
    
    return training_df
