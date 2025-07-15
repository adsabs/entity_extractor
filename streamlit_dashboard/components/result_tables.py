# streamlit_dashboard/components/result_tables.py
"""Result table components for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
from typing import Optional
try:
    from ..core_pipeline.utils import highlight_match
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core_pipeline.utils import highlight_match


def render_results_table(df_candidates: pd.DataFrame) -> None:
    """Render the main results table."""
    
    if df_candidates.empty:
        st.info("No matches found")
        return
    
    st.write(f"**Found {len(df_candidates)} matches**")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Entity filter
        unique_entities = df_candidates["label"].unique()
        selected_entities = st.multiselect(
            "Filter by Entity:",
            unique_entities,
            default=unique_entities[:5] if len(unique_entities) > 5 else unique_entities
        )
    
    with col2:
        # Match type filter
        unique_match_types = df_candidates["match_type"].unique()
        selected_match_types = st.multiselect(
            "Filter by Match Type:",
            unique_match_types,
            default=unique_match_types
        )
    
    with col3:
        # Section filter
        unique_sections = df_candidates["section"].unique()
        selected_sections = st.multiselect(
            "Filter by Section:",
            unique_sections,
            default=unique_sections
        )
    
    # Apply filters
    filtered_df = df_candidates[
        (df_candidates["label"].isin(selected_entities)) &
        (df_candidates["match_type"].isin(selected_match_types)) &
        (df_candidates["section"].isin(selected_sections))
    ]
    
    if filtered_df.empty:
        st.warning("No matches found with current filters")
        return
    
    # Display table
    st.write(f"Showing {len(filtered_df)} of {len(df_candidates)} matches")
    
    # Create display dataframe
    display_df = create_display_dataframe(filtered_df)
    
    # Interactive table
    selected_rows = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Context Preview": st.column_config.TextColumn(
                "Context Preview",
                width="large"
            ),
            "Entity": st.column_config.TextColumn(
                "Entity",
                width="small"
            ),
            "Match Type": st.column_config.TextColumn(
                "Match Type",
                width="small"
            ),
            "Section": st.column_config.TextColumn(
                "Section",
                width="small"
            )
        }
    )
    
    # Detailed view for selected rows
    if st.checkbox("Show detailed context"):
        render_detailed_context(filtered_df)
    
    # Export options
    render_export_options(filtered_df)


def create_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a display-friendly version of the dataframe."""
    
    display_df = df.copy()
    
    # Truncate context for preview
    display_df["Context Preview"] = display_df["context"].apply(
        lambda x: x[:100] + "..." if len(x) > 100 else x
    )
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        "label": "Entity",
        "match_type": "Match Type",
        "section": "Section",
        "bibcode": "Document ID"
    })
    
    # Select columns for display
    display_columns = [
        "Entity",
        "Match Type", 
        "Section",
        "Document ID",
        "Context Preview"
    ]
    
    return display_df[display_columns]


def render_detailed_context(df: pd.DataFrame) -> None:
    """Render detailed context for each match."""
    
    st.subheader("Detailed Context")
    
    for idx, row in df.iterrows():
        with st.expander(f"'{row['label']}' in {row['bibcode']} ({row['section']})"):
            
            # Match information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Entity", row["label"])
            
            with col2:
                st.metric("Match Type", row["match_type"])
            
            with col3:
                st.metric("Section", row["section"])
            
            # Context with highlighting
            st.subheader("Context")
            highlighted_context = highlight_match(row["context"], row["label"])
            st.markdown(highlighted_context)
            
            # Additional information
            if pd.notna(row.get("similarity_score")):
                st.metric("Similarity Score", f"{row['similarity_score']:.3f}")
            
            if pd.notna(row.get("ner_score")):
                st.metric("NER Score", f"{row['ner_score']:.3f}")
            
            if pd.notna(row.get("composite_score")):
                st.metric("Composite Score", f"{row['composite_score']:.3f}")
            
            # Document information
            st.subheader("Document Information")
            st.write(f"**Title:** {row['title']}")
            if row['abstract']:
                st.write(f"**Abstract:** {row['abstract'][:200]}...")


def render_export_options(df: pd.DataFrame) -> None:
    """Render export options for the results."""
    
    st.subheader("Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name="entity_matches.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name="entity_matches.json",
            mime="application/json"
        )
    
    with col3:
        # Show raw data
        if st.button("🔍 Show Raw Data"):
            st.subheader("Raw Data")
            st.dataframe(df, use_container_width=True)


def render_match_statistics(df: pd.DataFrame) -> None:
    """Render match statistics."""
    
    if df.empty:
        return
    
    st.subheader("Match Statistics")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", len(df))
    
    with col2:
        st.metric("Unique Entities", df["label"].nunique())
    
    with col3:
        st.metric("Unique Documents", df["bibcode"].nunique())
    
    with col4:
        avg_context_length = df["context"].str.len().mean()
        st.metric("Avg Context Length", f"{avg_context_length:.0f}")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matches by Type")
        match_type_counts = df["match_type"].value_counts()
        st.bar_chart(match_type_counts)
    
    with col2:
        st.subheader("Matches by Section")
        section_counts = df["section"].value_counts()
        st.bar_chart(section_counts)
