# streamlit_dashboard/components/autocomplete_simple.py
"""Simple autocomplete using st.selectbox with built-in search."""

import streamlit as st
from .autocomplete import load_ascl_software_names


def render_simple_autocomplete_search() -> str:
    """Render simple autocomplete using selectbox with built-in search."""
    # Load available software names
    software_names = load_ascl_software_names()
    
    if not software_names:
        # Fallback to regular text input if ASCL data unavailable
        return st.sidebar.text_input(
            "Search for software:",
            placeholder="e.g., astropy, PyVO, LENSKY",
            help="Search for software mentions by name"
        )
    
    # Use selectbox with built-in search functionality
    selection = st.sidebar.selectbox(
        "🔍 Search software:",
        options=[""] + software_names,  # Empty string as default
        index=0,
        placeholder="Type to filter software options...",
        help="Click and start typing to filter through 3,657+ software packages from ASCL ontology",
        key="software_selectbox"
    )
    
    return selection if selection else ""
