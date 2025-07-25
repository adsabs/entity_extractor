# streamlit_dashboard/components/autocomplete.py
"""Autocomplete functionality for software search using ASCL ontology."""

import json
import streamlit as st
from pathlib import Path
from typing import List, Dict, Set


@st.cache_data
def load_ascl_software_names() -> List[str]:
    """Load and extract software names from ASCL ontology JSON file."""
    ascl_path = Path(__file__).parent.parent.parent / "ontologies" / "ASCL" / "ascl_bibcodelabels.json"
    
    if not ascl_path.exists():
        st.warning(f"ASCL ontology file not found at {ascl_path}")
        return []
    
    try:
        with open(ascl_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        software_names = set()
        
        for entry in data.values():
            title = entry.get('title', '')
            if ':' in title:
                # Extract software name (part before the colon)
                software_name = title.split(':')[0].strip()
                if software_name:
                    software_names.add(software_name)
            else:
                # If no colon, use the entire title
                if title.strip():
                    software_names.add(title.strip())
        
        return sorted(list(software_names))
    
    except Exception as e:
        st.error(f"Error loading ASCL ontology: {e}")
        return []


def filter_software_suggestions(query: str, software_names: List[str], max_suggestions: int = 10) -> List[str]:
    """Filter software names based on user query for autocomplete suggestions."""
    if not query:
        return []
    
    query_lower = query.lower()
    suggestions = []
    
    # Exact matches first
    for name in software_names:
        if name.lower().startswith(query_lower):
            suggestions.append(name)
    
    # Partial matches second (if not already included)
    for name in software_names:
        if query_lower in name.lower() and name not in suggestions:
            suggestions.append(name)
    
    return suggestions[:max_suggestions]


def render_autocomplete_search() -> str:
    """Render autocomplete search with dropdown functionality."""
    # Load available software names
    software_names = load_ascl_software_names()
    
    if not software_names:
        # Fallback to regular text input if ASCL data unavailable
        return st.sidebar.text_input(
            "Search for software:",
            placeholder="e.g., astropy, PyVO, LENSKY",
            help="Search for software mentions by name"
        )
    
    # Create a text input for typing (no value= parameter for real-time updates)
    typed_input = st.sidebar.text_input(
        "🔍 Search software:",
        placeholder="Start typing... (e.g., zeu, ast, isis)",
        help="Start typing to see matching software from ASCL ontology",
        key="typed_query"
    )
    
    # Prepare options for selectbox based on typed input
    if typed_input and len(typed_input) >= 1:
        # Get filtered suggestions
        filtered_options = filter_software_suggestions(typed_input, software_names, max_suggestions=50)
        
        if filtered_options:
            # Create dropdown options with the typed query as default
            dropdown_options = ["-- Select software or keep typing --"] + filtered_options
            
            # Show the dropdown
            selected_option = st.sidebar.selectbox(
                f"📋 Found {len(filtered_options)} matches:",
                options=dropdown_options,
                key=f"dropdown_{typed_input}",
                help="Select from the dropdown or continue typing to refine results"
            )
            
            # If user selected something from dropdown, return that
            if selected_option != "-- Select software or keep typing --":
                return selected_option
    
    elif typed_input == "":
        # Show some popular software as examples when nothing is typed
        popular_software = [
            "astropy", "numpy", "scipy", "matplotlib", "GADGET", "FLASH", 
            "ZEUS-2D", "ISIS", "IRAF", "CASA"
        ]
        available_popular = [name for name in popular_software if name in software_names]
        
        if available_popular:
            st.sidebar.markdown("**💡 Popular software (examples):**")
            selected_popular = st.sidebar.selectbox(
                "Select to search:",
                options=["-- Start typing above --"] + available_popular[:5],
                key="popular_selection"
            )
            
            if selected_popular != "-- Start typing above --":
                # Return the selection directly
                return selected_popular
    
    # Return the typed input if nothing was selected from dropdown
    return typed_input
