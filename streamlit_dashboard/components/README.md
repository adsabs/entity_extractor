# Dashboard Components

## Autocomplete Component

The `autocomplete.py` module provides intelligent search suggestions for the software search box using the ASCL (Astrophysics Source Code Library) ontology.

### Features

- **Smart extraction**: Automatically extracts software names from ASCL titles (text before the colon)
- **Fast filtering**: Provides real-time suggestions as users type
- **Prioritized matching**: Shows exact prefix matches first, then partial matches
- **Caching**: Uses Streamlit's caching to load ASCL data efficiently

### Usage

The autocomplete is automatically integrated into the main dashboard. When users type in the search box, they'll see suggestions from 3,657+ software packages in the ASCL ontology.

### Examples

- Typing "zeu" shows: ZEUS-2D, ZEUS-MP/2, Zeus21, zeus
- Typing "ast" shows: AST, ASTERIX, ASTRAEUS, and more
- Typing "isis" shows: ISIS

### Implementation Details

- Loads data from: `ontologies/ASCL/ascl_bibcodelabels.json`
- Extracts software names by splitting titles on ":" character
- Filters suggestions using case-insensitive matching
- Returns up to 10 suggestions per query
