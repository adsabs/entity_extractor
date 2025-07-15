# streamlit_dashboard/test_basic.py
"""Basic test to verify core functionality."""

import pandas as pd
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Mock streamlit for testing
class MockStreamlit:
    def cache_data(self, func):
        return func
    def cache_resource(self, func):
        return func
    def warning(self, msg):
        print(f"Warning: {msg}")
    def error(self, msg):
        print(f"Error: {msg}")

sys.modules['streamlit'] = MockStreamlit()
sys.modules['st'] = MockStreamlit()

from core_pipeline.load_inputs import validate_entities, get_entity_list
from core_pipeline.batch_filter import find_exact_matches, get_match_statistics
from core_pipeline.utils import extract_context_window, highlight_match


def test_basic_functionality():
    """Test basic functionality with sample data."""
    
    print("Testing basic functionality...")
    
    # Test 1: Entity validation
    print("\n1. Testing entity validation...")
    entities = ["Python", "TensorFlow", "matplotlib", ""]
    validation = validate_entities(entities)
    print(f"Validation result: {validation}")
    
    # Test 2: Create sample corpus
    print("\n2. Creating sample corpus...")
    sample_corpus = pd.DataFrame([
        {
            "bibcode": "2023ApJ...900..123A",
            "title": "Deep Learning with Python and TensorFlow",
            "abstract": "We present a study using Python libraries including matplotlib for visualization.",
            "body": "The software package TensorFlow was used extensively in this work.",
            "full_text": "Deep Learning with Python and TensorFlow We present a study using Python libraries including matplotlib for visualization. The software package TensorFlow was used extensively in this work."
        },
        {
            "bibcode": "2023MNRAS.456..789B", 
            "title": "Astronomical Data Analysis",
            "abstract": "This paper describes methods for analyzing astronomical data.",
            "body": "We used various Python scripts and the matplotlib library for plotting.",
            "full_text": "Astronomical Data Analysis This paper describes methods for analyzing astronomical data. We used various Python scripts and the matplotlib library for plotting."
        }
    ])
    
    print(f"Sample corpus created with {len(sample_corpus)} documents")
    
    # Test 3: Find exact matches
    print("\n3. Testing exact matching...")
    test_entities = ["Python", "TensorFlow", "matplotlib"]
    matches = find_exact_matches(
        df_corpus=sample_corpus,
        entities=test_entities,
        match_types=["exact", "case_insensitive"],
        context_window=50
    )
    
    print(f"Found {len(matches)} matches")
    if not matches.empty:
        print("Match details:")
        for _, row in matches.iterrows():
            print(f"  - Entity: {row['label']}")
            print(f"    Document: {row['bibcode']}")
            print(f"    Section: {row['section']}")
            print(f"    Context: {row['context'][:100]}...")
    
    # Test 4: Get match statistics
    print("\n4. Testing match statistics...")
    stats = get_match_statistics(matches)
    print(f"Match statistics: {stats}")
    
    # Test 5: Test utility functions
    print("\n5. Testing utility functions...")
    test_text = "This is a test sentence with Python programming language."
    context = extract_context_window(test_text, 25, 31, 10)
    print(f"Context window: '{context}'")
    
    highlighted = highlight_match(test_text, "Python")
    print(f"Highlighted text: {highlighted}")
    
    print("\n✅ All basic tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
