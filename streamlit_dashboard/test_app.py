#!/usr/bin/env python3
"""Test script for the updated streamlit dashboard."""

import sys
from pathlib import Path
import pandas as pd

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def test_load_data():
    """Test loading the software mentions data."""
    csv_path = Path("../optimized_extractor/results/exports/software_mentions_all.csv.gzip")
    
    if not csv_path.exists():
        print(f"❌ Data file not found at {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path, compression='gzip', nrows=100)
        print(f"✅ Successfully loaded {len(df)} sample records")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample term names: {df['term_name'].head(3).tolist()}")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_search_functionality():
    """Test the search functionality."""
    from app import search_term, load_software_mentions_data
    
    # Load a small sample
    csv_path = Path("../optimized_extractor/results/exports/software_mentions_all.csv.gzip")
    df = pd.read_csv(csv_path, compression='gzip', nrows=1000)
    
    # Test search
    results = search_term(df, "astropy")
    print(f"✅ Search for 'astropy' returned {len(results)} results")
    
    results = search_term(df, "PyVO")
    print(f"✅ Search for 'PyVO' returned {len(results)} results")
    
    # Test with a term that likely exists
    unique_terms = df['term_name'].str.extract(r'^([^:]+)')[0].str.strip().value_counts()
    if len(unique_terms) > 0:
        test_term = unique_terms.index[0]
        results = search_term(df, test_term)
        print(f"✅ Search for '{test_term}' returned {len(results)} results")
    
    return True

def test_ner_models():
    """Test NER model functionality."""
    from app import get_available_ner_models, run_ner_on_context
    
    models = get_available_ner_models()
    print(f"✅ Available NER models: {models}")
    
    # Test with sample context
    context = "We used Python and matplotlib to analyze the data from the JWST telescope."
    
    for model in models:
        print(f"\nTesting {model}...")
        try:
            # This will likely fail due to model loading, but let's see the error
            results = run_ner_on_context(context, model)
            print(f"✅ {model}: {results['status']}")
            if results['status'] == 'success':
                print(f"   Found {len(results['entities'])} entities")
            else:
                print(f"   Error: {results.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ {model}: Exception - {e}")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing updated Streamlit dashboard functionality...")
    print("=" * 60)
    
    print("\n1. Testing data loading...")
    test_load_data()
    
    print("\n2. Testing search functionality...")
    test_search_functionality()
    
    print("\n3. Testing NER models...")
    test_ner_models()
    
    print("\n✅ Tests completed!")
    print("\nTo run the dashboard:")
    print("cd streamlit_dashboard")
    print("source ../venv/bin/activate")
    print("streamlit run app.py")
