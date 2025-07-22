#!/usr/bin/env python3
# optimized_extractor/debug_test.py
"""Debug test to check term matching and preprocessing."""

import json
import sys
import re
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from optimized_extractor.preprocessing import BiblioPreprocessor
from optimized_extractor.extraction_engine import TermMatcher, DocumentProcessor

def test_preprocessing():
    """Test preprocessing steps."""
    print("=== TESTING PREPROCESSING ===")
    
    # Test ontology parsing
    ontology_path = Path("ontologies/ASCL/ascl.json")
    bibcode_db_path = Path("bibcode_lookup.db")
    
    preprocessor = BiblioPreprocessor(ontology_path, bibcode_db_path)
    
    # Parse ontology
    terms_info, term_bibcodes = preprocessor.parse_ontology()
    
    print(f"Parsed {len(terms_info)} terms")
    print(f"Total bibcode associations: {sum(len(v) for v in term_bibcodes.values())}")
    
    # Show sample terms
    print("\nSample terms:")
    for i, (term_id, term_data) in enumerate(list(terms_info.items())[:5]):
        print(f"  Term {term_id}: '{term_data['name']}'")
        print(f"    Bibcodes: {len(term_bibcodes.get(term_id, []))}")
        if term_bibcodes.get(term_id):
            print(f"    Sample bibcodes: {list(term_bibcodes[term_id])[:3]}")
    
    # Get unique bibcodes
    unique_bibcodes = preprocessor.get_unique_bibcodes(term_bibcodes)
    print(f"\nUnique bibcodes: {len(unique_bibcodes)}")
    
    return terms_info, term_bibcodes, unique_bibcodes

def test_term_matching():
    """Test term matching logic."""
    print("\n=== TESTING TERM MATCHING ===")
    
    # Load a few terms
    with open("ontologies/ASCL/ascl.json", 'r') as f:
        ontology = json.load(f)
    
    # Create a small subset for testing
    test_terms = {}
    test_terms_list = ["1", "2", "3"]  # First few terms
    for term_id in test_terms_list:
        if term_id in ontology:
            test_terms[term_id] = ontology[term_id]
    
    print(f"Testing with {len(test_terms)} terms:")
    for term_id, term_data in test_terms.items():
        print(f"  {term_id}: '{term_data['title']}'")
    
    # Test matcher
    matcher = TermMatcher(test_terms)
    
    # Test sample text
    test_texts = [
        "This paper describes LENSKY software for galactic microlensing.",
        "We used the Bahcall-Soneira Galaxy Model in our analysis.",
        "The PMCode particle mesh code was utilized for simulations.",
        "No software mentioned here.",
        "CMBFAST microwave anisotropy calculations were performed.",
    ]
    
    print(f"\nTesting against {len(test_texts)} sample texts:")
    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: '{text}'")
        matches = matcher.find_matches(text, list(test_terms.keys()))
        for term_id, match_list in matches.items():
            if match_list:
                term_name = test_terms[term_id]['title']
                print(f"  MATCH: {term_id} ('{term_name}') - {len(match_list)} matches")
                for match in match_list:
                    print(f"    '{text[match.start():match.end()]}'")

def test_simple_patterns():
    """Test simpler patterns that might match better."""
    print("\n=== TESTING SIMPLE PATTERNS ===")
    
    # Extract just the main software names (before colons)
    with open("ontologies/ASCL/ascl.json", 'r') as f:
        ontology = json.load(f)
    
    simple_terms = {}
    for term_id, term_data in list(ontology.items())[:10]:
        title = term_data.get('title', '')
        # Extract name before colon if present
        simple_name = title.split(':')[0].strip() if ':' in title else title
        simple_terms[term_id] = {'name': simple_name, 'title': title}
        
    print("Simple software names to test:")
    for term_id, term_data in simple_terms.items():
        print(f"  {term_id}: '{term_data['name']}' (from '{term_data['title']}')")
    
    # Test patterns
    matcher = TermMatcher(simple_terms)
    
    test_texts = [
        "We used LENSKY for our galactic microlensing calculations.",
        "The BSGMODEL was employed in this study.",
        "PMCode simulations were run on the cluster.",
        "Results from CMBFAST are shown in Figure 1.",
        "ISIS method was applied for image subtraction.",
        "This is a test with no software names.",
    ]
    
    print(f"\nTesting simple patterns against {len(test_texts)} texts:")
    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: '{text}'")
        matches = matcher.find_matches(text, list(simple_terms.keys()))
        total_matches = sum(len(match_list) for match_list in matches.values())
        if total_matches > 0:
            for term_id, match_list in matches.items():
                if match_list:
                    term_name = simple_terms[term_id]['name']
                    print(f"  ✓ MATCH: {term_id} ('{term_name}') - {len(match_list)} matches")
                    for match in match_list:
                        print(f"    '{text[match.start():match.end()]}'")
        else:
            print("  No matches found")

def main():
    """Run all tests."""
    print("DEBUGGING SOFTWARE MENTION EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Test preprocessing
    terms_info, term_bibcodes, unique_bibcodes = test_preprocessing()
    
    # Test term matching
    test_term_matching()
    
    # Test simple patterns
    test_simple_patterns()
    
    print("\n" + "=" * 60)
    print("DEBUG TESTS COMPLETED")

if __name__ == "__main__":
    main()
