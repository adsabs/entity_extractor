#!/usr/bin/env python3
"""
Test script to verify individual match extraction works correctly.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from extraction_engine import DocumentProcessor, TermMatcher
import json

def test_individual_matches():
    """Test that individual matches are extracted correctly."""
    
    # Sample document with multiple matches
    doc_data = {
        'bibcode': '2023TEST001A',
        'title': 'Using SUSHI and SUSHI tools for analysis',
        'abstract': 'The SUSHI algorithm provides excellent results.',
        'body': 'We implemented SUSHI in our pipeline. The SUSHI framework is robust. SUSHI performs well under various conditions. Our SUSHI implementation handles edge cases.'
    }
    
    # Mock terms info
    terms_info = {
        '123': {
            'name': 'SUSHI: Software for Understanding Scientific Hierarchies',
            'bibcodes': ['2023TEST001A']
        }
    }
    
    # Create processor
    processor = DocumentProcessor(terms_info)
    
    # Process document
    results = processor.process_document(doc_data, ['123'])
    
    print(f"Total results: {len(results)}")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Term: {result.term_name}")
        print(f"  Location: {result.match_location}")
        print(f"  Match count: {result.match_count}")
        print(f"  In title: {result.in_title}")
        print(f"  In abstract: {result.in_abstract}")
        print(f"  Context: {result.context[:100]}...")
        print("-"*40)
    
    # Verify we have separate entries for each match location
    locations = [r.match_location for r in results]
    print(f"Match locations: {locations}")
    
    # Count body matches (should be 4 separate entries)
    body_matches = [r for r in results if r.match_location == 'body']
    print(f"Body match entries: {len(body_matches)}")
    
    return results

if __name__ == "__main__":
    test_individual_matches()
