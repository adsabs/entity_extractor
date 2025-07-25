#!/usr/bin/env python3
"""Test script to verify the search fix for exact vs fuzzy matching."""

import sys
import pandas as pd
from pathlib import Path

# Add the streamlit_dashboard to path
sys.path.append(str(Path(__file__).parent / "streamlit_dashboard"))

from app import search_term

def test_search_behavior():
    """Test the new search behavior for exact vs fuzzy matching."""
    
    # Create test data simulating the issue
    test_data = pd.DataFrame({
        'term_name': [
            'ZEUS: A Computational Fluid Dynamics Code',
            'ZEUS-2D: A radiation magnetohydrodynamics code',  
            'ZEUS-3D: A magnetohydrodynamics code',
            'ZEUS-MP: A multi-physics code',
            'Astropy: Community Python library',
            'NumPy: Numerical computing library'
        ],
        'bibcode': ['2001test1', '2002test2', '2003test3', '2004test4', '2005test5', '2006test6'],
        'context': ['context1', 'context2', 'context3', 'context4', 'context5', 'context6']
    })
    
    # Add the software_key column (simulating what happens in load_software_mentions_data)
    test_data['software_key'] = (
        test_data['term_name']
        .fillna('')
        .apply(lambda x: str(x).split(':')[0].strip().lower())
    )
    
    print("Test Data:")
    print(test_data[['term_name', 'software_key']])
    print("\n" + "="*80 + "\n")
    
    # Test 1: Exact search for "zeus" should only return the exact ZEUS match
    print("TEST 1: Searching for 'zeus' (exact match)")
    zeus_results = search_term(test_data, 'zeus')
    print(f"Results ({len(zeus_results)} found):")
    for _, row in zeus_results.iterrows():
        print(f"  - {row['term_name']}")
    
    expected_zeus = ['ZEUS: A Computational Fluid Dynamics Code']
    actual_zeus = zeus_results['term_name'].tolist()
    
    if actual_zeus == expected_zeus:
        print("✅ PASS: Only exact 'ZEUS' match returned")
    else:
        print(f"❌ FAIL: Expected {expected_zeus}, got {actual_zeus}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 2: Fuzzy search for "zeus-" should return ZEUS-2D, ZEUS-3D, ZEUS-MP  
    print("TEST 2: Searching for 'zeus-' (fuzzy match)")
    zeus_dash_results = search_term(test_data, 'zeus-')
    print(f"Results ({len(zeus_dash_results)} found):")
    for _, row in zeus_dash_results.iterrows():
        print(f"  - {row['term_name']}")
    
    expected_variants = [
        'ZEUS-2D: A radiation magnetohydrodynamics code',
        'ZEUS-3D: A magnetohydrodynamics code', 
        'ZEUS-MP: A multi-physics code'
    ]
    actual_variants = zeus_dash_results['term_name'].tolist()
    
    if set(actual_variants) == set(expected_variants):
        print("✅ PASS: All ZEUS variants returned for fuzzy search")
    else:
        print(f"❌ FAIL: Expected {expected_variants}, got {actual_variants}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 3: Partial search for "zeu" should return all ZEUS variants (fallback to fuzzy)
    print("TEST 3: Searching for 'zeu' (partial fuzzy match)")
    zeu_results = search_term(test_data, 'zeu')
    print(f"Results ({len(zeu_results)} found):")
    for _, row in zeu_results.iterrows():
        print(f"  - {row['term_name']}")
    
    expected_all_zeus = [
        'ZEUS: A Computational Fluid Dynamics Code',
        'ZEUS-2D: A radiation magnetohydrodynamics code',
        'ZEUS-3D: A magnetohydrodynamics code',
        'ZEUS-MP: A multi-physics code'
    ]
    actual_all_zeus = zeu_results['term_name'].tolist()
    
    if set(actual_all_zeus) == set(expected_all_zeus):
        print("✅ PASS: All ZEUS entries returned for partial search")
    else:
        print(f"❌ FAIL: Expected {expected_all_zeus}, got {actual_all_zeus}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 4: Exact search for "astropy" should work
    print("TEST 4: Searching for 'astropy' (exact match)")
    astropy_results = search_term(test_data, 'astropy')
    print(f"Results ({len(astropy_results)} found):")
    for _, row in astropy_results.iterrows():
        print(f"  - {row['term_name']}")
    
    expected_astropy = ['Astropy: Community Python library']
    actual_astropy = astropy_results['term_name'].tolist()
    
    if actual_astropy == expected_astropy:
        print("✅ PASS: Only exact 'Astropy' match returned")
    else:
        print(f"❌ FAIL: Expected {expected_astropy}, got {actual_astropy}")

if __name__ == "__main__":
    test_search_behavior()
