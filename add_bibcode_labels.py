#!/usr/bin/env python3
"""
Add bibcode_label column to software_mentions_all_individual_contexts.csv
based on ASCL bibcode classifications in ascl_bibcodelabels.json
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Set
import sys

def load_bibcode_labels(json_path: Path) -> Dict[str, Set[str]]:
    """Load and flatten all bibcodes from the ASCL JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_positive = set()
    all_negative = set()
    all_uncurated = set()
    
    # Iterate through all ASCL entries
    for entry_id, entry_data in data.items():
        # Add bibcodes from each category
        all_positive.update(entry_data.get('positive_bibcodes', []))
        all_negative.update(entry_data.get('negative_bibcodes', []))
        all_uncurated.update(entry_data.get('uncurated_bibcodes', []))
    
    return {
        'positive': all_positive,
        'negative': all_negative, 
        'uncurated': all_uncurated
    }

def get_bibcode_label(bibcode: str, bibcode_sets: Dict[str, Set[str]]) -> str:
    """Determine the label for a given bibcode."""
    if bibcode in bibcode_sets['positive']:
        return 'positive'
    elif bibcode in bibcode_sets['negative']:
        return 'negative'
    elif bibcode in bibcode_sets['uncurated']:
        return 'uncurated'
    else:
        return 'unknown'

def main():
    # Paths - use the full gzipped dataset
    json_path = Path('ontologies/ASCL/ascl_bibcodelabels.json')
    csv_path = Path('optimized_extractor/results/exports/software_mentions_all.csv.gzip')
    output_path = Path('optimized_extractor/results/exports/software_mentions_all_with_labels.csv.gzip')
    
    # Check if files exist
    if not json_path.exists():
        print(f"❌ Error: JSON file not found at {json_path}")
        sys.exit(1)
    
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    print("📖 Loading bibcode classifications...")
    bibcode_sets = load_bibcode_labels(json_path)
    
    print(f"Found {len(bibcode_sets['positive'])} positive bibcodes")
    print(f"Found {len(bibcode_sets['negative'])} negative bibcodes") 
    print(f"Found {len(bibcode_sets['uncurated'])} uncurated bibcodes")
    
    print("📊 Loading CSV data...")
    df = pd.read_csv(csv_path, compression='gzip')
    print(f"Loaded {len(df):,} rows from CSV")
    
    print("🏷️  Adding bibcode_label column...")
    df['bibcode_label'] = df['bibcode'].apply(lambda x: get_bibcode_label(x, bibcode_sets))
    
    # Show label distribution
    label_counts = df['bibcode_label'].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,}")
    
    print(f"💾 Saving to {output_path}...")
    df.to_csv(output_path, index=False, compression='gzip')
    
    print("✅ Done!")

if __name__ == '__main__':
    main()
