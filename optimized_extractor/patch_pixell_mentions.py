#!/usr/bin/env python3
# optimized_extractor/patch_pixell_mentions.py
"""
Quick patch to add the missing pixell mentions to the existing dataset.
"""

import pandas as pd
import json
import sqlite3
from pathlib import Path
from optimized_extractor.extraction_engine import DocumentProcessor

def patch_missing_pixell():
    """Add missing pixell mentions to the existing dataset."""
    
    # Load existing results
    df = pd.read_csv("optimized_extractor/results/exports/software_mentions_all.csv")
    print(f"Current dataset: {len(df)} mentions")
    
    # Current pixell mentions
    current_pixell = df[df['term_name'].str.contains('Pixell', case=False, na=False)]
    print(f"Current pixell mentions: {len(current_pixell)}")
    
    # Missing bibcodes (the ones we identified)
    missing_bibcodes = ['1973MarBi..23..129S', '1990CaJZ...68.1525H', '2004RuJMB..30...28T', 
                       '2004RuJMB..30..101T', '2009RuJMB..35..388T', '2010CaJZ...88.1149T']
    
    # Set up terms info for pixell
    terms_info = {
        '2791': {
            'title': 'Pixell: Rectangular pixel map manipulation and harmonic analysis library',
            'name': 'Pixell: Rectangular pixel map manipulation and harmonic analysis library'
        }
    }
    
    processor = DocumentProcessor(terms_info)
    corpus_base_path = Path('/home/scixmuse/scix_data/ads_metadata_by_year_full/')
    
    # Get file locations
    conn = sqlite3.connect('bibcode_lookup.db')
    cursor = conn.cursor()
    
    new_records = []
    
    for bibcode in missing_bibcodes:
        cursor.execute('SELECT filename, line_number, byte_offset FROM bibcode_lookup WHERE bibcode = ?', (bibcode,))
        result = cursor.fetchone()
        
        if result:
            filename, line_number, byte_offset = result
            file_path = corpus_base_path / filename
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(byte_offset)
                    line = f.readline()
                    doc = json.loads(line)
                
                # Extract with fixed logic
                results = processor.process_document(doc, ['2791'])
                
                for result in results:
                    new_records.append({
                        'term_id': result.term_id,
                        'term_name': result.term_name,
                        'bibcode': result.bibcode,
                        'title': result.title,
                        'abstract': result.abstract,
                        'context': result.context,
                        'match_count': result.match_count,
                        'in_title': result.in_title,
                        'in_abstract': result.in_abstract,
                        'match_location': result.match_location
                    })
                    print(f"✅ Added: {bibcode} -> {result.match_location}")
                    
            except Exception as e:
                print(f"❌ Error processing {bibcode}: {e}")
    
    conn.close()
    
    if new_records:
        # Add new records to existing DataFrame
        new_df = pd.DataFrame(new_records)
        combined_df = pd.concat([df, new_df], ignore_index=True)
        
        # Save updated dataset
        output_path = "optimized_extractor/results/exports/software_mentions_all_patched.csv"
        combined_df.to_csv(output_path, index=False)
        
        print(f"\\nPatched dataset saved to: {output_path}")
        print(f"Original mentions: {len(df)}")
        print(f"Added mentions: {len(new_records)}")
        print(f"Total mentions: {len(combined_df)}")
        
        # Verify pixell count
        patched_pixell = combined_df[combined_df['term_name'].str.contains('Pixell', case=False, na=False)]
        print(f"Pixell mentions after patch: {len(patched_pixell)}")
        
        return combined_df
    else:
        print("No new records to add")
        return df

if __name__ == "__main__":
    patch_missing_pixell()
