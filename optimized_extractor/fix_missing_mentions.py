#!/usr/bin/env python3
# optimized_extractor/fix_missing_mentions.py
"""
Fix missing mentions by re-extracting terms that only had title/abstract matches.
This addresses the bug where only body matches were being captured.
"""

import pandas as pd
import json
import sqlite3
from pathlib import Path
import logging
from optimized_extractor.extraction_engine import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_missing_mentions():
    """Re-extract mentions for all terms to capture title/abstract-only matches."""
    
    # Load existing results
    results_path = Path("optimized_extractor/results/exports/software_mentions_all.csv")
    df = pd.read_csv(results_path)
    
    logger.info(f"Current dataset has {len(df)} mentions")
    
    # Load terms info and preprocessing results
    with open("optimized_extractor/results/preprocessed/terms_info.json", 'r') as f:
        terms_info = json.load(f)
    
    with open("optimized_extractor/results/preprocessed/file_assignments.json", 'r') as f:
        file_assignments = json.load(f)
    
    logger.info(f"Processing {len(terms_info)} terms across {len(file_assignments)} files")
    
    # Set up document processor
    processor = DocumentProcessor(terms_info)
    corpus_base_path = Path('/home/scixmuse/scix_data/ads_metadata_by_year_full/')
    
    new_results = []
    processed_count = 0
    
    # Process each file's assignments
    for filename, assignments in file_assignments.items():
        file_path = corpus_base_path / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Processing {filename} with {len(assignments)} assignments")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for assignment in assignments:
                    try:
                        # Read the document
                        f.seek(assignment['byte_offset'])
                        line = f.readline()
                        doc_data = json.loads(line)
                        
                        # Extract with fixed logic
                        doc_results = processor.process_document(doc_data, assignment['terms'])
                        
                        # Convert to records
                        for result in doc_results:
                            new_results.append({
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
                        
                        processed_count += 1
                        if processed_count % 1000 == 0:
                            logger.info(f"Processed {processed_count} documents, found {len(new_results)} mentions")
                        
                    except Exception as e:
                        logger.error(f"Error processing {assignment['bibcode']}: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
    
    logger.info(f"Extraction completed. Found {len(new_results)} total mentions")
    
    # Create new DataFrame
    new_df = pd.DataFrame(new_results)
    
    # Save the corrected results
    output_path = Path("optimized_extractor/results/exports/software_mentions_all_fixed.csv")
    new_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved corrected results to {output_path}")
    logger.info(f"Original mentions: {len(df)}")
    logger.info(f"Fixed mentions: {len(new_df)}")
    logger.info(f"Difference: +{len(new_df) - len(df)}")
    
    return new_df

def main():
    """Main function."""
    logger.info("Starting missing mentions fix...")
    new_df = find_missing_mentions()
    
    # Quick verification with pixell
    pixell_results = new_df[new_df['term_name'].str.contains('Pixell', case=False, na=False)]
    logger.info(f"Pixell mentions in fixed dataset: {len(pixell_results)}")
    
    logger.info("Fix completed!")

if __name__ == "__main__":
    main()
