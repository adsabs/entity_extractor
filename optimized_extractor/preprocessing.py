# optimized_extractor/preprocessing.py
"""Preprocessing module for optimized software mention extraction."""

import json
import re
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiblioPreprocessor:
    """Handles preprocessing of ASCL ontology and bibcode resolution."""
    
    def __init__(self, ontology_path: Path, bibcode_db_path: Path):
        self.ontology_path = ontology_path
        self.bibcode_db_path = bibcode_db_path
        self.bibcode_url_pattern = re.compile(r'abs/([^/?#]+)')
        
    def extract_bibcode_from_url(self, entry: str) -> str:
        """Extract bibcode from URL or return as-is if not URL."""
        if isinstance(entry, str) and 'abs/' in entry:
            match = self.bibcode_url_pattern.search(entry)
            return match.group(1) if match else entry
        return entry if isinstance(entry, str) else None
        
    def parse_ontology(self) -> Tuple[Dict[str, Dict], Dict[str, Set[str]]]:
        """
        Parse ASCL ontology and extract term-bibcode mappings.
        
        Returns:
            terms_info: {term_id: {'name': str, 'title': str, 'abstract': str}}
            term_bibcodes: {term_id: set of bibcodes}
        """
        logger.info(f"Loading ontology from {self.ontology_path}")
        
        with open(self.ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
        
        terms_info = {}
        term_bibcodes = defaultdict(set)
        
        for term_id, term_data in ontology.items():
            # Store term metadata
            terms_info[term_id] = {
                'name': term_data.get('title', ''),
                'title': term_data.get('title', ''),
                'abstract': term_data.get('abstract', ''),
                'ascl_id': term_data.get('ascl_id', '')
            }
            
            # Extract bibcodes from all relevant fields
            bibcode_fields = ['positive_bibcodes', 'negative_bibcodes', 'used_in', 'described_in', 'cited_in']
            
            for field in bibcode_fields:
                entries = term_data.get(field, [])
                if entries and entries != False:  # Skip false entries
                    if isinstance(entries, list):
                        for entry in entries:
                            bibcode = self.extract_bibcode_from_url(entry)
                            if bibcode:
                                term_bibcodes[term_id].add(bibcode)
                    elif isinstance(entries, str):
                        bibcode = self.extract_bibcode_from_url(entries)
                        if bibcode:
                            term_bibcodes[term_id].add(bibcode)
        
        logger.info(f"Parsed {len(terms_info)} terms with {sum(len(v) for v in term_bibcodes.values())} total bibcode associations")
        return terms_info, dict(term_bibcodes)
    
    def get_unique_bibcodes(self, term_bibcodes: Dict[str, Set[str]]) -> Set[str]:
        """Get deduplicated set of all bibcodes."""
        all_bibcodes = set()
        for bibcodes in term_bibcodes.values():
            all_bibcodes.update(bibcodes)
        logger.info(f"Found {len(all_bibcodes)} unique bibcodes")
        return all_bibcodes
    
    def resolve_bibcodes_bulk(self, bibcodes: Set[str]) -> pd.DataFrame:
        """
        Bulk resolve bibcodes to file locations using single JOIN operation.
        
        Returns:
            DataFrame with columns: bibcode, filename, line_number, byte_offset, year
        """
        logger.info(f"Resolving {len(bibcodes)} bibcodes via bulk database join")
        
        # Create connection
        conn = sqlite3.connect(self.bibcode_db_path)
        
        try:
            # Create temporary table for bibcodes
            conn.execute("CREATE TEMP TABLE temp_bibcodes (bibcode TEXT PRIMARY KEY)")
            
            # Insert bibcodes in batches
            batch_size = 10000
            bibcode_list = list(bibcodes)
            for i in range(0, len(bibcode_list), batch_size):
                batch = bibcode_list[i:i + batch_size]
                conn.executemany("INSERT INTO temp_bibcodes VALUES (?)", [(b,) for b in batch])
            
            # Perform bulk join
            query = """
            SELECT t.bibcode, l.filename, l.line_number, l.byte_offset, l.year
            FROM temp_bibcodes t
            JOIN bibcode_lookup l ON t.bibcode = l.bibcode
            """
            
            result_df = pd.read_sql_query(query, conn)
            logger.info(f"Successfully resolved {len(result_df)} out of {len(bibcodes)} bibcodes")
            
            return result_df
            
        finally:
            conn.close()
    
    def create_work_assignments(self, 
                               resolved_bibcodes: pd.DataFrame,
                               term_bibcodes: Dict[str, Set[str]]) -> Dict[str, List[Dict]]:
        """
        Group resolved bibcodes by filename and create work assignments.
        
        Returns:
            {filename: [{'bibcode': str, 'byte_offset': int, 'line_number': int, 
                        'terms': [term_ids that mention this bibcode]}]}
        """
        logger.info("Creating work assignments grouped by file")
        
        # Create reverse mapping: bibcode -> list of term_ids
        bibcode_to_terms = defaultdict(list)
        for term_id, bibcodes in term_bibcodes.items():
            for bibcode in bibcodes:
                bibcode_to_terms[bibcode].append(term_id)
        
        # Group by filename and add term information
        file_assignments = defaultdict(list)
        
        for _, row in resolved_bibcodes.iterrows():
            bibcode = row['bibcode']
            filename = row['filename']
            
            assignment = {
                'bibcode': bibcode,
                'byte_offset': row['byte_offset'],
                'line_number': row['line_number'],
                'year': row['year'],
                'terms': bibcode_to_terms.get(bibcode, [])
            }
            
            file_assignments[filename].append(assignment)
        
        # Sort by byte offset for sequential processing
        for filename in file_assignments:
            file_assignments[filename].sort(key=lambda x: x['byte_offset'])
        
        logger.info(f"Created assignments for {len(file_assignments)} files")
        return dict(file_assignments)
    
    def save_preprocessing_results(self, 
                                 terms_info: Dict,
                                 term_bibcodes: Dict,
                                 file_assignments: Dict,
                                 output_dir: Path):
        """Save preprocessing results for the extraction phase."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save terms info
        with open(output_dir / 'terms_info.json', 'w') as f:
            json.dump(terms_info, f, indent=2)
        
        # Save term-bibcode mappings
        # Convert sets to lists for JSON serialization
        term_bibcodes_serializable = {k: list(v) for k, v in term_bibcodes.items()}
        with open(output_dir / 'term_bibcodes.json', 'w') as f:
            json.dump(term_bibcodes_serializable, f, indent=2)
        
        # Save file assignments
        with open(output_dir / 'file_assignments.json', 'w') as f:
            json.dump(file_assignments, f, indent=2)
        
        logger.info(f"Saved preprocessing results to {output_dir}")


def main():
    """Main preprocessing function."""
    # Paths
    ontology_path = Path("ontologies/ASCL/ascl.json")
    bibcode_db_path = Path("bibcode_lookup.db")
    output_dir = Path("optimized_extractor/preprocessed")
    
    # Initialize preprocessor
    preprocessor = BiblioPreprocessor(ontology_path, bibcode_db_path)
    
    # Run preprocessing pipeline
    logger.info("Starting preprocessing pipeline")
    
    # Parse ontology
    terms_info, term_bibcodes = preprocessor.parse_ontology()
    
    # Get unique bibcodes
    unique_bibcodes = preprocessor.get_unique_bibcodes(term_bibcodes)
    
    # Bulk resolve bibcodes
    resolved_bibcodes = preprocessor.resolve_bibcodes_bulk(unique_bibcodes)
    
    # Create work assignments
    file_assignments = preprocessor.create_work_assignments(resolved_bibcodes, term_bibcodes)
    
    # Save results
    preprocessor.save_preprocessing_results(terms_info, term_bibcodes, file_assignments, output_dir)
    
    logger.info("Preprocessing completed successfully")
    
    # Print summary statistics
    total_assignments = sum(len(assignments) for assignments in file_assignments.values())
    logger.info(f"Summary: {len(terms_info)} terms, {len(unique_bibcodes)} unique bibcodes, "
               f"{len(resolved_bibcodes)} resolved, {len(file_assignments)} files, "
               f"{total_assignments} total assignments")


if __name__ == "__main__":
    main()
