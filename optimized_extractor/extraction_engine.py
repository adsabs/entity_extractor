# optimized_extractor/extraction_engine.py
"""Parallel extraction engine for software mention contexts."""

import json
import re
import orjson
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Single extraction result."""
    term_id: str
    term_name: str
    bibcode: str
    title: str
    abstract: str
    context: str
    match_count: int
    in_title: bool
    in_abstract: bool
    match_location: str

class TermMatcher:
    """Efficient term matching with pre-compiled regex patterns."""
    
    def __init__(self, terms_info: Dict[str, Dict]):
        self.terms_info = terms_info
        self.regex_cache = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for all terms."""
        logger.info(f"Compiling regex patterns for {len(self.terms_info)} terms")
        
        for term_id, term_data in self.terms_info.items():
            # Get the term name - try 'name' first, then 'title'
            term_name = term_data.get('name') or term_data.get('title', '')
            if term_name:
                # Extract just the main software name (before colon if present)
                simple_name = term_name.split(':')[0].strip() if ':' in term_name else term_name
                if simple_name:
                    # Create case-insensitive word boundary pattern
                    escaped_term = re.escape(simple_name.lower())
                    pattern = rf'\b{escaped_term}\b'
                    self.regex_cache[term_id] = re.compile(pattern, re.IGNORECASE)
    
    def find_matches(self, text: str, term_ids: List[str]) -> Dict[str, List[re.Match]]:
        """Find all matches for given terms in text."""
        matches = {}
        for term_id in term_ids:
            if term_id in self.regex_cache:
                matches[term_id] = list(self.regex_cache[term_id].finditer(text))
        return matches

class ContextExtractor:
    """Extract context windows around matches."""
    
    @staticmethod
    def extract_context_window(text: str, match_start: int, match_end: int, 
                             window_words: int = 100) -> str:
        """Extract context window of specified word count around match."""
        # Simple word tokenization
        words = text.split()
        
        # Find word positions
        char_to_word = {}
        char_pos = 0
        for word_idx, word in enumerate(words):
            for i in range(len(word)):
                char_to_word[char_pos + i] = word_idx
            char_pos += len(word) + 1  # +1 for space
        
        # Get word indices for match
        start_word_idx = char_to_word.get(match_start, 0)
        end_word_idx = char_to_word.get(match_end - 1, len(words) - 1)
        
        # Extract window
        context_start = max(0, start_word_idx - window_words)
        context_end = min(len(words), end_word_idx + window_words + 1)
        
        return ' '.join(words[context_start:context_end])

class DocumentProcessor:
    """Process individual documents for extraction."""
    
    def __init__(self, terms_info: Dict[str, Dict]):
        self.matcher = TermMatcher(terms_info)
        self.extractor = ContextExtractor()
        self.terms_info = terms_info
    
    def process_document(self, doc_data: Dict[str, Any], 
                        relevant_terms: List[str]) -> List[ExtractionResult]:
        """Process a single document and extract all relevant matches."""
        results = []
        
        # Extract text fields and handle both strings and lists
        def extract_text_field(field_value):
            """Extract text from field that could be string, list, or None."""
            if field_value is None:
                return ''
            if isinstance(field_value, str):
                return field_value
            if isinstance(field_value, list):
                # Join list elements into single string
                return ' '.join(str(item) for item in field_value if item)
            return str(field_value)
        
        title = extract_text_field(doc_data.get('title'))
        abstract = extract_text_field(doc_data.get('abstract'))
        body = extract_text_field(doc_data.get('body'))
        bibcode = doc_data.get('bibcode', '')
        
        # Find matches in each field
        title_matches = self.matcher.find_matches(title, relevant_terms)
        abstract_matches = self.matcher.find_matches(abstract, relevant_terms)
        body_matches = self.matcher.find_matches(body, relevant_terms)
        
        # Process each term
        for term_id in relevant_terms:
            if term_id not in self.terms_info:
                continue
                
            term_name = self.terms_info[term_id].get('name') or self.terms_info[term_id].get('title', '')
            
            # Get all matches for this term
            title_term_matches = title_matches.get(term_id, [])
            abstract_term_matches = abstract_matches.get(term_id, [])
            body_term_matches = body_matches.get(term_id, [])
            
            # Create result entry for each individual match
            match_results = []
            
            # Process title matches
            for match in title_term_matches:
                context = self.extractor.extract_context_window(
                    title, match.start(), match.end()
                )
                match_results.append({
                    'location': 'title',
                    'context': context,
                    'match_count': 1,
                    'in_title': True,
                    'in_abstract': bool(abstract_term_matches),
                    'in_body': bool(body_term_matches)
                })
            
            # Process abstract matches
            for match in abstract_term_matches:
                context = self.extractor.extract_context_window(
                    abstract, match.start(), match.end()
                )
                match_results.append({
                    'location': 'abstract',
                    'context': context,
                    'match_count': 1,
                    'in_title': bool(title_term_matches),
                    'in_abstract': True,
                    'in_body': bool(body_term_matches)
                })
            
            # Process body matches
            for match in body_term_matches:
                context = self.extractor.extract_context_window(
                    body, match.start(), match.end()
                )
                match_results.append({
                    'location': 'body',
                    'context': context,
                    'match_count': 1,
                    'in_title': bool(title_term_matches),
                    'in_abstract': bool(abstract_term_matches),
                    'in_body': True
                })
            
            # Create ExtractionResult for each match
            for match_result in match_results:
                result = ExtractionResult(
                    term_id=term_id,
                    term_name=term_name,
                    bibcode=bibcode,
                    title=title,
                    abstract=abstract,
                    context=match_result['context'],
                    match_count=match_result['match_count'],
                    in_title=match_result['in_title'],
                    in_abstract=match_result['in_abstract'],
                    match_location=match_result['location']
                )
                
                results.append(result)
        
        return results

def process_file_worker(args):
    """Worker function for processing a single file."""
    filename, assignments, terms_info, corpus_base_path, worker_id = args
    
    logger.info(f"Worker {worker_id}: Processing {filename} with {len(assignments)} assignments")
    
    processor = DocumentProcessor(terms_info)
    results = []
    
    try:
        file_path = Path(corpus_base_path) / filename
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for assignment in assignments:
                try:
                    # Seek to byte offset
                    f.seek(assignment['byte_offset'])
                    line = f.readline()
                    
                    # Parse JSON
                    doc_data = orjson.loads(line)
                    
                    # Process document
                    doc_results = processor.process_document(doc_data, assignment['terms'])
                    results.extend(doc_results)
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Worker {worker_id}: JSON decode error for {assignment['bibcode']}: {e}")
                except Exception as e:
                    logger.error(f"Worker {worker_id}: Error processing {assignment['bibcode']}: {e}")
    
    except Exception as e:
        logger.error(f"Worker {worker_id}: Error processing file {filename}: {e}")
    
    logger.info(f"Worker {worker_id}: Completed {filename}, extracted {len(results)} results")
    return results

class ExtractionEngine:
    """Main extraction engine coordinating parallel processing."""
    
    def __init__(self, corpus_base_path: Path, num_workers: Optional[int] = None):
        self.corpus_base_path = corpus_base_path
        self.num_workers = num_workers or min(mp.cpu_count(), 32)
        
    def run_extraction(self, 
                      preprocessed_dir: Path,
                      output_dir: Path) -> Path:
        """Run the main extraction pipeline."""
        logger.info(f"Starting extraction with {self.num_workers} workers")
        
        # Load preprocessing results
        logger.info("Loading preprocessing results")
        with open(preprocessed_dir / 'terms_info.json', 'r') as f:
            terms_info = json.load(f)
        
        with open(preprocessed_dir / 'file_assignments.json', 'r') as f:
            file_assignments = json.load(f)
        
        # Prepare worker arguments
        worker_args = []
        for worker_id, (filename, assignments) in enumerate(file_assignments.items()):
            worker_args.append((filename, assignments, terms_info, 
                              str(self.corpus_base_path), worker_id))
        
        # Run parallel processing
        logger.info(f"Starting parallel processing with {len(worker_args)} file groups")
        
        all_results = []
        with mp.Pool(processes=self.num_workers) as pool:
            worker_results = pool.map(process_file_worker, worker_args)
            
            # Flatten results
            for results in worker_results:
                all_results.extend(results)
        
        logger.info(f"Extraction completed. Total results: {len(all_results)}")
        
        # Convert to DataFrame and save
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._save_results(all_results, output_dir)
    
    def _save_results(self, results: List[ExtractionResult], output_dir: Path) -> Path:
        """Save results to Parquet format."""
        logger.info(f"Saving {len(results)} results to Parquet")
        
        # Handle empty results
        if not results:
            logger.warning("No results to save. Creating empty DataFrame with expected schema.")
            # Create empty DataFrame with proper schema
            df = pd.DataFrame(columns=[
                'term_id', 'term_name', 'bibcode', 'title', 'abstract', 
                'context', 'match_count', 'in_title', 'in_abstract', 'match_location'
            ])
            # Set proper dtypes
            df = df.astype({
                'term_id': 'string',
                'term_name': 'string', 
                'bibcode': 'string',
                'title': 'string',
                'abstract': 'string',
                'context': 'string',
                'match_count': 'int64',
                'in_title': 'bool',
                'in_abstract': 'bool',
                'match_location': 'string'
            })
        else:
            # Convert to records
            records = []
            for result in results:
                records.append({
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
            
            # Create DataFrame
            df = pd.DataFrame(records)
        
        # Save as Parquet
        parquet_path = output_dir / 'software_mentions_extracted.parquet'
        df.to_parquet(parquet_path, compression='snappy', index=False)
        
        logger.info(f"Saved results to {parquet_path}")
        logger.info(f"Result summary: {len(df)} total mentions, "
                   f"{df['term_id'].nunique()} unique terms, "
                   f"{df['bibcode'].nunique()} unique bibcodes")
        
        return parquet_path


def main():
    """Main extraction function."""
    # Paths
    preprocessed_dir = Path("optimized_extractor/preprocessed")
    corpus_base_path = Path("data")  # Adjust based on your corpus location
    output_dir = Path("optimized_extractor/results")
    
    # Initialize and run extraction
    engine = ExtractionEngine(corpus_base_path)
    result_path = engine.run_extraction(preprocessed_dir, output_dir)
    
    logger.info(f"Extraction pipeline completed. Results saved to: {result_path}")


if __name__ == "__main__":
    main()
