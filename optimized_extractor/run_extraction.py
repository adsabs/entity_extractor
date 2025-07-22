#!/usr/bin/env python3
# optimized_extractor/run_extraction.py
"""
Main orchestration script for optimized software mention extraction.

This script runs the complete pipeline:
1. Preprocessing: Parse ontology, resolve bibcodes, create work assignments
2. Extraction: Parallel processing of documents to extract contexts
3. Output: Generate Parquet file and optional CSV exports

Usage:
    python run_extraction.py [options]
"""

import argparse
import logging
import time
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from optimized_extractor.preprocessing import BiblioPreprocessor
from optimized_extractor.extraction_engine import ExtractionEngine
from optimized_extractor.output_formatter import export_all_formats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run optimized software mention extraction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input paths
    parser.add_argument(
        '--ontology-path',
        type=Path,
        default=Path('ontologies/ASCL/ascl.json'),
        help='Path to ASCL ontology JSON file'
    )
    
    parser.add_argument(
        '--bibcode-db',
        type=Path,
        default=Path('bibcode_lookup.db'),
        help='Path to bibcode lookup database'
    )
    
    parser.add_argument(
        '--corpus-path',
        type=Path,
        default=Path('/home/scixmuse/scix_data/ads_metadata_by_year_full/'),
        help='Base path to corpus JSONL files'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('optimized_extractor/results'),
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--export-single-csv',
        action='store_true',
        default=True,
        help='Export single compressed CSV file'
    )
    
    parser.add_argument(
        '--export-term-csvs',
        action='store_true',
        default=False,
        help='Export separate CSV files for each term'
    )
    
    parser.add_argument(
        '--compression',
        choices=['gzip', 'bz2', 'xz', None],
        default='gzip',
        help='Compression for CSV exports'
    )
    
    # Processing configuration
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing if already done'
    )
    
    parser.add_argument(
        '--preprocessing-only',
        action='store_true',
        help='Run only preprocessing step'
    )
    
    # Debugging
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run with validation only'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def validate_inputs(args):
    """Validate input paths and configuration."""
    logger.info("Validating inputs...")
    
    # Check required files exist
    required_files = [args.ontology_path, args.bibcode_db]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Check corpus directory
    if not args.corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {args.corpus_path}")
    
    logger.info("Input validation passed")

def run_preprocessing(args):
    """Run the preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1: PREPROCESSING")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Initialize preprocessor
    preprocessor = BiblioPreprocessor(args.ontology_path, args.bibcode_db)
    
    # Create output directory
    preprocessed_dir = args.output_dir / 'preprocessed'
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # Run preprocessing steps
    logger.info("Step 1/4: Parsing ontology")
    terms_info, term_bibcodes = preprocessor.parse_ontology()
    
    logger.info("Step 2/4: Getting unique bibcodes")
    unique_bibcodes = preprocessor.get_unique_bibcodes(term_bibcodes)
    
    logger.info("Step 3/4: Resolving bibcodes via bulk database join")
    resolved_bibcodes = preprocessor.resolve_bibcodes_bulk(unique_bibcodes)
    
    logger.info("Step 4/4: Creating work assignments")
    file_assignments = preprocessor.create_work_assignments(resolved_bibcodes, term_bibcodes)
    
    # Save results
    preprocessor.save_preprocessing_results(
        terms_info, term_bibcodes, file_assignments, preprocessed_dir
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed:.1f} seconds")
    
    # Print summary
    total_assignments = sum(len(assignments) for assignments in file_assignments.values())
    logger.info(f"Preprocessing summary:")
    logger.info(f"  Terms: {len(terms_info)}")
    logger.info(f"  Unique bibcodes: {len(unique_bibcodes)}")
    logger.info(f"  Resolved bibcodes: {len(resolved_bibcodes)}")
    logger.info(f"  Files to process: {len(file_assignments)}")
    logger.info(f"  Total assignments: {total_assignments}")
    
    return preprocessed_dir

def run_extraction(args, preprocessed_dir):
    """Run the extraction pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 2: EXTRACTION")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Initialize extraction engine
    engine = ExtractionEngine(args.corpus_path, args.num_workers)
    
    # Run extraction
    result_path = engine.run_extraction(preprocessed_dir, args.output_dir)
    
    elapsed = time.time() - start_time
    logger.info(f"Extraction completed in {elapsed:.1f} seconds")
    
    return result_path

def run_output_formatting(args, parquet_path):
    """Run output formatting and export."""
    logger.info("=" * 60)
    logger.info("PHASE 3: OUTPUT FORMATTING")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Create export directory
    export_dir = args.output_dir / 'exports'
    
    # Export formats
    exported_paths, info = export_all_formats(
        parquet_path=parquet_path,
        output_dir=export_dir,
        export_single_csv=args.export_single_csv,
        export_term_csvs=args.export_term_csvs,
        compression=args.compression
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Output formatting completed in {elapsed:.1f} seconds")
    
    return exported_paths, info

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return
        
        total_start_time = time.time()
        
        # Phase 1: Preprocessing
        if not args.skip_preprocessing:
            preprocessed_dir = run_preprocessing(args)
        else:
            preprocessed_dir = args.output_dir / 'preprocessed'
            if not preprocessed_dir.exists():
                raise FileNotFoundError("Preprocessing directory not found. Run without --skip-preprocessing first.")
            logger.info(f"Using existing preprocessing results from {preprocessed_dir}")
        
        if args.preprocessing_only:
            logger.info("Preprocessing-only mode. Stopping here.")
            return
        
        # Phase 2: Extraction
        parquet_path = run_extraction(args, preprocessed_dir)
        
        # Phase 3: Output formatting
        exported_paths, info = run_output_formatting(args, parquet_path)
        
        # Final summary
        total_elapsed = time.time() - total_start_time
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {total_elapsed:.1f} seconds ({total_elapsed/3600:.2f} hours)")
        logger.info(f"Dataset info: {info}")
        logger.info(f"Main result file: {parquet_path}")
        
        if exported_paths.get('single_csv'):
            logger.info(f"Single CSV export: {exported_paths['single_csv']}")
        if exported_paths.get('term_csvs'):
            logger.info(f"Term-specific CSVs: {len(exported_paths['term_csvs'])} files in exports/csvs_by_term/")
        
        logger.info("All results saved in: " + str(args.output_dir))
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
