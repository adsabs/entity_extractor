# optimized_extractor/output_formatter.py
"""Output formatting utilities for different export formats."""

import pandas as pd
from pathlib import Path
import logging
from typing import Optional
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutputFormatter:
    """Handle different output format conversions and exports."""
    
    def __init__(self, parquet_path: Path):
        self.parquet_path = parquet_path
        self.df = None
        
    def load_data(self):
        """Load the Parquet data."""
        logger.info(f"Loading data from {self.parquet_path}")
        self.df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(self.df)} records")
        
    def export_single_csv(self, output_path: Path, compression: Optional[str] = 'gzip'):
        """Export all data to a single CSV file."""
        if self.df is None:
            self.load_data()
            
        logger.info(f"Exporting single CSV to {output_path}")
        
        if compression:
            output_path = output_path.with_suffix(f'.csv.{compression}')
            self.df.to_csv(output_path, index=False, compression=compression)
        else:
            self.df.to_csv(output_path, index=False)
            
        logger.info(f"Exported {len(self.df)} records to {output_path}")
        return output_path
    
    def export_csv_by_term(self, output_dir: Path, compression: Optional[str] = 'gzip'):
        """Export separate CSV files for each term."""
        if self.df is None:
            self.load_data()
            
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting CSVs by term to {output_dir}")
        
        unique_terms = self.df['term_id'].unique()
        exported_files = []
        
        for term_id in unique_terms:
            term_data = self.df[self.df['term_id'] == term_id]
            
            # Create safe filename
            safe_term_id = str(term_id).replace('/', '_').replace('\\', '_')
            filename = f"term_{safe_term_id}.csv"
            
            if compression:
                filename += f".{compression}"
                
            file_path = output_dir / filename
            
            if compression:
                term_data.to_csv(file_path, index=False, compression=compression)
            else:
                term_data.to_csv(file_path, index=False)
                
            exported_files.append(file_path)
        
        logger.info(f"Exported {len(exported_files)} term-specific CSV files")
        return exported_files
    
    def export_summary_stats(self, output_path: Path):
        """Export summary statistics."""
        if self.df is None:
            self.load_data()
            
        logger.info(f"Generating summary statistics")
        
        # Calculate statistics
        stats = {
            'total_mentions': len(self.df),
            'unique_terms': self.df['term_id'].nunique(),
            'unique_bibcodes': self.df['bibcode'].nunique(),
            'avg_matches_per_bibcode': self.df.groupby('bibcode')['match_count'].sum().mean(),
            'mentions_by_location': self.df['match_location'].value_counts().to_dict(),
            'top_terms_by_mentions': self.df.groupby(['term_id', 'term_name']).size().nlargest(20).reset_index().to_dict('records'),
            'mentions_with_title_presence': self.df['in_title'].sum(),
            'mentions_with_abstract_presence': self.df['in_abstract'].sum()
        }
        
        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        logger.info(f"Summary statistics saved to {output_path}")
        return stats
    
    def get_data_info(self) -> dict:
        """Get basic information about the dataset."""
        if self.df is None:
            self.load_data()
            
        return {
            'total_records': len(self.df),
            'unique_terms': self.df['term_id'].nunique(),
            'unique_bibcodes': self.df['bibcode'].nunique(),
            'columns': list(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'file_size_mb': self.parquet_path.stat().st_size / 1024 / 1024
        }


def export_all_formats(parquet_path: Path, 
                      output_dir: Path,
                      export_single_csv: bool = True,
                      export_term_csvs: bool = False,
                      compression: str = 'gzip'):
    """Export data in all requested formats."""
    formatter = OutputFormatter(parquet_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_paths = {}
    
    # Export single CSV if requested
    if export_single_csv:
        csv_path = output_dir / 'software_mentions_all'
        exported_paths['single_csv'] = formatter.export_single_csv(csv_path, compression)
    
    # Export term-specific CSVs if requested
    if export_term_csvs:
        term_csv_dir = output_dir / 'csvs_by_term'
        exported_paths['term_csvs'] = formatter.export_csv_by_term(term_csv_dir, compression)
    
    # Always export summary statistics
    stats_path = output_dir / 'summary_statistics.json'
    exported_paths['summary'] = formatter.export_summary_stats(stats_path)
    
    # Print dataset info
    info = formatter.get_data_info()
    logger.info(f"Dataset info: {info}")
    
    return exported_paths, info


def main():
    """Main export function for testing."""
    parquet_path = Path("optimized_extractor/results/software_mentions_extracted.parquet")
    output_dir = Path("optimized_extractor/exports")
    
    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        return
    
    # Export all formats
    exported_paths, info = export_all_formats(
        parquet_path=parquet_path,
        output_dir=output_dir,
        export_single_csv=True,
        export_term_csvs=True,
        compression='gzip'
    )
    
    logger.info(f"Export completed. Files: {exported_paths}")


if __name__ == "__main__":
    main()
