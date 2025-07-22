# Optimized Software Mention Extraction Pipeline

High-performance pipeline for extracting software mention contexts from 25M+ scientific literature records in under 24 hours.

## Overview

This pipeline processes 3,685 software terms from the ASCL ontology against a large corpus of scientific literature, extracting 100-word context windows, metadata, and occurrence statistics for each match.

### Key Performance Features

- **Sequential I/O**: Eliminates random disk seeks through bulk database operations
- **Parallel Processing**: Multi-core extraction with configurable worker pools
- **Memory Efficient**: Streaming processing with minimal memory footprint
- **Compressed Output**: Parquet format with optional CSV exports
- **Sub-24hr Execution**: Optimized for completion within a workday

## Architecture

```
Input: ASCL Ontology (3,685 terms) + Bibcode Database (25M+ records) + Literature Corpus
   ↓
Phase 1: Preprocessing (30-60 min)
   ├── Parse ontology and extract bibcodes
   ├── Bulk database join for file locations
   └── Group by file, sort by byte offset
   ↓
Phase 2: Parallel Extraction (6-12 hours)
   ├── Worker pool processes files sequentially
   ├── Regex matching with pre-compiled patterns
   ├── Context extraction (100-word windows)
   └── Metadata collection (title, abstract, counts)
   ↓
Phase 3: Output Generation (30 min)
   ├── Aggregate results to Parquet
   └── Optional CSV exports (single or by-term)
```

## Installation

```bash
# Install dependencies
pip install -r optimized_extractor/requirements.txt

# Ensure your corpus data and bibcode_lookup.db are accessible
ls bibcode_lookup.db ontologies/ASCL/ascl.json data/
```

## Usage

### Basic Execution

```bash
# Run complete pipeline
python optimized_extractor/run_extraction.py

# With custom paths
python optimized_extractor/run_extraction.py \
    --ontology-path ontologies/ASCL/ascl.json \
    --bibcode-db bibcode_lookup.db \
    --corpus-path data \
    --output-dir results
```

### Advanced Options

```bash
# Use specific number of workers
python optimized_extractor/run_extraction.py --num-workers 16

# Export term-specific CSV files
python optimized_extractor/run_extraction.py --export-term-csvs

# Skip preprocessing if already done
python optimized_extractor/run_extraction.py --skip-preprocessing

# Preprocessing only (for testing)
python optimized_extractor/run_extraction.py --preprocessing-only

# Dry run validation
python optimized_extractor/run_extraction.py --dry-run
```

### Output Format Options

```bash
# Single CSV with gzip compression (default)
python optimized_extractor/run_extraction.py --export-single-csv --compression gzip

# Separate CSV per term
python optimized_extractor/run_extraction.py --export-term-csvs

# No CSV compression
python optimized_extractor/run_extraction.py --compression None
```

## Output Structure

### Primary Output: Parquet File
- **Location**: `results/software_mentions_extracted.parquet`
- **Format**: Compressed columnar format
- **Size**: ~5-10GB (vs 50-100GB raw CSV)

### Schema
| Column | Type | Description |
|--------|------|-------------|
| term_id | string | Software term identifier |
| term_name | string | Actual software name |
| bibcode | string | Publication bibcode |
| title | string | Publication title |
| abstract | string | Publication abstract |
| context | string | 100-word window around match |
| match_count | int | Times term appears in body |
| in_title | bool | Term found in title |
| in_abstract | bool | Term found in abstract |
| match_location | string | Primary location (title/abstract/body) |

### Optional CSV Exports
- **Single CSV**: `results/exports/software_mentions_all.csv.gz`
- **By-term CSVs**: `results/exports/csvs_by_term/term_{id}.csv.gz`
- **Summary Stats**: `results/exports/summary_statistics.json`

## Performance Optimization

### Database Setup
Ensure bibcode_lookup.db has proper indexing:
```sql
CREATE INDEX IF NOT EXISTS idx_bibcode ON bibcode_lookup(bibcode);
```

### System Requirements
- **CPU**: 8+ cores recommended (scales to 32+)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: Fast SSD for corpus files (300-500 MB/s)
- **Disk Space**: 100GB for intermediate files + output

### Expected Performance
- **8 cores**: 16-18 hours
- **16 cores**: 10-12 hours
- **32 cores**: 6-8 hours

## Pipeline Phases

### Phase 1: Preprocessing (30-60 minutes)
1. **Ontology parsing**: Extract all bibcodes from ASCL terms
2. **Bibcode cleaning**: Handle URLs, deduplicate
3. **Bulk database join**: Single operation for all file locations
4. **Work assignment**: Group by file, sort by byte offset

### Phase 2: Extraction (6-12 hours)
1. **Worker initialization**: Pre-compile regex patterns
2. **File processing**: Sequential reads, zero random seeks
3. **Context extraction**: 100-word windows around matches
4. **Result collection**: Stream to Parquet shards

### Phase 3: Output (30 minutes)
1. **Result aggregation**: Merge all worker outputs
2. **Format conversion**: Generate requested export formats
3. **Statistics**: Summary analytics and metadata

## Troubleshooting

### Common Issues

**Missing bibcode_lookup.db index**:
```bash
sqlite3 bibcode_lookup.db "CREATE INDEX IF NOT EXISTS idx_bibcode ON bibcode_lookup(bibcode);"
```

**Memory errors**:
- Reduce number of workers: `--num-workers 4`
- Check available RAM and corpus file sizes

**Slow performance**:
- Verify corpus files on fast storage (not network mount)
- Check CPU utilization and I/O wait times
- Ensure no other heavy processes running

**File not found errors**:
- Verify all input paths exist and are accessible
- Check corpus file permissions
- Use absolute paths if relative paths fail

### Resume from Interruption

The pipeline saves intermediate results, allowing partial recovery:

```bash
# Skip preprocessing if completed
python optimized_extractor/run_extraction.py --skip-preprocessing
```

## Module Reference

- **`preprocessing.py`**: Ontology parsing and bibcode resolution
- **`extraction_engine.py`**: Parallel document processing
- **`output_formatter.py`**: Export format conversion
- **`run_extraction.py`**: Main orchestration script

## Performance Monitoring

Monitor progress through logs:
- Preprocessing: Database join progress
- Extraction: Per-worker file completion
- Output: Result aggregation status

For system monitoring:
```bash
# CPU and memory usage
htop

# I/O monitoring
iotop

# Disk space
df -h
```
