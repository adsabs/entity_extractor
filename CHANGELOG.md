# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] – 2025-01-29

### Fixed
- **OEG NER incorrectly split multi-char terms** ("Zeus" → "ze", "us"). Implemented span-aware classification that queries the model but only accepts tokens that overlap the literal match boundaries.
- Updated both `score_filtered_contexts.py` and `debug_score_subset.py` to load NER pipelines with `aggregation_strategy="none"` so token-level offsets are preserved.
- **Dashboard import paths** normalized to prevent module resolution errors when running `streamlit run app.py`.
- **Dashboard data file path** fixed to correctly locate the software mentions CSV file.

### Added
- **Span-aware NER function** `run_ner_on_span()` – guarantees that every literal occurrence of a known term is classified as a single entity even if the model fragments it during tokenization.
- Test script `test_ner_fix.py` to validate the span-aware NER functionality.

### Changed
- NER model initialization now uses `token-classification` task with `aggregation_strategy="none"` instead of `"simple"` to preserve character offsets for span-aware processing.
