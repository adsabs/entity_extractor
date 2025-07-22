# optimized_extractor/__init__.py
"""Optimized software mention extraction pipeline."""

__version__ = "1.0.0"
__author__ = "Entity Extractor Team"
__description__ = "High-performance pipeline for extracting software mentions from scientific literature"

from .preprocessing import BiblioPreprocessor
from .extraction_engine import ExtractionEngine, DocumentProcessor, TermMatcher, ContextExtractor
from .output_formatter import OutputFormatter, export_all_formats

__all__ = [
    "BiblioPreprocessor",
    "ExtractionEngine", 
    "DocumentProcessor",
    "TermMatcher",
    "ContextExtractor",
    "OutputFormatter",
    "export_all_formats"
]
