# streamlit_dashboard/core_pipeline/utils.py
"""Utility functions for the core pipeline."""

import json
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Generator


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> Generator[Dict[str, Any], None, None]:
    """Load JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def count_lines(path: Path) -> int:
    """Count lines in a file."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Get SQLite database connection."""
    return sqlite3.connect(db_path)


def dataframe_to_db(df: pd.DataFrame, table_name: str, db_path: Path) -> None:
    """Save DataFrame to SQLite database."""
    conn = get_db_connection(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()


def db_to_dataframe(query: str, db_path: Path) -> pd.DataFrame:
    """Load DataFrame from SQLite database."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def extract_context_window(text: str, match_start: int, match_end: int, window_size: int = 100) -> str:
    """Extract context window around a match."""
    start = max(0, match_start - window_size)
    end = min(len(text), match_end + window_size)
    return text[start:end]


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    import re
    return re.sub(r'\s+', ' ', text.lower().strip())


def highlight_match(text: str, match: str) -> str:
    """Highlight match in text for display."""
    import re
    pattern = re.compile(re.escape(match), re.IGNORECASE)
    return pattern.sub(f"**{match}**", text)
