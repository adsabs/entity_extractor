#!/usr/bin/env python3
"""Launch the Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Launch the Streamlit app
    print("🚀 Starting Entity Extractor Dashboard...")
    print("Open your browser and navigate to: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n👋 Shutting down dashboard...")
