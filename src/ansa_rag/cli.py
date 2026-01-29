#!/usr/bin/env python3
"""
CLI entry points for ANSA RAG package.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_app():
    """Run the Streamlit app."""
    # Find the app.py file relative to this package
    package_dir = Path(__file__).parent
    app_path = package_dir.parent / "app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)
    
    # Run streamlit with the app file
    subprocess.run(["streamlit", "run", str(app_path)])


def run_db_insert():
    """Run the database insertion utility."""
    from ansa_rag.database.db_insert import main
    main()


def run_query():
    """Run the database query utility."""
    from ansa_rag.database.query_db import main
    main()


if __name__ == "__main__":
    # For testing purposes
    if len(sys.argv) < 2:
        print("Usage: python cli.py [app|db_insert|query]")
        sys.exit(1)
    
    command = sys.argv[1]
    if command == "app":
        run_app()
    elif command == "db_insert":
        run_db_insert()
    elif command == "query":
        run_query()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)