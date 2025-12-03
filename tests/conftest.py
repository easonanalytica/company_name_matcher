"""Pytest configuration and fixtures."""

import sys
from pathlib import Path


# Add scripts directory to Python path so test_validate_data.py can import from scripts.validate_data
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
