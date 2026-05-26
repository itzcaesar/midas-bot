"""Shared pytest configuration: put project root and src/ on sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)
