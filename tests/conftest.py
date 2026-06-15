"""Pytest configuration — put the repo root on sys.path.

Lets tests import top-level modules (`validation`, `api_server`) and the
`raganything` package regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
