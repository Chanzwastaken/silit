"""Shared pytest configuration for the SILIT test suite.

`app/` and `model/` are not installed packages, so they are added to
sys.path here to let tests import `config`, `predictor` and
`dataset_loader` directly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _path in (ROOT / 'app', ROOT / 'model', ROOT):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
