"""Root conftest: make repo root (main.py) and src/ importable during tests.

Keeps tests runnable both after ``poetry install`` and directly with ``pytest``.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
_SRC = _ROOT / "src"

for path in (_ROOT, _SRC):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)
