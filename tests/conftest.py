from __future__ import annotations

import sys
from pathlib import Path

# Make `src/agentic_memory/` importable when running tests from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
