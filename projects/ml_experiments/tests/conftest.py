from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ML_EXPERIMENTS_SRC = ROOT / "projects" / "ml_experiments" / "src"
SHARED_SRC = ROOT / "shared" / "birdsys_core" / "src"

for path in (str(ML_EXPERIMENTS_SRC), str(SHARED_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)
