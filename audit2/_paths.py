"""Paths for the validation tests in audit2/. The tests (t*.py, l1_checks.py) import the
project's own functions through `project()`.

Data root: MSS_DATA_ROOT (default: data/ in the repository), as in scripts/phase1/_common.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MSS_DATA_ROOT", str(ROOT / "data"))
DATA = Path(os.environ["MSS_DATA_ROOT"])
MUSDB_TEST = DATA / "musdb18hq" / "test"
STEMS_V1 = DATA / "stems_out"
P2DATA = ROOT / "data" / "phase2"
P3DATA = ROOT / "data" / "phase3" / "v3"
P4DATA = ROOT / "data" / "phase4" / "p2det"
RES = ROOT / "results"
OUT = RES / "audit2"
OUT.mkdir(parents=True, exist_ok=True)
SR = 44100
SEED = 20260925


def project():
    """Make the project's src/ and script folders importable."""
    for p in (ROOT / "src", ROOT / "scripts" / "phase1", ROOT / "scripts" / "phase2",
              ROOT / "scripts" / "phase3", ROOT / "scripts" / "phase4", ROOT / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def safe_name(s: str) -> str:
    return "".join("_" if c in '<>:"/\\|?*' else c for c in s).strip()


def tracks() -> list[str]:
    return sorted(p.name for p in MUSDB_TEST.iterdir() if p.is_dir())


def dump(name: str, obj) -> None:
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
