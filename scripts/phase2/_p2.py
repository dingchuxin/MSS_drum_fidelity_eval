"""Shared helpers for the Phase 2 analyses (see scripts/phase1/_common.py for paths)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from _common import *  # noqa: F401,F403,E402
import _common as C  # noqa: E402

OUT2 = C.ROOT / "results" / "phase2"
OUT2.mkdir(parents=True, exist_ok=True)
# new separations (float WAV) and caches live under the gitignored data/ of this checkout
P2DATA = Path(os.environ.get("MSS_P2_DATA", C.ROOT / "data" / "phase2"))
MSST_DIR = Path(os.environ.get("MSST_DIR", C.DATA.parent / "external" / "MSST"))
DESCS = ["level_db", "variation", "crest", "impulsiveness", "env_skew", "env_flux", "attack_slope"]


def run_metadata(extra: dict | None = None) -> dict:
    """Package versions, MSST commit and GPU driver, captured at run time."""
    import importlib
    meta = {"python": sys.version.split()[0], "executable": sys.executable, "seed_base": C.SEED}
    for m in ["numpy", "scipy", "librosa", "soundfile", "torch", "museval", "mir_eval",
              "audio_separator", "demucs", "tensorflow", "spleeter"]:
        try:
            mod = importlib.import_module(m)
            meta[m] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    try:
        import importlib.metadata as im
        meta["audio_separator"] = im.version("audio-separator")
    except Exception:
        pass
    try:
        meta["msst_commit"] = subprocess.run(["git", "-C", str(MSST_DIR), "rev-parse", "HEAD"],
                                             capture_output=True, text=True).stdout.strip()
    except Exception:
        meta["msst_commit"] = "?"
    try:
        meta["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                      "--format=csv,noheader"], capture_output=True,
                                     text=True).stdout.strip()
    except Exception:
        meta["gpu"] = "?"
    try:
        import torch
        meta["cuda"] = torch.version.cuda
        meta["cudnn"] = torch.backends.cudnn.version()
        meta["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:
        pass
    meta["repo_commit"] = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip()
    if extra:
        meta.update(extra)
    return meta


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
