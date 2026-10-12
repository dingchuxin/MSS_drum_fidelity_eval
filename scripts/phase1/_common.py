"""Shared paths/helpers for the Phase 1 analyses (cached stems only; no separation).

Data lives outside the repo checkout on some machines, so the data root is configurable:
    MSS_DATA_ROOT=<dir containing musdb18hq/ and stems_out/>   (default: <repo>/data)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(os.environ.get("MSS_DATA_ROOT", ROOT / "data"))
MUSDB_TEST = DATA / "musdb18hq" / "test"
STEMS = DATA / "stems_out"
OUT = ROOT / "results" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

SR = 44100
SEED = 20260925
MODELS = ["spleeter", "htdemucs", "bs_roformer", "scnet"]
LABEL = {"spleeter": "Spleeter", "htdemucs": "HT-Demucs",
         "bs_roformer": "BS-Roformer", "scnet": "SCNet-XL"}
SOURCES = ["drums", "bass", "other", "vocals"]


def safe_name(s: str) -> str:
    return "".join("_" if c in '<>:"/\|?*' else c for c in s).strip()


def tracks() -> list[str]:
    return sorted(p.name for p in MUSDB_TEST.iterdir() if p.is_dir())


def load(path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    assert sr == SR, (path, sr)
    return x


def ref_stem(track: str, stem: str = "drums") -> np.ndarray:
    return load(MUSDB_TEST / track / f"{stem}.wav")


def est_stem(model: str, track: str, stem: str = "drums") -> np.ndarray:
    return load(STEMS / model / safe_name(track) / f"{stem}.wav")


def mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def versions() -> dict[str, str]:
    import importlib
    out = {"python": sys.version.split()[0]}
    for m in ["numpy", "scipy", "pandas", "librosa", "mir_eval", "museval", "soundfile",
              "matplotlib"]:
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:  # pragma: no cover
            out[m] = "n/a"
    return out
