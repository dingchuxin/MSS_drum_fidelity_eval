"""Shared helpers for Phase 4 (two added analyses, values outside the pipeline, reproducibility).

CPU only, cached stems only. Onsets: the Phase 2 detector (timing.detect) and greedy
closest-first matching at +/-50 ms, reused from the Phase 2 caches (definition unchanged).
Stroke measures and descriptors: the v3 definitions (Phase 3). New caches carry a definition
tag: data/phase4/<TAG>/...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase3"))
from _p3 import *  # noqa: F401,F403,E402
import _p3 as P  # noqa: E402

ROOT = P.ROOT
SR = P.SR
SEED = P.SEED
tracks = P.tracks
safe_name = P.safe_name
mono = P.mono
ref_stem = P.ref_stem
holm = P.holm
rank_biserial = P.rank_biserial
wilcoxon_p = P.wilcoxon_p
P2MEAS = P.P2MEAS
P2DATA = P.P2DATA

TAG = "p2det"              # onset envelopes and picks of the Phase 2 detector (timing.detect)
DATA4 = ROOT / "data" / "phase4" / TAG
OUT4 = ROOT / "results" / "phase4"
for _d in (DATA4, OUT4):
    _d.mkdir(parents=True, exist_ok=True)

MODELS4 = ["v1_spleeter", "v1_htdemucs", "v1_bs_roformer", "v1_scnet"]
LABELS4 = MODELS4 + ["htdemucs_b1"]
FLOORS4 = ["leak_20dB", "white_20dB"]          # generic floors at 20 dB, as in Phase 2
NICE4 = {**P.NICE, "floor_leak_20dB": "leakage floor 20 dB", "floor_white_20dB": "white-noise floor 20 dB",
         "leak_20dB": "leakage floor 20 dB", "white_20dB": "white-noise floor 20 dB"}
NB = 10_000


def p2_onsets(label: str, track: str) -> tuple[np.ndarray, np.ndarray]:
    """Phase 2 reference and estimate onsets (s) of a model label or a generic floor."""
    import pickle
    if label.startswith("floor_") or label in FLOORS4:
        c = label.replace("floor_", "")
        fl = pickle.load(open(P2DATA / "floors" / f"{safe_name(track)}.pkl", "rb"))
        return np.asarray(fl["ref"]), np.asarray(fl[c])
    r = pickle.load(open(P2MEAS / label / f"{safe_name(track)}.pkl", "rb"))
    return np.asarray(r["ref_onsets"]), np.asarray(r["est_onsets"])


def boot_median_ci(v, rng, n=NB):
    """Median and 95 % percentile bootstrap CI of the median (resampling tracks)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(v), size=(n, len(v)))
    lo, hi = np.percentile(np.median(v[idx], axis=1), [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi)


def versions() -> dict:
    import importlib
    import subprocess
    out = {"python": sys.version.split()[0]}
    for m in ["numpy", "scipy", "pandas", "librosa", "numba", "soundfile", "matplotlib", "mir_eval"]:
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = "?"
    out["repo_commit_at_run"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                               capture_output=True, text=True).stdout.strip()
    return out
