"""Detector floor with the Phase 2 detector: the Phase 1 J1 perturbations (white noise at
40/30/20 dB SNR; sum of the other three true stems at 30/20 dB), same seeds
([20260925, track index, condition index]), onsets from timing.detect.

    python floors.py [--jobs 3]  -> data/phase2/floors/<track>.pkl {ref, <cond>: onsets}
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import _p2 as P  # noqa: E402
import measure as M  # noqa: E402
from mss_eval import timing as T  # noqa: E402

sys.path.insert(0, str(P.ROOT / "scripts" / "phase1"))
from j1_timing import PERTURB, _perturb  # noqa: E402


def one(ti_t):
    ti, t = ti_t
    f = P.P2DATA / "floors" / f"{P.safe_name(t)}.pkl"
    if f.exists():
        return t
    ref = P.mono(P.ref_stem(t))
    others = sum(P.mono(P.ref_stem(t, s)) for s in P.SOURCES if s != "drums")
    out = {"ref": M.ref_info(t, False)["onsets"]}
    for ci, (kind, db) in enumerate(PERTURB):
        rng = np.random.default_rng([P.SEED, ti, ci])
        out[f"{kind}_{db}dB"] = T.detect(_perturb(ref, others, kind, db, rng), P.SR)
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    from multiprocessing import Pool
    with Pool(a.jobs) as p:
        for i, t in enumerate(p.imap_unordered(one, list(enumerate(P.tracks()))), 1):
            print(f"[{i}/50] {t}", flush=True)
