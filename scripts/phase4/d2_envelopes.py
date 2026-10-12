"""D2 inputs: the Phase 2 detector's raw onset-strength envelope (timing.onset_envelope, hop 64)
of every signal, cached, so that peak picking can be repeated at several thresholds.

Signals per track: the true drum stem (as Phase 2: full length, stereo -> mono mean); the four
default runs and htdemucs_b1 (as Phase 2: trimmed to the reference length); the generic floors
leak_20dB and white_20dB, regenerated with the Phase 2 seeds ([20260925, track index,
condition index], scripts/phase2/floors.py).

Check: timing.detect_delta(delta=0.07) on the cached envelope must reproduce the stored Phase 2
onsets exactly (reference, estimates and floors); the result is stored with the cache.

    python d2_envelopes.py [--jobs 3]  -> data/phase4/p2det/env/<track>.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import _p4 as P  # noqa: E402
from mss_eval import timing as T  # noqa: E402

sys.path.insert(0, str(P.ROOT / "scripts" / "phase1"))
sys.path.insert(0, str(P.ROOT / "scripts" / "phase2"))
from j1_timing import PERTURB, _perturb  # noqa: E402

OUT = P.DATA4 / "env"


def one(ti_t):
    ti, t = ti_t
    f = OUT / f"{P.safe_name(t)}.pkl"
    if f.exists():
        return t, None
    import measure as M2
    ref2 = P.ref_stem(t)
    res = {"track": t, "env": {}, "reproduces_p2": {}}
    sigs = {"ref": ref2}
    for lab in P.LABELS4:
        est = P.load(M2.stem_dir(lab, t) / "drums.wav")
        n = min(len(est), len(ref2))
        sigs[lab] = est[:n]
    ref = P.mono(ref2)
    others = sum(P.mono(P.ref_stem(t, s)) for s in P.SOURCES if s != "drums")
    for ci, (kind, db) in enumerate(PERTURB):
        c = f"{kind}_{db}dB"
        if c in P.FLOORS4:
            rng = np.random.default_rng([P.SEED, ti, ci])
            sigs[f"floor_{c}"] = _perturb(ref, others, kind, db, rng)
    for name, y in sigs.items():
        env = T.onset_envelope(y, P.SR, T.HOP_FINE)
        res["env"][name] = env
        on = T.detect_delta(y, P.SR, 0.07, env=env)
        if name == "ref":
            stored = P.p2_onsets(P.LABELS4[0], t)[0]
        else:
            stored = P.p2_onsets(name, t)[1]
        res["reproduces_p2"][name] = bool(len(on) == len(stored) and np.array_equal(on, stored))
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    return t, res["reproduces_p2"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    from multiprocessing import Pool
    with Pool(a.jobs, maxtasksperchild=4) as p:
        for i, (t, ok) in enumerate(p.imap_unordered(one, list(enumerate(P.tracks()))), 1):
            print(f"[{i}/50] {t} {ok}", flush=True)
