"""C2 detections: the two alternative detectors on every signal used for the Phase 2 timing
headlines (reference; v1 Spleeter, HT-Demucs, BS-Roformer, SCNet-XL; B1 HT-Demucs) and on the
five generic floors (white noise 40/30/20 dB, other-stems leakage 30/20 dB; Phase 2 seeds).

  rise    timing.detect_refined(mode='rise', min_sep_s=0.03)
  energy  timing.detect_energy with the Phase 2 events (from the Phase 2 caches: same
          detector, same signals, so the events are identical)
  p2      the Phase 2 onsets themselves (from the caches), for the matching-rule variant

    python c2_detect.py [--jobs 2]   -> data/phase3/v3/c2/<track>.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import _p3 as P  # noqa: E402

sys.path.insert(0, str(P.ROOT / "scripts" / "phase1"))
from j1_timing import PERTURB, _perturb  # noqa: E402

from mss_eval import timing as T  # noqa: E402

LAB = ["v1_spleeter", "v1_htdemucs", "v1_bs_roformer", "v1_scnet", "htdemucs_b1"]
COND = [f"{k}_{db}dB" for k, db in PERTURB]
OUT = P.DATA3 / "c2"


def est_signal(label, track):
    import measure as M2
    return P.load(M2.stem_dir(label, track) / "drums.wav")


def one(ti_t):
    ti, t = ti_t
    f = OUT / f"{P.safe_name(t)}.pkl"
    if f.exists():
        return t
    ref = P.mono(P.ref_stem(t))
    others = sum(P.mono(P.ref_stem(t, s)) for s in P.SOURCES if s != "drums")
    fl2 = pickle.load(open(P.P2DATA / "floors" / f"{P.safe_name(t)}.pkl", "rb"))
    res = {"track": t, "on": {}}
    sigs = {"ref": (ref, pickle.load(open(P.P2MEAS / LAB[0] / f"{P.safe_name(t)}.pkl", "rb"))["ref_onsets"])}
    for lab in LAB:
        p2 = pickle.load(open(P.P2MEAS / lab / f"{P.safe_name(t)}.pkl", "rb"))
        y = P.mono(est_signal(lab, t))[: len(ref)]
        sigs[lab] = (y, p2["est_onsets"])
    for ci, (kind, db) in enumerate(PERTURB):
        rng = np.random.default_rng([P.SEED, ti, ci])
        c = f"{kind}_{db}dB"
        sigs[f"floor_{c}"] = (_perturb(ref, others, kind, db, rng), fl2[c])
    for name, (y, ev) in sigs.items():
        res["on"][("p2", name)] = np.asarray(ev)
        res["on"][("energy", name)] = T.detect_energy(y, P.SR, events=ev)
        res["on"][("rise", name)] = T.detect_refined(y, P.SR, mode="rise", min_sep_s=0.03)
    # check: the regenerated floor signal reproduces the cached Phase 2 floor onsets (2 tracks)
    if ti in (0, 17):
        c = COND[1]
        res["floor_recheck"] = bool(np.allclose(T.detect(sigs[f"floor_{c}"][0], P.SR), fl2[c]))
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    print(t, flush=True)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--reverse", action="store_true", help="process tracks in reverse order "
                    "(a second process can then work from the other end; finished tracks are skipped)")
    a = ap.parse_args()
    from multiprocessing import Pool
    jobs = list(enumerate(P.tracks()))
    if a.reverse:
        jobs = jobs[::-1]
    with Pool(a.jobs, maxtasksperchild=4) as p:
        list(p.imap_unordered(one, jobs))
