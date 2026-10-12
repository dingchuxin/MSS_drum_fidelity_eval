"""J0 + J1: onset-detector resolution and the timing battery, from cached drum stems.

Per track (drums only), with both detectors (v1 = the published pipeline, hop 512 +
backtracking; refined = hop 64 + parabolic peak interpolation, see src/mss_eval/timing.py):

  J0 shift test   the reference stem delayed by known integer sample counts; the error of
                  the recovered delay (matched onsets, +/-50 ms) is pure detector
                  quantisation, since the signal is otherwise identical.
  J1 floor        the reference stem under small perturbations (white noise at 40/30/20 dB
                  SNR; the sum of the other three reference stems at 30/20 dB, i.e. an
                  interference-only 'separation' at a given SIR) vs the clean reference.
  J1 battery      each model's estimate vs the reference: onset F/P/R at +/-5..50 ms and
                  signed deviations (estimate - reference) of pairs matched at +/-50 ms.

Per-track results are cached (MSS_PHASE1_CACHE, default <repo>/data/phase1_cache; gitignored)
so the summary step (j1_summarize.py) can be rerun without re-detecting.

    python scripts/phase1/j1_timing.py [--jobs 4]
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import _common as C  # noqa: E402
from mss_eval import timing as T  # noqa: E402

CACHE = Path(os.environ.get("MSS_PHASE1_CACHE", C.ROOT / "data" / "phase1_cache")) / "j1"
TOLS_MS = [5, 10, 20, 30, 50]
SHIFTS = [7, 23, 64, 111, 256, 441]          # samples: 0.16-10 ms, off both hop grids
PERTURB = [("white", 40), ("white", 30), ("white", 20), ("leak", 30), ("leak", 20)]
MATCH_S = 0.05


def _prf(ref, est):
    import mir_eval
    out = {}
    for tol in TOLS_MS:
        if len(ref) == 0:
            f = p = r = np.nan
        else:
            f, p, r = mir_eval.onset.f_measure(ref, est, window=tol / 1000)
        out[tol] = (f, p, r)
    return out


def _pairs(ref, est):
    i, j = T.match_pairs(ref, est, MATCH_S)
    return ref[i], est[j]


def _perturb(ref: np.ndarray, others: np.ndarray, kind: str, db: float,
             rng: np.random.Generator) -> np.ndarray:
    p_ref = np.mean(ref ** 2)
    if kind == "white":
        n = rng.standard_normal(ref.shape)
    else:                                             # interference from the true mix
        n = others.copy()
    n *= np.sqrt(p_ref / (np.mean(n ** 2) + 1e-20) / 10 ** (db / 10))
    return ref + n


def one(ti_track):
    ti, track = ti_track
    out_f = CACHE / f"{C.safe_name(track)}.pkl"
    if out_f.exists():
        return track
    ref_st = C.ref_stem(track)
    ref = C.mono(ref_st)
    others = sum(C.mono(C.ref_stem(track, s)) for s in C.SOURCES if s != "drums")
    res = {"track": track, "n_samples": len(ref)}

    det = {"refined": lambda y: T.detect_refined(y, C.SR), "v1": lambda y: T.detect_v1(y, C.SR)}
    ons_ref = {k: f(ref) for k, f in det.items()}
    res["ref_onsets"] = ons_ref

    # J0 shift test (both detectors)
    shift = {}
    for k, f in det.items():
        for s in SHIFTS:
            y = np.concatenate([np.zeros(s), ref])[: len(ref)]
            o = f(y)
            r, e = _pairs(ons_ref[k], o)
            shift[(k, s)] = (e - r) * 1000 - s / C.SR * 1000     # error in ms
    res["shift_err_ms"] = shift

    # J1 floor (refined detector; v1 also, for comparison)
    floor = {}
    for ci, (kind, db) in enumerate(PERTURB):
        rng = np.random.default_rng([C.SEED, ti, ci])
        y = _perturb(ref, others, kind, db, rng)
        for k, f in det.items():
            o = f(y)
            floor[(k, f"{kind}_{db}dB")] = {"onsets": o, "prf": _prf(ons_ref[k], o),
                                            "pairs": _pairs(ons_ref[k], o)}
    res["floor"] = floor

    # J1 battery
    bat = {}
    for m in C.MODELS:
        est = C.mono(C.est_stem(m, track))
        for k, f in det.items():
            o = f(est)
            bat[(k, m)] = {"onsets": o, "prf": _prf(ons_ref[k], o),
                           "pairs": _pairs(ons_ref[k], o)}
    res["battery"] = bat
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(out_f, "wb") as fh:
        pickle.dump(res, fh)
    return track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    from multiprocessing import Pool
    jobs = list(enumerate(C.tracks()))
    with Pool(a.jobs, maxtasksperchild=5) as p:
        for i, t in enumerate(p.imap_unordered(one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {t}", flush=True)


if __name__ == "__main__":
    main()
