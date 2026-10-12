"""C0.4 (and the C2 detector validation on real stems): known-shift precision, OFF-GRID
shifts only (no multiple of the 64-sample hop).

Each of the 50 reference drum stems is delayed by 7, 23, 37, 111, 300, 441 samples (integer:
prepend zeros) and 0.5, 32.5 samples (fractional: FFT phase ramp, audit/synth.frac_delay);
the recovered delay error is measured over onsets paired at +/-50 ms (greedy closest-first)
between the clean and the shifted signal, for
  p2      timing.detect (Phase 2 detector)                                   -> C0.4
  rise    timing.detect_refined(mode='rise', min_sep_s=0.03)                 -> C2(a)
  energy  timing.detect_energy (Phase 2 events, broadband energy onset fn.)   -> C2(b)

    python c0_shift_test.py [--jobs 2]
Output: results/phase3/c0_shift_test.csv (per detector x shift), c0_shift_test_by_track.csv
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _p3 as P  # noqa: E402

sys.path.insert(0, str(P.ROOT / "audit"))
import synth as S  # noqa: E402

from mss_eval import timing as T  # noqa: E402

SHIFTS = [7, 23, 37, 111, 300, 441, 0.5, 32.5]
CACHE = P.DATA3 / "shift_test"


def dets(y):
    ev = T.detect(y, P.SR)
    return {"p2": ev, "energy": T.detect_energy(y, P.SR, events=ev),
            "rise": T.detect_refined(y, P.SR, mode="rise", min_sep_s=0.03)}


def one(track):
    f = CACHE / f"{P.safe_name(track)}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    ref = P.mono(P.ref_stem(track))
    base = dets(ref)
    out = {"track": track, "err": {}}
    for s in SHIFTS:
        y = S.frac_delay(ref, s) if s != int(s) else np.concatenate([np.zeros(int(s)), ref])[: len(ref)]
        d = dets(y)
        for k in base:
            i, j = T.match_pairs(base[k], d[k], 0.05)
            out["err"][(k, s)] = ((d[k][j] - base[k][i]) * 1000 - s / P.SR * 1000,
                                  len(base[k]) + len(d[k]) - 2 * len(i))
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    print(track, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    from multiprocessing import Pool
    with Pool(a.jobs, maxtasksperchild=4) as p:
        res = p.map(one, P.tracks())
    rows, trk = [], []
    for k in ("p2", "rise", "energy"):
        for s in SHIFTS:
            e = np.concatenate([r["err"][(k, s)][0] for r in res])
            un = sum(r["err"][(k, s)][1] for r in res)
            ae = np.abs(e)
            rows.append({"detector": k, "shift_samples": s, "shift_ms": s / P.SR * 1000,
                         "n_pairs": len(e), "n_unpaired": un, "mean_err_ms": e.mean(),
                         "median_abs_err_ms": np.median(ae), "p95_abs_err_ms": np.percentile(ae, 95),
                         "p99_abs_err_ms": np.percentile(ae, 99), "max_abs_err_ms": ae.max(),
                         "share_abs_err_gt_1ms": np.mean(ae > 1)})
            for r in res:
                ee = np.abs(r["err"][(k, s)][0])
                trk.append({"detector": k, "shift_samples": s, "track": r["track"],
                            "p95_abs_err_ms": np.percentile(ee, 95) if len(ee) else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT3 / "c0_shift_test.csv", index=False, float_format="%.5g")
    pd.DataFrame(trk).to_csv(P.OUT3 / "c0_shift_test_by_track.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 200):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
