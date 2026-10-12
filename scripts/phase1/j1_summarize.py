"""Summarise the J0/J1 per-track cache written by j1_timing.py.

Outputs (results/phase1/):
  j0_resolution.csv              per-detector resolution (known-shift test, pooled over tracks)
                                 and perturbation-floor statistics
  onset_deviation_by_track.csv   per-track deviation summaries (both detectors)
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import _common as C
from j1_timing import CACHE, PERTURB, TOLS_MS

COND = [f"{k}_{db}dB" for k, db in PERTURB]
DET = ["refined", "v1"]


def load():
    res = []
    for t in C.tracks():
        with open(CACHE / f"{C.safe_name(t)}.pkl", "rb") as fh:
            res.append(pickle.load(fh))
    return res


def dev_stats(d: np.ndarray) -> dict:
    if len(d) == 0:
        return {k: np.nan for k in ["median_dev_ms", "q25_dev_ms", "q75_dev_ms", "iqr_dev_ms",
                                    "median_abs_dev_ms", "frac_abs_gt_5ms", "frac_abs_gt_10ms",
                                    "frac_abs_gt_20ms"]}
    q25, med, q75 = np.percentile(d, [25, 50, 75])
    a = np.abs(d)
    return {"median_dev_ms": med, "q25_dev_ms": q25, "q75_dev_ms": q75, "iqr_dev_ms": q75 - q25,
            "median_abs_dev_ms": np.median(a), "frac_abs_gt_5ms": np.mean(a > 5),
            "frac_abs_gt_10ms": np.mean(a > 10), "frac_abs_gt_20ms": np.mean(a > 20)}


def main():
    res = load()

    # ---- J0 shift test, pooled over tracks ---------------------------------------------
    pooled = {k: [] for k in DET}
    for r in res:
        for (k, s), e in r["shift_err_ms"].items():
            pooled[k].append(e)
    rrows = []
    for k in DET:
        e = np.abs(np.concatenate(pooled[k]))
        row = {"detector": k, "hop_samples": 64 if k == "refined" else 512,
               "frame_period_ms": (64 if k == "refined" else 512) / C.SR * 1000,
               "shift_n_pairs": len(e), "shift_median_abs_err_ms": np.median(e),
               "shift_p95_abs_err_ms": np.percentile(e, 95), "shift_p99_abs_err_ms": np.percentile(e, 99)}
        for c in COND:
            d = np.concatenate([(r["floor"][(k, c)]["pairs"][1] - r["floor"][(k, c)]["pairs"][0]) * 1000
                                for r in res])
            row[f"floor_{c}_median_abs_ms"] = np.median(np.abs(d))
            row[f"floor_{c}_p95_abs_ms"] = np.percentile(np.abs(d), 95)
        rrows.append(row)
    rdf = pd.DataFrame(rrows)
    rdf.to_csv(C.OUT / "j0_resolution.csv", index=False, float_format="%.4g")

    # ---- per-track deviation ------------------------------------------------------------
    trk_rows = []
    for r in res:
        t = r["track"]
        for k in DET:
            n_ref = len(r["ref_onsets"][k])
            for src, blob in [(m, r["battery"][(k, m)]) for m in C.MODELS] + \
                             [(c, r["floor"][(k, c)]) for c in COND]:
                kind = "model" if src in C.MODELS else "floor"
                rt, et = blob["pairs"]
                d = (et - rt) * 1000
                trk_rows.append({"detector": k, "kind": kind, "model": src, "track": t,
                                 "n_ref": n_ref, "n_est": len(blob["onsets"]), "n_pairs": len(d),
                                 **{f"f_{tol}ms": blob["prf"][tol][0] for tol in TOLS_MS},
                                 **dev_stats(d)})
    trk = pd.DataFrame(trk_rows)
    trk.to_csv(C.OUT / "onset_deviation_by_track.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 250):
        print(rdf.T.to_string())


if __name__ == "__main__":
    main()
