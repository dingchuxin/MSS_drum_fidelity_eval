"""C4: models vs floors at their own measured levels (Phase 2 detector, greedy matching).

Generic floors: the Phase 2 floors (white noise 40/30/20 dB, other-stems leakage 30/20 dB),
which were detected with the Phase 2 detector (data/phase2/floors; labelled 'P2 detector').
Matched floors (c4_floors.py): per track and model, (a) leakage at the model's SIR, (b) white
noise at its SDR, (c) leakage at SIR + white noise at SAR.
Per signal and track: F at +/-5/10/20/50 ms, median |dev|, share > 5/8/10/14/20 ms (of matched
pairs), unmatched reference / estimate shares. Each model is compared with its own three
floors track by track (paired two-sided Wilcoxon, rank-biserial r > 0 = model larger), Holm
over all tests. HT-Demucs: floors at the B1 levels; compared with both B1 and v1 timing.

    python c4_analyze.py
Outputs: results/phase3/c4_by_track.csv, c4_summary.csv, c4_tests.csv
"""
from __future__ import annotations

import pickle

import mir_eval
import numpy as np
import pandas as pd

import _p3 as P
from mss_eval import timing as T

TOLS = [5, 10, 20, 50]
THR = [5, 8, 10, 14, 20]
MODEL_OF = {"v1_spleeter": "spleeter", "v1_htdemucs": "htdemucs", "htdemucs_b1": "htdemucs",
            "v1_bs_roformer": "bs_roformer", "v1_scnet": "scnet"}


def meas(ref, est):
    ir, ie = T.match_pairs(ref, est, 0.05)
    a = np.abs((est[ie] - ref[ir]) * 1000)
    r = {"median_abs_dev_ms": np.median(a) if len(a) else np.nan,
         **{f"share_gt_{k}ms": np.mean(a > k) if len(a) else np.nan for k in THR},
         "share_ref_unmatched": 1 - len(ir) / max(len(ref), 1),
         "share_est_unmatched": 1 - len(ir) / max(len(est), 1)}
    for tol in TOLS:
        r[f"f_{tol}ms"] = mir_eval.onset.f_measure(ref, est, window=tol / 1000)[0]
    return r


def main():
    rows = []
    for t in P.tracks():
        c4 = pickle.load(open(P.DATA3 / "c4" / f"{P.safe_name(t)}.pkl", "rb"))
        fl = pickle.load(open(P.P2DATA / "floors" / f"{P.safe_name(t)}.pkl", "rb"))
        ref = fl["ref"]
        for lab, m in MODEL_OF.items():
            p2 = pickle.load(open(P.P2MEAS / lab / f"{P.safe_name(t)}.pkl", "rb"))
            assert np.allclose(p2["ref_onsets"], ref)
            rows.append({"track": t, "signal": lab, "kind": "model", **meas(ref, p2["est_onsets"])})
        for m in ("spleeter", "htdemucs", "bs_roformer", "scnet"):
            for k in ("leak", "noise", "leak_noise"):
                rows.append({"track": t, "signal": f"floor_{m}_{k}", "kind": "matched floor", **meas(ref, c4["on"][(m, k)]),
                             **{f"level_{x}": c4["levels"][m][x] for x in ("sdr", "sir", "sar")}})
        for c in ("white_40dB", "white_30dB", "white_20dB", "leak_30dB", "leak_20dB"):
            rows.append({"track": t, "signal": f"generic_{c} (P2 detector)", "kind": "generic floor", **meas(ref, fl[c])})
    bt = pd.DataFrame(rows)
    bt.to_csv(P.OUT3 / "c4_by_track.csv", index=False, float_format="%.5g")
    cols = [c for c in bt.columns if c.startswith(("f_", "median_", "share_", "level_"))]
    summ = bt.groupby(["kind", "signal"])[cols].median().reset_index()
    summ.to_csv(P.OUT3 / "c4_summary.csv", index=False, float_format="%.5g")
    tests = []
    for lab, m in MODEL_OF.items():
        a = bt[bt.signal == lab].set_index("track")
        for k in ("leak", "noise", "leak_noise"):
            b = bt[bt.signal == f"floor_{m}_{k}"].set_index("track").loc[a.index]
            for c in ("share_gt_5ms", "share_gt_10ms", "share_gt_20ms", "median_abs_dev_ms", "f_5ms",
                      "share_ref_unmatched", "share_est_unmatched"):
                d = (a[c] - b[c]).to_numpy(float)
                tests.append({"model": lab, "floor": k, "measure": c, "median_model": a[c].median(),
                              "median_floor": b[c].median(), "median_paired_diff": np.nanmedian(d),
                              "n_model_gt_floor": int(np.nansum(d > 0)), "n": int(np.isfinite(d).sum()),
                              "p": P.wilcoxon_p(d), "rank_biserial": P.rank_biserial(d)})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = P.holm(tests.p)
    tests.to_csv(P.OUT3 / "c4_tests.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(summ.round(4).to_string(index=False))
        print(tests[tests.measure.isin(["share_gt_10ms", "share_gt_5ms", "share_ref_unmatched"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
