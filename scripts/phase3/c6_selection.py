"""C6: the Experiment 2 selection effect.

Phase 2's Factor 1 drift used only reference onsets in R matched at ALL 12 chunk positions.
From the Phase 2 B3 caches (per run: the matched estimate time of each reference onset in R;
onsets and matching unchanged):
  1  per model, the share of reference onsets matched at all 12 positions, at 1-11, at none
     (and the comparison of onsets matched at 6-11 positions with those matched at all 12);
  2  status change: per onset, the number of positions whose match status differs from o = 0;
     the share of onsets whose status changes at least once;
  3  drift (SD ddof 1, range, share of range > 5 ms) of fully matched onsets, and of partially
     matched onsets over the positions where they ARE matched (>= 2 positions);
  4  the overlap swap repeated with the status-change measure and with drift including
     partially matched onsets, Holm over the 10 tests;
  5  the same for Factor 2 (3 sub-hop positions).

    python c6_selection.py
Outputs: results/phase3/c6_*.csv
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import _p3 as P

M4 = ["spleeter", "htdemucs", "bs_roformer", "scnet"]
F1 = [f"f1_o{k:02d}" for k in range(12)]
SW = [f"swap_o{k:02d}" for k in range(12)]
F2 = ["f1_o00", "f2_s1", "f2_s2"]


def per_onset(r, conds):
    E = np.vstack([r["runs"][c]["est_time_per_ref_onset"] for c in conds]) * 1000
    ok = np.isfinite(E)
    n = ok.sum(0)
    changes = (ok != ok[0:1]).sum(0)
    sd = np.full(E.shape[1], np.nan)
    rg = np.full(E.shape[1], np.nan)
    for j in range(E.shape[1]):
        v = E[ok[:, j], j]
        if len(v) >= 2:
            sd[j] = np.std(v, ddof=1)
            rg[j] = v.max() - v.min()
    return n, changes, sd, rg


def main():
    man = pd.read_csv(P.OUT2 / "b3_manifest.csv")
    recs = []
    for t in man.track:
        for m in M4:
            r = pickle.load(open(P.P2DATA / "b3_measure" / m / f"{P.safe_name(t)}.pkl", "rb"))
            for fac, conds in (("F1", F1), ("F2", F2), ("swap", SW)):
                if not all(c in r["runs"] for c in conds):
                    continue
                n, ch, sd, rg = per_onset(r, conds)
                recs.append(pd.DataFrame({"model": m, "track": t, "factor": fac, "n_positions": len(conds),
                                          "n_matched": n, "status_changes": ch, "sd_ms": sd, "range_ms": rg}))
    po = pd.concat(recs, ignore_index=True)
    po.to_csv(P.OUT3 / "c6_per_onset.csv.gz", index=False, float_format="%.4g")
    rows = []
    for (m, fac), g in po.groupby(["model", "factor"]):
        K = g.n_positions.iloc[0]
        full = g[g.n_matched == K]
        part = g[(g.n_matched >= 2) & (g.n_matched < K)]
        mid = g[(g.n_matched >= 6) & (g.n_matched < K)] if K == 12 else part
        rows.append({"model": m, "factor": fac, "n_onsets": len(g),
                     "share_matched_all": np.mean(g.n_matched == K),
                     "share_matched_some": np.mean((g.n_matched > 0) & (g.n_matched < K)),
                     "share_matched_none": np.mean(g.n_matched == 0),
                     "share_status_changes_ge1": np.mean(g.status_changes >= 1),
                     "mean_status_changes": g.status_changes.mean(),
                     "full_median_sd_ms": full.sd_ms.median(), "full_median_range_ms": full.range_ms.median(),
                     "full_share_range_gt_5ms": np.mean(full.range_ms > 5),
                     "partial_n": len(part), "partial_median_sd_ms": part.sd_ms.median(),
                     "partial_median_range_ms": part.range_ms.median(),
                     "partial_share_range_gt_5ms": np.mean(part.range_ms > 5),
                     "matched_6to11_median_sd_ms": mid.sd_ms.median() if K == 12 else np.nan,
                     "matched_6to11_share_range_gt_5ms": np.mean(mid.range_ms > 5) if K == 12 else np.nan,
                     "all_ge2_median_sd_ms": g[g.n_matched >= 2].sd_ms.median(),
                     "all_ge2_share_range_gt_5ms": np.mean(g[g.n_matched >= 2].range_ms > 5)})
    summ = pd.DataFrame(rows)
    summ.to_csv(P.OUT3 / "c6_selection_summary.csv", index=False, float_format="%.5g")
    # per track measures for the swap comparison
    pt = po.groupby(["model", "factor", "track"]).agg(
        share_changed=("status_changes", lambda v: np.mean(v >= 1)),
        median_sd_incl_partial=("sd_ms", "median")).reset_index()
    comps = [("bs_roformer", "F1", "bs_roformer", "swap", "BS-Roformer: overlap 2 (native) vs 4"),
             ("scnet", "F1", "scnet", "swap", "SCNet-XL: overlap 4 (native) vs 2"),
             ("bs_roformer", "swap", "scnet", "F1", "at overlap 4: BS-Roformer vs SCNet-XL"),
             ("bs_roformer", "F1", "scnet", "swap", "at overlap 2: BS-Roformer vs SCNet-XL"),
             ("bs_roformer", "F1", "scnet", "F1", "native: BS-Roformer (2) vs SCNet-XL (4)")]
    rows = []
    for m1, f1, m2, f2, lab in comps:
        for meas in ("share_changed", "median_sd_incl_partial"):
            a = pt[(pt.model == m1) & (pt.factor == f1)].set_index("track")[meas]
            b = pt[(pt.model == m2) & (pt.factor == f2)].set_index("track")[meas].loc[a.index]
            d = (a - b).to_numpy(float)
            rows.append({"comparison": lab, "measure": meas, "median_a": a.median(), "median_b": b.median(),
                         "median_paired_diff": np.nanmedian(d), "n": int(np.isfinite(d).sum()),
                         "p": P.wilcoxon_p(d), "rank_biserial": P.rank_biserial(d)})
    sw = pd.DataFrame(rows)
    sw["p_holm_10"] = P.holm(sw.p)
    sw.to_csv(P.OUT3 / "c6_overlap_swap_status.csv", index=False, float_format="%.5g")
    # the stored per-onset selection table (results/audit), for comparison
    aud = pd.read_csv(P.ROOT / "results" / "audit" / "l1_b3_selection_per_onset.csv.gz")
    print(aud.head().to_string())
    with pd.option_context("display.width", 250):
        print(summ.round(4).to_string(index=False))
        print(sw.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
