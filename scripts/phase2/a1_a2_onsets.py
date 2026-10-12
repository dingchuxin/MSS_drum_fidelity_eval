"""A1 (which onsets fail: substitutions, recurrence across models) and A2 (threshold table),
refined detector, drums. The regressions on stroke properties are in scripts/phase3/c1_c7_onsets.py.

Unit: reference onset. A matched pair is 'displaced' if |est - ref| > 10 ms. A displaced
pair is a 'likely substitution' if an UNMATCHED reference or estimate onset lies within
+/-50 ms of either member of the pair (a missed or extra stroke paired with the wrong
neighbour) rather than a true shift. For A2 the same neighbour test is applied to every
pair with |dev| > 5 ms (the smallest threshold), and 'without substitutions' drops those
pairs from numerator and denominator.

Outputs (results/phase2/):
  a1_substitution.csv, a1_recurrence.csv, a2_thresholds.csv,
  fig_timing_error_distribution.csv (figure-ready)
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import _p2 as P
import measure as M
from mss_eval import timing as T

import os
LABELS = os.environ.get("A1_LABELS", "v1_spleeter,v1_htdemucs,v1_bs_roformer,v1_scnet,htdemucs_b1").split(",")
FLOORS = ["white_40dB", "white_30dB", "white_20dB", "leak_30dB", "leak_20dB"]
THR = [5, 8, 10, 14, 20]
NB = 10_000


def _subst(ref, est, ir, ie, window=0.05):
    """Per matched pair: is an unmatched ref/est onset within +/-window of either member?"""
    um_r = np.setdiff1d(np.arange(len(ref)), ir)
    um_e = np.setdiff1d(np.arange(len(est)), ie)
    cand = np.sort(np.concatenate([ref[um_r], est[um_e]]))
    out = np.zeros(len(ir), bool)
    if len(cand) == 0:
        return out
    for k, (a, b) in enumerate(zip(ref[ir], est[ie])):
        for x in (a, b):
            p = np.searchsorted(cand, x)
            for q in (p - 1, p):
                if 0 <= q < len(cand) and abs(cand[q] - x) <= window:
                    out[k] = True
    return out


def pairs_table(label: str) -> pd.DataFrame:
    rows = []
    for t in P.tracks():
        r = pickle.load(open(M.MEAS / label / f"{P.safe_name(t)}.pkl", "rb"))
        ref, est, ir, ie = r["ref_onsets"], r["est_onsets"], r["i_ref"], r["i_est"]
        dev = (est[ie] - ref[ir]) * 1000
        rows.append(pd.DataFrame({"label": label, "track": t, "i_ref": ir, "dev_ms": dev,
                                  "neighbour_unmatched": _subst(ref, est, ir, ie)}))
    return pd.concat(rows, ignore_index=True)


def floor_pairs(cond: str) -> pd.DataFrame:
    rows = []
    for t in P.tracks():
        r = pickle.load(open(P.P2DATA / "floors" / f"{P.safe_name(t)}.pkl", "rb"))
        ref, est = r["ref"], r[cond]
        ir, ie = T.match_pairs(ref, est, 0.05)
        rows.append(pd.DataFrame({"label": f"floor_{cond}", "track": t, "i_ref": ir,
                                  "dev_ms": (est[ie] - ref[ir]) * 1000,
                                  "neighbour_unmatched": _subst(ref, est, ir, ie)}))
    return pd.concat(rows, ignore_index=True)


def boot_ci(per_track: pd.DataFrame, num: str, den: str, rng):
    """95 % CIs (tracks resampled) for the pooled share and the median per-track share."""
    n = per_track[num].to_numpy(float)
    d = per_track[den].to_numpy(float)
    idx = rng.integers(0, len(n), size=(NB, len(n)))
    pooled = n[idx].sum(1) / np.maximum(d[idx].sum(1), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(d > 0, n / d, np.nan)
    med = np.nanmedian(sh[idx], axis=1)
    return (np.percentile(pooled, [2.5, 97.5]), np.percentile(med, [2.5, 97.5]),
            n.sum() / d.sum(), np.nanmedian(sh))


def main():
    feats = pickle.load(open(P.P2DATA / "a1_features.pkl", "rb"))
    allp = pd.concat([pairs_table(l) for l in LABELS] + [floor_pairs(c) for c in FLOORS],
                     ignore_index=True)
    allp["abs_dev"] = allp.dev_ms.abs()
    allp["displaced"] = allp.abs_dev > 10
    allp["subst10"] = allp.displaced & allp.neighbour_unmatched
    allp["subst5"] = (allp.abs_dev > 5) & allp.neighbour_unmatched
    rng = np.random.default_rng(P.SEED)

    # ---- A2 thresholds ------------------------------------------------------------------
    a2 = []
    for lab, g in allp.groupby("label", sort=False):
        for excl in (False, True):
            h = g[~g.subst5] if excl else g
            for thr in THR:
                pt = h.groupby("track").agg(num=("abs_dev", lambda v: (v > thr).sum()),
                                            den=("abs_dev", "size")).reindex(P.tracks(), fill_value=0)
                (plo, phi), (mlo, mhi), pooled, med = boot_ci(pt, "num", "den", rng)
                a2.append({"label": lab, "without_substitutions": excl, "threshold_ms": thr,
                           "pooled_share": pooled, "pooled_ci_lo": plo, "pooled_ci_hi": phi,
                           "median_track_share": med, "median_ci_lo": mlo, "median_ci_hi": mhi,
                           "n_pairs": int(pt.den.sum())})
    a2 = pd.DataFrame(a2)
    a2.to_csv(P.OUT2 / "a2_thresholds.csv", index=False, float_format="%.5g")

    # figure-ready: distribution of signed deviations with the thresholds
    hist_edges = np.arange(-50, 50.5, 0.5)
    fr = []
    for lab, g in allp.groupby("label", sort=False):
        h, _ = np.histogram(g.dev_ms, bins=hist_edges)
        fr.append(pd.DataFrame({"label": lab, "bin_lo_ms": hist_edges[:-1], "bin_hi_ms": hist_edges[1:],
                                "count": h, "density": h / h.sum() / 0.5}))
    pd.concat(fr).to_csv(P.OUT2 / "fig_timing_error_distribution.csv", index=False, float_format="%.6g")

    # ---- A1 ------------------------------------------------------------------------------
    mp = allp[allp.label.isin(LABELS)].merge(feats, on=["track", "i_ref"], how="left")
    assert mp.env_strength.notna().all()
    sub_rows = []
    for lab, g in mp.groupby("label", sort=False):
        d = g[g.displaced]
        sub_rows.append({"label": lab, "n_matched": len(g), "n_displaced_gt_10ms": len(d),
                         "share_displaced": len(d) / len(g),
                         "n_likely_substitution": int(d.subst10.sum()),
                         "share_of_displaced_likely_substitution": d.subst10.mean(),
                         "share_displaced_excl_substitution": (g.displaced & ~g.subst10).sum() / (len(g) - g.subst10.sum())})
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(P.OUT2 / "a1_substitution.csv", index=False, float_format="%.5g")

    # cross-model recurrence
    rec = []
    labs = LABELS
    wide = mp.pivot_table(index=["track", "i_ref"], columns="label", values="displaced", aggfunc="first")
    for a in labs:
        for b in labs:
            if a == b:
                continue
            w = wide[[a, b]].dropna()
            w = w.astype(bool)
            base = w[b].mean()
            cond = w.loc[w[a], b].mean() if w[a].any() else np.nan
            rec.append({"A": a, "B": b, "n_both_matched": len(w), "n_displaced_A": int(w[a].sum()),
                        "P_B": base, "P_B_given_A": cond, "lift": cond / base if base else np.nan})
    rec = pd.DataFrame(rec)
    rec.to_csv(P.OUT2 / "a1_recurrence.csv", index=False, float_format="%.5g")

    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(a2[(a2.threshold_ms == 10)].round(4).to_string(index=False))
        print(sub.round(4).to_string(index=False))
        print(rec.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
