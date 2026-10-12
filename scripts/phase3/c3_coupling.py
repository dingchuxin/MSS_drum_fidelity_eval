"""C3: timing-shape coupling per matched stroke (Phase 2 detector; v3 stroke measures).

Per matched reference stroke and model: signed displacement dev = est - ref (ms); v3 per-stroke
changes est - ref in rise time (ms), peak level (dB), pre-stroke level (minimum env10 in the
20 ms before the env10 peak; dB) and level jump (peak - pre-stroke; dB); covariates: reference
stroke level relative to the track's drum level (v3 peak) and onset strength.
Strokes with a pre-stroke level in digital silence (< -150 dB in either signal) have NaN
pre/jump changes (pair_v3).

  1  within-track Spearman of dev with each change: median over tracks and the share of tracks
     with p < 0.05 (by sign);
  2  OLS of |dev| on |changes| + covariates with track fixed effects (demeaned): within R^2
     and partial R^2 per term ((SSR_without - SSR_full) / SSR_without);
  3  displaced strokes (|dev| > 10 ms): the share with a large |change| (above that measure's
     90th percentile among non-displaced strokes), per measure and for any measure;
  4  predicted vs observed displacement from the rise-time change, using the detector's local
     shape sensitivity from synthetic test 1b of audit/l2_t1_onsets.py (results/audit/l2_t1_summary.json,
     local slopes in ms per ms of attack for 0.5-1, 1-2, ..., 15-20 ms attacks). The attack is
     inferred as rise / 0.8 (10-90 % of a linear ramp); rise_ms cannot resolve attacks below
     about 2 ms (a 0.5 ms attack reads ~2.3 ms on synthetic strokes), so predictions for strokes whose
     reference or estimate rise is below 2 ms are flagged and reported separately.

    python c3_coupling.py
Outputs: results/phase3/c3_*.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _p3 as P

CH = ["d_rise_ms", "d_peak_db", "d_pre_db", "d_jump_db"]


def slope_fn():
    s = json.load(open(P.ROOT / "results" / "audit" / "l2_t1_summary.json"))["1b"]["p2_detect"]["local_slope_median"]
    edges, vals = [], []
    for k, v in s.items():
        a, b = (float(x) for x in k.split("-"))
        edges.append((a, b))
        vals.append(v)
    return edges, vals


def predicted_shift(a0, a1, edges, vals):
    """Integral of the local slope from attack a0 to a1 (ms); slopes held constant outside
    0.5-20 ms at the end values."""
    lo, hi = np.minimum(a0, a1), np.maximum(a0, a1)
    sign = np.sign(a1 - a0)
    tot = np.zeros_like(lo)
    bounds = [(-np.inf, edges[0][0], vals[0])] + [(a, b, v) for (a, b), v in zip(edges, vals)] + \
             [(edges[-1][1], np.inf, vals[-1])]
    for a, b, v in bounds:
        seg = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None)
        tot += seg * v
    return sign * tot


def main():
    df = P.onset_table()
    m = df[df.matched].copy()
    m["abs_dev"] = m.dev_ms.abs()
    rows1, rows2, rows3, rows4 = [], [], [], []
    edges, vals = slope_fn()
    for lab in P.LABELS:
        g = m[m.label == lab]
        # 1 within-track Spearman
        for c in CH:
            rs = []
            for t, h in g.groupby("track"):
                ok = h[c].notna() & h.dev_ms.notna()
                if ok.sum() < 20:
                    continue
                r, p = spearmanr(h.dev_ms[ok], h[c][ok])
                rs.append((r, p))
            rs = np.array(rs)
            rows1.append({"label": lab, "change": c, "n_tracks": len(rs), "median_rho": np.median(rs[:, 0]),
                          "share_tracks_p05": np.mean(rs[:, 1] < 0.05),
                          "share_tracks_p05_positive": np.mean((rs[:, 1] < 0.05) & (rs[:, 0] > 0)),
                          "share_tracks_p05_negative": np.mean((rs[:, 1] < 0.05) & (rs[:, 0] < 0))})
        # 2 FE regression of |dev| on |changes| + covariates
        X = pd.DataFrame({"abs_d_rise": g.d_rise_ms.abs(), "abs_d_peak": g.d_peak_db.abs(),
                          "abs_d_pre": g.d_pre_db.abs(), "abs_d_jump": g.d_jump_db.abs(),
                          "ref_level_rel": g.peak_rel_db, "onset_strength": g.env_strength})
        d = pd.concat([X, g[["abs_dev", "track"]]], axis=1).dropna()
        dm = d.drop(columns="track") - d.drop(columns="track").groupby(d.track).transform("mean")
        y = dm.abs_dev.to_numpy()
        terms = list(X.columns)

        def ssr(cols):
            A = dm[cols].to_numpy()
            b = np.linalg.lstsq(A, y, rcond=None)[0]
            return float(((y - A @ b) ** 2).sum())
        sst = float((y ** 2).sum())
        full = ssr(terms)
        r = {"label": lab, "n_strokes": len(d), "within_R2": 1 - full / sst}
        for tname in terms:
            red = ssr([x for x in terms if x != tname])
            r[f"partialR2_{tname}"] = (red - full) / red
        r["within_R2_changes_only"] = 1 - ssr(["abs_d_rise", "abs_d_peak", "abs_d_pre", "abs_d_jump"]) / sst
        rows2.append(r)
        # 3 displaced strokes with a large change
        nd = g[~g.displaced]
        dd = g[g.displaced]
        r = {"label": lab, "n_displaced": len(dd)}
        anyl = np.zeros(len(dd), bool)
        for c in CH:
            thr = nd[c].abs().quantile(0.9)
            big = (dd[c].abs() > thr).to_numpy()
            r[f"threshold_{c}"] = thr
            r[f"share_displaced_large_{c}"] = np.nanmean(big) if len(dd) else np.nan
            r[f"share_nondisplaced_large_{c}"] = 0.1
            anyl |= big
        r["share_displaced_any_large"] = anyl.mean() if len(dd) else np.nan
        nd_any = np.zeros(len(nd), bool)
        for c in CH:
            nd_any |= (nd[c].abs() > nd[c].abs().quantile(0.9)).to_numpy()
        r["share_nondisplaced_any_large"] = nd_any.mean()
        rows3.append(r)
        # 4 predicted displacement from the rise change
        ref_rise = g.ref_rise_ms.to_numpy()
        est_rise = ref_rise + g.d_rise_ms.to_numpy()
        pred = predicted_shift(ref_rise / 0.8, est_rise / 0.8, edges, vals)
        ok = np.isfinite(pred) & np.isfinite(g.dev_ms.to_numpy())
        resolved = ok & (ref_rise >= 2) & (est_rise >= 2)
        obs = g.dev_ms.to_numpy()
        for nm, sel in (("all matched", ok), ("both rises >= 2 ms", resolved),
                        ("displaced > 10 ms", ok & g.displaced.to_numpy())):
            if sel.sum() < 10:
                continue
            rr, _ = spearmanr(pred[sel], obs[sel])
            A = np.column_stack([np.ones(sel.sum()), pred[sel]])
            b = np.linalg.lstsq(A, obs[sel], rcond=None)[0]
            res = obs[sel] - A @ b
            rows4.append({"label": lab, "subset": nm, "n": int(sel.sum()), "spearman_pred_obs": rr,
                          "R2_obs_on_pred": 1 - (res ** 2).sum() / ((obs[sel] - obs[sel].mean()) ** 2).sum(),
                          "ols_slope_obs_on_pred": b[1], "median_abs_pred_ms": np.median(np.abs(pred[sel])),
                          "median_abs_obs_ms": np.median(np.abs(obs[sel])),
                          "share_abs_pred_gt_5ms": np.mean(np.abs(pred[sel]) > 5),
                          "share_same_sign": np.mean(np.sign(pred[sel]) == np.sign(obs[sel])),
                          "share_rise_below_2ms_either": float(np.mean((ref_rise[ok] < 2) | (est_rise[ok] < 2)))})
        print(lab, flush=True)
    for nm, rows in (("c3_within_track_spearman", rows1), ("c3_fe_regression", rows2),
                     ("c3_displaced_large_change", rows3), ("c3_predicted_vs_observed", rows4)):
        pd.DataFrame(rows).to_csv(P.OUT3 / f"{nm}.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 250):
        for rows in (rows1, rows2, rows3, rows4):
            print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
