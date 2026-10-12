"""C1 (onset spacing within tracks) and C7 (failure including missed onsets).

Unit: reference onset (Phase 2 detector), per label (v1 Spleeter, HT-Demucs, BS-Roformer,
SCNet-XL; HT-Demucs B1). displaced = matched at +/-50 ms and |dev| > 10 ms.
Predictors as Phase 2's A1 (standardised; sim_any 0/1), with the C0.5 features (band shares
and drum-to-rest on [t - 20, t + 50 ms]; v3 stroke peak). Logistic regressions with CR1
standard errors clustered by track; 'FE' adds track fixed effects (tracks whose outcome
does not vary are dropped).

    python c1_c7_onsets.py
Outputs: results/phase3/c1_*.csv, c7_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _p3 as P

TERMS = ["env_strength", "peak_rel_db", "log10_ioi_min_ms", "drum_to_rest_db", "share_low", "share_high", "sim_any"]


def fits(g, y, tag):
    out = []
    for suffix, fe, name in (("_p2", False, "P2 features, no FE"), ("", False, "P3 features, no FE"),
                             ("", True, "P3 features, track FE")):
        Z = P.design(g, suffix)
        r = P.logit(y, Z.to_numpy(), list(Z.columns), g.track.to_numpy(), fe=fe)
        r.insert(0, "model_spec", name)
        r.insert(0, "outcome", tag)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def main():
    df = P.onset_table()
    rng = np.random.default_rng(P.SEED)
    c1_fit, c1_within, c7_fit, c7_peak = [], [], [], []
    for lab in P.LABELS:
        g = df[(df.label == lab)]
        m = g[g.matched].copy()
        # C1.2 / A1 logit on matched onsets
        c1_fit.append(fits(m, m.displaced.to_numpy(float), "displaced | matched").assign(label=lab))
        # C1.3 within-track rates by spacing quintile (track-specific quintiles)
        rows = []
        for t, h in m.groupby("track"):
            if len(h) < 25:
                continue
            q = pd.qcut(h.ioi_min_ms.rank(method="first"), 5, labels=False)
            base = h.displaced.mean()
            for k in range(5):
                sel = h[q == k]
                rows.append({"track": t, "quintile": k + 1, "rate_minus_track_rate": sel.displaced.mean() - base,
                             "median_ioi_ms": sel.ioi_min_ms.median()})
        w = pd.DataFrame(rows)
        for k, h in w.groupby("quintile"):
            v = h.rate_minus_track_rate.to_numpy()
            lo, hi = P.boot_ci(v, np.mean, rng)
            c1_within.append({"label": lab, "quintile": k, "median_ioi_ms_in_quintile": h.median_ioi_ms.median(),
                              "mean_rate_minus_track_rate": v.mean(), "ci_lo": lo, "ci_hi": hi, "n_tracks": len(v)})
        # C7 failure over all reference onsets
        fail = (g.displaced | ~g.matched).to_numpy(float)
        c7_fit.append(fits(g, fail, "failed (displaced or unmatched) | all").assign(label=lab))
        c7_fit.append(fits(g, (~g.matched).to_numpy(float), "unmatched | all").assign(label=lab))
        # C7 d(peak) of unmatched vs matched strokes (v3, all reference strokes)
        rows = []
        for t, h in g.groupby("track"):
            um, mm = h[~h.matched].d_peak_db.dropna(), h[h.matched].d_peak_db.dropna()
            if len(um) >= 5:
                rows.append({"track": t, "n_unmatched": len(um), "d_peak_unmatched": um.median(),
                             "d_peak_matched": mm.median()})
        w = pd.DataFrame(rows)
        d = (w.d_peak_unmatched - w.d_peak_matched).to_numpy()
        c7_peak.append({"label": lab, "n_tracks_ge5_unmatched": len(w), "total_unmatched": int((~g.matched).sum()),
                        "median_d_peak_unmatched_db": w.d_peak_unmatched.median(),
                        "median_d_peak_matched_db": w.d_peak_matched.median(),
                        "median_paired_diff_db": np.median(d), "n_tracks_unmatched_lower": int((d < 0).sum()),
                        "p": P.wilcoxon_p(d), "rank_biserial": P.rank_biserial(d),
                        "pooled_median_d_peak_unmatched": g[~g.matched].d_peak_db.median(),
                        "pooled_median_d_peak_matched": g[g.matched].d_peak_db.median()})
        print(lab, flush=True)
    fit = pd.concat(c1_fit, ignore_index=True)
    fit.to_csv(P.OUT3 / "c1_logit.csv", index=False, float_format="%.5g")
    pd.DataFrame(c1_within).to_csv(P.OUT3 / "c1_within_track_spacing.csv", index=False, float_format="%.5g")
    f7 = pd.concat(c7_fit, ignore_index=True)
    f7.to_csv(P.OUT3 / "c7_logit.csv", index=False, float_format="%.5g")
    c7p = pd.DataFrame(c7_peak)
    c7p["p_holm"] = P.holm(c7p.p)
    c7p.to_csv(P.OUT3 / "c7_peak_unmatched_vs_matched.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(pd.DataFrame(c1_within).round(4).to_string(index=False))
        piv = fit[fit.term != "intercept"].pivot_table(index=["label", "term"], columns="model_spec", values="odds_ratio")
        print(piv.round(3).to_string())
        f = f7[(f7.model_spec == "P3 features, track FE")]
        print(f.pivot_table(index=["label", "term"], columns="outcome", values="odds_ratio").round(3).to_string())
        print(f.pivot_table(index=["label", "term"], columns="outcome", values="p").round(4).to_string())
        print(c7p.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
