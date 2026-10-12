"""D2: are some missed strokes a detector-threshold effect?

Inputs: the cached raw onset-strength envelopes (d2_envelopes.py). Peak picking with
timing.detect_delta (a copy of the Phase 2 detector with the threshold `delta` as a parameter;
delta = 0.07 reproduces the stored Phase 2 onsets exactly, checked in d2_envelopes.py and again
here against the Phase 2 per-track values). Matching: closest-first within +/-50 ms.
Signals: v1_spleeter, v1_htdemucs, v1_bs_roformer, v1_scnet, htdemucs_b1 (check), and the
generic floors leak_20dB (the main comparison) and white_20dB.

1. Threshold sensitivity, delta = 0.03, 0.05, 0.07, 0.10, two variants:
     both      reference and estimate re-detected at delta;
     ref_p2    reference fixed at delta = 0.07 (the Phase 2 reference onsets), estimate at delta;
   per track: missed share (1 - n_pairs/n_ref), extra share (1 - n_pairs/n_est), displaced share
   (|dev| > 10 ms among matched), typical deviation (median |dev|); summarised as medians over
   tracks with 95 % bootstrap CIs.
2. Threshold-free check for every reference onset missed at delta = 0.07 (Phase 2 onsets):
   in the estimate's normalised envelope (as _peak_pick normalises it), the true local maxima
   (x[n] > x[n-1], x[n] >= x[n+1]) within +/-50 ms of the reference onset; the one with the
   largest margin above the picker's moving mean (timing.pick_margin: the quantity compared
   with delta) gives the height and the time offset. Classes (hierarchical, exhaustive):
     paired_elsewhere  an estimate onset lies within +/-50 ms but is paired with a neighbouring
                       reference stroke (greedy one-to-one matching);
     above_delta       a peak reaches delta but no onset was kept (the picker's 30 ms wait or
                       the 30 ms merge of refined events);
     near_miss         delta/2 <= margin < delta;
     weak_peak         margin < delta/2;
     no_peak           no local maximum within +/-50 ms.
   The first two classes make the shares cover every missed stroke. Also stored per missed stroke: the
   reference envelope's own margin at the stroke (largest margin of a local maximum within
   +/-5 ms of the reference onset).
3. Normalisation: per track, the estimate's un-normalised envelope maximum over the
   reference's (and the same for the envelope range, max - min); Spearman rho with the missed
   share within each model (Holm over the models). Added check: the estimate re-picked at
   delta = 0.07 after normalising its envelope with the REFERENCE's min and max, so that both
   signals share one threshold in envelope units; missed share with that normalisation.

    python d2_threshold.py [--jobs 3]
Outputs: results/phase4/d2_threshold_by_track.csv, d2_missed_classes.csv,
         d2_missed_strokes.csv.gz, d2_normalisation_by_track.csv, d2_summary.csv
"""
from __future__ import annotations

import argparse
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _p4 as P  # noqa: E402
from mss_eval import timing as T  # noqa: E402

DELTAS = [0.03, 0.05, 0.07, 0.10]
SIGNALS = P.LABELS4 + [f"floor_{c}" for c in P.FLOORS4]
HOP = T.HOP_FINE
CLASSES = ["paired_elsewhere", "above_delta", "near_miss", "weak_peak", "no_peak"]


def norm(env):
    """_peak_pick's normalisation to [0, 1] over the whole signal."""
    e = env - env.min()
    return e / (e.max() + 1e-12)


def counts(ref, est):
    i, j = T.match_pairs(ref, est, 0.05)
    a = np.abs(est[j] - ref[i]) * 1000
    return {"n_ref": len(ref), "n_est": len(est), "n_pairs": len(i),
            "missed": 1 - len(i) / len(ref) if len(ref) else np.nan,
            "extra": 1 - len(i) / len(est) if len(est) else np.nan,
            "displaced_gt10": float(np.mean(a > 10)) if len(a) else np.nan,
            "median_abs_dev_ms": float(np.median(a)) if len(a) else np.nan}


def local_max(x):
    n = np.arange(1, len(x) - 1)
    return n[(x[n] > x[n - 1]) & (x[n] >= x[n + 1])]


def one(t):
    r = pickle.load(open(P.DATA4 / "env" / f"{P.safe_name(t)}.pkl", "rb"))
    env = r["env"]
    rows, strokes, normrows = [], [], []
    ref_on = {d: T.detect_delta(None, P.SR, d, env=env["ref"]) for d in DELTAS}
    ref_n = norm(env["ref"])
    ref_m = T.pick_margin(ref_n, P.SR, HOP)
    ref_lm = local_max(ref_n)
    for s in SIGNALS:
        est_on = {d: T.detect_delta(None, P.SR, d, env=env[s]) for d in DELTAS}
        for d in DELTAS:
            rows.append({"variant": "both", "delta": d, "signal": s, "track": t, **counts(ref_on[d], est_on[d])})
            rows.append({"variant": "ref_p2", "delta": d, "signal": s, "track": t, **counts(ref_on[0.07], est_on[d])})
        # 2. threshold-free check at delta = 0.07
        delta = 0.07
        ref, est = ref_on[delta], est_on[delta]
        i, j = T.match_pairs(ref, est, 0.05)
        missed = np.setdiff1d(np.arange(len(ref)), i)
        xn = norm(env[s])
        mg = T.pick_margin(xn, P.SR, HOP)
        lm = local_max(xn)
        for k in missed:
            tk = ref[k]
            f0, f1 = tk * P.SR / HOP - 0.05 * P.SR / HOP, tk * P.SR / HOP + 0.05 * P.SR / HOP
            near_est = est[(est >= tk - 0.05) & (est <= tk + 0.05)]
            cand = lm[(lm >= f0) & (lm <= f1)]
            rc = ref_lm[np.abs(ref_lm - tk * P.SR / HOP) <= 0.005 * P.SR / HOP]
            row = {"signal": s, "track": t, "i_ref": int(k), "t_ref_s": tk,
                   "n_est_onsets_within_50ms": len(near_est),
                   "ref_margin": float(ref_m[rc].max()) if len(rc) else np.nan,
                   "n_local_max": len(cand)}
            if len(cand):
                b = cand[np.argmax(mg[cand])]
                row.update(est_margin=float(mg[b]), est_height=float(xn[b]),
                           offset_ms=float((b * HOP / P.SR - tk) * 1000))
            else:
                row.update(est_margin=np.nan, est_height=np.nan, offset_ms=np.nan)
            if len(near_est):
                c = "paired_elsewhere"
            elif not len(cand):
                c = "no_peak"
            elif row["est_margin"] >= delta:
                c = "above_delta"
            elif row["est_margin"] >= delta / 2:
                c = "near_miss"
            else:
                c = "weak_peak"
            row["class"] = c
            strokes.append(row)
        # 3. normalisation
        e, er = env[s], env["ref"]
        xr = (e - er.min()) / (er.max() - er.min() + 1e-12)        # estimate in the reference's units
        import librosa
        kw = dict(pre_max=0.03 * P.SR // HOP, post_max=0.00 * P.SR // HOP + 1, pre_avg=0.10 * P.SR // HOP,
                  post_avg=0.10 * P.SR // HOP + 1, wait=0.03 * P.SR // HOP, delta=0.07)
        on_refnorm = T._refine_peak(xr, librosa.util.peak_pick(xr, **kw), P.SR, HOP)
        c07 = counts(ref, est)
        crn = counts(ref, on_refnorm)
        normrows.append({"signal": s, "track": t, "ref_env_max": float(er.max()), "est_env_max": float(e.max()),
                         "ref_env_min": float(er.min()), "est_env_min": float(e.min()),
                         "max_ratio": float(e.max() / er.max()),
                         "range_ratio": float((e.max() - e.min()) / (er.max() - er.min())),
                         "missed": c07["missed"], "extra": c07["extra"],
                         "missed_refnorm": crn["missed"], "extra_refnorm": crn["extra"],
                         "displaced_gt10_refnorm": crn["displaced_gt10"]})
    return rows, strokes, normrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    from multiprocessing import Pool
    with Pool(a.jobs) as p:
        res = p.map(one, P.tracks())
    by = pd.DataFrame([r for x in res for r in x[0]])
    st = pd.DataFrame([r for x in res for r in x[1]])
    nr = pd.DataFrame([r for x in res for r in x[2]])

    # delta = 0.07 must reproduce the stored Phase 2 per-track values
    chk = []
    for lab in P.LABELS4:
        bt = pd.read_csv(P.ROOT / "results" / "phase2" / "measure" / f"{lab}_by_track.csv").set_index("track")
        g = by[(by.variant == "both") & (by.delta == 0.07) & (by.signal == lab)].set_index("track").loc[bt.index]
        chk.append({"signal": lab,
                    "n_pairs_equal": bool((g.n_pairs == bt.n_pairs).all()),
                    "n_ref_equal": bool((g.n_ref == bt.n_ref).all()),
                    "n_est_equal": bool((g.n_est == bt.n_est).all()),
                    "max_abs_diff_median_abs_dev_ms": float((g.median_abs_dev_ms - bt.median_abs_dev_ms).abs().max()),
                    "max_abs_diff_share_gt10": float((g.displaced_gt10 - bt.share_gt_10ms).abs().max())})
    chk = pd.DataFrame(chk)
    print(chk.to_string())
    assert chk.n_pairs_equal.all() and chk.n_est_equal.all() and chk.n_ref_equal.all()

    by.to_csv(P.OUT4 / "d2_threshold_by_track.csv", index=False, float_format="%.6g")
    st.to_csv(P.OUT4 / "d2_missed_strokes.csv.gz", index=False, float_format="%.6g")
    nr.to_csv(P.OUT4 / "d2_normalisation_by_track.csv", index=False, float_format="%.6g")

    cls = []
    for s, g in st.groupby("signal", sort=False):
        n_missed_total = len(g)
        r = {"signal": s, "n_missed": n_missed_total, "n_tracks": g.track.nunique()}
        for c in CLASSES:
            r[f"share_{c}"] = float((g["class"] == c).mean())
        # per-track shares, median over tracks (tracks with >= 5 missed strokes)
        pt = g.groupby("track")["class"].value_counts(normalize=True).unstack(fill_value=0)
        pt = pt[g.groupby("track").size().loc[pt.index] >= 5]
        for c in CLASSES:
            r[f"median_track_share_{c}"] = float(pt[c].median()) if c in pt else 0.0
        r["n_tracks_ge5_missed"] = len(pt)
        r["median_est_margin_near_or_weak"] = float(g.loc[g["class"].isin(["near_miss", "weak_peak"]), "est_margin"].median())
        r["median_abs_offset_ms_with_peak"] = float(g.loc[g["class"].isin(["above_delta", "near_miss", "weak_peak"]), "offset_ms"].abs().median())
        r["median_ref_margin_missed"] = float(g.ref_margin.median())
        cls.append(r)
    pd.DataFrame(cls).to_csv(P.OUT4 / "d2_missed_classes.csv", index=False, float_format="%.5g")

    rng = np.random.default_rng(P.SEED)
    summ = []
    for (v, d, s), g in by.groupby(["variant", "delta", "signal"], sort=False):
        r = {"part": "threshold", "variant": v, "delta": d, "signal": s}
        for c in ("missed", "extra", "displaced_gt10", "median_abs_dev_ms"):
            med, lo, hi = P.boot_median_ci(g[c], rng)
            r[f"{c}_median"], r[f"{c}_ci_lo"], r[f"{c}_ci_hi"] = med, lo, hi
        summ.append(r)
    from scipy.stats import spearmanr
    rho_rows = []
    for s, g in nr.groupby("signal", sort=False):
        rr = {"part": "normalisation", "signal": s,
              "max_ratio_median": float(g.max_ratio.median()), "max_ratio_min": float(g.max_ratio.min()),
              "max_ratio_max": float(g.max_ratio.max()), "range_ratio_median": float(g.range_ratio.median()),
              "missed_median": float(g.missed.median()), "missed_refnorm_median": float(g.missed_refnorm.median()),
              "extra_refnorm_median": float(g.extra_refnorm.median()),
              "tracks_fewer_missed_refnorm": int((g.missed_refnorm < g.missed).sum()),
              "tracks_more_missed_refnorm": int((g.missed_refnorm > g.missed).sum()),
              "p_wilcoxon_missed_refnorm_vs_p2": P.wilcoxon_p(g.missed_refnorm - g.missed)}
        rr["rho_max_ratio_missed"], rr["p_rho_max_ratio"] = spearmanr(g.max_ratio, g.missed)
        rr["rho_range_ratio_missed"], rr["p_rho_range_ratio"] = spearmanr(g.range_ratio, g.missed)
        rho_rows.append(rr)
    rho = pd.DataFrame(rho_rows)
    m = rho.signal.isin(P.MODELS4)
    rho.loc[m, "p_holm_max_ratio_models"] = P.holm(rho.loc[m, "p_rho_max_ratio"])
    rho.loc[m, "p_holm_range_ratio_models"] = P.holm(rho.loc[m, "p_rho_range_ratio"])
    summ = pd.concat([pd.DataFrame(summ), rho,
                      chk.assign(part="reproduction_delta_0.07")], ignore_index=True)
    summ.to_csv(P.OUT4 / "d2_summary.csv", index=False, float_format="%.6g")
    print(pd.DataFrame(cls).round(3).to_string())
    print(rho.round(3).to_string())


if __name__ == "__main__":
    main()
