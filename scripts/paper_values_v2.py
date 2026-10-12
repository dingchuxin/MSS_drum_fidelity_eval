"""Values reported in the article that no result file stores directly, computed from the result files.

The script reads only stored result files (unrounded) and writes

  <out>/paper_values_v2.csv         name, value, source (unrounded values)
  <out>/paper_values_v2_checks.csv  every value that has a stored or expected value:
                                             name, value, expected, rel_diff, above_1pct, note

Items D3.1-D3.8 (row names start with "D3.<item>"):
  1  pairwise model comparisons on f_5ms, median_abs_dev_ms, share_gt_10ms and the share missed
     (1 - n_pairs/n_ref), 6 pairs x 4 measures, two-sided Wilcoxon, Holm over the 24 tests
  2  BS-Roformer vs SCNet-XL on five v3 stroke measures (all reference strokes), Holm over 5
  3  within-model Spearman rho: d attack_slope vs d valley_db_all and vs d rise_ms_all
  4  BS-Roformer vs SCNet-XL on median_dev_ms (Phase 2 detector), uncorrected p
  5  bootstrap CIs of rho(SI-SDR or BSS Eval SDR, outcome) for the 15 A4 outcomes, after
     rebuilding the outcomes from the per-track files and checking rho against the stored file
  6  median over tracks of d_accent_matched / ref_accent_matched and d_valley_db_all /
     ref_valley_db_all, in %
  7  the error-budget figure's CIs (median over tracks of median_abs_dev_ms and of the share
     missed, 10,000-resample bootstrap), with the same generator and draw order as the figure
     script (scripts/make_figures_v2.py, fig_budget), so the values are the figure's
  8  the seeded-run check in SD units: median_abs_track_diff / reference unit, shape measures
Items E2a-E2f (row names start with "E2<item>"):
  a  missed strokes' extra loss in stroke peak (C7), in dB and in reference SD units
  b  Experiment 2: median over passages of the per-passage median drift, factors F1 and swap
  c  D1: share of the 60-250 ms strokes with all three onsets matched, pooled over tracks
  d  D1: the true stem's SD of d_k, median over tracks
  e  D1, Spleeter: Spearman rho of the IOI-profile changes with the missed and extra shares
  f  D2: statistics of the missed-stroke classes (near misses, weak peaks, local maxima)
Expected values are checked at the precision they are quoted (`quoted_decimals` in the checks file).

Conventions: pairs are (first, second) in the order Spleeter, HT-Demucs, BS-Roformer, SCNet-XL;
the paired difference is first - second; rank-biserial r is positive when the first is larger
(zero differences dropped); Wilcoxon p from scipy.stats.wilcoxon (two-sided, default method).
Seeds: numpy default_rng(20260925), one generator per item.

    python scripts/paper_values_v2.py [--out DIR]
<out> defaults to results/phase5.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "phase5"
SEED = 20260925
MODELS = ["spleeter", "htdemucs", "bs_roformer", "scnet"]
SHORT = {"spleeter": "Spleeter", "htdemucs": "HT-Demucs", "bs_roformer": "BS-Roformer", "scnet": "SCNet-XL"}
SRC_P2 = "results/phase2/measure/v1_{m}_by_track.csv"
SRC_P3 = "results/phase3/rerun/measure/v1_{m}_by_track.csv"

rows: list[dict] = []
checks: list[dict] = []


def put(name, value, source, expected=None, note="", tol=0.01, dec=None):
    """dec: digits after the decimal point at which `expected` is quoted, when a trailing zero
    makes it ambiguous (e.g. 0.190); otherwise inferred from the value."""
    value = float(value)
    rows.append({"name": name, "value": value, "source": source})
    if expected is not None:
        rel = abs(value - expected) / abs(expected) if expected != 0 else abs(value)
        # precision at which the expected value is quoted (digits after the decimal point)
        if dec is None:
            s = np.format_float_positional(expected, trim="-")
            dec = len(s.split(".")[1]) if "." in s else 0
            if "e" in repr(expected):                # e.g. 4.5e-13: two significant digits
                dec = -int(np.floor(np.log10(abs(expected)))) + 1
        same = bool(np.isclose(round(value, dec), round(expected, dec), rtol=0, atol=10 ** (-dec) / 10))
        checks.append({"name": name, "value": value, "expected": expected, "rel_diff": rel,
                       "above_1pct": bool(rel > tol), "quoted_decimals": dec,
                       "equal_at_quoted_precision": same, "note": note})


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired(a, b):
    d = (a - b).dropna()
    return {"median_first": float(a.median()), "median_second": float(b.median()),
            "median_paired_diff": float(d.median()), "n_first_larger": int((d > 0).sum()),
            "n_tracks": int(len(d)), "rank_biserial": rank_biserial(d), "p": float(wilcoxon(d).pvalue)}


def p2(m):
    df = pd.read_csv(ROOT / SRC_P2.format(m=m)).set_index("track").sort_index()
    df["share_missed"] = 1 - df.n_pairs / df.n_ref
    return df


def p3(m):
    return pd.read_csv(ROOT / SRC_P3.format(m=m)).set_index("track").sort_index()


def item1():
    D = {m: p2(m) for m in MODELS}
    tests = []
    for meas in ("f_5ms", "median_abs_dev_ms", "share_gt_10ms", "share_missed"):
        for a, b in combinations(MODELS, 2):
            tests.append((meas, a, b, paired(D[a][meas], D[b][meas])))
    ph = holm([t[3]["p"] for t in tests])
    exp = {("median_abs_dev_ms", "htdemucs", "bs_roformer"): (35, 0.002),
           ("median_abs_dev_ms", "bs_roformer", "scnet"): (19, 0.027),
           ("median_abs_dev_ms", "htdemucs", "scnet"): (None, 0.17),
           ("share_gt_10ms", "bs_roformer", "scnet"): (7, None)}
    for (meas, a, b, r), h in zip(tests, ph):
        src = f"{SRC_P2.format(m=a)}; {SRC_P2.format(m=b)}" + ("" if meas != "share_missed" else " (1 - n_pairs/n_ref)")
        base = f"D3.1 {meas} {a} vs {b}"
        e = exp.get((meas, a, b), (None, None))
        for k, v in r.items():
            put(f"{base} {k}", v, src, expected=e[0] if k == "n_first_larger" else None)
        put(f"{base} p_holm24", h, src, expected=e[1],
            note="expected value is quoted at 1-2 significant digits" if e[1] is not None else "")


def item2():
    bs, sc = p3("bs_roformer"), p3("scnet")
    meas = [("jump_db", "d_jump_db_all", 0.22, 37), ("valley_db", "d_valley_db_all", 0.38, 42),
            ("peak_db", "d_peak_db_all", 0.07, 33), ("pre_db", "d_pre_db_all", -0.09, 16),
            ("rise_ms", "d_rise_ms_all", -0.003, 19)]
    res = [(m, col, e1, e2, paired(bs[col], sc[col])) for m, col, e1, e2 in meas]
    ph = holm([r[4]["p"] for r in res])
    src = f"{SRC_P3.format(m='bs_roformer')}; {SRC_P3.format(m='scnet')}"
    for (m, col, e1, e2, r), h in zip(res, ph):
        base = f"D3.2 {col} bs_roformer vs scnet"
        for k, v in r.items():
            ex = e1 if k == "median_paired_diff" else e2 if k == "n_first_larger" else None
            put(f"{base} {k}", v, src, expected=ex,
                note="expected value is quoted at 1-2 significant digits" if k == "median_paired_diff" else "")
        put(f"{base} p_holm5", h, src)


def item3():
    rv, rr = [], []
    for m in MODELS:
        d = p3(m)
        r1 = spearmanr(d.d_attack_slope, d.d_valley_db_all, nan_policy="omit")[0]
        r2 = spearmanr(d.d_attack_slope, d.d_rise_ms_all, nan_policy="omit")[0]
        rv.append(r1), rr.append(r2)
        put(f"D3.3 rho d_attack_slope~d_valley_db_all {m}", r1, SRC_P3.format(m=m))
        put(f"D3.3 rho d_attack_slope~d_rise_ms_all {m}", r2, SRC_P3.format(m=m))
    src = SRC_P3.format(m="{4 models}")
    put("D3.3 rho d_attack_slope~d_valley_db_all min over models", min(rv), src, expected=0.645)
    put("D3.3 rho d_attack_slope~d_valley_db_all max over models", max(rv), src, expected=0.772)
    put("D3.3 rho d_attack_slope~d_rise_ms_all min over models", min(rr), src, expected=-0.111)
    put("D3.3 rho d_attack_slope~d_rise_ms_all max over models", max(rr), src, expected=0.071)


def item4():
    bs, sc = p2("bs_roformer"), p2("scnet")
    r = paired(bs.median_dev_ms, sc.median_dev_ms)
    src = f"{SRC_P2.format(m='bs_roformer')}; {SRC_P2.format(m='scnet')}"
    put("D3.4 median_dev_ms bs_roformer vs scnet n_bs_lower", r["n_tracks"] - r["n_first_larger"], src, expected=46)
    put("D3.4 median_dev_ms bs_roformer vs scnet median_paired_diff_ms", r["median_paired_diff"], src, expected=-0.66,
        note="expected value is quoted at 2 significant digits")
    put("D3.4 median_dev_ms bs_roformer vs scnet p_uncorrected", r["p"], src, expected=4.5e-13,
        note="expected value is quoted at 2 significant digits")
    put("D3.4 median_dev_ms bs_roformer vs scnet rank_biserial", r["rank_biserial"], src)


# ----------------------------------------------------------------- item 5
DESCS = ["level_db", "variation", "crest", "impulsiveness", "env_skew", "env_flux", "attack_slope"]
STROKE = ["peak_db", "floor_db", "valley_db", "rise_ms", "accent"]
OUTS = ["f_5ms", "share_gt_10ms", "median_abs_dev_ms"] + [f"abs_{m}_sd" for m in DESCS + STROKE]


def dcol(m):
    return f"d_{m}" if m in DESCS else ("d_accent_all" if m == "accent" else f"d_{m}_all")


def a4_frames():
    bss = pd.read_csv(R / "phase1" / "bss_drums_pertrack.csv")
    U = pd.read_csv(R / "phase3" / "rerun" / "units.csv").set_index("measure")
    fr = []
    for m in MODELS:
        df = p3(m)
        b = bss[bss.model == m].set_index("track").loc[df.index]
        df["sdr"] = b.sdr.to_numpy()
        df["f_5ms"] = p2(m).f_5ms.loc[df.index]
        for k in DESCS + STROKE:
            df[f"abs_{k}_sd"] = df[dcol(k)].abs() / U.loc[k, "sd"]
        df["model"] = f"v1_{m}"
        fr.append(df.reset_index())
    return pd.concat(fr, ignore_index=True)


def spearman_rows(X, Y):
    """Row-wise Spearman rho of two (B, n) arrays (average ranks for ties)."""
    rx, ry = rankdata(X, axis=1), rankdata(Y, axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def item5():
    D = a4_frames()
    st = pd.read_csv(R / "phase3" / "rerun" / "a4_sdr_vs_fidelity_v3.csv")
    st = st[st.model_set == "v1"]
    src = ("results/phase3/rerun/measure/v1_{model}_by_track.csv; results/phase2/measure/v1_{model}_by_track.csv (f_5ms); "
           "results/phase1/bss_drums_pertrack.csv (sdr); results/phase3/rerun/units.csv; "
           "stored: results/phase3/rerun/a4_sdr_vs_fidelity_v3.csv")
    rng = np.random.default_rng(SEED)
    tr = np.array(sorted(D.track.unique()))
    tix = {t: np.where(D.track.to_numpy() == t)[0] for t in tr}
    for pr in ("si_sdr", "sdr"):
        for oc in OUTS:
            los, his = [], []
            for m in MODELS:
                g = D[D.model == f"v1_{m}"]
                ok = (g[pr].notna() & g[oc].notna()).to_numpy()
                x, y = g[pr].to_numpy()[ok], g[oc].to_numpy()[ok]
                rho = spearmanr(x, y)[0]
                s = st[(st.predictor == pr) & (st.outcome == oc) & (st.scope == "within_model") & (st.model == f"v1_{m}")]
                put(f"D3.5 rho {pr}~{oc} within {m}", rho, src, expected=float(s.rho.iloc[0]),
                    note="stored value (5 significant digits) must be reproduced")
                idx = rng.integers(0, len(x), size=(10_000, len(x)))
                bs = spearman_rows(x[idx], y[idx])
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                los.append(lo), his.append(hi)
                put(f"D3.5 rho {pr}~{oc} within {m} ci_lo", lo, src)
                put(f"D3.5 rho {pr}~{oc} within {m} ci_hi", hi, src)
            exp_lo = -0.49 if (pr, oc) == ("si_sdr", "share_gt_10ms") else None
            exp_hi = 0.18 if (pr, oc) == ("si_sdr", "share_gt_10ms") else None
            put(f"D3.5 rho {pr}~{oc} within-model CI span lo (min over models)", min(los), src, expected=exp_lo,
                note="bootstrap bound; expected quoted at 2 decimals" if exp_lo else "")
            put(f"D3.5 rho {pr}~{oc} within-model CI span hi (max over models)", max(his), src, expected=exp_hi,
                note="bootstrap bound; expected quoted at 2 decimals" if exp_hi else "")
            # demeaned: stored value, then a track-resampling bootstrap with model means recomputed
            x, y = D[pr], D[oc]
            xd = x - D.groupby("model")[pr].transform("mean")
            yd = y - D.groupby("model")[oc].transform("mean")
            ok = (xd.notna() & yd.notna()).to_numpy()
            rho = spearmanr(xd[ok], yd[ok])[0]
            s = st[(st.predictor == pr) & (st.outcome == oc) & (st.scope == "demeaned")]
            e = -0.10 if (pr, oc) == ("si_sdr", "share_gt_10ms") else None
            put(f"D3.5 rho {pr}~{oc} demeaned", rho, src, expected=float(s.rho.iloc[0]),
                note="stored value (5 significant digits) must be reproduced")
            if e is not None:
                put(f"D3.5 rho {pr}~{oc} demeaned (quoted)", rho, src, expected=e,
                    note="expected quoted at 2 decimals")
            xa, ya, mod = x.to_numpy(), y.to_numpy(), D.model.to_numpy()
            bs = []
            for _ in range(2000):
                sel = np.concatenate([tix[t] for t in rng.choice(tr, len(tr))])
                xs, ys, ms = xa[sel], ya[sel], mod[sel]
                xm = pd.Series(xs).groupby(ms).transform("mean").to_numpy()
                ym = pd.Series(ys).groupby(ms).transform("mean").to_numpy()
                u, v = xs - xm, ys - ym
                k = np.isfinite(u) & np.isfinite(v)
                bs.append(spearmanr(u[k], v[k])[0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            put(f"D3.5 rho {pr}~{oc} demeaned ci_lo", lo, src, expected=-0.31 if e is not None else None,
                note="bootstrap bound; expected quoted at 2 decimals" if e is not None else "")
            put(f"D3.5 rho {pr}~{oc} demeaned ci_hi", hi, src, expected=0.13 if e is not None else None,
                note="bootstrap bound; expected quoted at 2 decimals" if e is not None else "")


def item6():
    ea = {"bs_roformer": 4.8, "scnet": 6.6, "htdemucs": 9.3, "spleeter": 20.3}
    ev = {"bs_roformer": 5.4, "scnet": 2.5, "htdemucs": 6.7, "spleeter": 2.4}
    for m in MODELS:
        d = p3(m)
        a = 100 * (d.d_accent_matched / d.ref_accent_matched).median()
        v = 100 * (d.d_valley_db_all / d.ref_valley_db_all).median()
        put(f"D3.6 median d_accent_matched/ref_accent_matched % {m}", a, SRC_P3.format(m=m), expected=ea[m],
            note="expected quoted at 1 decimal")
        put(f"D3.6 median d_valley_db_all/ref_valley_db_all % {m}", v, SRC_P3.format(m=m), expected=ev[m],
            note="expected quoted at 1 decimal")


def item7():
    """Same generator and draw order as make_figures_v2.fig_budget: per model (Spleeter,
    HT-Demucs, BS-Roformer, SCNet-XL) first the typical deviation, then the share missed."""
    rng = np.random.default_rng(SEED)

    def boot_median(v, B=10000):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        i = rng.integers(0, len(v), (B, len(v)))
        meds = np.median(v[i], axis=1)
        return np.median(v), np.percentile(meds, 2.5), np.percentile(meds, 97.5)

    for m in MODELS:
        pt = pd.read_csv(ROOT / SRC_P2.format(m=m))
        typ = boot_median(pt.median_abs_dev_ms)
        mis = boot_median(100 * (1 - pt.n_pairs / pt.n_ref))
        for k, v in zip(("median", "ci_lo", "ci_hi"), typ):
            put(f"D3.7 budget typical deviation ms {m} {k}", v, SRC_P2.format(m=m))
        for k, v in zip(("median", "ci_lo", "ci_hi"), mis):
            put(f"D3.7 budget share missed % {m} {k}", v, SRC_P2.format(m=m) + " (100 * (1 - n_pairs/n_ref))")


def item8():
    b = pd.read_csv(R / "phase3" / "rerun" / "b1_vs_v1_measures_v3.csv").set_index("measure")
    U = pd.read_csv(R / "phase3" / "rerun" / "units.csv").set_index("measure")
    src = "results/phase3/rerun/b1_vs_v1_measures_v3.csv; results/phase3/rerun/units.csv"
    vals = {}
    for meas in b.index:
        if not meas.startswith("d_"):
            continue
        base = meas[2:].replace("_all", "").replace("_matched", "")
        unit = U.loc[base, "sd"]
        vals[meas] = b.loc[meas, "median_abs_track_diff"] / unit
        put(f"D3.8 b1 vs v1 median_abs_track_diff in SD units {meas}", vals[meas], src)
    mx = max(vals, key=vals.get)
    put(f"D3.8 b1 vs v1 max over shape measures (SD units; at {mx})", vals[mx], src, expected=0.033,
        note="expected quoted at 2 significant digits; expected measure d_crest")
    return mx


# ----------------------------------------------------------------- items E2a-E2f
def e2a():
    """Missed strokes' extra loss in stroke peak (C7), in reference SD units of the stroke peak."""
    c = pd.read_csv(R / "phase3" / "c7_peak_unmatched_vs_matched.csv").set_index("label")
    unit = pd.read_csv(R / "phase3" / "rerun" / "units.csv").set_index("measure").loc["peak_db", "sd"]
    src = ("results/phase3/c7_peak_unmatched_vs_matched.csv (median_paired_diff_db); "
           "results/phase3/rerun/units.csv (peak_db)")
    put("E2a stroke-peak unit (dB)", unit, "results/phase3/rerun/units.csv (peak_db)", expected=4.606, dec=3)
    exp = {"v1_bs_roformer": (1.460, 0.32), "v1_scnet": (1.576, 0.34), "v1_spleeter": (1.808, 0.39),
           "v1_htdemucs": (2.484, 0.54), "htdemucs_b1": (2.496, 0.54)}
    for lab, (e_db, e_sd) in exp.items():
        loss = -c.loc[lab, "median_paired_diff_db"]          # unmatched minus matched is negative: a loss
        put(f"E2a extra peak loss of missed strokes dB {lab}", loss, src, expected=e_db, dec=3)
        put(f"E2a extra peak loss of missed strokes SD units {lab}", loss / unit, src, expected=e_sd, dec=2)


def e2b():
    """Experiment 2: median over passages of the per-passage median drift (median_sd_ms)."""
    d = pd.read_csv(R / "phase2" / "b3_timing_drift_by_track.csv")
    src = "results/phase2/b3_timing_drift_by_track.csv (median_sd_ms)"
    exp = {("F1", "spleeter"): 0.385, ("F1", "htdemucs"): 0.251, ("F1", "bs_roformer"): 0.190,
           ("F1", "scnet"): 0.159, ("swap", "scnet"): 0.311, ("swap", "bs_roformer"): 0.185}
    for (fac, m), e in exp.items():
        g = d[(d.factor == fac) & (d.model == m)]
        put(f"E2b median per-passage drift ms {fac} {m} (n={len(g)})", g.median_sd_ms.median(), src,
            expected=e, dec=3, note="swap: SCNet-XL at 50 % overlap, BS-Roformer at 75 %" if fac == "swap" else "")


def e2c():
    """D1: share of the 60-250 ms strokes whose three onsets are all matched, pooled over tracks."""
    d = pd.read_csv(R / "phase4" / "d1_statistic_by_track.csv")
    src = ("results/phase4/d1_statistic_by_track.csv "
           "(sum ioi60_250__n_triplets_matched / sum ioi60_250__n_triplets_ref)")
    exp = {"v1_spleeter": 54.2, "v1_htdemucs": 79.7, "v1_bs_roformer": 89.3, "v1_scnet": 86.6,
           "floor_leak_20dB": 92.6, "floor_white_20dB": 79.9}
    for sig, e in exp.items():
        g = d[d.signal == sig]
        v = 100 * g.ioi60_250__n_triplets_matched.sum() / g.ioi60_250__n_triplets_ref.sum()
        put(f"E2c D1 strokes used (pooled) % {sig}", v, src, expected=e, dec=1)


def e2d():
    """D1: the true stem's SD of d_k (all strokes with two neighbours), median over tracks."""
    d = pd.read_csv(R / "phase4" / "d1_statistic_by_track.csv")
    g = d[d.signal == "v1_spleeter"]           # the reference statistic is the same for every signal
    for s in ("v1_htdemucs", "v1_bs_roformer", "v1_scnet", "floor_leak_20dB"):
        h = d[d.signal == s].set_index("track")
        for col in ("ioi60_250__ref_all_sd_ms", "all__ref_all_sd_ms"):
            assert np.allclose(h[col].loc[g.track], g[col], equal_nan=True)
    for col, e in (("ioi60_250__ref_all_sd_ms", 19.73), ("all__ref_all_sd_ms", 104.79)):
        put(f"E2d D1 true-stem SD of d_k ms, median over tracks, {col}", g[col].median(),
            f"results/phase4/d1_statistic_by_track.csv ({col}; identical for every signal)", expected=e, dec=2)


def e2e():
    """D1, Spleeter: Spearman rho across tracks of the IOI-profile changes with missed and extra shares."""
    d = pd.read_csv(R / "phase4" / "d1_statistic_by_track.csv")
    g = d[d.signal == "v1_spleeter"]
    share = {"missed": 1 - g.n_pairs / g.n_ref, "extra": 1 - g.n_pairs / g.n_est}
    src = "results/phase4/d1_statistic_by_track.csv (signal v1_spleeter; n_pairs, n_ref, n_est)"
    exp = {("change_ioi_iqr_ms", "missed"): 0.27, ("change_ioi_iqr_ms", "extra"): 0.24,
           ("change_ioi_share_lt60", "missed"): -0.04, ("change_ioi_share_lt60", "extra"): 0.69}
    for (col, what), e in exp.items():
        rho, pval = spearmanr(g[col], share[what])
        put(f"E2e Spleeter rho {col}~{what} share", rho, src, expected=e, dec=2)
        put(f"E2e Spleeter rho {col}~{what} share p", pval, src)


def e2f():
    """D2: statistics of the missed-stroke classes at delta = 0.07."""
    d = pd.read_csv(R / "phase4" / "d2_missed_strokes.csv.gz")
    src = "results/phase4/d2_missed_strokes.csv.gz"
    exp = {"v1_spleeter": (51.5, 3.87, 0.0167, 20.7, 6.96, 4),
           "v1_htdemucs": (44.8, 3.71, 0.0145, 18.2, 12.11, 4),
           "v1_bs_roformer": (41.7, 2.28, 0.0065, 21.5, 13.27, 4),
           "v1_scnet": (42.3, 3.46, 0.0149, 18.2, 9.63, 4)}
    for sig in ["v1_spleeter", "v1_htdemucs", "v1_bs_roformer", "v1_scnet", "htdemucs_b1",
                "floor_leak_20dB", "floor_white_20dB"]:
        g = d[d.signal == sig]
        near, weak = g[g["class"] == "near_miss"], g[g["class"] == "weak_peak"]
        e = exp.get(sig, (None,) * 6)
        put(f"E2f share near miss or peak >= delta % {sig}",
            100 * g["class"].isin(["near_miss", "above_delta"]).mean(), src, expected=e[0], dec=1)
        put(f"E2f near misses median |offset| ms {sig}", near.offset_ms.abs().median(), src, expected=e[1], dec=2)
        put(f"E2f weak peaks median margin {sig}", weak.est_margin.median(), src, expected=e[2], dec=4)
        put(f"E2f weak peaks share margin <= 0 % {sig}", 100 * (weak.est_margin <= 0).mean(), src,
            expected=e[3], dec=1)
        put(f"E2f weak peaks median |offset| ms {sig}", weak.offset_ms.abs().median(), src, expected=e[4], dec=2)
        put(f"E2f median n_local_max, all missed strokes {sig}", g.n_local_max.median(), src,
            expected=e[5], dec=0)
        for c in ("near_miss", "weak_peak"):
            put(f"E2f median n_local_max, {c} {sig}", g[g["class"] == c].n_local_max.median(), src)


def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser(description="Recompute the article's values from the result files.")
    ap.add_argument("--out", default=str(OUT), help="output folder (default: results/phase5)")
    OUT = Path(ap.parse_args().out)
    item1(); item2(); item3(); item4(); item5(); item6(); item7()
    mx = item8()
    e2a(); e2b(); e2c(); e2d(); e2e(); e2f()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "paper_values_v2.csv", index=False, float_format="%.6g")
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "paper_values_v2_checks.csv", index=False, float_format="%.6g")
    print(f"{len(rows)} values, {len(ck)} checked, {int(ck.above_1pct.sum())} differ by more than 1 %")
    print(ck[ck.above_1pct].to_string())
    print("D3.8 maximum at", mx)


if __name__ == "__main__":
    main()
