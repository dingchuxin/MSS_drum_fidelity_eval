"""C0.6: the analyses that use the v3 stroke measures and envelope descriptors.

    python c0_rerun.py [a3 j3 a4 a5 b1 b2 b3 corr all]

Inputs: results/phase3/rerun/measure/<label>[_bl]_by_track.csv (measure3.py),
data/phase3/v3/b3 (b3_measure3.py), and from earlier stages results/phase2/measure,
results/phase2/b3_manifest.csv, b3_timing_drift_by_track.csv, results/phase1/bss_drums_pertrack.csv
and results/tables/benchmark_raw.csv (read only).
Summaries (C0.2): medians with 95 % bootstrap CIs over tracks (10,000 resamples, seed
20260925) are primary; means are reported only with their leave-one-track-out (LOTO) range,
and tracks whose removal moves a mean by > 0.1 SD, or changes whether its CI excludes 0 (LOTO
CIs: 2,000 resamples), are named. If one track changes a reference between-track SD by
> 25 %, 1.4826 x MAD is also given as the unit.
Outputs: results/phase3/rerun/*.csv and results/phase3/fig_sdr_vs_fidelity_v3.csv
"""
from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import _p3 as P

RR = P.RERUN
MD = RR / "measure"
V1 = ["v1_spleeter", "v1_htdemucs", "v1_bs_roformer", "v1_scnet"]
MAIN = V1 + ["htdemucs_b1"]
DESCS = P.V3DESCS
STROKE = ["peak_db", "floor_db", "valley_db", "rise_ms", "accent"]
EXTRA = ["pre_db", "jump_db"]
NB, NB_LOTO = 10_000, 2_000


def load(label):
    return pd.read_csv(MD / f"{label}_by_track.csv").set_index("track").loc[P.tracks()]


def dcol(m, st="all"):
    if m in DESCS:
        return f"d_{m}"
    if m == "accent":
        return f"d_accent_{st}"
    return f"d_{m}_{st}"


def refcol(m):
    if m in DESCS:
        return f"ref_{m}"
    if m == "accent":
        return "ref_accent_all"
    return f"ref_{m}_all"


# ---------------------------------------------------------------------- units
def units(ref_df: pd.DataFrame, measures) -> pd.DataFrame:
    rows = []
    for m in measures:
        v = ref_df[refcol(m)].to_numpy(float)
        v = v[np.isfinite(v)]
        sd = np.std(v, ddof=1)
        loo = np.array([np.std(np.delete(v, i), ddof=1) for i in range(len(v))])
        ch = np.abs(loo - sd) / sd
        tr = ref_df.index[np.isfinite(ref_df[refcol(m)])][int(np.argmax(ch))]
        mad = 1.4826 * np.median(np.abs(v - np.median(v)))
        rows.append({"measure": m, "sd": sd, "max_loo_sd_change": ch.max(), "most_influential_track": tr,
                     "sd_without_that_track": loo[int(np.argmax(ch))], "mad_unit": mad,
                     "use_mad_too": bool(ch.max() > 0.25)})
    return pd.DataFrame(rows).set_index("measure")


def summ(v, rng, unit, tracks):
    """Median/mean summaries of per-track values v (natural units) in SD units."""
    ok = np.isfinite(v)
    v, tracks = v[ok], np.asarray(tracks)[ok]
    z = v / unit
    out = {"n": len(v), "median_nat": np.median(v), "median_sd": np.median(z)}
    out["median_ci_lo"], out["median_ci_hi"] = P.boot_ci(z, np.median, rng)
    out["mean_sd"] = z.mean()
    out["mean_ci_lo"], out["mean_ci_hi"] = P.boot_ci(z, np.mean, rng)
    loo = np.array([np.delete(z, i).mean() for i in range(len(z))])
    out["mean_loto_min"], out["mean_loto_max"] = loo.min(), loo.max()
    excl = (out["mean_ci_lo"] > 0) or (out["mean_ci_hi"] < 0)
    infl = []
    for i in np.argsort(-np.abs(loo - z.mean()))[:5]:
        zz = np.delete(z, i)
        lo, hi = P.boot_ci(zz, np.mean, rng, NB_LOTO)
        moved = abs(zz.mean() - z.mean()) > 0.1
        flips = ((lo > 0) or (hi < 0)) != excl
        if moved or flips:
            infl.append(f"{tracks[i]} ({'moves ' + format(zz.mean() - z.mean(), '+.2f') if moved else ''}"
                        f"{'; ' if moved and flips else ''}{'CI status flips' if flips else ''})")
    out["mean_influential_tracks"] = "; ".join(infl)
    return out


def sd_table(labels, measures, sets=("all", "matched"), ref_label=None, tag=""):
    rng = np.random.default_rng(P.SEED)
    ref_df = load(ref_label or labels[0])
    U = units(ref_df, measures)
    rows = []
    for lab in labels:
        df = load(lab)
        for m in measures:
            for st in (sets if (m in STROKE or m in EXTRA) else ("-",)):
                v = df[dcol(m, st if st != "-" else "all")].to_numpy(float)
                r = {"label": lab, "measure": m, "stroke_set": st, "unit_sd": U.loc[m, "sd"],
                     **summ(v, rng, U.loc[m, "sd"], df.index)}
                if U.loc[m, "use_mad_too"]:
                    z = v[np.isfinite(v)] / U.loc[m, "mad_unit"]
                    r.update({"unit_mad": U.loc[m, "mad_unit"], "median_mad_units": np.median(z),
                              "mean_mad_units": z.mean()})
                rows.append(r)
    out = pd.DataFrame(rows)
    U.to_csv(RR / f"units{tag}.csv", float_format="%.5g")
    return out, U


# ---------------------------------------------------------------------- A3 tables
def a3_j2():
    main, U = sd_table(MAIN, DESCS + STROKE + EXTRA)
    main.to_csv(RR / "sd_units_v3.csv", index=False, float_format="%.5g")
    # digital-silence summary
    sil = []
    for lab in MAIN:
        df = load(lab)
        sil.append({"label": lab, "strokes_silent_either": int(df.n_silent_either.sum()),
                    "tracks_with_silent_strokes": int((df.n_silent_either > 0).sum()),
                    "ref_silent_est_not": int(df.n_ref_silent_est_not.sum()),
                    "est_silent_ref_not": int(df.n_est_silent_ref_not.sum()),
                    "tracks_ref_silent_est_not": int((df.n_ref_silent_est_not > 0).sum()),
                    "total_reference_strokes": int(df.n_ref.sum())})
    pd.DataFrame(sil).to_csv(RR / "a3_digital_silence.csv", index=False)
    # floor / valley hypothesis tests (one-sided), Holm over labels x 2 per set
    tests = []
    for st in ("all", "matched"):
        rr = []
        for lab in MAIN:
            df = load(lab)
            for m, alt in (("floor_db", "less"), ("valley_db", "greater")):
                d = df[dcol(m, st)].to_numpy(float)
                d = d[np.isfinite(d)]
                rr.append({"label": lab, "stroke_set": st, "measure": m, "hypothesis": alt,
                           "median_d_nat": np.median(d), "n": len(d),
                           "n_in_direction": int((d < 0).sum() if alt == "less" else (d > 0).sum()),
                           "p_one_sided": P.wilcoxon_p(d, alt), "rank_biserial": P.rank_biserial(d)})
        rr = pd.DataFrame(rr)
        rr["p_holm"] = P.holm(rr.p_one_sided)
        tests.append(rr)
    tests = pd.concat(tests)
    tests.to_csv(RR / "a3_floor_valley_tests_v3.csv", index=False, float_format="%.5g")
    # accent tests (two-sided)
    acc = []
    for st in ("all", "matched"):
        for lab in MAIN:
            d = load(lab)[f"d_accent_{st}"].to_numpy(float)
            d = d[np.isfinite(d)]
            acc.append({"label": lab, "stroke_set": st, "median_d_accent_db": np.median(d),
                        "n_positive": int((d > 0).sum()), "n": len(d), "p": P.wilcoxon_p(d),
                        "rank_biserial": P.rank_biserial(d)})
    acc = pd.DataFrame(acc)
    acc["p_holm"] = P.holm(acc.p)
    acc.to_csv(RR / "a3_accent_tests_v3.csv", index=False, float_format="%.5g")

    print(main[main.stroke_set.isin(["-", "all"])][["label", "measure", "median_sd", "median_ci_lo",
                                                    "median_ci_hi", "mean_sd", "mean_loto_min", "mean_loto_max",
                                                    "mean_influential_tracks"]].round(3).to_string(index=False))
    print(U.round(3).to_string())
    print(pd.DataFrame(sil).to_string(index=False))
    print(tests.round(4).to_string(index=False))


# ---------------------------------------------------------------------- J3 families
def j3():
    raw = pd.read_csv(P.ROOT / "results" / "tables" / "benchmark_raw.csv")
    D = {m: load(f"v1_{m}") for m in P.MODELS}
    key = ("bs_roformer", "scnet")
    pairs = list(combinations(P.MODELS, 2))

    def wide(col, f=lambda s: s):
        return pd.DataFrame({m: f(D[m][col]) for m in P.MODELS})

    fam = {"B_analyst_abs_key": ({f"abs_d_{x}": wide(f"d_{x}", np.abs) for x in DESCS}, [key]),
           "Bstar_analyst_abs_all_pairs": ({f"abs_d_{x}": wide(f"d_{x}", np.abs) for x in DESCS}, pairs),
           "C_analyst_signed_key": ({f"d_{x}": wide(f"d_{x}") for x in DESCS}, [key])}
    onsetF = raw.pivot(index="track", columns="model", values="onset_f_measure").loc[P.tracks()]
    fam["B_analyst_abs_key"][0]["onset_f_measure_v1_50ms"] = onsetF
    fam["Bstar_analyst_abs_all_pairs"][0]["onset_f_measure_v1_50ms"] = onsetF
    out = []
    for name, (mets, prs) in fam.items():
        rows = []
        for mt, w in mets.items():
            for a, b in prs:
                d = (w[a] - w[b]).to_numpy(float)
                rows.append({"family": name, "metric": mt, "model_a": a, "model_b": b,
                             "median_a": w[a].median(), "median_b": w[b].median(), "median_diff": np.median(d),
                             "n_a_gt_b": int((d > 0).sum()), "n": len(d), "p": P.wilcoxon_p(d),
                             "rank_biserial": P.rank_biserial(d)})
        rows = pd.DataFrame(rows)
        rows["p_holm"] = P.holm(rows.p)
        out.append(rows)
    new = pd.concat(out, ignore_index=True)
    new.to_csv(RR / "j3_stats_v3.csv", index=False, float_format="%.5g")
    print(new[new.family != "Bstar_analyst_abs_all_pairs"].round(4).to_string(index=False))


# ---------------------------------------------------------------------- A4
def a4():
    bss = pd.read_csv(P.ROOT / "results" / "phase1" / "bss_drums_pertrack.csv")
    U = pd.read_csv(RR / "units.csv").set_index("measure")
    frames = {}
    for lab in MAIN:
        df = load(lab).copy()
        if lab.startswith("v1_"):
            b = bss[bss.model == lab[3:]].set_index("track").loc[P.tracks()]
            for k in ("sdr", "sir", "sar"):
                df[k] = b[k].to_numpy()
        p2 = pd.read_csv(P.OUT2 / "measure" / f"{lab}_by_track.csv").set_index("track").loc[P.tracks()]
        df["f_5ms"] = p2["f_5ms"]
        for m in DESCS + STROKE:
            df[f"abs_{m}_sd"] = df[dcol(m)].abs() / U.loc[m, "sd"]
        df["model"] = lab
        frames[lab] = df.reset_index()
    outs = ["f_5ms", "share_gt_10ms", "median_abs_dev_ms"] + [f"abs_{m}_sd" for m in DESCS + STROKE]
    preds = ["si_sdr", "sdr", "sir", "sar"]
    rng = np.random.default_rng(P.SEED)
    rows = []
    for set_name, labs in (("v1", V1), ("v1_with_htdemucs_b1", [V1[0], "htdemucs_b1", V1[2], V1[3]])):
        Dd = pd.concat([frames[l] for l in labs], ignore_index=True)
        trk = Dd.track.to_numpy()
        ut = np.array(P.tracks())
        tidx = {t: np.where(trk == t)[0] for t in ut}
        for pr in preds:
            for oc in outs:
                for lab in labs:
                    if set_name != "v1" and lab != "htdemucs_b1":
                        continue
                    g = Dd[Dd.model == lab]
                    ok = g[pr].notna() & g[oc].notna()
                    rho, p = spearmanr(g[pr][ok], g[oc][ok])
                    rows.append({"model_set": set_name, "predictor": pr, "outcome": oc, "scope": "within_model",
                                 "model": lab, "n": int(ok.sum()), "rho": rho, "p": p})
                x, y = Dd[pr], Dd[oc]
                xd = x - Dd.groupby("model")[pr].transform("mean")
                yd = y - Dd.groupby("model")[oc].transform("mean")
                xr = Dd.groupby("model")[pr].rank()
                yr = Dd.groupby("model")[oc].rank()
                for scope, xx, yy in (("pooled", x, y), ("demeaned", xd, yd), ("within_model_ranks", xr, yr)):
                    ok = (xx.notna() & yy.notna()).to_numpy()
                    rho, p = spearmanr(xx[ok], yy[ok])
                    r = {"model_set": set_name, "predictor": pr, "outcome": oc, "scope": scope, "model": "all",
                         "n": int(ok.sum()), "rho": rho, "p": p}
                    if scope == "pooled":
                        bs = []
                        xa, ya = xx.to_numpy(), yy.to_numpy()
                        for _ in range(2000):
                            sel = np.concatenate([tidx[t] for t in rng.choice(ut, len(ut))])
                            s = sel[np.isfinite(xa[sel]) & np.isfinite(ya[sel])]
                            bs.append(spearmanr(xa[s], ya[s])[0])
                        r["cluster_ci_lo"], r["cluster_ci_hi"] = np.percentile(bs, [2.5, 97.5])
                    rows.append(r)
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for (ms, pr, sc), idx in res.groupby(["model_set", "predictor", "scope"]).groups.items():
        res.loc[idx, "p_holm"] = P.holm(res.loc[idx, "p"])
    res["holm_family"] = res.model_set + ":" + res.predictor + ":" + res.scope
    res.to_csv(RR / "a4_sdr_vs_fidelity_v3.csv", index=False, float_format="%.5g")
    pd.concat([frames[l] for l in MAIN]).to_csv(P.OUT3 / "fig_sdr_vs_fidelity_v3.csv", index=False, float_format="%.6g")
    v = res[res.model_set == "v1"]
    for sc in ("pooled", "demeaned", "within_model_ranks"):
        print(sc)
        print(v[v.scope == sc].pivot(index="outcome", columns="predictor", values="rho").round(2).to_string())
    w = v[v.scope == "within_model"]
    print("within-model significant after Holm:", int((w.p_holm < 0.05).sum()), "of", len(w))
    print(w[w.predictor == "si_sdr"].pivot(index="outcome", columns="model", values="rho").round(2).to_string())


# ---------------------------------------------------------------------- A5
def a5():
    keys = ["f_5ms", "share_gt_10ms", "median_dev_ms", "median_abs_dev_ms",
            *[f"d_{x}" for x in DESCS], *[dcol(m) for m in STROKE]]
    rows = []
    for cond in ("", "_bl"):
        Dd = {}
        for l in V1:
            df = load(l + cond).copy()
            if "f_5ms" not in df:
                p2 = pd.read_csv(P.OUT2 / "measure" / f"{l}{cond}_by_track.csv").set_index("track").loc[P.tracks()]
                df["f_5ms"] = p2["f_5ms"]
            Dd[l] = df
        for other in V1[1:]:
            for k in keys:
                a, b = Dd["v1_spleeter"][k], Dd[other][k]
                ok = a.notna() & b.notna()
                d = (a - b)[ok].to_numpy()
                rows.append({"condition": "band_limited" if cond else "full_band", "contrast": f"spleeter - {other[3:]}",
                             "measure": k, "median_paired_diff": np.median(d), "p": P.wilcoxon_p(d),
                             "rank_biserial": P.rank_biserial(d)})
    r = pd.DataFrame(rows)
    r["p_holm"] = np.nan
    for c, idx in r.groupby("condition").groups.items():
        r.loc[idx, "p_holm"] = P.holm(r.loc[idx, "p"])
    w = r.pivot_table(index=["contrast", "measure"], columns="condition", values=["median_paired_diff", "p_holm", "rank_biserial"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    sf, sb = w.p_holm_full_band < 0.05, w.p_holm_band_limited < 0.05
    same = np.sign(w.median_paired_diff_full_band) == np.sign(w.median_paired_diff_band_limited)
    w["verdict"] = np.select([sf & sb & same, sf & ~sb, ~sf & sb, sf & sb & ~same],
                             ["survives", "vanishes under band limit", "appears under band limit", "reverses"],
                             "n.s. in both")
    r.to_csv(RR / "a5_bandlimit_tests_v3.csv", index=False, float_format="%.5g")
    w.to_csv(RR / "a5_bandlimit_summary_v3.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(w[["contrast", "measure", "median_paired_diff_full_band", "median_paired_diff_band_limited",
                 "p_holm_full_band", "p_holm_band_limited", "verdict"]].round(3).to_string(index=False))


# ---------------------------------------------------------------------- B1 vs v1
def b1():
    b, v = load("htdemucs_b1"), load("v1_htdemucs")
    bss = pd.read_csv(P.ROOT / "results" / "phase1" / "bss_drums_pertrack.csv")
    bb = bss[bss.model == "htdemucs"].set_index("track").loc[P.tracks()]
    for k in ("sdr", "sir", "sar"):
        v[k] = bb[k].to_numpy()
    cols = ["si_sdr", "sdr", "sir", "sar", "share_gt_10ms", "median_abs_dev_ms", "median_dev_ms",
            *[f"d_{x}" for x in DESCS], *[dcol(m) for m in STROKE], *[dcol(m, "matched") for m in STROKE]]
    U = pd.read_csv(RR / "units.csv").set_index("measure")
    rows = []
    for c in cols:
        d = (b[c] - v[c]).to_numpy(float)
        rows.append({"measure": c, "median_v1": v[c].median(), "median_b1": b[c].median(),
                     "median_abs_track_diff": np.nanmedian(np.abs(d)), "p_wilcoxon": P.wilcoxon_p(d),
                     "rank_biserial": P.rank_biserial(d), "between_track_sd_v1": v[c].std(ddof=1)})
    out = pd.DataFrame(rows)
    out["median_abs_diff_over_between_track_sd"] = out.median_abs_track_diff / out.between_track_sd_v1
    out.to_csv(RR / "b1_vs_v1_measures_v3.csv", index=False, float_format="%.5g")
    print(out.round(4).to_string(index=False))


# ---------------------------------------------------------------------- B2
def b2():
    A, B = "scnet_xl_ihf", "scnet_masked_xl_ihf"
    da, db = load(A), load(B)
    p2a = pd.read_csv(P.OUT2 / "measure" / f"{A}_by_track.csv").set_index("track").loc[P.tracks()]
    p2b = pd.read_csv(P.OUT2 / "measure" / f"{B}_by_track.csv").set_index("track").loc[P.tracks()]
    for df, p2 in ((da, p2a), (db, p2b)):
        for k in ("f_5ms", "f_10ms", "iqr_dev_ms"):
            df[k] = p2[k]
    key = ["d_attack_slope", "d_env_flux", "median_dev_ms"]
    sec = ["si_sdr", "sdr", "sir", "sar", "f_5ms", "f_10ms", "share_gt_10ms", "median_abs_dev_ms", "iqr_dev_ms",
           *[f"d_{x}" for x in DESCS if x not in ("attack_slope", "env_flux")],
           *[dcol(m) for m in STROKE], *[dcol(m, "matched") for m in STROKE]]
    out = []
    for fam, keys in (("key", key), ("secondary", sec)):
        rr = []
        for k in keys:
            x, y = db[k], da[k]
            d = (x - y).to_numpy(float)
            rr.append({"family": fam, "measure": k, "median_direct": y.median(), "median_masked": x.median(),
                       "median_paired_diff": np.nanmedian(d), "n_masked_gt_direct": int(np.nansum(d > 0)),
                       "n": int(np.isfinite(d).sum()), "p": P.wilcoxon_p(d), "rank_biserial": P.rank_biserial(d)})
        rr = pd.DataFrame(rr)
        rr["p_holm"] = P.holm(rr.p)
        out.append(rr)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(RR / "b2_masking_tests_v3.csv", index=False, float_format="%.5g")
    print(res.round(4).to_string(index=False))


# ---------------------------------------------------------------------- B3
def b3():
    import pickle
    M4 = ["spleeter", "htdemucs", "bs_roformer", "scnet"]
    F1 = [f"f1_o{k:02d}" for k in range(12)]
    SW = [f"swap_o{k:02d}" for k in range(12)]
    F2 = ["f1_o00", "f2_s1", "f2_s2"]
    SH = [f"d_{x}" for x in DESCS] + [f"d_stroke_{m}_all" for m in ("peak_db", "floor_db", "valley_db", "rise_ms")] + ["d_accent_all"]
    REFK = {f"d_{x}": f"ref_{x}" for x in DESCS}
    REFK.update({f"d_stroke_{m}_all": f"ref_stroke_{m}_all" for m in ("peak_db", "floor_db", "valley_db", "rise_ms")})
    REFK["d_accent_all"] = "ref_accent_all"
    man = pd.read_csv(P.OUT2 / "b3_manifest.csv")
    recs, refv = [], {}
    f3 = []
    for t in man.track:
        for m in M4:
            r = pickle.load(open(P.DATA3 / "b3" / m / f"{P.safe_name(t)}.pkl", "rb"))
            if t not in refv:
                refv[t] = {k: r["runs"]["f1_o00"][v] for k, v in REFK.items()}
            for name, conds in (("F1", F1), ("F2", F2), ("swap", SW)):
                vals = pd.DataFrame([{k: r["runs"][c][k] for k in SH} for c in conds if c in r["runs"]])
                if len(vals) < 2:
                    continue
                row = {"model": m, "track": t, "factor": name}
                for k in SH:
                    row[f"sd_{k}"] = vals[k].std(ddof=1)
                    row[f"meanabs_{k}"] = vals[k].abs().mean()
                recs.append(row)
            for L, c, full in ((5, "f3_L05", "full_R"), (10, "f3_L10", "full_pm1"), (20, "f3_L20", "full_pm1")):
                row = {"model": m, "track": t, "length_s": L}
                for k in SH:
                    row[f"diff_{k}"] = r["runs"][c][k] - r[full][k]
                f3.append(row)
    shp = pd.DataFrame(recs)
    f3 = pd.DataFrame(f3)
    rv = pd.DataFrame(refv).T
    sd_ref = {k: rv[k].astype(float).std(ddof=1) for k in SH}
    for k in SH:
        shp[f"sdunits_sd_{k}"] = shp[f"sd_{k}"] / sd_ref[k]
        shp[f"sdunits_meanabs_{k}"] = shp[f"meanabs_{k}"] / sd_ref[k]
        f3[f"sdunits_diff_{k}"] = f3[f"diff_{k}"] / sd_ref[k]
    shp.to_csv(RR / "b3_shape_drift_by_track_v3.csv", index=False, float_format="%.5g")
    f3.to_csv(RR / "b3_factor3_by_track_v3.csv", index=False, float_format="%.5g")
    # overlap swap: 20 tests (timing from Phase 2, unchanged definition; shape v3), Holm over 20
    tim = pd.read_csv(P.OUT2 / "b3_timing_drift_by_track.csv")
    piv = tim.pivot_table(index="track", columns=["model", "factor"], values="median_sd_ms")
    comps = [("bs_roformer", "F1", "bs_roformer", "swap", "BS-Roformer: overlap 2 (native) vs 4"),
             ("scnet", "F1", "scnet", "swap", "SCNet-XL: overlap 4 (native) vs 2"),
             ("bs_roformer", "swap", "scnet", "F1", "at overlap 4: BS-Roformer vs SCNet-XL"),
             ("bs_roformer", "F1", "scnet", "swap", "at overlap 2: BS-Roformer vs SCNet-XL"),
             ("bs_roformer", "F1", "scnet", "F1", "native: BS-Roformer (2) vs SCNet-XL (4)")]
    rows = []
    for m1, f1, m2, f2, lab in comps:
        tabs = [("timing median SD (ms)", piv)]
        for v in ("attack_slope", "env_flux", "variation"):
            tabs.append((f"shape SD units {v}", shp.pivot_table(index="track", columns=["model", "factor"],
                                                               values=f"sdunits_sd_d_{v}")))
        for meas, tab in tabs:
            x, y = tab[(m1, f1)], tab[(m2, f2)]
            d = (x - y).to_numpy(float)
            rows.append({"comparison": lab, "measure": meas, "median_a": x.median(), "median_b": y.median(),
                         "median_paired_diff": np.nanmedian(d), "n": int(np.isfinite(d).sum()),
                         "p": P.wilcoxon_p(d), "rank_biserial": P.rank_biserial(d)})
    sw = pd.DataFrame(rows)
    sw["p_holm_20"] = P.holm(sw.p)
    sw.to_csv(RR / "b3_overlap_swap_tests_v3.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(shp.groupby(["model", "factor"])[[f"sdunits_sd_{k}" for k in SH]].median().round(3).T.to_string())
        print(shp.groupby(["model", "factor"])[[f"sdunits_meanabs_{k}" for k in SH]].median().round(3).T.to_string())
        print(f3.groupby(["model", "length_s"])[[f"sdunits_diff_{k}" for k in SH]].apply(lambda d: d.abs().median()).round(3).to_string())
        print(sw.round(4).to_string(index=False))


# ---------------------------------------------------------------------- correlations / rank agreement
def corr():
    rows = []
    for st in ("all", "matched"):
        for lab in MAIN:
            df = load(lab)
            for y in ("valley_db", "accent", "floor_db", "peak_db"):
                yc = dcol(y, st)
                ok = df.d_variation.notna() & df[yc].notna()
                rho, p = spearmanr(df.d_variation[ok], df[yc][ok])
                rows.append({"stroke_set": st, "label": lab, "x": "d_variation (v3)", "y": yc, "rho": rho, "p": p, "n": int(ok.sum())})
    c = pd.DataFrame(rows)
    c["p_holm"] = np.nan
    for st, idx in c.groupby("stroke_set").groups.items():
        c.loc[idx, "p_holm"] = P.holm(c.loc[idx, "p"])
    c.to_csv(RR / "a3_variation_correlations_v3.csv", index=False, float_format="%.5g")
    # rank agreement of the variation bias between the four default runs
    piv = pd.DataFrame({m: load(f"v1_{m}").d_variation for m in P.MODELS})
    cm = piv.corr("spearman")
    off = cm.values[np.triu_indices(4, 1)]
    cm.to_csv(RR / "a1c_rank_agreement_v3.csv", float_format="%.4g")
    print(c[c.stroke_set == "all"].round(3).to_string(index=False))
    print(cm.round(3).to_string(), "\nmean off-diagonal:", round(float(np.mean(off)), 3))


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    steps = {"a3": a3_j2, "j3": j3, "a4": a4, "a5": a5, "b1": b1, "b2": b2, "b3": b3, "corr": corr}
    for k, f in steps.items():
        if "all" in which or k in which:
            print("=" * 30, k)
            f()
