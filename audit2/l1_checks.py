"""Data checks on the stored caches and result files.

A  digitally silent estimate strokes (peak -240 dB): pair_v3 NaNs floor/valley/pre/jump but not
   the peak. Which strokes; effect on accent contrast (all strokes), the level-dependence slope
   (C5) and the Holm-tested accent change.
B  cache freshness: onset_table.pkl rebuilt from the measure caches; measure3.summarize re-run on
   every cache against results/phase3/rerun/measure/*_by_track.csv; strokes_v3 / dynamics_v3
   recomputed from audio for 2 tracks x 2 labels against the cache.
C  selection: all vs matched strokes (share matched; matched-minus-all difference of d(peak) and of
   accent contrast), and where the stroke peak sits in its window (at the window start = louder
   signal before the stroke).
D  small definitional checks: dependence of the reference SD unit on the paired estimate (floor);
   hard-coded '/50' denominators in printed summary tables; the hard-coded 0.1 in C3; Figure 4(a) pools
   onsets while Figure 2(b) uses per-track medians; the 20-ms look-back limit of rise time.

    OMP_NUM_THREADS=1 python audit2/l1_checks.py   -> results/audit2/l1_*.csv, l1_summary.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import pickle  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _paths as A  # noqa: E402

A.project()
import _p3 as P  # noqa: E402
import measure3 as M3  # noqa: E402
import c5_c8_strokes as C5  # noqa: E402
from mss_eval import strokes as SK, analyst  # noqa: E402

res = {}
TR = A.tracks()
LAB = P.LABELS
MEAS = A.P3DATA / "measure"


def cache(lab, t):
    return pickle.load(open(MEAS / lab / f"{A.safe_name(t)}.pkl", "rb"))


# ------------------------------------------------------------------ A silent estimate strokes
rows = []
for lab in sorted(p.name for p in MEAS.iterdir()):
    for t in TR:
        r = cache(lab, t)
        bad = r["est"]["peak_db"] < SK.SILENCE_DB
        for k in np.where(bad)[0]:
            rows.append({"label": lab, "track": t, "i_ref": int(k), "t_ref_s": float(r["ref_onsets"][k]),
                         "ref_peak_db": float(r["ref"]["peak_db"][k]), "est_peak_db": float(r["est"]["peak_db"][k]),
                         "matched": bool(r["matched"][k])})
sil = pd.DataFrame(rows)
sil.to_csv(A.OUT / "l1_silent_estimate_peaks.csv", index=False)
res["A_silent_estimate_peaks_by_label"] = sil.groupby("label").size().to_dict()

U = pd.read_csv(A.RES / "phase3" / "rerun" / "units.csv").set_index("measure")
acc_rows, slope_rows = [], []
df_on = P.onset_table()
for lab in ("v1_spleeter", "v1_bs_roformer"):
    bt = pd.read_csv(A.RES / "phase3" / "rerun" / "measure" / f"{lab}_by_track.csv").set_index("track").loc[TR]
    new = []
    for t in TR:
        r = cache(lab, t)
        pr, pe = r["ref"]["peak_db"], r["est"]["peak_db"]
        ok = np.isfinite(pr) & np.isfinite(pe) & (pe >= SK.SILENCE_DB)
        new.append(np.std(pe[ok], ddof=1) - np.std(pr[ok], ddof=1))
    new = np.array(new)
    old = bt.d_accent_all.to_numpy()
    for name, v in (("stored", old), ("silent strokes excluded", new)):
        acc_rows.append({"label": lab, "version": name, "median_d_accent_all_db": float(np.median(v)),
                         "mean_d_accent_all_db": float(np.mean(v)), "mean_sd_units": float(np.mean(v) / U.loc["accent", "sd"]),
                         "median_sd_units": float(np.median(v) / U.loc["accent", "sd"]),
                         "p_wilcoxon_two_sided": P.wilcoxon_p(v), "rank_biserial": P.rank_biserial(v),
                         "n_tracks_changed": int(np.sum(np.abs(new - old) > 1e-4 * np.maximum(np.abs(old), 1)))})
    # C5 slopes, all strokes, with and without the silent strokes
    g = df_on[df_on.label == lab].copy()
    g["ref_peak_rel"] = g.peak_rel_db
    silent = g.track.astype(str) + "|" + g.i_ref.astype(str)
    sl_ids = set((sil[sil.label == lab].track + "|" + sil[sil.label == lab].i_ref.astype(str)).tolist())
    gx = g[~silent.isin(sl_ids)]
    for name, h in (("stored (all strokes)", g), ("silent strokes excluded", gx)):
        s = C5.slopes(h, "d_peak_db", "ref_peak_rel")
        b, _, _ = C5.fe_slope(h, "d_peak_db", "ref_peak_rel", np.random.default_rng(0), nb=10)
        aff = s[s.track.isin(sil[sil.label == lab].track)]
        slope_rows.append({"label": lab, "version": name, "median_within_track_slope": float(s.slope.median()),
                           "pooled_fe_slope": b, "n_tracks_positive": int((s.slope > 0).sum()),
                           "affected_tracks_slopes": "; ".join(f"{a.track[:28]}: {a.slope:+.3f}" for a in aff.itertuples())})
acc = pd.DataFrame(acc_rows)
acc.to_csv(A.OUT / "l1_silent_effect_accent.csv", index=False, float_format="%.6g")
slp = pd.DataFrame(slope_rows)
slp.to_csv(A.OUT / "l1_silent_effect_slope.csv", index=False, float_format="%.6g")
res["A_accent"] = acc.to_dict("records")
res["A_slope"] = slp.to_dict("records")

# ------------------------------------------------------------------ B cache freshness
# B1 onset_table rebuilt
feats = pickle.load(open(A.P3DATA / "a1_features_v3.pkl", "rb"))
parts = []
for lab in LAB:
    for t in TR:
        r = cache(lab, t)
        d = {"label": lab, "track": t, "i_ref": np.arange(len(r["ref_onsets"])), "t": r["ref_onsets"],
             "matched": r["matched"], "dev_ms": r["dev_ms"]}
        for m in P.SMEAS3:
            d[f"ref_{m}"] = r["ref"][m]
            d[f"d_{m}"] = r["est"][m] - r["ref"][m]
        parts.append(pd.DataFrame(d))
reb = pd.concat(parts, ignore_index=True).merge(feats.drop(columns=["t"]), on=["track", "i_ref"], how="left")
reb["displaced"] = reb.matched & (reb.dev_ms.abs() > 10)
same_cols = [c for c in reb.columns if c in df_on.columns]
mism = {}
for c in same_cols:
    a, b = reb[c].to_numpy(), df_on[c].to_numpy()
    if a.dtype.kind in "fc":
        mism[c] = int(np.sum(~((a == b) | (np.isnan(a) & np.isnan(b)))))
    else:
        mism[c] = int(np.sum(a != b))
res["B1_onset_table_rebuild"] = {"shape_equal": reb.shape == df_on.shape, "columns_with_mismatch": {k: v for k, v in mism.items() if v}}
# B2 summaries vs stored by-track CSVs
rows = []
for lab in sorted(p.name for p in MEAS.iterdir()):
    f = A.RES / "phase3" / "rerun" / "measure" / f"{lab}_by_track.csv"
    if not f.exists():
        continue
    st = pd.read_csv(f).set_index("track").loc[TR]
    nw = pd.DataFrame([M3.summarize(cache(lab, t)) for t in TR]).set_index("track")
    for c in st.columns:
        if c not in nw or not pd.api.types.is_numeric_dtype(st[c]):
            continue
        x, y = nw[c].to_numpy(float), st[c].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        rel = np.abs(x[ok] - y[ok]) / np.maximum(np.abs(y[ok]), 1e-9)
        rows.append({"label": lab, "column": c, "max_rel_diff": float(rel.max()) if len(rel) else 0.0,
                     "nan_mismatch": int((np.isfinite(x) != np.isfinite(y)).sum())})
b2 = pd.DataFrame(rows)
b2.to_csv(A.OUT / "l1_by_track_vs_cache.csv", index=False, float_format="%.3g")
res["B2_by_track_vs_cache"] = {"n_columns_checked": len(b2), "max_rel_diff": float(b2.max_rel_diff.max()),
                               "n_nan_mismatch": int(b2.nan_mismatch.sum()),
                               "worst": b2.sort_values("max_rel_diff").tail(3).to_dict("records")}
# B3 recompute from audio with the current code (2 tracks x 2 labels)
import soundfile as sf  # noqa: E402
rows = []
for lab, mdir in (("v1_spleeter", "spleeter"), ("v1_bs_roformer", "bs_roformer")):
    for t in (TR[3], TR[41]):
        r = cache(lab, t)
        est = sf.read(str(A.STEMS_V1 / mdir / A.safe_name(t) / "drums.wav"), always_2d=True, dtype="float64")[0]
        ref = sf.read(str(A.MUSDB_TEST / t / "drums.wav"), always_2d=True, dtype="float64")[0]
        n = min(len(est), len(ref))
        est, ref = est[:n], ref[:n]
        rs, es = SK.strokes_v3(ref, A.SR, r["ref_onsets"]), SK.strokes_v3(est, A.SR, r["ref_onsets"])
        rs, es, _ = SK.pair_v3(rs, es)
        dv = analyst.dynamics_v3(est, ref, A.SR)
        mx = 0.0
        for m in P.SMEAS3:
            for a, b in ((rs[m], r["ref"][m]), (es[m], r["est"][m])):
                ok = np.isfinite(a) & np.isfinite(b)
                mx = max(mx, float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0)
                assert np.array_equal(np.isfinite(a), np.isfinite(b))
        dmx = max(abs(dv[k] - r["desc"][k]) for k in dv)
        rows.append({"label": lab, "track": t, "max_abs_diff_strokes": mx, "max_abs_diff_descriptors": dmx})
res["B3_recompute_from_audio"] = rows

# ------------------------------------------------------------------ C selection
rows = []
for lab in LAB:
    bt = pd.read_csv(A.RES / "phase3" / "rerun" / "measure" / f"{lab}_by_track.csv").set_index("track").loc[TR]
    g = df_on[df_on.label == lab]
    rows.append({"label": lab, "share_ref_strokes_matched_pooled": float(g.matched.mean()),
                 "median_track_share_matched": float(bt.n_matched.div(bt.n_ref).median()),
                 "median_d_peak_all": float(bt.d_peak_db_all.median()), "median_d_peak_matched": float(bt.d_peak_db_matched.median()),
                 "median_d_accent_all": float(bt.d_accent_all.median()), "median_d_accent_matched": float(bt.d_accent_matched.median()),
                 "median_d_valley_all": float(bt.d_valley_db_all.median()), "median_d_valley_matched": float(bt.d_valley_db_matched.median()),
                 "pooled_median_d_peak_unmatched": float(g[~g.matched].d_peak_db.median()),
                 "pooled_median_d_peak_matched": float(g[g.matched].d_peak_db.median())})
sel = pd.DataFrame(rows)
sel.to_csv(A.OUT / "l1_selection_all_vs_matched.csv", index=False, float_format="%.5g")
res["C_selection"] = sel.to_dict("records")
# where the peak sits: at the window start s_k (within one frame) = the envelope is falling into the window
rows = []
for lab in ("v1_spleeter", "v1_htdemucs", "v1_bs_roformer", "v1_scnet"):
    nr = ne = n = 0
    for t in TR:
        r = cache(lab, t)
        s = np.maximum(r["ref_onsets"] - 0.020, 0)
        fr = 64 / A.SR
        nr += int(np.sum(r["ref"]["peak_t"] - s < 1.01 * fr))
        ne += int(np.sum(r["est"]["peak_t"] - s < 1.01 * fr))
        n += len(s)
    rows.append({"label": lab, "n_strokes": n, "share_ref_peak_at_window_start": nr / n, "share_est_peak_at_window_start": ne / n})
res["C_peak_at_window_start"] = rows

# ------------------------------------------------------------------ D small checks
fl = {lab: pd.read_csv(A.RES / "phase3" / "rerun" / "measure" / f"{lab}_by_track.csv").set_index("track").loc[TR].ref_floor_db_all
      for lab in LAB}
fl = pd.DataFrame(fl)
res["D_ref_floor_unit_dependence"] = {"tracks_where_ref_floor_median_differs_between_labels":
                                      int((fl.max(axis=1) - fl.min(axis=1) > 1e-9).sum()),
                                      "max_diff_db": float((fl.max(axis=1) - fl.min(axis=1)).max()),
                                      "sd_by_label": fl.std(ddof=1).round(4).to_dict()}
c5 = pd.read_csv(A.RES / "phase3" / "c5_expansion.csv")
fv = pd.read_csv(A.RES / "phase3" / "rerun" / "a3_floor_valley_tests_v3.csv")
ac = pd.read_csv(A.RES / "phase3" / "rerun" / "a3_accent_tests_v3.csv")
c4t = pd.read_csv(A.RES / "phase3" / "c4_tests.csv")
res["D_denominator_50"] = {"c5_rows_with_n_tracks_below_50": int((c5.n_tracks < 50).sum()),
                           "c5_min_n_tracks": int(c5.n_tracks.min()),
                           "floor_valley_tests_rows_n_below_50": int((fv.n < 50).sum()),
                           "accent_tests_rows_n_below_50": int((ac.n < 50).sum()),
                           "c4_tests_rows_n_below_50": int((c4t.n < 50).sum())}
c3 = pd.read_csv(A.RES / "phase3" / "c3_displaced_large_change.csv")
g = df_on[df_on.matched]
act = {}
for lab in LAB:
    nd = g[(g.label == lab) & ~g.displaced]
    act[lab] = {c: float((nd[c].abs() > nd[c].abs().quantile(0.9)).mean()) for c in ["d_rise_ms", "d_peak_db", "d_pre_db", "d_jump_db"]}
res["D_c3_nondisplaced_large_share_actual_vs_hardcoded_0.1"] = act
h = pd.read_csv(A.RES / "phase2" / "fig_timing_error_distribution.csv")
rows = []
for lab in ("v1_spleeter", "v1_htdemucs", "v1_bs_roformer", "v1_scnet"):
    gg = h[h.label == lab]
    pooled = gg.loc[(gg.bin_lo_ms >= 10) | (gg.bin_hi_ms <= -10), "count"].sum() / gg["count"].sum()
    bt = pd.read_csv(A.RES / "phase2" / "measure" / f"{lab}_by_track.csv")
    pooled_exact = (bt.share_gt_10ms * bt.n_pairs).sum() / bt.n_pairs.sum()
    rows.append({"label": lab, "fig4a_pooled_share_gt10_from_histogram": float(pooled),
                 "pooled_share_gt10_from_by_track": float(pooled_exact),
                 "median_over_tracks_share_gt10 (fig2b)": float(bt.share_gt_10ms.median())})
res["D_fig4a_pooled_vs_fig2b_median"] = rows
rr = []
for lab in ("v1_bs_roformer",):
    for t in TR:
        rr.append(cache(lab, t)["ref"]["rise_ms"])
rr = np.concatenate(rr)
res["D_rise_lookback"] = {"ref_strokes": int(np.isfinite(rr).sum()), "share_rise_ge_14ms": float(np.nanmean(rr >= 14)),
                          "max_rise_ms": float(np.nanmax(rr)), "p99_rise_ms": float(np.nanpercentile(rr, 99))}
A.dump("l1_summary.json", res)
for k, v in res.items():
    print(k, v)
