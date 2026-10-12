"""B3 analysis: timing and shape drift of passage R across how the file was cut.

Reads data/phase2/b3_measure/<model>/<track>.pkl (b3_measure.py) and the stored spans.

  timing drift (F1)  per reference onset in R matched at all 12 chunk positions: SD (ddof 1)
                     and range of its estimate time; compared with the onset's mean |dev|
                     against the truth and with the repeat-run difference |o00 - rep_o00|
  sub-frame (F2)     the same over the 3 sub-hop window shifts (o = 0); swapped overlap (swap)
  F3                 excerpt-first minus full-track-first (R cut from the full-track output),
                     per measure and input length (shape differences also in units of the
                     between-passage SD of the reference's value on R); flag lengths at or
                     below the MSST reflect-padding thresholds (BS-Roformer 11 s, SCNet-XL 16.5 s)
  HT-Demucs          run-to-run: audio-separator defaults (r1 vs r2) vs B1 settings
                     (o00 vs rep_o00); sample and measure differences within R
  sanity             the o = 0 window output inside R vs R cut from the full-track output
The shape drift of Experiment 2 and the overlap-swap tests are computed with the v3 measures
by scripts/phase3/c0_rerun.py (step b3).

    python b3_analyze.py
Outputs: results/phase2/b3_timing_drift_by_track.csv, b3_factor3_by_track.csv,
         b3_htdemucs_runs.csv, b3_sanity_window_vs_full.csv, b3_per_onset_drift.csv.gz
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import soundfile as sf

import _p2 as P

M4 = ["spleeter", "htdemucs", "bs_roformer", "scnet"]
F1 = [f"f1_o{k:02d}" for k in range(12)]
SW = [f"swap_o{k:02d}" for k in range(12)]
F2 = ["f1_o00", "f2_s1", "f2_s2"]
SM = ["peak_db", "floor_db", "valley_db", "rise_ms", "decay_ms"]
SHAPE = [*[f"d_{x}" for x in P.DESCS], *[f"d_stroke_{m}" for m in SM], "d_accent_db"]
REFLECT = {"bs_roformer": 11.0, "scnet": 16.5}
B3 = P.P2DATA / "b3"


def load(model, track):
    f = P.P2DATA / "b3_measure" / model / f"{P.safe_name(track)}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def shape_vec(run):
    v = {f"d_{x}": run["d_profile"][x] for x in P.DESCS}
    v.update({f"d_stroke_{m}": run[f"d_stroke_{m}"] for m in SM})
    v["d_accent_db"] = run["d_accent_db"]
    return v


def ref_values(r):
    """Reference values on R (for SD units), from the R +/- 1 s span entry."""
    R0, R1 = r["R"]
    ref = r["ref"][f"{R0 - P.SR}_{R1 + P.SR}"]
    v = {f"d_{x}": ref["profile"][x] for x in P.DESCS}
    v.update({f"d_stroke_{m}": ref["strokes_median"].get(m, np.nan) for m in SM})
    v["d_accent_db"] = ref["accent_db"]
    return v


def timing_drift(r, conds):
    runs = r["runs"]
    if not all(c in runs for c in conds):
        return None
    E = np.vstack([runs[c]["est_time_per_ref_onset"] for c in conds]) * 1000     # ms
    ref = r["ref"][f"{r['R'][0] - P.SR}_{r['R'][1] + P.SR}"]["onsets_R"] * 1000
    ok = np.all(np.isfinite(E), axis=0)
    if ok.sum() == 0:
        return None
    sd = E[:, ok].std(axis=0, ddof=1)
    rg = E[:, ok].max(0) - E[:, ok].min(0)
    mad = np.abs(E[:, ok] - ref[ok]).mean(0)
    return {"n_onsets_R": E.shape[1], "n_matched_all": int(ok.sum()),
            "median_sd_ms": np.median(sd), "p90_sd_ms": np.percentile(sd, 90),
            "median_range_ms": np.median(rg), "max_range_ms": rg.max(),
            "share_range_gt_1ms": np.mean(rg > 1), "share_range_gt_5ms": np.mean(rg > 5),
            "median_mean_abs_dev_ms": np.median(mad), "per_onset_sd": sd, "per_onset_range": rg,
            "per_onset_mean_abs_dev": mad}


def repeat_floor(r, a="f1_o00", b="rep_o00"):
    x, y = r["runs"][a]["est_time_per_ref_onset"], r["runs"][b]["est_time_per_ref_onset"]
    ok = np.isfinite(x) & np.isfinite(y)
    d = np.abs(x[ok] - y[ok]) * 1000
    return {"repeat_median_abs_ms": np.median(d) if len(d) else np.nan,
            "repeat_max_abs_ms": d.max() if len(d) else np.nan,
            "repeat_matched_status_changes": int((np.isfinite(x) != np.isfinite(y)).sum())}


def audio_diff(model, track, c1, c2, R0, R1):
    x = [sf.read(str(B3 / model / P.safe_name(track) / f"{c}.wav"), dtype="float64", always_2d=True)[0]
         for c in (c1, c2)]
    a = R0 - (R0 - P.SR)
    s1, s2 = x[0][a:a + (R1 - R0)], x[1][a:a + (R1 - R0)]
    d = s1 - s2
    return {"max_abs_diff": float(np.abs(d).max()), "rms_diff": float(np.sqrt(np.mean(d ** 2))),
            "diff_rel_db": float(10 * np.log10(np.mean(d ** 2) / np.mean(s1 ** 2))) if np.any(d) else -np.inf,
            "bit_identical": bool(np.array_equal(s1, s2))}


def main():
    man = pd.read_csv(P.OUT2 / "b3_manifest.csv")
    man = man[man.qualifies]
    tim, f3, htd, sanity = [], [], [], []
    ref_rows = []
    per_onset = []
    for _, mr in man.iterrows():
        t = mr.track
        for m in M4:
            r = load(m, t)
            if r is None or not r["runs"]:
                continue
            R0, R1 = r["R"]
            rv = ref_values(r)
            ref_rows.append({"model": m, "track": t, **rv})
            for name, conds in (("F1", F1), ("F2", F2), ("swap", SW)):
                td = timing_drift(r, conds)
                if td is None:
                    continue
                row = {"model": m, "track": t, "factor": name,
                       **{k: v for k, v in td.items() if not k.startswith("per_onset")}}
                if name == "F1" and "rep_o00" in r["runs"]:
                    row.update(repeat_floor(r))
                tim.append(row)
                if name == "F1":
                    per_onset.append(pd.DataFrame({"model": m, "track": t, "sd_ms": td["per_onset_sd"],
                                                   "range_ms": td["per_onset_range"],
                                                   "mean_abs_dev_ms": td["per_onset_mean_abs_dev"]}))
            # F3
            for L, c, full in ((5, "f3_L05", "full_R"), (10, "f3_L10", "full_pm1"), (20, "f3_L20", "full_pm1")):
                if c not in r["runs"]:
                    continue
                ex, fu = r["runs"][c], r[full]
                x, y = ex["est_time_per_ref_onset"], fu["est_time_per_ref_onset"]
                ok = np.isfinite(x) & np.isfinite(y)
                row = {"model": m, "track": t, "length_s": L,
                       "below_reflect_threshold": L <= REFLECT.get(m, 0),
                       "median_abs_onset_diff_ms": np.median(np.abs(x[ok] - y[ok]) * 1000) if ok.any() else np.nan,
                       "max_abs_onset_diff_ms": np.max(np.abs(x[ok] - y[ok]) * 1000) if ok.any() else np.nan,
                       "matched_status_changes": int((np.isfinite(x) != np.isfinite(y)).sum())}
                sa, sb = shape_vec(ex), shape_vec(fu)
                for k in SHAPE:
                    row[f"diff_{k}"] = sa[k] - sb[k]
                f3.append(row)
            if m == "htdemucs":
                for c1, c2, lab in (("default_o00_r1", "default_o00_r2", "default"), ("f1_o00", "rep_o00", "b1_seeded")):
                    if c1 in r["runs"] and c2 in r["runs"]:
                        rr = {"track": t, "setting": lab, **audio_diff(m, t, c1, c2, R0, R1)}
                        x, y = r["runs"][c1]["est_time_per_ref_onset"], r["runs"][c2]["est_time_per_ref_onset"]
                        ok = np.isfinite(x) & np.isfinite(y)
                        rr["median_abs_onset_diff_ms"] = np.median(np.abs(x[ok] - y[ok])) * 1000 if ok.any() else np.nan
                        rr["max_abs_onset_diff_ms"] = np.max(np.abs(x[ok] - y[ok])) * 1000 if ok.any() else np.nan
                        a, b = shape_vec(r["runs"][c1]), shape_vec(r["runs"][c2])
                        rr.update({f"absdiff_{k}": abs(a[k] - b[k]) for k in SHAPE})
                        htd.append(rr)
            if m in ("bs_roformer", "scnet", "spleeter", "htdemucs") and "f1_o00" in r["runs"]:
                # sanity: o = 0 window output inside R vs R cut from the full-track output
                x = sf.read(str(B3 / m / P.safe_name(t) / "f1_o00.wav"), dtype="float64", always_2d=True)[0][P.SR:P.SR + (R1 - R0)]
                from b3_measure import full_stem
                y = full_stem(m, t)[R0:R1]
                n = min(len(x), len(y))
                from scipy.signal import correlate, correlation_lags
                xc = correlate(P.mono(x[:n]), P.mono(y[:n]), mode="full", method="fft")
                lags = correlation_lags(n, n, mode="full")
                win = np.abs(lags) <= 1000                   # +/-1000 samples is ample
                lag = int(lags[win][np.argmax(xc[win])])
                sanity.append({"model": m, "track": t,
                               "snr_window_vs_fulltrack_db": float(10 * np.log10(np.sum(y[:n] ** 2) / np.sum((x[:n] - y[:n]) ** 2))),
                               "xcorr_peak_lag_samples": lag})
    tim, f3 = pd.DataFrame(tim), pd.DataFrame(f3)
    refv = pd.DataFrame(ref_rows).drop_duplicates("track")
    sd_ref = {k: refv[k].std(ddof=1) for k in SHAPE}
    if len(f3):
        for k in SHAPE:
            f3[f"sdunits_diff_{k}"] = f3[f"diff_{k}"] / sd_ref[k]
    pre = "b3_"
    tim.to_csv(P.OUT2 / f"{pre}timing_drift_by_track.csv", index=False, float_format="%.5g")
    f3.to_csv(P.OUT2 / f"{pre}factor3_by_track.csv", index=False, float_format="%.5g")
    pd.DataFrame(htd).to_csv(P.OUT2 / f"{pre}htdemucs_runs.csv", index=False, float_format="%.5g")
    pd.DataFrame(sanity).to_csv(P.OUT2 / f"{pre}sanity_window_vs_full.csv", index=False, float_format="%.5g")
    if per_onset:
        po = pd.concat(per_onset)
        po.to_csv(P.OUT2 / f"{pre}per_onset_drift.csv.gz", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tim.groupby(["model", "factor"]).median(numeric_only=True).round(3).T.to_string())
        if len(f3):
            print(f3.groupby(["model", "length_s"])[["median_abs_onset_diff_ms", "max_abs_onset_diff_ms",
                                                     *[f"sdunits_diff_{k}" for k in ("d_attack_slope", "d_env_flux", "d_variation", "d_stroke_valley_db")]]].median().round(3).to_string())
        print(pd.DataFrame(htd).round(4).to_string())
        print(pd.DataFrame(sanity).round(2).to_string())


if __name__ == "__main__":
    main()
