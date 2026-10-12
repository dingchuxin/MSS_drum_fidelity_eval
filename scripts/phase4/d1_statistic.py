"""D1: what the timing errors do to statistics an analyst would report.

Onsets: the Phase 2 reference and estimate onsets (timing.detect), matched closest-first within
+/-50 ms (timing.match_pairs), exactly as in Phase 2; the generic floors leak_20dB and
white_20dB from the Phase 2 floor caches. Signals: v1_spleeter, v1_htdemucs, v1_bs_roformer,
v1_scnet, htdemucs_b1 (check), floor_leak_20dB, floor_white_20dB.

1. Local timing deviation d_k = t_k - (t_{k-1} + t_{k+1}) / 2 for every reference stroke with
   two neighbours. Reference: on the reference onsets. Estimate: on the matched estimate onsets
   of the same three strokes, only where all three are matched. The change (estimate - reference)
   is computed on that common set of strokes (the reference statistics over all strokes with
   two neighbours are reported as well, as ref_all_*). Per track and signal: SD (ddof 1), mean
   |d_k| and median |d_k|, plus the per-stroke error e_k = d_k(est) - d_k(ref) (median |e_k|).
   Stroke sets: all; both reference inter-onset intervals within 60-250 ms.
2. Mean against median of the absolute matched deviation |dev| per track; their ratio; the share
   of the sum of |dev| (hence of the mean) contributed by |dev| > 10 ms; the mean without them.
3. Inter-onset-interval profile from all detected onsets (unmatched included): median and IQR of
   consecutive IOIs and the share below 60 ms, per signal; change estimate - reference.
Summaries: median over tracks with a 95 % percentile bootstrap CI (10,000 resamples of tracks,
seed 20260925); for the SD of d_k also the share of tracks whose |change| exceeds 1 ms and 5 ms.
Yardstick helper: each track's tempo (librosa.feature.tempo on the true drum stem, as in C1),
to express the jazz-trio asynchrony range (1.5-5.6 % of a quarter note) in ms.

    python d1_statistic.py
Outputs: results/phase4/d1_statistic_by_track.csv, d1_summary.csv, d1_tempo_yardstick.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _p4 as P
from mss_eval import timing as T

SIGNALS = P.LABELS4 + [f"floor_{c}" for c in P.FLOORS4]
IOI_LO, IOI_HI = 0.060, 0.250


def dstats(d, prefix):
    d = np.asarray(d, float)
    if len(d) < 3:
        return {f"{prefix}sd_ms": np.nan, f"{prefix}mean_abs_ms": np.nan, f"{prefix}median_abs_ms": np.nan}
    return {f"{prefix}sd_ms": float(np.std(d, ddof=1)), f"{prefix}mean_abs_ms": float(np.mean(np.abs(d))),
            f"{prefix}median_abs_ms": float(np.median(np.abs(d)))}


def ioi_stats(on, prefix):
    ioi = np.diff(np.sort(on)) * 1000
    if len(ioi) < 2:
        return {f"{prefix}ioi_median_ms": np.nan, f"{prefix}ioi_iqr_ms": np.nan, f"{prefix}ioi_share_lt60": np.nan}
    q1, q3 = np.percentile(ioi, [25, 75])
    return {f"{prefix}ioi_median_ms": float(np.median(ioi)), f"{prefix}ioi_iqr_ms": float(q3 - q1),
            f"{prefix}ioi_share_lt60": float(np.mean(ioi < 60))}


def one(sig, track):
    ref, est = P.p2_onsets(sig, track)
    i, j = T.match_pairs(ref, est, 0.05)
    m = np.full(len(ref), np.nan)
    m[i] = est[j]
    row = {"signal": sig, "track": track, "n_ref": len(ref), "n_est": len(est), "n_pairs": len(i)}
    # 1. local timing deviation
    k = np.arange(1, len(ref) - 1)
    d_ref = (ref[k] - (ref[k - 1] + ref[k + 1]) / 2) * 1000
    d_est = (m[k] - (m[k - 1] + m[k + 1]) / 2) * 1000
    ioi_in = ((ref[k] - ref[k - 1]) >= IOI_LO) & ((ref[k] - ref[k - 1]) <= IOI_HI) & \
             ((ref[k + 1] - ref[k]) >= IOI_LO) & ((ref[k + 1] - ref[k]) <= IOI_HI)
    ok = np.isfinite(d_est)
    for sset, sel in (("all", np.ones(len(k), bool)), ("ioi60_250", ioi_in)):
        row[f"{sset}__n_triplets_ref"] = int(sel.sum())
        row[f"{sset}__n_triplets_matched"] = int((sel & ok).sum())
        row.update(dstats(d_ref[sel], f"{sset}__ref_all_"))
        row.update(dstats(d_ref[sel & ok], f"{sset}__ref_"))
        row.update(dstats(d_est[sel & ok], f"{sset}__est_"))
        for s in ("sd_ms", "mean_abs_ms", "median_abs_ms"):
            row[f"{sset}__change_{s}"] = row[f"{sset}__est_{s}"] - row[f"{sset}__ref_{s}"]
        e = d_est[sel & ok] - d_ref[sel & ok]
        row[f"{sset}__median_abs_err_ms"] = float(np.median(np.abs(e))) if len(e) else np.nan
        row[f"{sset}__share_abs_err_gt10"] = float(np.mean(np.abs(e) > 10)) if len(e) else np.nan
    # 2. mean against median of |dev|
    a = np.abs(est[j] - ref[i]) * 1000
    row["dev_mean_abs_ms"] = float(a.mean())
    row["dev_median_abs_ms"] = float(np.median(a))
    row["dev_mean_over_median"] = row["dev_mean_abs_ms"] / row["dev_median_abs_ms"]
    row["dev_share_gt10"] = float(np.mean(a > 10))
    row["dev_tail_share_of_mean"] = float(a[a > 10].sum() / a.sum())
    row["dev_mean_abs_ms_without_tail"] = float(a[a <= 10].mean())
    row["dev_median_abs_ms_without_tail"] = float(np.median(a[a <= 10]))
    # 3. IOI profile, all detected onsets
    row.update(ioi_stats(ref, "ref_"))
    row.update(ioi_stats(est, "est_"))
    for s in ("ioi_median_ms", "ioi_iqr_ms", "ioi_share_lt60"):
        row[f"change_{s}"] = row[f"est_{s}"] - row[f"ref_{s}"]
    return row


def tempo(track):
    import librosa
    y = P.mono(P.ref_stem(track)).astype(np.float32)
    return float(librosa.feature.tempo(y=y, sr=P.SR)[0])


def main():
    rng = np.random.default_rng(P.SEED)
    rows = [one(s, t) for s in SIGNALS for t in P.tracks()]
    df = pd.DataFrame(rows)
    # the matching must reproduce the Phase 2 pair counts
    for lab in P.LABELS4:
        bt = pd.read_csv(P.ROOT / "results" / "phase2" / "measure" / f"{lab}_by_track.csv").set_index("track")
        g = df[df.signal == lab].set_index("track")
        assert (g.n_pairs == bt.n_pairs.loc[g.index]).all(), lab
        assert np.allclose(g.dev_median_abs_ms, bt.median_abs_dev_ms.loc[g.index], rtol=1e-5), lab
    df.to_csv(P.OUT4 / "d1_statistic_by_track.csv", index=False, float_format="%.6g")

    stats = []
    for sset in ("all", "ioi60_250"):
        for s in ("sd_ms", "mean_abs_ms", "median_abs_ms"):
            stats += [(f"{sset}__ref_{s}", sset), (f"{sset}__est_{s}", sset), (f"{sset}__change_{s}", sset)]
        stats += [(f"{sset}__median_abs_err_ms", sset), (f"{sset}__share_abs_err_gt10", sset)]
    stats += [(c, "matched onsets") for c in ("dev_mean_abs_ms", "dev_median_abs_ms", "dev_mean_over_median",
                                              "dev_share_gt10", "dev_tail_share_of_mean",
                                              "dev_mean_abs_ms_without_tail", "dev_median_abs_ms_without_tail")]
    stats += [(f"{p}{s}", "all detected onsets") for p in ("ref_", "est_", "change_")
              for s in ("ioi_median_ms", "ioi_iqr_ms", "ioi_share_lt60")]
    out = []
    for sig in SIGNALS:
        g = df[df.signal == sig]
        for col, sset in stats:
            med, lo, hi = P.boot_median_ci(g[col], rng)
            r = {"signal": sig, "statistic": col, "stroke_set": sset, "n_tracks": int(g[col].notna().sum()),
                 "median": med, "ci_lo": lo, "ci_hi": hi, "mean": float(g[col].mean())}
            if col.endswith("__change_sd_ms"):
                v = g[col].dropna()
                r["share_tracks_abs_change_gt1ms"] = float((v.abs() > 1).mean())
                r["share_tracks_abs_change_gt5ms"] = float((v.abs() > 5).mean())
                r["share_tracks_increase_gt1ms"] = float((v > 1).mean())
            out.append(r)
    pd.DataFrame(out).to_csv(P.OUT4 / "d1_summary.csv", index=False, float_format="%.6g")

    tp = pd.DataFrame({"track": P.tracks()})
    tp["tempo_bpm_librosa"] = [tempo(t) for t in tp.track]
    tp["quarter_ms"] = 60000 / tp.tempo_bpm_librosa
    tp["jazz_async_lo_ms"] = 0.015 * tp.quarter_ms
    tp["jazz_async_hi_ms"] = 0.056 * tp.quarter_ms
    tp.to_csv(P.OUT4 / "d1_tempo_yardstick.csv", index=False, float_format="%.4g")
    print(tp.describe().round(2))


if __name__ == "__main__":
    main()
