"""Test 1: the v3 stroke measures (strokes.strokes_v3, strokes.pair_v3) and the
per-track summary (scripts/phase3/measure3.summarize, which defines accent contrast) on synthetic
stroke trains with analytically known 10-ms RMS levels (1 kHz sine carrier, see synth2.py).

1a  known peak, inter-stroke level, valley, pre-stroke level and level jump; IOIs 60 ms to 1.5 s
1b  a pure gain on the estimate (x0.5 = -6.02 dB) and a per-stroke gain (known accent change)
1c  rise time: linear attacks 0.5-20 ms, against 0.8 x attack and against an independent
    resolution-limited value (2.9-ms RMS of the noise-free power envelope, own code)
1d  digital silence: reference gap, estimate gap, pre-stroke silence, a silent estimate stroke
1e  window boundaries: 1-s cap, the t_k + 50 ms end of the peak window, the s_k+1 cap (IOI < 70 ms)

    python audit2/t1_strokes.py   -> results/audit2/t1_*.csv, t1_summary.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _paths as A
import synth2 as S

A.project()
from mss_eval import strokes as SK, analyst  # noqa: E402
import measure3 as M3  # noqa: E402

SR = S.SR
TOL_DB = 1e-6
res = {}


def train(onsets, amps, gaps, lead_gap, n_extra=1.2):
    """Strokes at `onsets` (s): A_k on [t_k, t_k + 12 ms), 0.9 A_k on [t_k + 12, t_k + 25 ms),
    then the gap level G_k until the next onset; G_-1 = lead_gap before the first stroke."""
    n = int((onsets[-1] + n_extra) * SR)
    secs = [(0.0, onsets[0], lead_gap)]
    nxt = list(onsets[1:]) + [n / SR]
    for t, a, g, tn in zip(onsets, amps, gaps, nxt):
        secs += [(t, t + 0.012, a), (t + 0.012, t + 0.025, 0.9 * a), (t + 0.025, tn, g)]
    return S.piecewise(n, secs) * S.sine(n)


def summary_row(ref, est, on, matched=None):
    rs, es = SK.strokes_v3(ref, SR, on), SK.strokes_v3(est, SR, on)
    rs, es, fl = SK.pair_v3(rs, es)
    matched = np.ones(len(on), bool) if matched is None else matched
    r = {"label": "synthetic", "track": "synthetic", "ref_onsets": np.asarray(on), "matched": matched,
         "dev_ms": np.where(matched, 0.0, np.nan), "ref": rs, "est": es, **fl,
         "desc": analyst.dynamics_v3(est, ref, SR), "si_sdr": 0.0, "n_est_onsets": int(matched.sum())}
    return rs, es, fl, M3.summarize(r)


# ------------------------------------------------------------------ 1a known values
ioi = np.array([150, 60, 100, 250, 150, 1500, 120, 65, 200, 150, 150]) / 1000
on = 0.3 + np.concatenate([[0], np.cumsum(ioi)])
on = np.round(on * SR) / SR
K = len(on)
amps = np.array([0.5, 0.2, 0.4, 0.15, 0.6, 0.3, 0.25, 0.5, 0.1, 0.45, 0.35, 0.2])
gaps = np.array([0.005, 0.01, 0.002, 0.02, 0.001, 0.008, 0.003, 0.012, 0.0005, 0.006, 0.004, 0.009])
lead = 0.007
ref = train(on, amps, gaps, lead)
rs = SK.strokes_v3(ref, SR, on)
exp = pd.DataFrame({"t": on, "peak_db": [S.level(a) for a in amps], "floor_db": [S.level(g) for g in gaps],
                    "pre_db": [S.level(g) for g in np.concatenate([[lead], gaps[:-1]])]})
exp["valley_db"] = exp.peak_db - exp.floor_db
exp["jump_db"] = exp.peak_db - exp.pre_db
rows = []
for m in ["peak_db", "floor_db", "valley_db", "pre_db", "jump_db"]:
    err = rs[m] - exp[m].to_numpy()
    rows.append({"measure": m, "max_abs_err_db": float(np.nanmax(np.abs(err))), "n": K,
                 "pass": bool(np.nanmax(np.abs(err)) < TOL_DB)})
t1a = pd.DataFrame(rows)
t1a.to_csv(A.OUT / "t1a_known_levels.csv", index=False)
detail = exp.copy()
for m in ["peak_db", "floor_db", "valley_db", "pre_db", "jump_db"]:
    detail[f"got_{m}"] = rs[m]
detail["ioi_next_ms"] = np.append(ioi * 1000, np.nan)
detail.to_csv(A.OUT / "t1a_known_levels_detail.csv", index=False, float_format="%.6f")
res["1a_known_levels"] = t1a.set_index("measure")[["max_abs_err_db", "pass"]].to_dict("index")

# ------------------------------------------------------------------ 1b gain
g = 0.5
est = g * ref
rr, ee, fl, row = summary_row(ref, est, on)
gdb = 20 * np.log10(g)
rows = []
for m, shift in [("peak_db", gdb), ("floor_db", gdb), ("pre_db", gdb), ("valley_db", 0.0), ("jump_db", 0.0),
                 ("rise_ms", 0.0)]:
    d = ee[m] - rr[m]
    rows.append({"test": "uniform gain x0.5", "measure": m, "expected_shift": shift,
                 "max_abs_dev_from_expected": float(np.nanmax(np.abs(d - shift))),
                 "summary_d_all": row[f"d_{m}_all"]})
rows.append({"test": "uniform gain x0.5", "measure": "accent (summary d_accent_all)", "expected_shift": 0.0,
             "max_abs_dev_from_expected": abs(row["d_accent_all"]), "summary_d_all": row["d_accent_all"]})
# per-stroke gain: stroke k's whole segment [t_k - 20 ms, t_k+1 - 20 ms) scaled by g_k
gk = np.array([1.0, 0.5, 1.2, 0.8, 1.0, 0.6, 1.4, 0.9, 1.1, 0.7, 1.3, 1.0])
gain = np.ones(len(ref))
bnd = np.round(np.append(on - 0.020, len(ref) / SR) * SR).astype(int)
for k in range(K):
    gain[bnd[k]:bnd[k + 1]] = gk[k]
gain[:bnd[0]] = gk[0]
est2 = ref * gain
rr, ee, fl, row = summary_row(ref, est2, on)
gkdb = 20 * np.log10(gk)
pk_ref = exp.peak_db.to_numpy()
acc_exp = np.std(pk_ref + gkdb, ddof=1) - np.std(pk_ref, ddof=1)
# only the peak is asserted here: inter-stroke and pre-stroke frames are centred 10-ms windows that
# straddle the segment boundary s_k+1, so a gain step exactly at s_k+1 mixes two gains in them
for m, shift in [("peak_db", gkdb)]:
    d = ee[m] - rr[m]
    rows.append({"test": "per-stroke gain", "measure": m, "expected_shift": float(np.median(shift)),
                 "max_abs_dev_from_expected": float(np.nanmax(np.abs(d - shift))),
                 "summary_d_all": row[f"d_{m}_all"]})
rows.append({"test": "per-stroke gain", "measure": "accent (summary d_accent_all)", "expected_shift": acc_exp,
             "max_abs_dev_from_expected": abs(row["d_accent_all"] - acc_exp), "summary_d_all": row["d_accent_all"]})
t1b = pd.DataFrame(rows)
t1b["pass"] = t1b.max_abs_dev_from_expected < 1e-6
t1b.to_csv(A.OUT / "t1b_gain.csv", index=False, float_format="%.8g")
res["1b_gain_all_pass"] = bool(t1b["pass"].all())
res["1b_gain"] = t1b[["test", "measure", "max_abs_dev_from_expected", "pass"]].to_dict("records")


# ------------------------------------------------------------------ 1c rise time
def own_e3_db_free(power, centres, win=128):
    """Independent centred 2.9-ms RMS of a power envelope: frame centred on sample c covers
    samples [c - 64, c + 64)."""
    c = np.concatenate([[0.0], np.cumsum(np.pad(power, (win, win)))])
    lo = centres - win // 2 + win
    return np.sqrt(np.maximum((c[lo + win] - c[lo]) / win, 0))


def own_rise(e, kp, lookback):
    lo = max(0, kp - lookback)
    seg = e[lo:kp + 1]
    base, P = seg.min(), e[kp]
    l10, l90 = base + 0.1 * (P - base), base + 0.9 * (P - base)
    # first frame >= 90 %, last frame < 10 % before it; linear interpolation of both crossings
    i90 = int(np.argmax(seg >= l90))
    t90 = lo + i90 - 1 + (l90 - seg[i90 - 1]) / (seg[i90] - seg[i90 - 1]) if i90 > 0 else float(lo + i90)
    below = np.where(seg[:i90 + 1] < l10)[0]
    if len(below):
        i10 = below[-1] + 1
        t10 = lo + i10 - 1 + (l10 - seg[i10 - 1]) / (seg[i10] - seg[i10 - 1])
    else:
        t10 = float(lo)
    return max(0.0, t90 - t10)


rows = []
for att in (0.5, 1, 2, 3, 5, 8, 10, 15, 20):
    for carrier in ("sine5000", "sine1000"):
        f = float(carrier[4:])
        starts = 0.3 + 0.25 * np.arange(10)
        n = int((starts[-1] + 0.5) * SR)
        a = sum(S.ramp_decay(n, t0, att, 40.0, 0.4) for t0 in starts)
        x = a * S.sine(n, f)
        # reference onsets 12 ms after the envelope start (the detector's typical latency)
        ons = starts + 0.012
        st = SK.strokes_v3(x, SR, ons)
        env = SK.Envelopes(x, SR)
        for k, t0 in enumerate(starts[1:-1], 1):
            # pipeline peak frame of env3 within the peak window, then the own 10-90 % on it
            i_s, i_pk = env.idx(ons[k] - 0.020), env.idx(ons[k] + 0.050) + 1
            kp = i_s + int(np.argmax(env.e3[i_s:i_pk]))
            own_on_pipeline_env = own_rise(env.e3, kp, env.idx(0.020)) * 64 / SR * 1000
            centres = np.arange(len(env.e3)) * 64
            e3_free = own_e3_db_free(0.5 * a ** 2, centres)
            kpf = i_s + int(np.argmax(e3_free[i_s:i_pk]))
            res_lim = own_rise(e3_free, kpf, env.idx(0.020)) * 64 / SR * 1000
            rows.append({"carrier": carrier, "attack_ms": att, "stroke": k, "true_rise_ms": 0.8 * att,
                         "rise_ms": st["rise_ms"][k], "resolution_limited_ms": res_lim,
                         "own_algorithm_on_pipeline_env3_ms": own_on_pipeline_env})
t1c = pd.DataFrame(rows)
t1c["err_vs_true"] = t1c.rise_ms - t1c.true_rise_ms
t1c["err_vs_res"] = t1c.rise_ms - t1c.resolution_limited_ms
t1c["err_vs_own_algo"] = t1c.rise_ms - t1c.own_algorithm_on_pipeline_env3_ms
t1c.to_csv(A.OUT / "t1c_rise.csv", index=False, float_format="%.5g")
s = t1c.groupby(["carrier", "attack_ms"])[["rise_ms", "true_rise_ms", "resolution_limited_ms", "err_vs_true",
                                            "err_vs_res", "err_vs_own_algo"]].median()
s.to_csv(A.OUT / "t1c_rise_summary.csv", float_format="%.4g")
res["1c_rise_max_abs_err_vs_own_algorithm_ms"] = float(t1c.err_vs_own_algo.abs().max())
res["1c_rise_median_err_vs_true_by_attack_sine5000"] = s.loc["sine5000", "err_vs_true"].round(3).to_dict()
res["1c_rise_median_err_vs_resolution_limited_by_attack_sine5000"] = s.loc["sine5000", "err_vs_res"].round(3).to_dict()
res["1c_rise_median_err_vs_true_by_attack_sine1000"] = s.loc["sine1000", "err_vs_true"].round(3).to_dict()

# ------------------------------------------------------------------ 1d digital silence
rows = []
# (i) reference gap of stroke 3 is digital silence (both signals): floor/valley NaN in both;
#     stroke 4's pre-stroke level sits in the silence: pre/jump NaN in both
g2 = gaps.copy()
g2[3] = 0.0
ref_s = train(on, amps, g2, lead)
est_s = 0.7 * ref_s
rr, ee, fl, row = summary_row(ref_s, est_s, on)
rows.append({"case": "reference gap silent (stroke 3)", "floor_nan_both": bool(np.isnan(rr["floor_db"][3]) and np.isnan(ee["floor_db"][3])),
             "valley_nan_both": bool(np.isnan(rr["valley_db"][3]) and np.isnan(ee["valley_db"][3])),
             "silent_ref_flag": bool(fl["silent_ref"][3]), "silent_est_flag": bool(fl["silent_est"][3]),
             "next_pre_nan_both": bool(np.isnan(rr["pre_db"][4]) and np.isnan(ee["pre_db"][4])),
             "other_strokes_untouched": bool(np.isfinite(rr["floor_db"][[0, 1, 2, 5, 6]]).all()),
             "summary_n_silent_either": row["n_silent_either"]})
# (ii) estimate-only gap silence (stroke 6)
est_z = 0.7 * ref.copy()
a0, a1 = int(round((on[6] + 0.025) * SR)), int(round(on[7] * SR))
est_z[a0:a1] = 0.0
rr, ee, fl, row = summary_row(ref, est_z, on)
rows.append({"case": "estimate gap silent (stroke 6)", "floor_nan_both": bool(np.isnan(rr["floor_db"][6]) and np.isnan(ee["floor_db"][6])),
             "valley_nan_both": bool(np.isnan(rr["valley_db"][6]) and np.isnan(ee["valley_db"][6])),
             "silent_ref_flag": bool(fl["silent_ref"][6]), "silent_est_flag": bool(fl["silent_est"][6]),
             "next_pre_nan_both": bool(np.isnan(rr["pre_db"][7]) and np.isnan(ee["pre_db"][7])),
             "other_strokes_untouched": bool(np.isfinite(rr["floor_db"][[0, 1, 2, 3, 4, 5]]).all()),
             "summary_n_silent_either": row["n_silent_either"]})
# (iii) the estimate is digitally silent over a whole stroke (stroke 8: its segment zeroed)
est_m = 0.7 * ref.copy()
a0, a1 = int(round((on[8] - 0.030) * SR)), int(round((on[9] - 0.020) * SR))   # incl. the 5-ms overhang
est_m[a0:a1] = 0.0
rr, ee, fl, row = summary_row(ref, est_m, on)
_, _, _, row_ok = summary_row(ref, 0.7 * ref, on)
rows.append({"case": "estimate silent over a whole stroke (stroke 8)", "est_peak_db": float(ee["peak_db"][8]),
             "peak_nan": bool(np.isnan(ee["peak_db"][8])), "floor_nan_both": bool(np.isnan(ee["floor_db"][8])),
             "d_accent_all_with_silent_stroke": row["d_accent_all"], "d_accent_all_without": row_ok["d_accent_all"],
             "d_peak_db_all_median_with": row["d_peak_db_all"], "d_peak_db_all_median_without": row_ok["d_peak_db_all"]})
t1d = pd.DataFrame(rows)
t1d.to_csv(A.OUT / "t1d_silence.csv", index=False, float_format="%.6g")
res["1d_silence"] = t1d.to_dict("records")


# ------------------------------------------------------------------ 1e window boundaries
def three(on3, extra, amps3=(0.3, 0.3, 0.3), gap=0.01, total=None):
    n = int(((total or on3[-1] + 1.3)) * SR)
    secs = [(0.0, on3[0], gap)]
    nxt = list(on3[1:]) + [n / SR]
    for t, a, tn in zip(on3, amps3, nxt):
        secs += [(t, t + 0.012, a), (t + 0.012, t + 0.025, 0.9 * a), (t + 0.025, tn, gap)]
    secs += extra
    return S.piecewise(n, secs) * S.sine(n)


rows = []
# 1-s cap: stroke 1 with IOI 1.5 s; a 100-ms dip at amplitude 1e-4 inside / outside [s_1, s_1 + 1 s)
o3 = np.round(np.array([0.3, 0.45, 1.95]) * SR) / SR
s1 = o3[1] - 0.020
for where, t0, expect_dip in (("inside cap", s1 + 0.85, True), ("after cap", s1 + 1.02, False)):
    x = three(o3, [(t0, t0 + 0.100, 1e-4)])
    st = SK.strokes_v3(x, SR, o3)
    exp_floor = S.level(1e-4) if expect_dip else S.level(0.01)
    rows.append({"test": f"1-s cap: dip {where}", "measure": "floor_db (stroke 1)", "got": st["floor_db"][1],
                 "expected": exp_floor})
# end of the peak window t_k + 50 ms: a burst of 3x amplitude at [t + 36, t + 52) (inside) or
# [t + 56, t + 70) (frames centred > t + 50 ms only)
o3 = np.round(np.array([0.3, 0.6, 0.9]) * SR) / SR
for where, (b0, b1), inside in (("ends inside window", (0.036, 0.052), True), ("starts after t+50 ms", (0.056, 0.070), False)):
    x = three(o3, [(o3[1] + b0, o3[1] + b1, 0.9)])
    st = SK.strokes_v3(x, SR, o3)
    rows.append({"test": f"peak window end: burst {where}", "measure": "peak_db (stroke 1)", "got": st["peak_db"][1],
                 "expected": S.level(0.9) if inside else S.level(0.3)})
# the s_k+1 cap for IOI < 70 ms: IOI 60 ms; burst at [t + 46, t + 58) lies before t + 50 ms but after
# s_k+1 + 5 ms (= t + 45 ms), so only frames past s_k+1 see it
o3 = np.round(np.array([0.3, 0.6, 0.66]) * SR) / SR
x = three(o3, [(o3[1] + 0.046, o3[1] + 0.058, 0.9)])
st = SK.strokes_v3(x, SR, o3)
rows.append({"test": "s_k+1 cap (IOI 60 ms): burst after s_k+1", "measure": "peak_db (stroke 1)", "got": st["peak_db"][1],
             "expected": S.level(0.3)})
rows.append({"test": "s_k+1 cap (IOI 60 ms): the burst belongs to the next stroke", "measure": "peak_db (stroke 2)",
             "got": st["peak_db"][2], "expected": S.level(0.9)})
# last stroke: segment = [s_K, s_K + 1 s) or the end of the signal
o3 = np.round(np.array([0.3, 0.6, 0.9]) * SR) / SR
x = three(o3, [(o3[2] + 0.60, o3[2] + 0.75, 1e-4)], total=o3[2] + 0.8)
st = SK.strokes_v3(x, SR, o3)
rows.append({"test": "last stroke: dip before the signal end", "measure": "floor_db (stroke 2)", "got": st["floor_db"][2],
             "expected": S.level(1e-4)})
t1e = pd.DataFrame(rows)
t1e["abs_err"] = (t1e.got - t1e.expected).abs()
t1e["pass"] = t1e.abs_err < 1e-6
t1e.to_csv(A.OUT / "t1e_windows.csv", index=False, float_format="%.6g")
res["1e_windows"] = t1e.to_dict("records")

A.dump("t1_summary.json", res)
print(t1a.to_string(index=False))
print(t1b.to_string(index=False))
print(s.round(3).to_string())
print(t1d.T.to_string())
print(t1e.to_string(index=False))
