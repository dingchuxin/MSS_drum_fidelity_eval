"""C0.1 validation of the v3 stroke measures on synthetic stroke trains with known levels, with
inter-onset intervals of 60, 100, 150 and 250 ms.

Each configuration is a train of 24 strokes (carrier x linear-attack / exponential-decay
envelope, audit/synth.py) on a stationary floor (or digital silence); peaks alternate between
0 dB and -accent dB (re 0.3 full scale). The pipeline's reference onsets are the Phase 2
detector's onsets of the signal itself (timing.detect), as in measure3.py.

Known values (per stroke k, from the noise-free total power P(t) = sum of every stroke's
squared envelope x amplitude^2 + floor^2):
  peak_true    10 log10 of the maximum of P in [start_k, start_k + attack + 1 ms]
  floor_true   10 log10 of the minimum of P between stroke k's envelope peak and the start
               of stroke k+1 (the true valley bottom = the true inter-stroke level)
  rise_true    0.8 x attack (10 -> 90 % of a linear amplitude ramp)
  accent_true  SD (ddof 1) of the stroke amplitudes in dB
Resolution-limited values: strokes_v3 applied to sqrt(P) (a signal whose windowed RMS is the
noise-free RMS envelope) at the same onsets, i.e. what an ideal implementation at the stated
10 ms / 2.9 ms resolution returns.

Output: results/phase3/c0_v3_validation.csv, c0_v3_validation_summary.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import _p3 as P

sys.path.insert(0, str(P.ROOT / "audit"))
import synth as S  # noqa: E402

from mss_eval import strokes as SK, timing as T  # noqa: E402

SR = S.SR
N = 24
IOIS = [0.06, 0.10, 0.15, 0.25]


def run():
    rows = []
    cfgs = [(c, a, tau, fl, acc, ioi) for c in ("sine5000", "white") for a in (0.5, 1, 2, 5, 10)
            for tau in (15, 40, 80) for fl in (None, -60, -40, -30) for acc in (0, 6) for ioi in IOIS]
    for ci, (carrier, att, tau, floor_db, accent, ioi) in enumerate(cfgs):
        rng = np.random.default_rng(P.SEED + ci)
        starts = 0.3 + ioi * np.arange(N)
        n = int((starts[-1] + 0.6) * SR)
        y = np.zeros(n)
        pw = np.zeros(n)
        amps = np.where(np.arange(N) % 2 == 0, 0.3, 0.3 * 10 ** (-accent / 20))
        dur = 0.6
        for t0, a in zip(starts, amps):
            S.place(y, S.event(att, tau, carrier, rng, dur_s=dur, amp=a), t0)
            env = S.envelope(int(dur * SR), att, tau)
            S.place(pw, (a * env) ** 2, t0)                  # carriers are unit-RMS
        F = 0.0 if floor_db is None else 0.3 * 10 ** (floor_db / 20)
        if F > 0:
            fl = S.carrier(len(y), "sine3100" if carrier.startswith("sine") else "white", rng)
            y = y + F * fl
            pw = pw + F ** 2
        on = T.detect(y, SR)
        i, j = T.match_pairs(starts, on, 0.05)
        meas = SK.strokes_v3(y, SR, on)
        res = SK.strokes_v3(np.sqrt(pw), SR, on)
        for ii, jj in zip(i, j):
            if ii == 0 or ii == N - 1:                     # skip edge strokes
                continue
            s0 = int(starts[ii] * SR)
            pk = int((starts[ii] + att / 1000) * SR)
            nxt = int(starts[ii + 1] * SR)
            peak_true = 10 * np.log10(pw[s0:pk + int(0.001 * SR)].max())
            floor_true = 10 * np.log10(pw[pk:nxt].min()) if pw[pk:nxt].min() > 0 else -np.inf
            rows.append({"carrier": carrier, "attack_ms": att, "tau_ms": tau,
                         "floor_db_re_loud_peak": floor_db if floor_db is not None else -np.inf,
                         "accent_db": accent, "ioi_ms": ioi * 1000, "stroke": ii,
                         "class": "loud" if ii % 2 == 0 else "soft",
                         "latency_ms": (on[jj] - starts[ii]) * 1000,
                         "peak_true": peak_true, "floor_true": floor_true, "rise_true": 0.8 * att,
                         "peak": meas["peak_db"][jj], "floor": meas["floor_db"][jj],
                         "rise": meas["rise_ms"][jj],
                         "peak_res": res["peak_db"][jj], "floor_res": res["floor_db"][jj],
                         "rise_res": res["rise_ms"][jj], "n_det": len(on), "cfg": ci,
                         "silence": floor_db is None})
        if ci % 100 == 0:
            print(ci, len(cfgs), flush=True)
    df = pd.DataFrame(rows)
    for m in ("peak", "floor", "rise"):
        df[f"{m}_err_true"] = df[m] - df[f"{m}_true"]
        df[f"{m}_err_res"] = df[m] - df[f"{m}_res"]
    df.loc[df.floor < SK.SILENCE_DB, ["floor_err_true", "floor_err_res"]] = np.nan
    df.to_csv(P.OUT3 / "c0_v3_validation.csv", index=False, float_format="%.4g")
    return df


def summarize(df):
    rows = []
    for ioi, g in df.groupby("ioi_ms"):
        for tau, h in g.groupby("tau_ms"):
            nz = h[~h.silence]
            rows.append({"ioi_ms": ioi, "tau_ms": tau, "measure": "peak",
                         "median_err_vs_true_db": h.peak_err_true.median(),
                         "median_err_vs_res_db": h.peak_err_res.median(),
                         "p90_abs_err_vs_res_db": h.peak_err_res.abs().quantile(0.9)})
            for fl, k in nz.groupby("floor_db_re_loud_peak"):
                rows.append({"ioi_ms": ioi, "tau_ms": tau, "measure": f"floor@{fl:g}dB",
                             "median_err_vs_true_db": k.floor_err_true.median(),
                             "median_err_vs_res_db": k.floor_err_res.median(),
                             "p90_abs_err_vs_res_db": k.floor_err_res.abs().quantile(0.9)})
            sil = h[h.silence]
            rows.append({"ioi_ms": ioi, "tau_ms": tau, "measure": "floor@silence",
                         "share_floor_below_-150dB": float((sil.floor < SK.SILENCE_DB).mean()),
                         "median_floor_db": float(sil.floor.median()),
                         "median_floor_true_db": float(sil.floor_true.median())})
        for att, h in g.groupby("attack_ms"):
            rows.append({"ioi_ms": ioi, "attack_ms": att, "measure": "rise",
                         "median_err_vs_true_ms": h.rise_err_true.median(),
                         "median_err_vs_res_ms": h.rise_err_res.median(),
                         "p90_abs_err_vs_res_ms": h.rise_err_res.abs().quantile(0.9)})
        a6 = g[g.accent_db == 6].groupby("cfg")
        meas_sd = a6.peak.std(ddof=1)
        true_sd = a6.peak_true.std(ddof=1)
        rows.append({"ioi_ms": ioi, "measure": "accent_sd",
                     "median_true_sd_db": float(true_sd.median()),
                     "median_measured_sd_db": float(meas_sd.median()),
                     "median_err_db": float((meas_sd - true_sd).median())})
        rows.append({"ioi_ms": ioi, "measure": "latency",
                     "median_latency_ms": g.latency_ms.median(),
                     "p95_latency_ms": g.latency_ms.quantile(0.95)})
    out = pd.DataFrame(rows)
    out.to_csv(P.OUT3 / "c0_v3_validation_summary.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    summarize(run())
