"""C2 detector validation on synthetic drum-like events (audit/synth.py), with the design of
tests 1a and 1b of audit/l2_t1_onsets.py; outputs are written to results/phase3.

  1a  latency: detected time - true event start (5 random drum-like tracks, 30 s) and the
      known-shift error for off-grid shifts (7, 23, 111, 441 samples; 0.5 and 32.5 by FFT);
  1b  shape sensitivity: the same start times with attacks of 0.5 ... 20 ms (4 spectra x 2 decays
      x 3 attack shapes x 2 level spreads, 40 events each); per-event OLS slope of the detected
      time on the attack (ms per ms) and local slopes between attack steps.
Detectors: rise (detect_refined mode='rise', min_sep_s=0.03) and energy (detect_energy). The
Phase 2 detector's values come from audit/l2_t1_onsets.py (results/audit/l2_t1_summary.json), repeated in the
output for comparison.

    python c2_validate_synth.py [--jobs 2]
Outputs: results/phase3/c2_validate_1a.csv, c2_validate_1b.csv.gz, c2_validate_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _p3 as P  # noqa: E402

sys.path.insert(0, str(P.ROOT / "audit"))
import synth as S  # noqa: E402

from mss_eval import timing as T  # noqa: E402

SR = S.SR
ATT = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]
SHIFTS = [7, 23, 111, 441, 0.5, 32.5]


def dets(y):
    ev = T.detect(y, SR)
    return {"p2": ev, "energy": T.detect_energy(y, SR, events=ev),
            "rise": T.detect_refined(y, SR, mode="rise", min_sep_s=0.03)}


def t1a(seed):
    rng = np.random.default_rng(1000 + seed)
    y, ev = S.drum_track(rng, dur_s=30.0)
    tt = np.array([e["t"] for e in ev])
    base = dets(y)
    rows = []
    for k, on in base.items():
        i, j = T.match_pairs(tt, on, 0.05)
        d = (on[j] - tt[i]) * 1000
        rows.append({"seed": seed, "detector": k, "shift_samples": 0.0, "n_true": len(tt), "n_det": len(on),
                     "latency_median_ms": np.median(d), "latency_p05_ms": np.percentile(d, 5),
                     "latency_p95_ms": np.percentile(d, 95)})
    for s in SHIFTS:
        ys = S.frac_delay(y, s) if s != int(s) else np.concatenate([np.zeros(int(s)), y])[: len(y)]
        o = dets(ys)
        for k in base:
            i, j = T.match_pairs(base[k], o[k], 0.05)
            e = (o[k][j] - base[k][i]) * 1000 - s / SR * 1000
            rows.append({"seed": seed, "detector": k, "shift_samples": s, "n_pairs": len(i),
                         "median_abs_err_ms": np.median(np.abs(e)), "p95_abs_err_ms": np.percentile(np.abs(e), 95),
                         "max_abs_err_ms": np.abs(e).max()})
    return rows


def t1b(cfg):
    kind, tau, shape, spread = cfg
    n_ev, ioi = 40, 0.4
    # deterministic seed (Python's salted hash() would differ between sessions)
    rng0 = np.random.default_rng([P.SEED, zlib.crc32(f"{kind}|{tau}|{shape}|{spread}".encode())])
    seeds = rng0.integers(0, 2**31, n_ev)
    lv = rng0.uniform(-spread, 0, n_ev)
    times = 0.5 + ioi * np.arange(n_ev)
    rows = []
    for att in ATT:
        y = np.zeros(int((times[-1] + 1.0) * SR))
        for e in range(n_ev):
            x = S.event(att, tau, kind, np.random.default_rng(seeds[e]), dur_s=0.35, shape=shape,
                        amp=0.3 * 10 ** (lv[e] / 20))
            S.place(y, x, times[e])
        d = dets(y)
        for k in ("energy", "rise", "p2"):
            on = d[k]
            i, j = T.match_pairs(times, on, 0.05)
            dt = np.full(n_ev, np.nan)
            dt[i] = (on[j] - times[i]) * 1000
            for e in range(n_ev):
                rows.append({"detector": k, "kind": kind, "tau_ms": tau, "shape": shape, "level_spread_db": spread,
                             "event": e, "attack_ms": att, "det_minus_start_ms": dt[e]})
    return rows


def summarize_1b(df):
    out = {}
    for k, g in df.groupby("detector"):
        w = g.pivot_table(index=["kind", "tau_ms", "shape", "level_spread_db", "event"], columns="attack_ms",
                          values="det_minus_start_ms").dropna()
        x = np.array(ATT, float)
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(X, w.to_numpy().T, rcond=None)[0][1]
        loc = {f"{a0}-{a1}": float(np.median((w[a1] - w[a0]) / (a1 - a0))) for a0, a1 in zip(ATT[:-1], ATT[1:])}
        by = pd.Series(beta, index=w.index).groupby(level=["kind", "shape"]).median()
        out[k] = {"n_events": int(len(w)), "slope_ms_per_ms_median": float(np.median(beta)),
                  "slope_p10": float(np.percentile(beta, 10)), "slope_p90": float(np.percentile(beta, 90)),
                  "local_slope_median": {kk: round(v, 3) for kk, v in loc.items()},
                  "slope_by_kind_shape": {f"{a}|{b}": round(float(v), 3) for (a, b), v in by.items()},
                  "latency_at_1ms_attack_median": float(np.median(w[1]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    from multiprocessing import Pool
    cfgs = [(k, tau, sh, sp) for k in S.KINDS for tau in (30, 100) for sh in ("lin", "cos", "exp") for sp in (0, 12)]
    with Pool(a.jobs) as p:
        r1 = sum(p.map(t1a, range(5)), [])
        r2 = sum(p.map(t1b, cfgs), [])
    d1 = pd.DataFrame(r1)
    d1.to_csv(P.OUT3 / "c2_validate_1a.csv", index=False, float_format="%.5g")
    d2 = pd.DataFrame(r2)
    d2.to_csv(P.OUT3 / "c2_validate_1b.csv.gz", index=False, float_format="%.5g")
    sh = d1[d1.shift_samples != 0].groupby("detector").agg(median_abs_err_ms=("median_abs_err_ms", "median"),
                                                          p95_abs_err_ms_worst_shift=("p95_abs_err_ms", "max"),
                                                          max_abs_err_ms=("max_abs_err_ms", "max"))
    lat = d1[d1.shift_samples == 0].groupby("detector")[["latency_median_ms", "latency_p05_ms", "latency_p95_ms"]].median()
    summ = {"1a_shift": sh.round(4).reset_index().to_dict("records"),
            "1a_latency_vs_true_start": lat.round(3).reset_index().to_dict("records"),
            "1b": summarize_1b(d2),
            "audit_p2_1b": json.load(open(P.ROOT / "results" / "audit" / "l2_t1_summary.json"))["1b"]["p2_detect"]}
    json.dump(summ, open(P.OUT3 / "c2_validate_summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ("1a_shift", "1a_latency_vs_true_start")}, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "slope_by_kind_shape"} for k, v in summ["1b"].items()},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
