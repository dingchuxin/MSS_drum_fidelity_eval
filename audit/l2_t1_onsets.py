"""Test 1: onset detection and matching on synthetic drum-like signals.

  1a  known shifts (integer and fractional samples) -> recovered shift error
  1b  same start times, different attack times -> detected-time sensitivity (ms per ms)
  1c  near-simultaneous events 5-60 ms apart -> merging / splitting
  1d  interference: a second signal with onsets at known offsets -> bias, extra events
  1e  matching: greedy closest-first (timing.match_pairs) vs maximum cardinality (mir_eval)

Uses the project's detectors: timing.detect (Phase 2), timing.detect_refined (Phase 1 J1)
and timing.detect_v1. Outputs results/audit/l2_t1*.csv and l2_t1_summary.json.

    python audit/l2_t1_onsets.py [a b c d e]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import _paths as A
import synth as S

A.project()
from mss_eval import timing as T  # noqa: E402

SR = S.SR
DET = {"p2_detect": lambda y: T.detect(y, SR),
       "p1_detect_refined": lambda y: T.detect_refined(y, SR),
       "v1": lambda y: T.detect_v1(y, SR)}


def match_true(true_t, det_t, w=0.05):
    i, j = T.match_pairs(np.asarray(true_t), np.asarray(det_t), w)
    return i, j


# ---------------------------------------------------------------------- 1a
def t1a():
    rows = []
    shifts = [1, 7, 23, 64, 111, 256, 441, 1000, 2205, 0.25, 0.5, 0.75, 32.5]
    for seed in range(5):
        rng = np.random.default_rng(1000 + seed)
        y, ev = S.drum_track(rng, dur_s=30.0)
        base = {k: f(y) for k, f in DET.items()}
        tt = np.array([e["t"] for e in ev])
        for k, on in base.items():                     # absolute latency vs true starts
            i, j = match_true(tt, on)
            d = (on[j] - tt[i]) * 1000
            rows.append({"seed": seed, "detector": k, "shift_samples": 0.0, "n_true": len(tt),
                         "n_det": len(on), "n_pairs": len(i), "latency_median_ms": np.median(d),
                         "latency_p05_ms": np.percentile(d, 5), "latency_p95_ms": np.percentile(d, 95)})
        for s in shifts:
            ys = S.frac_delay(y, s) if s != int(s) else np.concatenate([np.zeros(int(s)), y])[: len(y)]
            for k, f in DET.items():
                o = f(ys)
                i, j = T.match_pairs(base[k], o, 0.05)
                e = (o[j] - base[k][i]) * 1000 - s / SR * 1000
                rows.append({"seed": seed, "detector": k, "shift_samples": s, "n_true": len(tt),
                             "n_det": len(o), "n_base": len(base[k]), "n_pairs": len(i),
                             "n_unpaired": len(o) + len(base[k]) - 2 * len(i),
                             "mean_err_ms": e.mean(), "median_abs_err_ms": np.median(np.abs(e)),
                             "p95_abs_err_ms": np.percentile(np.abs(e), 95),
                             "max_abs_err_ms": np.abs(e).max()})
        print("1a seed", seed, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(A.OUT / "l2_t1a_shift.csv", index=False, float_format="%.5g")
    sh = df[df.shift_samples != 0]
    summ = sh.groupby("detector").agg(n_pairs=("n_pairs", "sum"), n_unpaired=("n_unpaired", "sum"),
                                      mean_err_ms=("mean_err_ms", "mean"),
                                      median_abs_err_ms=("median_abs_err_ms", "median"),
                                      p95_abs_err_ms=("p95_abs_err_ms", "max"),
                                      max_abs_err_ms=("max_abs_err_ms", "max"))
    lat = df[df.shift_samples == 0].groupby("detector")[["latency_median_ms", "latency_p05_ms", "latency_p95_ms"]].median()
    print(summ.round(4).to_string())
    print(lat.round(3).to_string())
    return {"shift": summ.round(4).reset_index().to_dict("records"),
            "latency_vs_true_start": lat.round(3).reset_index().to_dict("records")}


# ---------------------------------------------------------------------- 1b
ATT = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]


def t1b():
    rows = []
    n_ev, ioi = 40, 0.4
    for kind in S.KINDS:
        for tau in (30, 100):
            for shape in ("lin", "cos", "exp"):
                for lvl_spread in (0, 12):
                    # hash() of strings changes between Python sessions, so this test's values vary
                    # slightly between runs; the stored results hold the run reported in the article
                    rng0 = np.random.default_rng(abs(hash((kind, tau, shape, lvl_spread))) % 2**32)
                    seeds = rng0.integers(0, 2**31, n_ev)
                    lv = rng0.uniform(-lvl_spread, 0, n_ev)
                    times = 0.5 + ioi * np.arange(n_ev)
                    for att in ATT:
                        y = np.zeros(int((times[-1] + 1.0) * SR))
                        for e in range(n_ev):         # common random numbers across attacks
                            x = S.event(att, tau, kind, np.random.default_rng(seeds[e]),
                                        dur_s=0.35, shape=shape, amp=0.3 * 10 ** (lv[e] / 20))
                            S.place(y, x, times[e])
                        for k in ("p2_detect", "v1"):
                            on = DET[k](y)
                            i, j = match_true(times, on)
                            dt = np.full(n_ev, np.nan)
                            dt[i] = (on[j] - times[i]) * 1000
                            for e in range(n_ev):
                                rows.append({"detector": k, "kind": kind, "tau_ms": tau, "shape": shape,
                                             "level_spread_db": lvl_spread, "event": e,
                                             "level_db": lv[e], "attack_ms": att, "det_minus_start_ms": dt[e],
                                             "n_det": len(on)})
                print("1b", kind, tau, shape, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(A.OUT / "l2_t1b_attack.csv.gz", index=False, float_format="%.5g")
    return summarize_1b(df)


def summarize_1b(df):
    out = {}
    for k, g in df.groupby("detector"):
        # per event: OLS slope of detected time on attack over 0.5-20 ms, events detected at all attacks
        w = g.pivot_table(index=["kind", "tau_ms", "shape", "level_spread_db", "event"],
                          columns="attack_ms", values="det_minus_start_ms")
        w = w.dropna()
        x = np.array(ATT, float)
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(X, w.to_numpy().T, rcond=None)[0][1]
        loc = {}
        for a0, a1 in zip(ATT[:-1], ATT[1:]):
            loc[f"{a0}-{a1}"] = float(np.median((w[a1] - w[a0]) / (a1 - a0)))
        by = pd.Series(beta, index=w.index).groupby(level=["kind", "shape"]).median().round(3)
        out[k] = {"n_events": int(len(w)), "slope_ms_per_ms_median": float(np.median(beta)),
                  "slope_p10": float(np.percentile(beta, 10)), "slope_p90": float(np.percentile(beta, 90)),
                  "local_slope_median": {kk: round(v, 3) for kk, v in loc.items()},
                  "median_shift_0p5_to_5ms": float(np.median(w[5] - w[0.5])),
                  "median_shift_1_to_3ms": float(np.median(w[3] - w[1])),
                  "median_shift_2_to_10ms": float(np.median(w[10] - w[2])),
                  "slope_by_kind_shape": {f"{a}|{b}": float(v) for (a, b), v in by.items()},
                  "latency_at_1ms_attack_median": float(np.median(w[1]))}
        print(k, json.dumps(out[k], indent=1))
    return out


# ---------------------------------------------------------------------- 1c
def t1c():
    rows = []
    gaps = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
    for kind2 in ("same", "other"):
        for rel_db in (0, -6, -12):
            for g in gaps:
                rng = np.random.default_rng(7000 + g + (0 if kind2 == "same" else 100) + int(-rel_db))
                n = 30
                t1 = 0.5 + 0.6 * np.arange(n)
                y = np.zeros(int((t1[-1] + 1.0) * SR))
                for t in t1:
                    k1 = S.KINDS[rng.integers(4)]
                    k2 = k1 if kind2 == "same" else S.KINDS[(S.KINDS.index(k1) + 1 + rng.integers(3)) % 4]
                    S.place(y, S.event(1.0, 60, k1, rng, amp=0.3), t)
                    S.place(y, S.event(1.0, 60, k2, rng, amp=0.3 * 10 ** (rel_db / 20)), t + g / 1000)
                for k, f in DET.items():
                    on = f(y)
                    for t in t1:
                        sel = on[(on >= t - 0.025) & (on < t + g / 1000 + 0.06)]
                        rows.append({"detector": k, "second": kind2, "rel_db": rel_db, "gap_ms": g,
                                     "n_found": len(sel),
                                     "first_rel_t1_ms": (sel[0] - t) * 1000 if len(sel) else np.nan,
                                     "second_rel_t2_ms": (sel[1] - t - g / 1000) * 1000 if len(sel) > 1 else np.nan})
            print("1c", kind2, rel_db, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(A.OUT / "l2_t1c_near.csv", index=False, float_format="%.5g")
    s = df.groupby(["detector", "second", "rel_db", "gap_ms"]).agg(
        share_merged=("n_found", lambda v: np.mean(v == 1)), share_split=("n_found", lambda v: np.mean(v == 2)),
        share_more=("n_found", lambda v: np.mean(v > 2)), share_none=("n_found", lambda v: np.mean(v == 0)),
        first_rel_t1_ms=("first_rel_t1_ms", "median")).reset_index()
    s.to_csv(A.OUT / "l2_t1c_near_summary.csv", index=False, float_format="%.4g")
    piv = s[s.second == "same"].pivot_table(index=["detector", "rel_db"], columns="gap_ms", values="share_split")
    print(piv.round(2).to_string())
    return {"share_split_same_kind": {f"{d}|{r}": {int(g): round(float(v), 3) for g, v in row.items()}
                                      for (d, r), row in piv.iterrows()}}


# ---------------------------------------------------------------------- 1d
def interferer(kind: str, rng, amp: float) -> np.ndarray:
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    if kind == "tone220":                     # plucked/piano-like harmonic tone
        x = sum(np.sin(2 * np.pi * 220 * h * t) / h for h in range(1, 8))
        env = S.envelope(n, 2.0, 250)
    elif kind == "bass80":
        x = sum(np.sin(2 * np.pi * 80 * h * t) / h ** 1.5 for h in range(1, 6))
        env = S.envelope(n, 5.0, 300)
    else:                                     # a noise burst of a different band
        x = S.carrier(n, "mid", rng)
        env = S.envelope(n, 1.0, 40)
    x = x / np.sqrt(np.mean(x ** 2))
    return amp * x * env


def t1d():
    rows = []
    offsets = [-40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40]
    for ik in ("tone220", "bass80", "noise_mid"):
        for rel_db in (-20, -10, -6, 0):
            rng = np.random.default_rng(9000 + int(-rel_db) + len(ik))
            n = 44
            times = 0.5 + 0.5 * np.arange(n)
            clean = np.zeros(int((times[-1] + 1.0) * SR))
            kinds = [("high" if i % 2 else "low") for i in range(n)]
            for t, k in zip(times, kinds):
                S.place(clean, S.event(1.0, 60 if k == "low" else 25, k, rng, amp=0.3), t)
            for off in offsets:
                y = clean.copy()
                for t in times:
                    S.place(y, interferer(ik, rng, 0.3 * 10 ** (rel_db / 20)), t + off / 1000)
                for k in ("p2_detect", "v1"):
                    c = DET[k](clean)
                    o = DET[k](y)
                    i, j = T.match_pairs(c, o, 0.05)
                    d = (o[j] - c[i]) * 1000
                    for kk in ("low", "high"):
                        # clean events belonging to this drum kind
                        ci = np.array([np.argmin(np.abs(times - c[ii])) for ii in i])
                        m = np.array([kinds[q] == kk for q in ci], bool) if len(ci) else np.zeros(0, bool)
                        rows.append({"detector": k, "interferer": ik, "rel_db": rel_db, "offset_ms": off,
                                     "drum": kk, "n_clean": len(c), "n_interfered": len(o), "n_pairs": len(i),
                                     "median_shift_ms": np.median(d[m]) if m.any() else np.nan,
                                     "p90_abs_shift_ms": np.percentile(np.abs(d[m]), 90) if m.any() else np.nan,
                                     "share_abs_shift_gt_5ms": np.mean(np.abs(d[m]) > 5) if m.any() else np.nan})
            print("1d", ik, rel_db, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(A.OUT / "l2_t1d_interference.csv", index=False, float_format="%.4g")
    piv = df[df.detector == "p2_detect"].pivot_table(index=["interferer", "rel_db", "drum"], columns="offset_ms",
                                                     values="median_shift_ms")
    print(piv.round(2).to_string())
    extra = df[df.detector == "p2_detect"].assign(extra=lambda d: d.n_interfered - d.n_clean)
    return {"median_shift_ms_p2": {f"{a}|{b}|{c}": {int(o): round(float(v), 2) for o, v in row.items()}
                                   for (a, b, c), row in piv.iterrows()},
            "extra_events_p2_max": int(extra.extra.max())}


# ---------------------------------------------------------------------- 1e
def maxcard(ref, est, w):
    import mir_eval
    m = mir_eval.util.match_events(np.asarray(ref), np.asarray(est), w)
    return np.array([a for a, _ in m], int), np.array([b for _, b in m], int)


def t1e():
    out = {}
    # constructed case: greedy takes the closest pair and blocks two farther legal pairs
    ref, est = np.array([0.000, 0.080]), np.array([0.045, 0.120])
    gi, gj = T.match_pairs(ref, est, 0.05)
    mi, mj = maxcard(ref, est, 0.05)
    out["constructed"] = {"ref_s": ref.tolist(), "est_s": est.tolist(),
                          "greedy_pairs": [(int(a), int(b)) for a, b in zip(gi, gj)],
                          "greedy_dev_ms": ((est[gj] - ref[gi]) * 1000).round(3).tolist(),
                          "maxcard_pairs": [(int(a), int(b)) for a, b in zip(mi, mj)],
                          "maxcard_dev_ms": ((est[mj] - ref[mi]) * 1000).round(3).tolist()}
    print(out["constructed"])
    # random: dense references with jitter, deletions and insertions
    rows = []
    for rep in range(2000):
        rng = np.random.default_rng(rep)
        ref = np.sort(np.cumsum(rng.uniform(0.03, 0.2, 60)))
        keep = rng.random(len(ref)) > 0.1
        est = ref[keep] + rng.normal(0, 0.01, keep.sum())
        est = np.sort(np.concatenate([est, rng.uniform(0, ref[-1], rng.integers(0, 8))]))
        gi, gj = T.match_pairs(ref, est, 0.05)
        mi, mj = maxcard(ref, est, 0.05)
        rows.append({"rep": rep, "n_greedy": len(gi), "n_maxcard": len(mi),
                     "medabs_greedy_ms": np.median(np.abs(est[gj] - ref[gi])) * 1000,
                     "medabs_maxcard_ms": np.median(np.abs(est[mj] - ref[mi])) * 1000,
                     "share_gt10_greedy": np.mean(np.abs(est[gj] - ref[gi]) > 0.01),
                     "share_gt10_maxcard": np.mean(np.abs(est[mj] - ref[mi]) > 0.01)})
    df = pd.DataFrame(rows)
    df.to_csv(A.OUT / "l2_t1e_matching_random.csv", index=False, float_format="%.5g")
    out["random"] = {"reps": len(df), "share_reps_fewer_greedy_pairs": float((df.n_greedy < df.n_maxcard).mean()),
                     "max_pair_deficit": int((df.n_maxcard - df.n_greedy).max()),
                     "mean_share_gt10_greedy": float(df.share_gt10_greedy.mean()),
                     "mean_share_gt10_maxcard": float(df.share_gt10_maxcard.mean())}
    print(out["random"])
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c", "d", "e"]
    f = A.OUT / "l2_t1_summary.json"
    summ = json.load(open(f)) if f.exists() else {}
    for w in which:
        summ[f"1{w}"] = {"a": t1a, "b": t1b, "c": t1c, "d": t1d, "e": t1e}[w]()
        json.dump(summ, open(f, "w"), indent=1, default=float)
