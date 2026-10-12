"""J3: paired statistics across tracks (drums, 50 MUSDB18-HQ test tracks).

Families (Holm-corrected within each family):
  A  developer   SIR, SAR (and SDR, ISR) for all 6 model pairs   -> 24 tests
  B  analyst     |d_x| for the 8 dynamic-profile descriptors + onset-F (v1, +/-50 ms),
                 key contrast BS-Roformer vs SCNet-XL             -> 9 tests (8 corrected:
                 burst_rate is degenerate, reported but excluded from Holm in every family)
  B* analyst     same metrics, all 6 model pairs                  -> 54 tests (context)
  C  analyst     signed d_x (direction), BS-Roformer vs SCNet-XL  -> 8 tests
  D  timing      refined-detector timing metrics (from J1), BS-Roformer vs SCNet-XL,
                 if results/phase1/onset_deviation_by_track.csv exists

Test: two-sided Wilcoxon signed-rank on per-track paired differences (scipy, zero_method=
'wilcox', exact where available). Effect size: matched-pairs rank-biserial correlation
r = (R+ - R-) / (R+ + R-), zero differences dropped; r > 0 means model_a > model_b.
Output: results/phase1/stats_tests.csv
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

import _common as C

DESCS = ["level_db", "variation", "crest", "impulsiveness", "env_skew", "env_flux",
         "attack_slope", "burst_rate"]
KEY = ("bs_roformer", "scnet")


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = rankdata(np.abs(d))
    rp, rn = r[d > 0].sum(), r[d < 0].sum()
    return (rp - rn) / (rp + rn)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    ps = p[ok]
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    out[ok] = adj
    return out


def test(wide: pd.DataFrame, a: str, b: str) -> dict:
    x, y = wide[a].to_numpy(float), wide[b].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    if np.all(d == 0):
        return {"n": int(ok.sum()), "median_a": np.median(x[ok]), "median_b": np.median(y[ok]),
                "median_diff": 0.0, "n_a_gt_b": 0, "W": np.nan, "p": np.nan,
                "rank_biserial": np.nan}
    res = wilcoxon(x[ok], y[ok], zero_method="wilcox", alternative="two-sided")
    return {"n": int(ok.sum()), "median_a": np.median(x[ok]), "median_b": np.median(y[ok]),
            "median_diff": np.median(d), "n_a_gt_b": int((d > 0).sum()),
            "W": float(res.statistic), "p": float(res.pvalue), "rank_biserial": rank_biserial(d)}


def run_family(name: str, metrics: dict[str, pd.DataFrame], pairs) -> pd.DataFrame:
    rows = []
    for metric, wide in metrics.items():
        for a, b in pairs:
            rows.append({"family": name, "metric": metric, "model_a": a, "model_b": b,
                         **test(wide, a, b)})
    df = pd.DataFrame(rows)
    # burst_rate is zero on most reference stems (see J2): reported, but kept out of the
    # corrected family so a degenerate test does not inflate m
    inc = ~df.metric.str.contains("burst_rate") & np.isfinite(df.p)
    df["p_holm"] = np.nan
    df.loc[inc, "p_holm"] = holm(df.loc[inc, "p"].to_numpy())
    df["m_family"] = int(inc.sum())
    return df


def main():
    raw = pd.read_csv(C.ROOT / "results" / "tables" / "benchmark_raw.csv")
    raw = raw[raw.stem == "drums"]
    all_pairs = list(combinations(C.MODELS, 2))
    out = []

    bss_f = C.OUT / "bss_drums_pertrack.csv"
    if bss_f.exists():
        bss = pd.read_csv(bss_f)
        A = {m: bss.pivot(index="track", columns="model", values=m) for m in ["sir", "sar", "sdr", "isr"]}
        out.append(run_family("A_developer_all_pairs", A, all_pairs))
    else:
        print("!! bss_drums_pertrack.csv missing - family A skipped")

    def wide(col, f=lambda s: s):
        return raw.assign(v=f(raw[col])).pivot(index="track", columns="model", values="v")

    B = {f"abs_d_{x}": wide(f"d_{x}", np.abs) for x in DESCS}
    B["onset_f_measure_v1_50ms"] = wide("onset_f_measure")
    out.append(run_family("B_analyst_abs_key", B, [KEY]))
    out.append(run_family("Bstar_analyst_abs_all_pairs", B, all_pairs))
    out.append(run_family("C_analyst_signed_key", {f"d_{x}": wide(f"d_{x}") for x in DESCS}, [KEY]))

    tim_f = C.OUT / "onset_deviation_by_track.csv"
    if tim_f.exists():
        tim = pd.read_csv(tim_f)
        tim = tim[tim.detector == "refined"]
        D = {c: tim.pivot(index="track", columns="model", values=c)
             for c in ["f_5ms", "f_10ms", "f_20ms", "f_50ms", "median_dev_ms", "median_abs_dev_ms",
                       "iqr_dev_ms", "frac_abs_gt_5ms", "frac_abs_gt_10ms"]}
        out.append(run_family("D_timing_refined_key", D, [KEY]))
        out.append(run_family("Dstar_timing_refined_all_pairs", D, all_pairs))

    df = pd.concat(out, ignore_index=True)
    df["burst_rate_degenerate"] = df.metric.str.contains("burst_rate")
    df.to_csv(C.OUT / "stats_tests.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200):
        print(df[~df.family.str.contains("star|Dstar")][
            ["family", "metric", "model_a", "model_b", "median_a", "median_b", "median_diff",
             "p", "p_holm", "rank_biserial"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
