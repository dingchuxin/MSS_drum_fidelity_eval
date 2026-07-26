"""Emit every number the paper quotes, at the precision the paper prints, from unrounded data.

WHY THIS EXISTS. The result tables in results/tables/ are written with .round(3) for
readability. Reading a value off those tables and rounding it again for the paper is a
double-rounding bug: |d attack-slope| for BS-Roformer is 1.3246, stored as 1.325, and
re-rounding 1.325 to 2 dp yields 1.33 instead of the correct 1.32. Every value of the form
x.xx4y is exposed to this, and five such cells reached a draft before being caught.
Quote paper values from THIS script, never from a rounded table.

Everything is recomputed from the raw per-track sources:
  results/tables/benchmark_raw.csv        drums, 4 models x 50 tracks (SI-SDR + analyst)
  results/tables/benchmark_metrics.csv    aggregated BSS-Eval (SDR/SIR/SAR/ISR)
  results/tables/input_length_length.csv  A2 sweep, 3 models x 20 tracks x 4 levels

Usage:  python scripts/paper_values.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"

GEN = ["spleeter", "htdemucs", "bs_roformer", "scnet"]      # architecture generation order
LABEL = {"spleeter": "Spleeter", "htdemucs": "HT-Demucs",
         "bs_roformer": "BS-Roformer", "scnet": "SCNet-XL"}
LEVELS = ["region_only", "ctx_15s", "ctx_30s", "full"]
A2_MODELS = ["htdemucs", "bs_roformer", "scnet"]
STEMS = ["bass", "drums", "other", "vocals"]


def head(s: str):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def load():
    raw = pd.read_csv(T / "benchmark_raw.csv")
    agg = pd.read_csv(T / "benchmark_metrics.csv", header=[0, 1], index_col=[0, 1])
    a2 = pd.read_csv(T / "input_length_length.csv")
    return raw, agg, a2[a2.stem == "drums"].copy()


# --- developer view ---------------------------------------------------------

def table2(agg):
    head("TABLE 2 - median BSS-Eval SDR (dB) per stem, 50 tracks")
    t = agg[("sdr", "median")].unstack()[STEMS]
    print(t.reindex(["spleeter", "htdemucs", "scnet", "bs_roformer"]).round(2).to_string())
    print("\ndrums gap, the two modern models: "
          f"{abs(t.loc['bs_roformer', 'drums'] - t.loc['scnet', 'drums']):.2f} dB")
    # the generational story is NOT monotonic per stem - check before claiming it
    print("\nmonotonic across generations (2019 -> 2022 -> 2023 -> 2024)?")
    for s in STEMS:
        seq = [t.loc[m, s] for m in GEN]
        ok = all(seq[i] < seq[i + 1] for i in range(3))
        print(f"  {s:7s} " + " -> ".join(f"{v:5.2f}" for v in seq) + f"   {ok}")


def table6(agg):
    head("TABLE 6 - interference (SIR) vs artifact (SAR), drums, median")
    for m in ["spleeter", "htdemucs", "scnet", "bs_roformer"]:
        print(f"  {LABEL[m]:12s} SIR {agg.loc[(m, 'drums'), ('sir', 'median')]:5.1f}"
              f"   SAR {agg.loc[(m, 'drums'), ('sar', 'median')]:5.1f}")
    lp = agg[("labeling_precision", "median")]
    print(f"\nlabeling precision, all models x stems: {lp.min():.2f}-{lp.max():.2f}")


# --- A1 / B3 ---------------------------------------------------------------

def table3(raw):
    head("TABLE 3 - drums, medians over 50 tracks (ranks in parentheses)")
    g = raw.groupby("model")
    t = pd.DataFrame({
        "si_sdr": g.si_sdr.median(),
        "onset_f": g.onset_f_measure.median(),
        "abs_d_attack": g.d_attack_slope.apply(lambda x: x.abs().median()),
        "abs_d_variation": g.d_variation.apply(lambda x: x.abs().median()),
    })
    r = pd.DataFrame({
        "si_sdr": t.si_sdr.rank(ascending=False),
        "onset_f": t.onset_f.rank(ascending=False),
        "abs_d_attack": t.abs_d_attack.rank(),          # 1 = least distortion
        "abs_d_variation": t.abs_d_variation.rank(),
    }).astype(int)
    for m in ["bs_roformer", "scnet", "htdemucs", "spleeter"]:
        print(f"  {LABEL[m]:12s} SI-SDR {t.si_sdr[m]:5.2f} ({r.si_sdr[m]})   "
              f"onset-F {t.onset_f[m]:.2f} ({r.onset_f[m]})   "
              f"|d attack| {t.abs_d_attack[m]:.2f} ({r.abs_d_attack[m]})   "
              f"|d variation| {t.abs_d_variation[m]:.2f} ({r.abs_d_variation[m]})")
    print(f"\nattack-slope distortion ratio BS-Roformer / SCNet-XL: "
          f"{t.abs_d_attack['bs_roformer'] / t.abs_d_attack['scnet']:.2f}x")
    print(f"dynamics ratio HT-Demucs / BS-Roformer:              "
          f"{t.abs_d_variation['htdemucs'] / t.abs_d_variation['bs_roformer']:.2f}x")


def a1(raw):
    head("A1 - cross-model vs within-model spread (drums)")
    rows = []
    for c in [c for c in raw.columns if c.startswith("dyn_") or c.startswith("d_")]:
        cross = raw.groupby("model")[c].mean().std(ddof=1)
        within = raw.groupby("model")[c].std(ddof=1).mean()
        rows.append({"descriptor": c, "ratio": cross / within})
    sp = pd.DataFrame(rows).sort_values("ratio")
    dev = sp[sp.descriptor.str.startswith("d_")]
    print(f"  the 8 DEVIATION (d_*) descriptors: {dev.ratio.min():.2f}-{dev.ratio.max():.2f}"
          f"   (d_variation {dev.set_index('descriptor').ratio['d_variation']:.2f})")
    print(f"  all 16 raw + deviation:            {sp.ratio.min():.2f}-{sp.ratio.max():.2f}")
    print("  ^ quote the range whose scope you state; these are different families.")
    piv = raw.pivot_table(index="track", columns="model", values="d_variation").corr("spearman")
    off = piv.values[np.triu_indices_from(piv.values, k=1)]
    print(f"\n  between-model Spearman on per-track d_variation: "
          f"mean {off.mean():.2f}, range {off.min():.2f}-{off.max():.2f}")


def b3(raw):
    head(f"B3 - pooled Spearman vs SI-SDR (drums, n = {len(raw)})")
    for col, ab in [("onset_f_measure", False), ("d_attack_slope", True),
                    ("d_variation", True), ("d_crest", True)]:
        y = raw[col].abs() if ab else raw[col]
        rho, p = spearmanr(raw.si_sdr, y)
        ps = "< 1e-4" if p < 1e-4 else f"= {p:.3f}"
        print(f"  {('|' + col + '|') if ab else col:20s} rho = {rho:+.2f}   p {ps}")


# --- A2 --------------------------------------------------------------------

def drift(a2, descs):
    return a2.groupby(["model", "track"])[descs].std(ddof=0)      # population SD, ddof=0


def table4(a2):
    head("TABLE 4 - A2 drift (SD across the four windows), mean over 20 tracks")
    descs = ["dyn_attack_slope", "dyn_impulsiveness", "dyn_variation", "onset_f_measure"]
    mean = drift(a2, descs).groupby("model").mean()
    print("  model         attack-slope  impulsiveness  variation  onset-F")
    for m in A2_MODELS:
        r = mean.loc[m]
        print(f"  {LABEL[m]:12s}  {r.dyn_attack_slope:11.2f}  {r.dyn_impulsiveness:13.2f}"
              f"  {r.dyn_variation:9.2f}  {r.onset_f_measure:7.3f}")


def attention(a2):
    head("A2 attention test - BS-Roformer vs SCNet-XL, paired Wilcoxon (two-sided)")
    descs = ["dyn_attack_slope", "dyn_impulsiveness", "dyn_variation", "dyn_crest",
             "onset_f_measure"]
    piv = drift(a2, descs).reset_index().pivot(index="track", columns="model")
    for d in descs:
        bs, sc = piv[(d, "bs_roformer")], piv[(d, "scnet")]
        _, p = wilcoxon(bs, sc)
        ratio = (bs / sc.replace(0, np.nan)).median()
        print(f"  {d:20s} p = {p:.3f}   {(bs > sc).sum()}/{len(bs)} higher   "
              f"median ratio {ratio:.2f}x")
    bs, ht = piv[("dyn_attack_slope", "bs_roformer")], piv[("dyn_attack_slope", "htdemucs")]
    _, p = wilcoxon(bs, ht)
    print(f"\n  attack-slope vs HT-Demucs: p = {p:.2f}")

    # leave-one-out: how fragile is the headline at n = 20?
    bs, sc = piv[("dyn_attack_slope", "bs_roformer")], piv[("dyn_attack_slope", "scnet")]
    loo = sorted(wilcoxon(bs.drop(t), sc.drop(t))[1] for t in bs.index)
    above = sum(1 for p in loo if p >= 0.05)
    print(f"  leave-one-out p range: {min(loo):.3f}-{max(loo):.3f}")
    print(f"  omitting any one of {above} tracks (of {len(loo)}) lifts p above 0.05")


def table5(a2):
    head("TABLE 5 - mean fidelity of region R at each context level, 20 tracks")
    a2 = a2.assign(**{f"abs_{c}": a2[c].abs() for c in
                      ["d_crest", "d_variation", "d_attack_slope"]})
    for meas, lab in [("abs_d_crest", "|d crest|"), ("abs_d_variation", "|d variation|"),
                      ("onset_f_measure", "onset-F"), ("abs_d_attack_slope", "|d attack-slope|")]:
        t = a2.pivot_table(index="model", columns="level", values=meas)[LEVELS]
        print(f"  {lab}")
        for m in A2_MODELS:
            dp = 3 if meas == "onset_f_measure" else 2
            print(f"    {LABEL[m]:12s} " + "  ".join(f"{t.loc[m, l]:.{dp}f}" for l in LEVELS))


def friedman(a2):
    head("A2 across-level Friedman (20 tracks) - 4 descriptors x 3 models = 12 tests")
    a2 = a2.assign(**{f"abs_{c}": a2[c].abs() for c in
                      ["d_crest", "d_variation", "d_attack_slope"]})
    ps = []
    for meas, lab in [("abs_d_crest", "|d crest|"), ("abs_d_variation", "|d variation|"),
                      ("onset_f_measure", "onset-F"), ("abs_d_attack_slope", "|d attack-slope|")]:
        for m in A2_MODELS:
            piv = a2[a2.model == m].pivot_table(index="track", columns="level",
                                                values=meas)[LEVELS].dropna()
            _, p = friedmanchisquare(*[piv[l].values for l in LEVELS])
            ps.append(p)
            note = "  <- significant" if p < 0.05 else ("  <- marginal" if p < 0.1 else "")
            print(f"  {lab:18s} {LABEL[m]:12s} p = {p:.3f}{note}")
    k = len(ps)
    print(f"\n  {k} tests; smallest p = {min(ps):.3f}, Bonferroni x{k} = {min(ps) * k:.2f}")
    print(f"  above 0.1: {sum(1 for p in ps if p > 0.1)} of {k}")


def direction(a2):
    head("A2 direction - R-alone vs full: does context move R toward truth?")
    for m in A2_MODELS:
        dm = a2[a2.model == m]
        rr = dm[dm.level == "region_only"].set_index("track")
        ff = dm[dm.level == "full"].set_index("track")
        c = rr.index.intersection(ff.index)
        dv = ff.loc[c, "d_variation"].abs().mean() - rr.loc[c, "d_variation"].abs().mean()
        print(f"  {LABEL[m]:12s} |d variation| change {dv:+.2f}   "
              f"onset-F {rr.loc[c, 'onset_f_measure'].mean():.2f} -> "
              f"{ff.loc[c, 'onset_f_measure'].mean():.2f}")


def main():
    raw, agg, a2 = load()
    print(f"sources: benchmark_raw {len(raw)} rows (drums, {raw.track.nunique()} tracks x "
          f"{raw.model.nunique()} models) | A2 {len(a2)} drums rows "
          f"({a2.track.nunique()} tracks x {a2.model.nunique()} models x {a2.level.nunique()} levels)")
    table2(agg); table6(agg); table3(raw); a1(raw); b3(raw)
    table4(a2); attention(a2); table5(a2); friedman(a2); direction(a2)
    print("\nAll values above are computed from unrounded per-track data. Quote them as printed.")


if __name__ == "__main__":
    main()
