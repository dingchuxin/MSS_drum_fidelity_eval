"""A2 synthesis: quantify how a fixed region's separation drifts with input length,
and test the attention hypothesis (the transformer drifts more than the frame-local CNN).

Reads results/tables/input_length_length.csv (model x track x level x stem); drums stem.
  drift      = across-the-4-levels SD of each descriptor per (model, track); mean +- SE over tracks
  attention  = paired Wilcoxon of per-track drift, bs_roformer vs scnet (and vs htdemucs)
  direction  = region_only vs full: does more context move R TOWARD truth (|d_*| shrinks,
               onset-F grows) or merely change it?
Writes results/tables/a2_*.{md,csv} and results/figures/a2_drift.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mss_eval import report  # noqa: E402

CSV = ROOT / "results" / "tables" / "input_length_length.csv"
DESCS = ["dyn_attack_slope", "dyn_impulsiveness", "dyn_variation", "dyn_crest", "onset_f_measure"]
ORDER = ["htdemucs", "bs_roformer", "scnet"]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    return df[df.stem == "drums"].copy()


def drift_per_track(d: pd.DataFrame) -> pd.DataFrame:
    """Across-level SD of each descriptor, per (model, track) — the A2 instability signal."""
    return d.groupby(["model", "track"])[DESCS].std(ddof=0)


def drift_table(per: pd.DataFrame) -> pd.DataFrame:
    mean, sem = per.groupby("model").mean(), per.groupby("model").sem()
    out = pd.DataFrame(index=ORDER)
    for c in DESCS:
        out[c] = [f"{mean.loc[m, c]:.3f} ± {sem.loc[m, c]:.3f}" for m in ORDER]
    return out


def attention_test(per: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon
    piv = per.reset_index().pivot(index="track", columns="model")
    rows = []
    for desc in DESCS:
        for other in ["scnet", "htdemucs"]:
            bs, ot = piv[(desc, "bs_roformer")], piv[(desc, other)]
            try:
                _, p = wilcoxon(bs, ot)
            except ValueError:
                p = float("nan")
            rows.append({"descriptor": desc, "bs_roformer_vs": other,
                         "n_bs_greater": int((bs > ot).sum()), "of": len(bs),
                         "median_ratio": round(float((bs / ot.replace(0, np.nan)).median()), 2),
                         "wilcoxon_p": round(float(p), 4)})
    return pd.DataFrame(rows)


def direction_table(d: pd.DataFrame) -> pd.DataFrame:
    """region_only vs full: does context reduce the distortion (move R toward truth)?"""
    rows = []
    for m in ORDER:
        dm = d[d.model == m]
        rr = dm[dm.level == "region_only"].set_index("track")
        ff = dm[dm.level == "full"].set_index("track")
        common = rr.index.intersection(ff.index)
        row = {"model": m}
        for label, col in {"|d_variation|": "d_variation", "|d_attack_slope|": "d_attack_slope"}.items():
            a, b = rr.loc[common, col].abs().mean(), ff.loc[common, col].abs().mean()
            row[f"{label} R-only"] = round(a, 3)
            row[f"{label} full"] = round(b, 3)
            row[f"{label} Δ"] = round(b - a, 3)
        row["onsetF R-only"] = round(rr.loc[common, "onset_f_measure"].mean(), 3)
        row["onsetF full"] = round(ff.loc[common, "onset_f_measure"].mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def figure(per: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean, sem = per.groupby("model").mean(), per.groupby("model").sem()
    labels = {"dyn_attack_slope": "attack-slope", "dyn_impulsiveness": "impulsiveness",
              "dyn_variation": "variation", "onset_f_measure": "onset-F"}
    colors = {"htdemucs": "#8a8a8a", "bs_roformer": "#d1495b", "scnet": "#30638e"}
    cols = list(labels)
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 3.7))
    for ax, c in zip(axes, cols):
        ax.bar(ORDER, [mean.loc[m, c] for m in ORDER], yerr=[sem.loc[m, c] for m in ORDER],
               color=[colors[m] for m in ORDER], capsize=3)
        ax.set_title(labels[c], fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("drift of R across\ncontext lengths (SD)")
    fig.suptitle("A2 — drift of a fixed drums region with input length (mean ± SE, 20 tracks)\n"
                 "attack-slope: transformer > CNN (paired p=.044); impulsiveness: outlier-driven, n.s.; timing (onset-F): flat")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = ROOT / "results" / "figures" / "a2_drift.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure -> {out}")


def main():
    d = load()
    print(f"rows: {len(d)} | models: {sorted(d.model.unique())} | tracks: {d.track.nunique()}")
    per = drift_per_track(d)

    def show(title, frame, name, caption):
        print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
        print(frame.to_string())
        report.write_table(frame, name, caption)

    show("A2 DRIFT — drums region R across context lengths (mean ± SE across-level SD, 20 tracks)",
         drift_table(per), "a2_drift", "A2 drift of drums region across context lengths")
    show("ATTENTION TEST — bs_roformer drift vs scnet / htdemucs (paired, per track)",
         attention_test(per), "a2_attention_test", "A2 paired test: transformer vs CNN/hybrid drift")
    show("DIRECTION — region_only vs full: does more context move R toward truth?",
         direction_table(d), "a2_direction", "A2 region_only vs full fidelity")
    figure(per)
    print("\nA2 SYNTHESIS COMPLETE.")


if __name__ == "__main__":
    main()
