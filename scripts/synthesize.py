"""Synthesis layer: turn per-track benchmark metrics into the §4.7 claim-tests A1 and B3.

The full benchmark saved only aggregated medians, but the separated stems are cached
in data/stems_out/, so we regenerate the per-(model, track) DRUMS rows fast — SI-SDR +
the analyst dynamic/timing profile (the slow framewise BSS-Eval SDR/SIR/SAR already lives
in results/tables/benchmark_metrics.csv). Then:

  A1  the dynamic-profile distortion is MODEL-DEPENDENT
      A1a  cross-model spread of each descriptor vs the within-model between-track SD
      A1c  do models agree on WHICH tracks are most distorted? (between-model Spearman)
  B3  SI-SDR does NOT track the analyst-relevant distortions
      model-ranking flip (rank by SDR vs by analyst fidelity) + pooled Spearman (n=200)

Outputs: results/tables/synthesis_*.{md,csv} and results/figures/sdr_vs_analyst.png.
Focus stem: drums (rhythm study). Set FOCUS = STEMS below for the control-stem version.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mss_eval import analyst, metrics, report              # noqa: E402
from mss_eval.io import load_audio, load_config            # noqa: E402

STEMS = ["drums", "bass", "vocals", "other"]
FOCUS = ["drums"]                                          # rhythm study
RAW = ROOT / "results" / "tables" / "benchmark_raw.csv"


def _safe_name(s: str) -> str:
    return "".join("_" if c in '<>:"/\\|?*' else c for c in s).strip()


def build_raw() -> pd.DataFrame:
    import musdb
    cfg = load_config("datasets")
    sr = load_config("models").get("defaults", {}).get("sample_rate", 44100)
    root = ROOT / cfg["benchmark"]["root"]
    db = musdb.DB(root=str(root), subsets=cfg["benchmark"]["subset"], is_wav=True)
    stems_root = ROOT / "data" / "stems_out"
    models = sorted(d.name for d in stems_root.iterdir() if d.is_dir())
    print(f"models: {models} | focus stems: {FOCUS}", flush=True)
    rows = []
    for ti, t in enumerate(db, 1):
        refs = {s: t.targets[s].audio for s in FOCUS if s in t.targets}
        for model in models:
            cdir = stems_root / model / _safe_name(t.name)
            for s in FOCUS:
                p = cdir / f"{s}.wav"
                if not p.exists() or s not in refs:
                    continue
                est = load_audio(p, target_sr=sr)[0]
                row = {"model": model, "track": t.name, "stem": s,
                       "si_sdr": metrics.si_sdr(est, refs[s])}
                row.update(analyst.analyst_row(est, refs[s], sr))
                rows.append(row)
        print(f"  [{ti:2d}/50] {t.name[:44]}", flush=True)
    df = pd.DataFrame(rows)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW, index=False)
    print(f"raw table: {len(df)} rows -> {RAW}", flush=True)
    return df


# ---- A1: the dynamic profile is model-dependent ----------------------------

def a1_spread(df: pd.DataFrame, stem="drums") -> pd.DataFrame:
    d = df[df.stem == stem]
    descs = [c for c in d.columns if c.startswith("dyn_") or c.startswith("d_")]
    out = []
    for c in descs:
        cross = d.groupby("model")[c].mean().std(ddof=1)          # spread across model means
        within = d.groupby("model")[c].std(ddof=1).mean()        # typical between-track SD
        out.append({"descriptor": c, "cross_model_SD": round(cross, 3),
                    "within_model_SD": round(within, 3),
                    "ratio": round(cross / (within + 1e-9), 3)})
    return pd.DataFrame(out).sort_values("ratio", ascending=False).reset_index(drop=True)


def a1_rank_stability(df: pd.DataFrame, stem="drums", desc="d_variation") -> pd.DataFrame:
    piv = df[df.stem == stem].pivot_table(index="track", columns="model", values=desc)
    return piv.corr(method="spearman").round(3)


# ---- B3: SDR does not track analyst degradation ----------------------------

def b3_ranking(df: pd.DataFrame, stem="drums") -> pd.DataFrame:
    g = df[df.stem == stem].groupby("model")
    tab = pd.DataFrame({
        "si_sdr": g["si_sdr"].median(),
        "onset_f": g["onset_f_measure"].median(),
        "abs_d_attack": g["d_attack_slope"].apply(lambda x: x.abs().median()),
        "abs_d_variation": g["d_variation"].apply(lambda x: x.abs().median()),
    })
    tab["rank_SDR"] = tab["si_sdr"].rank(ascending=False).astype(int)
    tab["rank_onsetF"] = tab["onset_f"].rank(ascending=False).astype(int)
    tab["rank_transient"] = tab["abs_d_attack"].rank(ascending=True).astype(int)   # 1 = least distortion
    tab["rank_dynamics"] = tab["abs_d_variation"].rank(ascending=True).astype(int)
    return tab.round(3)


def b3_spearman(df: pd.DataFrame, stem="drums") -> pd.DataFrame:
    from scipy.stats import spearmanr
    d = df[df.stem == stem]
    specs = [("onset_f_measure", False, "higher SDR -> higher onset-F"),
             ("d_attack_slope", True, "higher SDR -> smaller |Δ attack|"),
             ("d_variation", True, "higher SDR -> smaller |Δ variation|"),
             ("d_crest", True, "higher SDR -> smaller |Δ crest|")]
    out = []
    for col, use_abs, note in specs:
        y = d[col].abs() if use_abs else d[col]
        rho, p = spearmanr(d["si_sdr"], y)
        out.append({"analyst_metric": (f"|{col}|" if use_abs else col),
                    "spearman_vs_SI_SDR": round(rho, 3), "p_value": round(p, 4),
                    "n": len(d), "expected_if_SDR_tracked_it": note})
    return pd.DataFrame(out)


def figure(df: pd.DataFrame, stem="drums"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.stem == stem]
    models = sorted(d.model.unique())
    colors = dict(zip(models, plt.cm.tab10.colors))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [("onset_f_measure", "onset F-measure  (timing fidelity, higher=better)", lambda s: s),
              ("d_attack_slope", "|Δ attack-slope|  (transient distortion, lower=better)", lambda s: s.abs())]
    for ax, (ycol, ylab, tf) in zip(axes, panels):
        for m in models:
            dm = d[d.model == m]
            ax.scatter(dm["si_sdr"], tf(dm[ycol]), s=18, alpha=0.6, color=colors[m], label=m)
        ax.set_xlabel("SI-SDR (dB)  —  developer metric")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="model")
    fig.suptitle(f"SI-SDR vs. analyst fidelity — {stem} stem (4 models x 50 tracks)\n"
                 "if SDR tracked what a drummer hears, these would trend cleanly")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "sdr_vs_analyst.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure -> {out}", flush=True)


def main():
    if RAW.exists():
        print(f"loading cached raw table: {RAW}")
        df = pd.read_csv(RAW)
    else:
        print("building per-track raw table from cached stems...")
        df = build_raw()

    def show(title, frame, name, caption, index=True):
        print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
        print(frame.to_string(index=index))
        report.write_table(frame, name, caption)

    show("A1a — cross-model spread of the DRUMS dynamic profile",
         a1_spread(df, "drums"), "synthesis_A1_spread",
         "A1a: cross-model vs within-model spread (drums stem)", index=False)

    a1c = a1_rank_stability(df, "drums", "d_variation")
    off = a1c.values[np.triu_indices_from(a1c.values, k=1)]
    show("A1c — do models agree on WHICH tracks are most distorted? (Spearman on d_variation)",
         a1c, "synthesis_A1c_rankstability",
         "A1c: between-model Spearman on d_variation (drums)")
    print(f"mean off-diagonal Spearman: {np.nanmean(off):.3f}")

    show("B3 — model RANKING: by SDR vs by analyst fidelity (drums)",
         b3_ranking(df, "drums"), "synthesis_B3_ranking",
         "B3: model ranking by SDR vs analyst metrics (drums)")

    show("B3 — pooled Spearman: does SI-SDR predict analyst distortion? (drums, n=200)",
         b3_spearman(df, "drums"), "synthesis_B3_spearman",
         "B3: pooled Spearman SI-SDR vs analyst metrics (drums)", index=False)

    figure(df, "drums")
    print("\nSYNTHESIS COMPLETE.")


if __name__ == "__main__":
    main()
