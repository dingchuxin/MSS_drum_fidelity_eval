"""Regenerate the paper figures in the shared house style (PDF for the paper + PNG to view).
Fig 1 (dissociation): SI-SDR vs timing | SI-SDR vs transient shape | rank-flip slopegraph.
Fig 2 (A2): paired transient drift (transformer vs CNN) | drift by descriptor (timing flat)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mss_eval import figstyle as fs   # noqa: E402

fs.apply()
T, F = ROOT / "results" / "tables", ROOT / "results" / "figures"
raw = pd.read_csv(T / "benchmark_raw.csv")                 # drums, 200 rows
a2 = pd.read_csv(T / "input_length_length.csv")
a2 = a2[a2.stem == "drums"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(F / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    d = raw
    xs = np.array([d.si_sdr.min(), d.si_sdr.max()])

    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)

    def scatter(ax, y, ylab, tag, clip=None, fit=True):
        yy = y.mask(y > clip) if clip is not None else y   # points above the cap are hidden, not clipped to it
        for m in fs.MODEL_ORDER:
            dm = d[d.model == m]
            ax.scatter(dm.si_sdr, yy[dm.index], s=9, alpha=0.55,
                       color=fs.MODEL_COLORS[m], linewidths=0, label=fs.MODEL_LABEL[m])
        if fit:
            b, a0 = np.polyfit(d.si_sdr, y, 1)
            ax.plot(xs, a0 + b * xs, color=fs.MUTE, lw=0.8, ls="--")
        rho, _ = spearmanr(d.si_sdr, y)
        ax.text(0.05, 0.95, rf"$\rho = {rho:+.2f}$", transform=ax.transAxes,
                va="top", fontsize=7.5, bbox=box)
        ax.set_xlabel("SI-SDR (dB)"); ax.set_ylabel(ylab)
        ax.set_title(tag)

    scatter(axes[0], d.onset_f_measure, "onset F-measure", "(a) timing")
    yatt = d.d_attack_slope.abs()
    clip = float(np.percentile(yatt, 95))
    above = int((yatt > clip).sum())
    scatter(axes[1], yatt, r"$|\Delta$ attack-slope$|$", "(b) transient shape", clip=clip, fit=False)
    axes[1].set_ylim(top=clip * 1.05)
    if above:
        axes[1].text(0.95, 0.95, f"$\\uparrow$ {above} above axis top", transform=axes[1].transAxes,
                     va="top", ha="right", fontsize=6.5, color=fs.MUTE, bbox=box)

    # (c) rank-flip slopegraph
    ax = axes[2]
    g = d.groupby("model")
    sdr_rank = g.si_sdr.median().rank(ascending=False)
    att_rank = g.d_attack_slope.apply(lambda x: x.abs().median()).rank(ascending=True)
    for m in fs.MODEL_ORDER:
        c = fs.MODEL_COLORS[m]
        ax.plot([0, 1], [sdr_rank[m], att_rank[m]], color=c, lw=1.8, marker="o", ms=4.5)
        ax.text(-0.16, sdr_rank[m], fs.MODEL_LABEL[m], ha="right", va="center", fontsize=7.5, color=c)
        ax.text(1.16, att_rank[m], fs.MODEL_LABEL[m], ha="left", va="center", fontsize=7.5, color=c)
    ax.set_xlim(-1.35, 2.35); ax.set_ylim(4.6, 0.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["by\nSDR", "by transient\nfidelity"], fontsize=7.5)
    ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("(c) the ranking inverts")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", ms=5, color=fs.MODEL_COLORS[m],
                      label=fs.MODEL_LABEL[m]) for m in fs.MODEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               handletextpad=0.3, columnspacing=1.6, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.6)
    save(fig, "fig1_dissociation")


def fig2():
    per = a2.groupby(["model", "track"])[
        ["dyn_attack_slope", "dyn_variation", "onset_f_measure"]].std(ddof=0)
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 2.7),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # (a) paired attack-slope drift: SCNet vs BS-Roformer
    ax = axes[0]
    piv = per.reset_index().pivot(index="track", columns="model")["dyn_attack_slope"]
    sc, bs = piv["scnet"], piv["bs_roformer"]
    for t in piv.index:
        up = bs[t] > sc[t]
        ax.plot([0, 1], [sc[t], bs[t]],
                color=fs.MODEL_COLORS["bs_roformer"] if up else fs.MUTE, lw=0.9, alpha=0.55)
    ax.scatter(np.zeros(len(sc)), sc, s=14, color=fs.MODEL_COLORS["scnet"], zorder=3, linewidths=0)
    ax.scatter(np.ones(len(bs)), bs, s=14, color=fs.MODEL_COLORS["bs_roformer"], zorder=3, linewidths=0)
    ax.plot([-0.16, 0.16], [sc.median()] * 2, color=fs.MODEL_COLORS["scnet"], lw=2.2)
    ax.plot([0.84, 1.16], [bs.median()] * 2, color=fs.MODEL_COLORS["bs_roformer"], lw=2.2)
    _, p = wilcoxon(bs, sc); n = int((bs > sc).sum())
    ax.set_xlim(-0.45, 1.45); ax.set_xticks([0, 1]); ax.set_xticklabels(["SCNet-XL", "BS-Roformer"])
    ax.set_ylabel("attack-slope drift across context")
    ax.text(0.03, 0.97, f"{n}/20 tracks higher\nfor BS-Roformer  ($p = {p:.3f}$)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
    ax.set_title("(a) transient drift: transformer vs CNN")

    # (b) drift by descriptor, per model — timing near zero, shape not
    ax = axes[1]
    descs = ["onset_f_measure", "dyn_variation", "dyn_attack_slope"]
    dlab = {"onset_f_measure": "onset-F\n(timing)", "dyn_variation": "local dynamic\nvariation",
            "dyn_attack_slope": "attack-slope"}
    mean, sem = per.groupby("model").mean(), per.groupby("model").sem()
    ys = np.arange(len(descs))
    for mi, m in enumerate(["htdemucs", "bs_roformer", "scnet"]):
        ax.errorbar(mean.loc[m, descs].values, ys + (mi - 1) * 0.22,
                    xerr=sem.loc[m, descs].values, fmt="o", ms=4.5, lw=1.1, capsize=2,
                    color=fs.MODEL_COLORS[m], label=fs.MODEL_LABEL[m])
    ax.set_yticks(ys); ax.set_yticklabels([dlab[d] for d in descs]); ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(left=0)
    ax.set_xlabel("drift of region R across context lengths (SD)")
    ax.legend(loc="lower right", handletextpad=0.2, labelspacing=0.3)
    ax.set_title("(b) timing stays put; shape drifts")

    fig.tight_layout(w_pad=1.8)
    save(fig, "fig2_a2_drift")


if __name__ == "__main__":
    fig1(); fig2()
    print("wrote fig1_dissociation and fig2_a2_drift (.pdf + .png) to results/figures/")
