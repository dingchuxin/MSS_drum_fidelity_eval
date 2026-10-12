"""Figures of the article and its online supplement.

Every figure is drawn from the result files of the MSS_drum_fidelity_eval repository
(results/...); no number is typed in by hand except the design parameters in the design figure,
which restate the Methods. Output: vector PDF at the text width (34 pc = 14.4 cm).

    python scripts/make_figures_v2.py [--data /path/to/MSS_drum_fidelity_eval] [--out DIR] [--formats pdf,tiff]

<DIR> defaults to results/phase5/figures. TIFF output: 1200 dpi, as for line art; the PDFs are the primary output.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ style
W = 34 / 6.0225  # text width in inches (34 pc)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CRIT = "#d03b3b"
MODELS = ["Spleeter", "HT-Demucs", "BS-RoFormer", "SCNet-XL"]
KEY = {"Spleeter": "spleeter", "HT-Demucs": "htdemucs", "BS-RoFormer": "bs_roformer", "SCNet-XL": "scnet"}
COL = {"Spleeter": "#2a78d6", "HT-Demucs": "#eb6834", "BS-RoFormer": "#1baf7a", "SCNet-XL": "#eda100"}
MRK = {"Spleeter": "o", "HT-Demucs": "s", "BS-RoFormer": "^", "SCNet-XL": "D"}
# line styles carry model identity in greyscale, together with the marker shapes
LS = {"Spleeter": "-", "HT-Demucs": (0, (4.5, 1.5)), "BS-RoFormer": (0, (1, 1.2)),
      "SCNet-XL": (0, (6, 1.5, 1.2, 1.5))}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def panel(ax, letter, title=None):
    ax.set_title(f"({letter})" + (f" {title}" if title else ""), fontsize=8, loc="left", pad=4)


def hgrid(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def vgrid(ax):
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Figure 1: design
def fig1(out):
    fig = plt.figure(figsize=(W, 4.9))
    # ---------------- panel (a): the comparison
    ax = fig.add_axes([0.0, 0.52, 1.0, 0.48]); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
    ax.text(0.3, 49.5, "(a) Main benchmark: the same measurements on the estimate and on the true stem",
            fontsize=8, fontweight="bold", va="top")

    def box(x, y, w, h, text, fc="#ffffff", fs=6.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc=fc, ec=MUTED, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.2)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.7, color=INK2, shrinkA=0, shrinkB=0))

    box(0.5, 32, 14, 9, "Mixture\n(MUSDB18-HQ\ntest track)")
    box(19, 32, 16, 9, "Separator\n(four models,\nfixed settings)", fc="#f3f2ee")
    box(40, 32, 15, 9, "Estimated\ndrum stem")
    box(40, 18, 15, 9, "True\ndrum stem")
    box(40, 3, 15, 10, "Floors: true\ndrum stem +\nleakage or\nwhite noise", fc="#f3f2ee")
    box(61, 20.5, 17, 20.5, "Onset detector,\nstroke windows,\ntiming and\nshape measures")
    box(61, 3, 17, 12, "Onset detector,\ntiming measures\nonly")
    box(83, 3, 16.5, 38, "Estimate (or\nfloor) minus\nreference,\nper track;\nmedian over\n50 tracks")
    g = 1.0  # gap between an arrow's ends and the box edges (box pad 0.2 included)
    arrow(14.5 + g, 36.5, 19 - g, 36.5); arrow(35 + g, 36.5, 40 - g, 36.5)
    arrow(55 + g, 36.5, 61 - g, 36.5)                 # estimate -> detector
    arrow(55 + g, 23.5, 61 - g, 23.5)                 # true stem -> detector (reference)
    arrow(55 + g, 19.5, 61 - g, 13.0)                 # true stem -> timing-only detector (reference)
    arrow(55 + g, 8, 61 - g, 8)                       # floors -> timing-only detector
    arrow(47.5, 18 - g, 47.5, 13 + g)                 # true stem -> floors
    arrow(78 + g, 30.75, 83 - g, 30.75); arrow(78 + g, 9, 83 - g, 9)
    ax.text(19.5, 27.0, "Timing: detected onsets matched\nwithin ±50 ms; typical deviation,\nshare displaced by > 10 ms,\nshare of strokes missed",
            fontsize=6.6, color=INK2, va="top", ha="center", multialignment="left", linespacing=1.25)
    ax.text(19.5, 13.0, "Shape: stroke peaks, inter-stroke\nlevels, gaps, accent contrast,\nenvelope descriptors (SD units)",
            fontsize=6.6, color=INK2, va="top", ha="center", multialignment="left", linespacing=1.25)

    # ---------------- panel (b): Experiment 2
    bx = fig.add_axes([0.0, 0.0, 1.0, 0.5]); bx.set_axis_off(); bx.set_xlim(0, 100); bx.set_ylim(0, 50)
    bx.text(0.3, 49.5, "(b) Experiment 2: a fixed 5-s passage R, cut from the mixture in different ways",
            fontsize=8, fontweight="bold", va="top")
    lo_t, hi_t = -24.5, 31.5

    def tx(t):
        return 2 + (t - lo_t) / (hi_t - lo_t) * 96

    y0 = 41
    bx.add_patch(Rectangle((tx(lo_t), y0 - 0.5), tx(hi_t) - tx(lo_t), 1.0, fc=AXIS, ec="none"))
    bx.text(tx(lo_t), y0 + 1.8, "track (mixture)", fontsize=6.6, color=INK2)
    bx.add_patch(Rectangle((tx(0), y0 - 2.2), tx(5) - tx(0), 4.4, fc="#1c5cab", ec="none"))
    bx.text(tx(2.5), y0 + 3.2, "R (5 s)", ha="center", fontsize=6.5, fontweight="bold")
    for o, yy, lab in ((0.0, 32.5, "42-s window at the first offset (0 s)"),
                       (10.9, 24.5, "42-s window at the last offset (10.9 s)")):
        a0, b0 = -12 - o, -12 - o + 42
        bx.add_patch(Rectangle((tx(a0), yy - 1.5), tx(b0) - tx(a0), 3.0, fc="#ffffff", ec=INK2, lw=0.7))
        bx.add_patch(Rectangle((tx(0), yy - 1.5), tx(5) - tx(0), 3.0, fc="#1c5cab", ec="none", alpha=0.35))
        k = 1
        while a0 + k * 11.9 < b0 - 0.01:
            c0 = a0 + k * 11.9
            bx.plot([tx(c0), tx(c0)], [yy - 2.3, yy + 2.3], color=INK, lw=0.9)
            k += 1
        bx.text(tx(a0), yy + 2.4, lab, fontsize=6.6, color=INK2, va="bottom")
    bx.text(2, 20.3, "Black ticks: chunk boundaries, anchored at the window's start; shown for Spleeter's 11.9-s chunks without overlap.",
            fontsize=6.6, color=INK2, va="center")
    bx.text(2, 17.8, "The 12 offsets (0 to 10.9 s) move R through one Spleeter chunk and at least one full period",
            fontsize=6.6, color=INK2, va="center")
    bx.text(2, 15.4, "of each other model's chunk grid.", fontsize=6.6, color=INK2, va="center")
    items = [
        ("Chunk position", "12 offsets of k \u00d7 11.9/12 s (k = 0, \u2026, 11)"),
        ("Sub-frame shift", "the window at the first offset, shifted by 1/3 and 2/3 of the model's STFT hop"),
        ("Excerpt first", "R separated alone (5 s) and in 10-s and 20-s windows; compared with the full-track output"),
        ("Overlap swap", "overlaps exchanged: BS-RoFormer at 75 % (native 50 %), SCNet-XL at 50 % (native 75 %)"),
        ("Randomness", "HT-Demucs seeded in every condition, and two unseeded runs at the package defaults"),
    ]
    for i, (h, t) in enumerate(items):
        yy = 12.2 - i * 2.75
        bx.text(2, yy, h + ":", fontsize=6.6, fontweight="bold", va="center")
        bx.text(18, yy, t, fontsize=6.6, va="center", color=INK2)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2: excerpt
def fig2(root, out):
    """Excerpt figure. Envelopes from the 3-s excerpt audio; onsets and their status from the full-track
    analysis (results/phase4/d5_excerpt_strokes.csv), so the figure shows what every statistic counts."""
    import soundfile as sf
    sys.path.insert(0, str(root / "src"))
    from mss_eval import strokes as S
    d = root / "results" / "phase1" / "figure_excerpt"
    st = pd.read_csv(root / "results" / "phase4" / "d5_excerpt_strokes.csv")
    st = st[st.excerpt == "current"]
    t0 = float(st.t0_s.iloc[0])
    counted = ["within 10 ms", "displaced", "missed"]
    ref = np.sort(st[st.full_status.isin(counted)].t_ref_s.unique())
    rows = [("Reference", "reference")] + [(m, KEY[m]) for m in MODELS]
    env = {}
    for lab, k in rows:
        x, sr = sf.read(str(d / f"{k}.wav"), always_2d=True)
        e = S.rms_env(x.mean(1), S.W10)
        env[lab] = 20 * np.log10(e + 1e-12)
    tt = np.arange(len(env["Reference"])) * S.HOP / sr + t0
    fig, axes = plt.subplots(5, 1, figsize=(W, 5.0), sharex=True, gridspec_kw=dict(hspace=0.42))
    lo, hi = -72, -2
    res = {}
    for ax, (lab, k) in zip(axes, rows):
        ax.fill_between(tt, lo, env[lab], color="#d8d6cf", lw=0)
        ax.plot(tt, env[lab], color=INK2, lw=0.6)
        for r in ref:
            ax.axvline(r, color=MUTED, lw=0.5, alpha=0.6, zorder=0)
        ax.set_ylim(lo, hi + 12)
        ax.set_yticks([-60, -40, -20]); ax.tick_params(axis="y", labelsize=6.5)
        ax.set_title(lab if lab != "Reference" else "True drum stem", fontsize=7, loc="left", pad=1.5,
                     fontweight="bold")
        ytop = hi + 6
        if lab == "Reference":
            ax.plot(ref, np.full(len(ref), ytop), "v", ms=4.2, color=INK, mec="white", mew=0.5)
            continue
        g = st[st.model == f"v1_{k}"]
        mt = g[g.full_status.isin(["within 10 ms", "displaced"])]
        t_est = (mt.t_ref_s + mt.full_dev_ms / 1000).to_numpy()
        dev = mt.full_dev_ms.to_numpy()
        disp = np.abs(dev) > 10
        ax.plot(t_est[~disp], np.full((~disp).sum(), ytop), "v", ms=4.2, color=INK, mec="white", mew=0.5)
        ax.plot(t_est[disp], np.full(disp.sum(), ytop), "D", ms=4.6, color=CRIT, mec="white", mew=0.5)
        for tj, dv in zip(t_est[disp], dev[disp]):
            ax.text(tj, ytop - 7.5, f"{abs(dv):.0f} ms", fontsize=6.6, color=CRIT, ha="center", va="top",
                    bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
        extra = g[g.full_status == "extra"].t_extra_s.to_numpy()
        ax.plot(extra, np.full(len(extra), ytop), "v", ms=4.2, mfc="white", mec=INK, mew=0.7)
        missed = g[g.full_status == "missed"].t_ref_s.to_numpy()
        ax.plot(missed, np.full(len(missed), ytop), "x", ms=5, color=CRIT, mew=1.1)
        res[lab] = dict(n_ref=len(ref), matched=len(mt), extra=len(extra), missed=len(missed),
                        displaced=np.round(dev[disp], 1).tolist())
    axes[2].set_ylabel("10-ms RMS level (dBFS)")
    axes[-1].set_xlabel("Time in track (s)")
    axes[-1].set_xlim(t0 - 0.02, t0 + 3.02)
    handles = [Line2D([], [], ls="", marker="v", ms=4.5, color=INK, label="within 10 ms"),
               Line2D([], [], ls="", marker="D", ms=4.6, color=CRIT, label="displaced > 10 ms"),
               Line2D([], [], ls="", marker="v", ms=4.5, mfc="white", mec=INK, label="extra onset")]
    if any(r["missed"] for r in res.values()):   # legend entry only if a missed stroke is drawn
        handles.append(Line2D([], [], ls="", marker="x", ms=5, color=CRIT, mew=1.1, label="missed stroke"))
    handles.append(Line2D([], [], color=MUTED, lw=0.8, label="true-stem onset"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.53, 1.0), fontsize=6.5,
               handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(top=0.915, bottom=0.075, left=0.08, right=0.975)
    fig.savefig(out)
    plt.close(fig)
    return res


# ------------------------------------------------------------------ Figure 3: timing
def fig3(root, out):
    R = root / "results"
    h = pd.read_csv(R / "phase2" / "fig_timing_error_distribution.csv")
    xs = np.arange(0.5, 50.0, 0.5)

    def ccdf(lab):
        g = h[h.label == lab]
        tot = g["count"].sum()
        return np.array([g.loc[(g.bin_lo_ms >= x) | (g.bin_hi_ms <= -x), "count"].sum() / tot for x in xs])

    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw=dict(width_ratios=[1.22, 1.1], wspace=0.34))
    a.axvspan(8, 14, color="#eeeeea", lw=0, zorder=0)
    a.text(11, 0.62, "drummers'\nswing\nthresholds", ha="center", va="top", fontsize=6.6, color=INK2)
    for lab, name, ls in (("floor_white_20dB", "white noise 20 dB below", (0, (7, 2, 1.5, 2, 1.5, 2))),
                          ("floor_leak_20dB", "leakage 20 dB below", (0, (0.8, 2.2)))):
        a.plot(xs, 100 * ccdf(lab), color=MUTED, lw=1.0, ls=ls, label=name)
    for k, m in enumerate(MODELS):
        y = 100 * ccdf(f"v1_{KEY[m]}")
        a.plot(xs, y, color=COL[m], lw=1.4, ls=LS[m], label=m)
        sel = slice(4 + k, None, 5)          # a marker every 2.5 ms, staggered between models
        a.plot(xs[sel], y[sel], ls="", marker=MRK[m], ms=3.2, color=COL[m], mec="white", mew=0.5)
    a.set_yscale("log"); a.set_ylim(0.05, 100); a.set_xlim(0, 50)
    a.set_yticks([0.1, 1, 10, 100]); a.set_yticklabels(["0.1", "1", "10", "100"])
    a.set_xlabel("Absolute deviation $x$ of matched detected onsets (ms)")
    a.set_ylabel("Onsets deviating by more than $x$ (%)")
    hgrid(a)
    panel(a, "a")
    hs, ls_ = a.get_legend_handles_labels()
    # legend: models with their markers
    mh = [Line2D([], [], color=COL[m], lw=1.4, ls=LS[m], marker=MRK[m], ms=3.2, mec="white", mew=0.5)
          for m in MODELS]
    a.legend(mh + hs[:2], MODELS + ls_[:2], loc="upper right", fontsize=6.6, handlelength=2.8,
             borderaxespad=0.2, labelspacing=0.3)

    # (b) per-track difference to the leakage floor at the model's own SIR
    c = pd.read_csv(R / "phase3" / "c4_by_track.csv")
    rng = np.random.default_rng(20260925)
    lo_all, hi_all = 0.0, 0.0
    for k, m in enumerate(MODELS):
        mm = c[c.signal == f"v1_{KEY[m]}"].set_index("track").share_gt_10ms
        fl = c[c.signal == f"floor_{KEY[m]}_leak"].set_index("track").share_gt_10ms.loc[mm.index]
        dd = 100 * (mm - fl)
        x = k + rng.uniform(-0.18, 0.18, len(dd))
        b.plot(x, dd, ls="", marker=MRK[m], ms=3.0, color=COL[m], alpha=0.85, mec="white", mew=0.3)
        med = np.median(dd)
        b.plot([k - 0.3, k + 0.3], [med, med], color=INK, lw=1.2)
        b.text(k + 0.33, med, f"{med:+.1f}".replace("-", "\u2212"), fontsize=6.6, va="center", color=INK,
               bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.8))
        above = int((dd > 0).sum())
        b.text(k, 44.5, f"{above}/50", ha="center", va="bottom", fontsize=6.6, color=INK2)
        lo_all = min(lo_all, dd.min()); hi_all = max(hi_all, dd.max())
    b.axhline(0, color=INK2, lw=0.6)
    b.set_xticks(range(4)); b.set_xticklabels(["Spleeter", "HT-\nDemucs", "BS-\nRoFormer", "SCNet-\nXL"], fontsize=6.6)
    b.set_xlim(-0.5, 3.8)
    assert lo_all > -32 and hi_all < 44, (lo_all, hi_all)   # every track is inside the axis
    b.set_ylim(-32, 55)
    b.set_yticks([-30, -20, -10, 0, 10, 20, 30, 40])
    b.text(1.65, 50.5, "tracks above the floor", ha="center", va="bottom", fontsize=6.6, color=INK2)
    b.set_ylabel("Share displaced > 10 ms: model\nminus its leakage floor (points)")
    hgrid(b)
    panel(b, "b")
    fig.subplots_adjust(top=0.88, bottom=0.17, left=0.09, right=0.99)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 4: shape forest
def fig4(root, out):
    R = root / "results" / "phase3" / "rerun"
    d = pd.read_csv(R / "sd_units_v3.csv")
    env = [("level_db", "-", "Level"), ("crest", "-", "Crest factor"), ("impulsiveness", "-", "Waveform kurtosis"),
           ("variation", "-", "Envelope variation"), ("env_skew", "-", "Envelope skewness"),
           ("env_flux", "-", "Envelope flux"), ("attack_slope", "-", "10-ms rise (P90)")]
    stk = [("peak_db", "all", "Stroke peak"), ("floor_db", "all", "Inter-stroke level"),
           ("valley_db", "all", "Valley depth"), ("pre_db", "all", "Pre-stroke level"),
           ("jump_db", "all", "Level jump"), ("rise_ms", "all", "Rise time"),
           ("accent", "all", "Accent contrast"), ("accent", "matched", "Accent contrast,\nmatched strokes")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.5), gridspec_kw=dict(wspace=0.78))
    off = {m: o for m, o in zip(MODELS, (-0.27, -0.09, 0.09, 0.27))}
    for ax, rows, lab in ((a, env, "a"), (b, stk, "b")):
        for r, (meas, st, name) in enumerate(rows):
            if r % 2 == 0:
                ax.axhspan(r - 0.5, r + 0.5, color="#f5f4f0", lw=0, zorder=0)
            for m in MODELS:
                g = d[(d.label == f"v1_{KEY[m]}") & (d.measure == meas) & (d.stroke_set == st)].iloc[0]
                y = r + off[m]
                ax.plot([g.median_ci_lo, g.median_ci_hi], [y, y], color=COL[m], lw=1.0, solid_capstyle="round")
                ax.plot(g.median_sd, y, ls="", marker=MRK[m], ms=3.8, color=COL[m], mec="white", mew=0.5)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for _, _, n in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Change (between-track SDs)")
        ax.set_xlim(-0.62, 1.08)
        vgrid(ax)
        panel(ax, lab, "Envelope descriptors" if lab == "a" else "Stroke measures")
    mh = [Line2D([], [], color=COL[m], lw=1.0, marker=MRK[m], ms=3.8, mec="white", mew=0.5) for m in MODELS]
    fig.legend(mh, MODELS, loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.0), fontsize=6.8)
    fig.subplots_adjust(top=0.93, bottom=0.2, left=0.2, right=0.985)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 5: SDR vs fidelity
def fig5(root, out):
    from scipy.stats import spearmanr
    R = root / "results" / "phase3"
    f = pd.read_csv(R / "fig_sdr_vs_fidelity_v3.csv")
    U = pd.read_csv(R / "rerun" / "units.csv").set_index("measure")
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.3), sharey="row", gridspec_kw=dict(wspace=0.12, hspace=0.42))
    for k, m in enumerate(MODELS):
        g = f[f.label == f"v1_{KEY[m]}"]
        x = g.si_sdr.to_numpy()
        for r, (col, lab) in enumerate((("abs_level", "|Change in level| (SDs)"),
                                        ("share", "Share displaced > 10 ms (%)"))):
            ax = axes[r, k]
            if col == "abs_level":
                y = (g.est_level_db - g.ref_level_db).abs().to_numpy() / U.loc["level_db", "sd"]
            else:
                y = 100 * g.share_gt_10ms.to_numpy()
            ax.plot(x, y, ls="", marker=MRK[m], ms=3.0, color=COL[m], mec="white", mew=0.3, alpha=0.9)
            rho = spearmanr(x, y)[0]
            ax.text(0.97, 0.95, "$\\rho$ = " + f"{rho:+.2f}".replace("-", "\u2212"), transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.6)
            hgrid(ax)
            if r == 0:
                ax.set_title(m, fontsize=7.5, loc="center", fontweight="bold")
            if k == 0:
                ax.set_ylabel(lab)
            if r == 1:
                ax.set_xlabel("SI-SDR (dB)")
    fig.subplots_adjust(top=0.92, bottom=0.13, left=0.09, right=0.99)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 6: Experiment 2
def fig6(root, out):
    R = root / "results"
    po = pd.read_csv(R / "phase2" / "b3_per_onset_drift.csv.gz")
    sel = pd.read_csv(R / "phase3" / "c6_selection_summary.csv")
    tdr = pd.read_csv(R / "phase2" / "b3_timing_drift_by_track.csv")
    fig = plt.figure(figsize=(W, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.95, 1.0], wspace=0.5)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    # (a) ECDF of per-onset drift (fully matched onsets)
    # SCNet-XL is drawn first, so that BS-RoFormer's markers, which fall at nearly the same points, stay visible
    for m in ["Spleeter", "HT-Demucs", "SCNet-XL", "BS-RoFormer"]:
        v = np.sort(po[po.model == KEY[m]].sd_ms.to_numpy())
        y = np.arange(1, len(v) + 1) / len(v) * 100
        a.plot(v, y, color=COL[m], lw=1.3, ls=LS[m], label=m)
        idx = np.searchsorted(y, [25, 50, 75, 95])
        big = m == "SCNet-XL"
        a.plot(v[idx], y[idx], ls="", marker=MRK[m], ms=4.6 if big else 3.4, color=COL[m],
               mfc="white" if big else COL[m], mec=COL[m] if big else "white", mew=0.9 if big else 0.5)
    a.axvline(0.2, color=INK2, lw=0.6, ls=(0, (3, 2)))
    a.text(0.17, 99, "detector\nprecision\n(P95)", fontsize=6.6, color=INK2, va="top", ha="right")
    a.set_xscale("log"); a.set_xlim(0.01, 30)
    a.set_xticks([0.01, 0.1, 1, 10]); a.set_xticklabels(["0.01", "0.1", "1", "10"])
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_xlabel("SD across 12 chunk positions (ms)")
    a.set_ylabel("Fully matched onsets (%)")
    a.set_ylim(0, 100)
    hgrid(a)
    panel(a, "a")
    mh = [Line2D([], [], color=COL[m], lw=1.3, ls=LS[m], marker=MRK[m], ms=4.6 if m == "SCNet-XL" else 3.4,
                 mfc="white" if m == "SCNet-XL" else COL[m], mec=COL[m] if m == "SCNet-XL" else "white",
                 mew=0.9 if m == "SCNet-XL" else 0.5) for m in MODELS]
    a.legend(mh, MODELS, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=6.6,
             labelspacing=0.3, handlelength=2.6, columnspacing=1.0, handletextpad=0.4)
    # (b) detection status changes with chunk position
    for k, m in enumerate(MODELS):
        s = sel[(sel.model == KEY[m]) & (sel.factor == "F1")].iloc[0]
        some, none = 100 * s.share_matched_some, 100 * s.share_matched_none
        b.plot(some, k - 0.17, ls="", marker=MRK[m], ms=4.2, color=COL[m], mec="white", mew=0.5)
        b.plot(none, k + 0.17, ls="", marker=MRK[m], ms=4.0, mfc="white", mec=COL[m], mew=1.0)
        b.text(some + 0.6, k - 0.17, f"{some:.1f}", fontsize=6.6, color=INK2, va="center")
        b.text(none + 0.6, k + 0.17, f"{none:.1f}", fontsize=6.6, color=INK2, va="center")
    b.set_yticks(range(4)); b.set_yticklabels(MODELS); b.set_ylim(3.55, -0.55)
    b.tick_params(axis="y", length=0)
    b.set_xlim(0, 15)
    b.set_xlabel("Reference onsets (%)")
    vgrid(b)
    panel(b, "b")
    b.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.2, color=INK2, label="detected at some positions only"),
                      Line2D([], [], ls="", marker="o", ms=4.0, mfc="white", mec=INK2, mew=1.0, label="never detected")],
             loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=6.6, handletextpad=0.2, labelspacing=0.3)
    # (c) overlap swap: per-passage median drift at 50 % vs 75 % overlap
    for k, (m, nat) in enumerate((("BS-RoFormer", "50"), ("SCNet-XL", "75"))):
        g = tdr[tdr.model == KEY[m]].pivot_table(index="track", columns="factor", values="median_sd_ms")
        v50 = g["F1"] if nat == "50" else g["swap"]
        v75 = g["swap"] if nat == "50" else g["F1"]
        x0, x1 = 2 * k, 2 * k + 1
        for p, q in zip(v50, v75):
            c.plot([x0, x1], [p, q], color=COL[m], lw=0.4, alpha=0.45)
        c.plot([x0, x1], [np.median(v50), np.median(v75)], color=INK, lw=1.3, marker=MRK[m], ms=4.4,
               mfc=COL[m], mec=INK, mew=0.6)
        c.text(x0 + 0.5, -0.27, m, transform=c.get_xaxis_transform(), ha="center", va="top", fontsize=6.6)
    c.set_xticks([0, 1, 2, 3]); c.set_xticklabels(["50 %\n(native)", "75 %", "50 %", "75 %\n(native)"])
    c.set_xlim(-0.45, 3.55)
    c.set_ylim(0, None)
    c.set_ylabel("Median SD per passage (ms)")
    hgrid(c)
    panel(c, "c")
    fig.subplots_adjust(top=0.9, bottom=0.3, left=0.075, right=0.975)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2: the error budget at a glance
def fig_budget(root, out):
    """Five terms of the error budget per model, with 95 % bootstrap CIs of the median over tracks."""
    R = root / "results"
    a2 = pd.read_csv(R / "phase2" / "a2_thresholds.csv")
    c4 = pd.read_csv(R / "phase3" / "c4_summary.csv").set_index("signal")
    sdu = pd.read_csv(R / "phase3" / "rerun" / "sd_units_v3.csv")
    rng = np.random.default_rng(20260925)

    def boot_median(v, B=10000):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        i = rng.integers(0, len(v), (B, len(v)))
        meds = np.median(v[i], axis=1)
        return np.median(v), np.percentile(meds, 2.5), np.percentile(meds, 97.5)

    terms = {}
    for m in MODELS:
        k = KEY[m]
        pt = pd.read_csv(R / "phase2" / "measure" / f"v1_{k}_by_track.csv")
        typ = boot_median(pt.median_abs_dev_ms)
        assert abs(typ[0] / c4.loc[f"v1_{k}", "median_abs_dev_ms"] - 1) < 1e-4   # summary file is rounded
        g = a2[(a2.label == f"v1_{k}") & (~a2.without_substitutions) & (a2.threshold_ms == 10)].iloc[0]
        disp = (100 * g.median_track_share, 100 * g.median_ci_lo, 100 * g.median_ci_hi)
        mis = boot_median(100 * (1 - pt.n_pairs / pt.n_ref))
        assert abs(mis[0] / (100 * c4.loc[f"v1_{k}", "share_ref_unmatched"]) - 1) < 1e-4
        acc = sdu[(sdu.label == f"v1_{k}") & (sdu.measure == "accent") & (sdu.stroke_set == "matched")].iloc[0]
        val = sdu[(sdu.label == f"v1_{k}") & (sdu.measure == "valley_db") & (sdu.stroke_set == "all")].iloc[0]
        terms[m] = dict(
            typ=typ,
            disp=disp, disp_floor=100 * c4.loc[f"floor_{k}_leak", "share_gt_10ms"],
            mis=mis, mis_floor=100 * c4.loc[f"floor_{k}_leak", "share_ref_unmatched"],
            acc=(acc.median_nat, acc.median_ci_lo * acc.unit_sd, acc.median_ci_hi * acc.unit_sd),
            val=(val.median_nat, val.median_ci_lo * val.unit_sd, val.median_ci_hi * val.unit_sd))

    fig = plt.figure(figsize=(W, 3.35))
    gs = fig.add_gridspec(2, 3, hspace=0.85, wspace=0.16)
    spec = [("typ", None, "(a) Typical deviation", "ms", (0, 15.5)),
            ("disp", "disp_floor", "(b) Displaced > 10 ms", "% of matched onsets", (0, 18)),
            ("mis", "mis_floor", "(c) Missed strokes", "% of reference strokes", (0, 18)),
            ("acc", None, "(d) Accent contrast", "increase, dB (matched strokes)", (-0.1, 1.6)),
            ("val", None, "(e) Gaps between strokes", "deepening, dB (all strokes)", (-0.2, 2.4))]
    ypos = {m: i for i, m in enumerate(MODELS)}
    for n, (key, fkey, title, xlab, xlim) in enumerate(spec):
        ax = fig.add_subplot(gs[n // 3, n % 3])
        if key == "typ":
            ax.axvspan(8, 14, color="#eeeeea", lw=0, zorder=0)
            ax.text(11, -0.62, "drummers'\nswing\nthresholds", ha="center", va="top", fontsize=6.5, color=INK2)
        if key in ("acc", "val"):
            ax.axvline(0, color=INK2, lw=0.6)
        for m in MODELS:
            med, lo, hi = terms[m][key]
            y = ypos[m]
            ax.plot([lo, hi], [y, y], color=COL[m], lw=1.1, solid_capstyle="round", zorder=2)
            if fkey:   # floors first, so that the model's marker stays visible where the two overlap
                ax.plot(terms[m][fkey], y, ls="", marker="o", ms=4.4, mfc="white", mec=INK2, mew=0.9, zorder=3)
            # dark edges keep the lighter hues (SCNet-XL) visible in greyscale
            ax.plot(med, y, ls="", marker=MRK[m], ms=4.8, color=COL[m], mec=INK, mew=0.6, zorder=4)
        ax.set_ylim(len(MODELS) - 0.4, -0.6)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS if n % 3 == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=7.5, loc="left", pad=3)
        vgrid(ax)
    lx = fig.add_subplot(gs[1, 2]); lx.set_axis_off()
    handles = [Line2D([], [], color=INK2, lw=1.1, marker="o", ms=4.8, mfc=INK2, mec=INK, mew=0.6,
                      label="median over tracks,\nwith 95 % CI"),
               Line2D([], [], ls="", marker="o", ms=4.4, mfc="white", mec=INK2, mew=0.9,
                      label="leakage floor at\nthe model's SIR")]
    lx.legend(handles=handles, loc="center left", fontsize=6.6, frameon=False, handlelength=2.2,
              labelspacing=1.0, borderaxespad=0.0)
    fig.subplots_adjust(top=0.93, bottom=0.12, left=0.13, right=0.985)
    fig.savefig(out)
    plt.close(fig)
    return terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "phase5" / "figures"))
    ap.add_argument("--formats", default="pdf", help="comma-separated: pdf, tiff (1200 dpi)")
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in a.formats.split(","):
        ext = "." + fmt.strip()
        kw = {}
        if fmt.strip() == "tiff":
            kw = dict(dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
            mpl.rcParams["savefig.dpi"] = 1200
        _save = plt.Figure.savefig
        plt.Figure.savefig = lambda self, fn, _s=_save, _kw=kw: _s(self, fn, **_kw)
        fig1(out / f"fig1_design{ext}")
        terms = fig_budget(root, out / f"fig2_budget{ext}")
        info = fig2(root, out / f"fig3_excerpt{ext}")
        fig3(root, out / f"fig4_timing{ext}")
        fig4(root, out / f"fig5_shape{ext}")
        fig5(root, out / f"fig6_sdr{ext}")
        fig6(root, out / f"fig7_exp2{ext}")
        plt.Figure.savefig = _save
    for k, v in info.items():
        print(k, v)
    for m, t in terms.items():
        print(m, {k: (np.round(v, 3) if isinstance(v, tuple) else round(v, 3)) for k, v in t.items()})


if __name__ == "__main__":
    main()
