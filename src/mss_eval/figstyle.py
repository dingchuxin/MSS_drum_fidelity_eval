"""Shared figure house style: a muted, low-saturation palette, serif type, minimal ink,
double-column sizing. Call apply() before plotting; use MODEL_COLORS/MODEL_LABEL so every
figure identifies a model by the same colour."""
from __future__ import annotations

import matplotlib as mpl

# muted, low-saturation, low-contrast palette — consistent across every figure
MODEL_COLORS = {
    "spleeter":    "#9a938c",   # taupe grey        (2019 freq-domain baseline)
    "htdemucs":    "#b0995f",   # muted ochre        (2022 hybrid)
    "bs_roformer": "#a5695c",   # muted brick        (2023 transformer)
    "scnet":       "#5f7d8c",   # muted steel blue   (2024 CNN)
}
MODEL_LABEL = {"spleeter": "Spleeter", "htdemucs": "HT-Demucs",
               "bs_roformer": "BS-Roformer", "scnet": "SCNet-XL"}
MODEL_ORDER = ["spleeter", "htdemucs", "scnet", "bs_roformer"]

INK = "#2b2b2b"    # soft near-black for text/data
MUTE = "#9a9a9a"   # light grey for spines, guides, secondary marks

COL = 3.4          # single-column width (in), for reference
FULL = 7.0         # double-column full width (in)


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.edgecolor": MUTE, "axes.linewidth": 0.7, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTE, "ytick.color": MUTE,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.titlelocation": "left", "axes.titlepad": 6,
    })
