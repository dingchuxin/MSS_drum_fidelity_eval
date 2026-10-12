"""D5: the strokes of the excerpt shown in the article's excerpt figure, from the full-track analysis.

The excerpt is the track and 3-s window in results/phase1/figure_excerpt/manifest.json (chosen
by scripts/phase1/j6_excerpt.py). Per model and reference stroke in the window: the status in
the full-track Phase 2 analysis (reference onsets and matched estimate onsets, timing.detect +
closest-first +/-50 ms, as every statistic uses): within 10 ms, displaced (> 10 ms) or missed,
with the deviation; and the estimate onsets in the window without a partner (extra).

    python d5_excerpt.py
Output: results/phase4/d5_excerpt_strokes.csv (read by scripts/make_figures_v2.py)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _p4 as P
from mss_eval import timing as T

CUR = P.ROOT / "results" / "phase1" / "figure_excerpt"


def full_track_status(track, t0, t1):
    rows = []
    for lab in P.MODELS4:
        ref, est = P.p2_onsets(lab, track)
        i, j = T.match_pairs(ref, est, 0.05)
        m = dict(zip(i, j))
        for k in np.where((ref >= t0) & (ref < t1))[0]:
            dev = (est[m[k]] - ref[k]) * 1000 if k in m else np.nan
            rows.append({"model": lab, "t_ref_s": ref[k], "full_dev_ms": dev,
                         "full_status": "missed" if k not in m else ("displaced" if abs(dev) > 10 else "within 10 ms")})
        ex = np.setdiff1d(np.where((est >= t0) & (est < t1))[0], j)
        for e in ex:
            rows.append({"model": lab, "t_ref_s": np.nan, "t_extra_s": est[e], "full_status": "extra"})
    return pd.DataFrame(rows)


def main():
    man = json.load(open(CUR / "manifest.json"))
    df = full_track_status(man["track"], man["t0_s"], man["t1_s"])
    df.insert(0, "t0_s", man["t0_s"])
    df.insert(0, "track", man["track"])
    df.insert(0, "excerpt", "current")
    df.to_csv(P.OUT4 / "d5_excerpt_strokes.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(df.drop(columns=["track"]).to_string())


if __name__ == "__main__":
    main()
