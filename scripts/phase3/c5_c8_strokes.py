"""C5 (expansion hypothesis) and C8 (stroke-definition comparison), v3 stroke measures.

C5  Within each track, OLS of d(peak) = est - ref on the reference stroke peak relative to the
    track's drum level (dB); likewise d(floor) on the reference floor (inter-stroke level).
    Per model: median within-track slope, 95 % bootstrap CI over tracks, tracks with a
    positive slope; pooled fixed-effects slope (track-demeaned OLS) with a track-cluster
    bootstrap CI. Both for all reference strokes and matched strokes.
    A POSITIVE d(peak) slope means d rises with the reference level, i.e. quiet strokes lose
    more level than loud ones (expansion).
C8  Three definitions, per model: Phase 2 (estimate at its own onset, segment to the next
    reference onset, matched strokes), v3 matched, v3 all reference strokes. Per-track
    median d of peak, floor, valley, rise and accent contrast; median over tracks, sign
    counts, two-sided one-sample Wilcoxon over tracks, Holm per definition (25 tests).

    python c5_c8_strokes.py
Outputs: results/phase3/c5_*.csv, c8_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _p3 as P

SETS = ("all", "matched")


def slopes(df, y, x):
    rows = []
    for t, h in df.groupby("track"):
        ok = h[x].notna() & h[y].notna()
        if ok.sum() < 10 or h.loc[ok, x].std() == 0:
            continue
        b = np.polyfit(h.loc[ok, x], h.loc[ok, y], 1)[0]
        rows.append({"track": t, "slope": b, "n": int(ok.sum())})
    return pd.DataFrame(rows)


def fe_slope(df, y, x, rng, nb=2000):
    d = df[[y, x, "track"]].dropna()
    xd = d[x] - d.groupby("track")[x].transform("mean")
    yd = d[y] - d.groupby("track")[y].transform("mean")
    b = float((xd * yd).sum() / (xd ** 2).sum())
    tr = d.track.to_numpy()
    ut = np.unique(tr)
    idx = {t: np.where(tr == t)[0] for t in ut}
    xa, ya = xd.to_numpy(), yd.to_numpy()
    bs = []
    for _ in range(nb):
        s = np.concatenate([idx[t] for t in rng.choice(ut, len(ut))])
        bs.append((xa[s] * ya[s]).sum() / (xa[s] ** 2).sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return b, lo, hi


def c5(df, rng):
    rows = []
    for lab in P.LABELS:
        g = df[df.label == lab].copy()
        g["ref_peak_rel"] = g.peak_rel_db                                  # v3 ref peak - track level
        for st in SETS:
            h = g if st == "all" else g[g.matched]
            for y, x, name in (("d_peak_db", "ref_peak_rel", "d(peak) ~ ref peak rel. track level"),
                               ("d_floor_db", "ref_floor_db", "d(floor) ~ ref floor")):
                s = slopes(h, y, x)
                lo, hi = P.boot_ci(s.slope.to_numpy(), np.median, rng)
                b, blo, bhi = fe_slope(h, y, x, rng)
                rows.append({"label": lab, "stroke_set": st, "regression": name, "n_tracks": len(s),
                             "median_within_track_slope": s.slope.median(), "ci_lo": lo, "ci_hi": hi,
                             "n_tracks_positive": int((s.slope > 0).sum()), "p_wilcoxon_slopes": P.wilcoxon_p(s.slope),
                             "pooled_fe_slope": b, "fe_ci_lo": blo, "fe_ci_hi": bhi})
    out = pd.DataFrame(rows)
    out.to_csv(P.OUT3 / "c5_expansion.csv", index=False, float_format="%.5g")
    return out


def c8():
    rows = []
    MEAS = [("peak_db", "d_stroke_peak_db", "d_peak_db"), ("floor_db", "d_stroke_floor_db", "d_floor_db"),
            ("valley_db", "d_stroke_valley_db", "d_valley_db"), ("rise_ms", "d_stroke_rise_ms", "d_rise_ms"),
            ("accent", "d_accent_db", "d_accent")]
    for lab in P.LABELS:
        p2 = pd.read_csv(P.OUT2 / "measure" / f"{lab}_by_track.csv").set_index("track").loc[P.tracks()]
        p3 = pd.read_csv(P.RERUN / "measure" / f"{lab}_by_track.csv").set_index("track").loc[P.tracks()]
        for m, c2, c3 in MEAS:
            for defn, v in (("Phase 2 (own onset, matched)", p2[c2]),
                            ("v3 matched", p3[f"{c3}_matched"]),
                            ("v3 all reference strokes", p3[f"{c3}_all"])):
                x = v.to_numpy(float)
                x = x[np.isfinite(x)]
                rows.append({"definition": defn, "label": lab, "measure": m, "n_tracks": len(x),
                             "median_d": np.median(x), "n_negative": int((x < 0).sum()),
                             "n_positive": int((x > 0).sum()), "p": P.wilcoxon_p(x),
                             "rank_biserial": P.rank_biserial(x)})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for d, idx in out.groupby("definition").groups.items():
        out.loc[idx, "p_holm"] = P.holm(out.loc[idx, "p"])
    out.to_csv(P.OUT3 / "c8_definitions.csv", index=False, float_format="%.5g")
    return out


def main():
    rng = np.random.default_rng(P.SEED)
    df = P.onset_table()
    a = c5(df, rng)
    b = c8()
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(a.round(4).to_string(index=False))
        w = b.pivot_table(index=["label", "measure"], columns="definition", values="median_d")
        print(w.round(3).to_string())
        w = b.pivot_table(index=["label", "measure"], columns="definition", values="p_holm")
        print(w.round(4).to_string())
        w = b.assign(signs=b.n_negative.astype(str) + "-/" + b.n_positive.astype(str) + "+").pivot_table(
            index=["label", "measure"], columns="definition", values="signs", aggfunc="first")
        print(w.to_string())


if __name__ == "__main__":
    main()
