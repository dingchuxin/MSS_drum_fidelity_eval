"""C0.5: A1 onset features recomputed on a window that includes the attack.

Phase 2 measured band shares and the drum-to-rest level in [t, t + 50 ms] from the detected
onset t, which lies ~12-15 ms after the physical onset (audit/l2_t1_onsets.py, test 1a), i.e. mostly after the
attack. Here both are measured in [t - 20 ms, t + 50 ms] (70 ms, Hann-windowed FFT of the true
drum stem; the same band limits: < 150 Hz, 150-2000 Hz, 2-5 kHz, > 5 kHz). The drum class
proxy is recomputed from the new shares (argmax of share / track-median share over low, mid,
high). The stroke-level covariate `peak_rel_db` is recomputed from the v3 reference peak
(strokes_v3, measured from t - 20 ms; the Phase 2 value used the peak read from the detected
onset). Onset strength, nearest-neighbour spacing and the +/-30 ms coincidence
flags are unchanged (same definition) and reused from Phase 2.

Output: data/phase3/v3/a1_features_v3.pkl (Phase 2 columns kept with suffix _p2).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import _p3 as P

BANDS = [(0, 150), (150, 2000), (2000, 5000), (5000, 22050)]
BNAMES = ["low", "mid", "upper_mid", "high"]
PRE, POST = 0.020, 0.050


def main():
    f2 = pickle.load(open(P.P2DATA / "a1_features.pkl", "rb"))
    out = []
    n = int((PRE + POST) * P.SR)
    freqs = np.fft.rfftfreq(n, 1 / P.SR)
    w = np.hanning(n)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]
    for t in P.tracks():
        g = f2[f2.track == t].sort_values("i_ref").copy()
        m3 = pickle.load(open(P.DATA3 / "measure" / "v1_bs_roformer" / f"{P.safe_name(t)}.pkl", "rb"))
        assert len(m3["ref_onsets"]) == len(g) and np.allclose(m3["ref_onsets"], g.t.to_numpy())
        ref = P.mono(P.ref_stem(t))
        rest = sum(P.mono(P.ref_stem(t, s)) for s in P.SOURCES if s != "drums")
        sh = np.zeros((len(g), 4))
        d2r = np.zeros(len(g))
        for k, tt in enumerate(g.t.to_numpy()):
            a = int(round((tt - PRE) * P.SR))
            a0 = max(a, 0)
            seg = ref[a0:a + n]
            rseg = rest[a0:a + n]
            if len(seg) < n:
                pad = (a0 - a, n - len(seg) - (a0 - a))
                seg, rseg = np.pad(seg, pad), np.pad(rseg, pad)
            spec = np.abs(np.fft.rfft(seg * w)) ** 2
            tot = spec.sum() + 1e-20
            sh[k] = [spec[mk].sum() / tot for mk in masks]
            d2r[k] = 10 * np.log10((np.sum(seg ** 2) + 1e-20) / (np.sum(rseg ** 2) + 1e-20))
        for c in ["share_low", "share_mid", "share_upper_mid", "share_high", "drum_to_rest_db",
                  "peak_rel_db", "drum_class"]:
            g[f"{c}_p2"] = g[c]
        for i, b in enumerate(BNAMES):
            g[f"share_{b}"] = sh[:, i]
        g["drum_to_rest_db"] = d2r
        g["peak_rel_db"] = m3["ref"]["peak_db"] - m3["ref_level_db"]
        rel = g[["share_low", "share_mid", "share_high"]] / g[["share_low", "share_mid", "share_high"]].median()
        g["drum_class"] = rel.idxmax(axis=1).str.replace("share_", "")
        out.append(g)
        print(t, flush=True)
    df = pd.concat(out, ignore_index=True)
    pickle.dump(df, open(P.DATA3 / "a1_features_v3.pkl", "wb"))
    print(df[["share_low", "share_low_p2", "share_high", "share_high_p2", "drum_to_rest_db",
              "drum_to_rest_db_p2", "peak_rel_db", "peak_rel_db_p2"]].describe().round(3).to_string())
    print(pd.crosstab(df.drum_class_p2, df.drum_class))


if __name__ == "__main__":
    main()
