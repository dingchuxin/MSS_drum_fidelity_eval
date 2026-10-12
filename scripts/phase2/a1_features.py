"""A1 inputs: one row per reference drum onset (refined detector, Phase 1 J1 cache), with

  env_strength     refined-detector onset envelope at the onset (envelope normalised to
                   [0, 1] over the track, as in peak picking)
  peak_rel_db      stroke peak level (A3, 10 ms RMS, 0-50 ms) minus the track's drum level
                   (10 log10 mean square of the whole true drum stem)
  ioi_min_ms       distance to the nearest neighbouring reference onset
  band shares      energy of the true drum stem in [t, t + 50 ms] (Hann-windowed FFT) in
                   < 150 Hz, 150-2000 Hz, 2000-5000 Hz and > 5000 Hz, as fractions of the total
  drum_class       argmax over {low (<150), mid (150-2k), high (>5k)} of the band share divided
                   by that band's median share over the track's onsets (a within-track proxy:
                   raw shares favour 'low' because of the spectral tilt)
  sim_bass/other/vocals   a refined onset of that true stem within +/-30 ms
  drum_to_rest_db  10 log10(E_drums / E_bass+other+vocals) in [t, t + 50 ms]

Output: data/phase2/a1_features.pkl (DataFrame).     python a1_features.py [--jobs 3]
"""
from __future__ import annotations

import argparse
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _p2 as P  # noqa: E402
import measure as M  # noqa: E402
from mss_eval import timing as T  # noqa: E402

BANDS = [(0, 150), (150, 2000), (2000, 5000), (5000, 22050)]
BNAMES = ["low", "mid", "upper_mid", "high"]
WIN = 0.050
SIM = 0.030


def one(track: str) -> pd.DataFrame:
    f = P.P2DATA / "a1feat" / f"{P.safe_name(track)}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    ri = M.ref_info(track, False)
    on = ri["onsets"]
    ref = P.ref_stem(track)
    refm = P.mono(ref)
    rest = sum(P.mono(P.ref_stem(track, s)) for s in P.SOURCES if s != "drums")
    env = T.onset_envelope(ref, P.SR)
    env = (env - env.min()) / (env.max() - env.min() + 1e-12)
    k = on * P.SR / T.HOP_FINE
    strength = np.interp(k, np.arange(len(env)), env)
    other_on = {s: T.detect(P.ref_stem(track, s), P.SR) for s in ["bass", "other", "vocals"]}
    n = int(WIN * P.SR)
    freqs = np.fft.rfftfreq(n, 1 / P.SR)
    w = np.hanning(n)
    rows = []
    for i, t in enumerate(on):
        a = int(round(t * P.SR))
        seg = refm[a:a + n]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        spec = np.abs(np.fft.rfft(seg * w)) ** 2
        tot = spec.sum() + 1e-20
        bands = {f"share_{nm}": spec[(freqs >= lo) & (freqs < hi)].sum() / tot
                 for nm, (lo, hi) in zip(BNAMES, BANDS)}
        rseg = rest[a:a + n]
        d2r = 10 * np.log10((np.sum(seg ** 2) + 1e-20) / (np.sum(rseg ** 2) + 1e-20))
        ioi = min(t - on[i - 1] if i else np.inf, on[i + 1] - t if i + 1 < len(on) else np.inf)
        row = {"track": track, "i_ref": i, "t": t, "env_strength": strength[i],
               "peak_rel_db": ri["strokes"]["peak_db"].iloc[i] - ri["level_db"],
               "ioi_min_ms": ioi * 1000, "drum_to_rest_db": d2r, **bands}
        for s, o in other_on.items():
            row[f"sim_{s}"] = bool(len(o) and np.min(np.abs(o - t)) <= SIM)
        rows.append(row)
    df = pd.DataFrame(rows)
    rel = df[[f"share_{b}" for b in ["low", "mid", "high"]]] / df[[f"share_{b}" for b in ["low", "mid", "high"]]].median()
    df["drum_class"] = rel.idxmax(axis=1).str.replace("share_", "")
    df["sim_any"] = df[["sim_bass", "sim_other", "sim_vocals"]].any(axis=1)
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(df, open(f, "wb"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    from multiprocessing import Pool
    with Pool(a.jobs) as p:
        dfs = p.map(one, P.tracks())
    df = pd.concat(dfs, ignore_index=True)
    pickle.dump(df, open(P.P2DATA / "a1_features.pkl", "wb"))
    print(df.describe().round(3).to_string())
    print(df.drum_class.value_counts())


if __name__ == "__main__":
    main()
