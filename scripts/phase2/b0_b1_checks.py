"""B0 repeat-run check; B1 determinism and B1-vs-v1 comparison (HT-Demucs seeded vs default run).

    python b0_b1_checks.py b0      -> results/phase2/b0_repeat_runs.csv
    python b0_b1_checks.py b1      -> results/phase2/b1_determinism.csv, b1_vs_v1_by_track.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import soundfile as sf

import _p2 as P

S = P.P2DATA / "stems"
B0T = [0, 17, 34]


def rd(p):
    x, _ = sf.read(str(p), dtype="float64", always_2d=True)
    return x


def q16(x):
    """Exactly the v1 caching path: soundfile float -> PCM_16 WAV -> float64."""
    import io
    buf = io.BytesIO()
    sf.write(buf, x, P.SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    y, _ = sf.read(buf, dtype="float64", always_2d=True)
    return y


def sample_diff(a, b):
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    rms_a = np.sqrt(np.mean(a[:n] ** 2))
    rms_d = np.sqrt(np.mean(d ** 2))
    return {"max_abs_diff": float(np.abs(d).max()), "rms_diff": float(rms_d),
            "diff_rel_db": float(20 * np.log10(rms_d / rms_a)) if rms_d > 0 else -np.inf,
            "bit_identical": bool(np.array_equal(a[:n], b[:n]))}


def b0():
    runs = {"bs_roformer": ("b0_bs_roformer_run1", "b0_bs_roformer_run2"),
            "scnet": ("scnet_float", "b0_scnet_run2"),
            "spleeter": ("b0_spleeter_run1", "b0_spleeter_run2")}
    rows = []
    T_ = P.tracks()
    for m, (r1, r2) in runs.items():
        for ti in B0T:
            t = T_[ti]
            for stem in P.SOURCES:
                a = rd(S / r1 / P.safe_name(t) / f"{stem}.wav")
                b = rd(S / r2 / P.safe_name(t) / f"{stem}.wav")
                v = P.est_stem(m, t, stem)
                rows.append({"model": m, "track": t, "stem": stem, "comparison": "run1_vs_run2",
                             **sample_diff(a, b)})
                rows.append({"model": m, "track": t, "stem": stem, "comparison": "run1_vs_v1_int16",
                             **sample_diff(a, v)})
                q = q16(a)
                rows.append({"model": m, "track": t, "stem": stem, "comparison": "run1_vs_run1_quantised_int16",
                             **sample_diff(a, q)})
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT2 / "b0_repeat_runs.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 250):
        print(df.groupby(["model", "comparison"]).agg(max_abs=("max_abs_diff", "max"),
                                                       rms=("rms_diff", "max"),
                                                       rel_db=("diff_rel_db", "max"),
                                                       identical=("bit_identical", "all")).to_string())


def b1():
    T_ = P.tracks()
    det = []
    for ti in B0T[:2]:
        t = T_[ti]
        for stem in P.SOURCES:
            a = rd(S / "htdemucs_b1" / P.safe_name(t) / f"{stem}.wav")
            b = rd(S / "htdemucs_b1_repeat" / P.safe_name(t) / f"{stem}.wav")
            det.append({"track": t, "stem": stem, **sample_diff(a, b)})
    pd.DataFrame(det).to_csv(P.OUT2 / "b1_determinism.csv", index=False, float_format="%.6g")
    print(pd.DataFrame(det).to_string())
    rows = []
    for t in T_:
        for stem in P.SOURCES:
            b = rd(S / "htdemucs_b1" / P.safe_name(t) / f"{stem}.wav")
            v = P.est_stem("htdemucs", t, stem)
            n = min(len(b), len(v))
            b, v = b[:n], v[:n]
            g = float(np.sum(v * b) / np.sum(b * b))            # LS gain v ~ g * b
            resid = v - g * b
            r = {"track": t, "stem": stem, "ls_gain_v1_over_b1": g,
                 "b1_peak": float(np.abs(b).max()), "v1_peak": float(np.abs(v).max()),
                 "snr_v1_vs_gain_matched_b1_db": float(10 * np.log10(np.sum((g * b) ** 2) / np.sum(resid ** 2)))}
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT2 / "b1_vs_v1_by_track.csv", index=False, float_format="%.6g")
    print(df[df.stem == "drums"].snr_v1_vs_gain_matched_b1_db.describe().round(2).to_string())

if __name__ == "__main__":
    {"b0": b0, "b1": b1}[sys.argv[1]]()
