"""A0: clipping audit of every cached v1 stem (all models, all four instruments).

A 16-bit stem is counted as clipped at |x| >= 32767/32768 (the largest positive code).
Output: results/phase2/a0_clipping_audit.csv (+ summary printed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf

import _p2 as P

FS = 32767 / 32768


def main():
    rows = []
    for m in P.MODELS:
        for t in P.tracks():
            for s in P.SOURCES:
                f = P.STEMS / m / P.safe_name(t) / f"{s}.wav"
                x, _ = sf.read(str(f), dtype="int16", always_2d=True)
                a = np.abs(x.astype(np.int32))
                rows.append({"model": m, "track": t, "stem": s,
                             "n_full_scale": int((a >= 32767).sum()),
                             "peak": float(a.max() / 32768),
                             "at_0p9_peak": bool(abs(a.max() / 32768 - 0.9) < 2e-4)})
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT2 / "a0_clipping_audit.csv", index=False)
    print(df.groupby(["model", "stem"]).agg(n_clipped_stems=("n_full_scale", lambda v: (v > 0).sum()),
                                           samples=("n_full_scale", "sum"),
                                           max_peak=("peak", "max")).to_string())
    print(df[df.n_full_scale > 0].to_string(index=False))
    print(df[df.at_0p9_peak].to_string(index=False))


if __name__ == "__main__":
    main()
