"""Input-length sweep (§4.7 claim A2): does separating a fixed target region R
change with how much surrounding context the model is given?

Mechanism (works for benchmark AND ecological audio):
  1. Fix a measured region R = [r0, r1] within a track.
  2. Build a set of input windows, each CONTAINING R, of growing context.
  3. Separate each window; extract the output aligned to R; measure on R only.
  4. Drift of R's metrics across windows = the A2 effect.

Two ways to define the growing context, sharing all downstream code:
  - length mode: musically-neutral seconds of context (needs no annotation; MUSDB-ready).
  - phrase mode: musical phrase units (the thesis design; needs phrase boundaries — the
    SOURCE of those boundaries is deferred, this module just consumes a list of times).

The contrast between the two modes is itself the evidence for whether musical coherence
matters beyond raw duration.
"""
from __future__ import annotations

import bisect

import numpy as np

from . import analyst
from .separate import separate


def build_windows(region: tuple[float, float], track_len_s: float, spec: dict
                  ) -> list[tuple[str, tuple[float, float]]]:
    """Return [(level_name, (w0_s, w1_s))], each window ⊇ region, growing in context.

    spec = {mode: 'length'|'phrase', levels: [...], boundaries?: [phrase times]}.
    length level: {name, length: <seconds>|'full'};
    phrase level: {name, phrases: <count>|0.5|'full'} (requires spec['boundaries']).
    """
    r0, r1 = region
    mode = spec.get("mode", "length")
    windows: list[tuple[str, tuple[float, float]]] = []

    if mode == "length":
        for lv in spec["levels"]:
            L = lv.get("length")
            if L in (None, "full"):
                w0, w1 = 0.0, track_len_s
            else:
                L = max(float(L), r1 - r0)
                c = 0.5 * (r0 + r1)
                w1 = min(track_len_s, max(c + L / 2, r1))
                w0 = max(0.0, w1 - L)
            windows.append((lv["name"], (w0, w1)))

    elif mode == "phrase":
        b = sorted(spec["boundaries"])            # phrase boundary times (s), incl. 0 & end
        if not b or b[0] > 0:
            b = [0.0] + b
        if b[-1] < track_len_s:
            b = b + [track_len_s]
        i = max(0, bisect.bisect_right(b, r0) - 1)  # phrase index containing r0
        for lv in spec["levels"]:
            c = lv.get("phrases")
            if c in (None, "full"):
                w0, w1 = 0.0, track_len_s
            elif float(c) < 1:                      # partial phrase (e.g. 0.5)
                mid = b[i] + float(c) * (b[min(i + 1, len(b) - 1)] - b[i])
                w0, w1 = b[i], max(mid, r1)
            else:
                j = min(len(b) - 1, i + int(c))
                w0, w1 = min(b[i], r0), max(b[j], r1)
            windows.append((lv["name"], (w0, w1)))
    else:
        raise ValueError(f"unknown sweep mode '{mode}'")
    return windows


def sweep_region(mixture: np.ndarray, sr: int, model_cfg: dict,
                 region: tuple[float, float],
                 windows: list[tuple[str, tuple[float, float]]],
                 references: dict[str, np.ndarray] | None = None,
                 cache_dir=None) -> list[dict]:
    """Separate each window, extract region R's output, measure. One row per level×stem.

    Raw `dyn_*` on R (no reference needed) is the core A2 signal — it shows drift on
    ecological audio too. With `references` (benchmark) we add onset-F, dynamics Δ vs.
    truth, and labeling precision, all evaluated on R.
    """
    r0, r1 = int(region[0] * sr), int(region[1] * sr)
    ref_R = {k: (v[r0:r1] if v.ndim == 1 else v[r0:r1]) for k, v in (references or {}).items()}
    rows: list[dict] = []

    for level, (w0, w1) in windows:
        s0, s1 = int(w0 * sr), int(w1 * sr)
        # the full-track window == the benchmark separation, already cached: reuse it
        full = cache_dir is not None and s0 == 0 and s1 >= len(mixture)
        sep = separate(mixture[s0:s1], sr, model_cfg, input_range=(w0, w1),
                       cache_dir=cache_dir if full else None)
        a, b = r0 - s0, r1 - s0                    # R's indices within this window's output
        # guard: a model may return a slightly different length than its input
        est_R = {k: v[a:min(b, len(v))] for k, v in sep.stems.items()}

        lab: dict[str, float] = {}
        if ref_R:
            common = {k: est_R[k] for k in est_R if k in ref_R}
            if common:
                lab, _ = analyst.labeling_precision(common, {k: ref_R[k] for k in common}, sr)

        for stem, sig in est_R.items():
            row = {"model": model_cfg.get("name"), "level": level,
                   "input_len_s": round(w1 - w0, 3), "stem": stem}
            row.update({f"dyn_{k}": v for k, v in analyst.dynamic_profile(sig, sr).items()})
            if stem in ref_R:
                row.update(analyst.dynamics_fidelity(sig, ref_R[stem], sr))
                row.update({f"onset_{k}": v for k, v in
                            analyst.onset_fmeasure(sig, ref_R[stem], sr).items()})
                row["labeling_precision"] = lab.get(stem, float("nan"))
            rows.append(row)
    return rows
