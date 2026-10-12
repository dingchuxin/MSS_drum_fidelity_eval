"""Reference-based separation metrics (benchmark / MUSDB18 half).

These require ground-truth stems, so they only apply where reference signals
exist. SI-SDR is implemented directly; BSS-Eval (SDR/SIR/SAR, MDX-compatible)
is delegated to `museval` when available so numbers match published leaderboards.

All functions take float arrays shaped (samples,) or (samples, channels).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR (Le Roux et al., 2019), in dB.

    Invariant to a global gain on the estimate — useful because every MSS model
    in 4.7 alters absolute level, so plain SDR would conflate gain error with
    separation error.
    """
    est = _to_mono(np.asarray(estimate, dtype=np.float64))
    ref = _to_mono(np.asarray(reference, dtype=np.float64))
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + _EPS)
    target = alpha * ref
    noise = est - target
    return 10.0 * np.log10((np.dot(target, target) + _EPS) / (np.dot(noise, noise) + _EPS))


def bss_eval(estimates: dict[str, np.ndarray],
             references: dict[str, np.ndarray],
             sample_rate: int = 44100) -> dict[str, dict[str, float]]:
    """SDR/SIR/SAR/ISR per stem via museval (BSS-Eval v4), framewise-median.

    `estimates` and `references` are dicts keyed by stem name with matching keys.
    Returns {stem: {sdr, sir, sar, isr}} in dB (median over frames, ignoring NaNs).
    Raises ImportError with guidance if museval is not installed.
    """
    try:
        import museval
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "museval is required for BSS-Eval metrics. `pip install museval`.\n"
            "For a lightweight run use si_sdr() instead."
        ) from e

    names = [k for k in references if k in estimates]
    ref = np.stack([_as_2d(references[k]) for k in names])   # (nsrc, nsampl, nchan)
    est = np.stack([_as_2d(estimates[k]) for k in names])
    n = min(ref.shape[1], est.shape[1])
    ref, est = ref[:, :n], est[:, :n]

    sdr, isr, sir, sar, _ = museval.metrics.bss_eval(ref, est, window=sample_rate, hop=sample_rate)
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        out[name] = {
            "sdr": float(np.nanmedian(sdr[i])),
            "sir": float(np.nanmedian(sir[i])),
            "sar": float(np.nanmedian(sar[i])),
            "isr": float(np.nanmedian(isr[i])),
        }
    return out


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


_NON_METRIC = {"model", "stem", "track", "onset_n_ref", "onset_n_est"}


def summarize(rows: list[dict]) -> "object":
    """Collapse per-track metric rows into a per-model/per-stem table (pandas).

    Aggregates every numeric metric column (developer + analyst) with median/mean/std;
    `.std` across tracks shows how much a metric varies between tracks.
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    keys = [c for c in ("model", "stem") if c in df.columns]
    metric_cols = [c for c in df.select_dtypes("number").columns if c not in _NON_METRIC]
    return df.groupby(keys)[metric_cols].agg(["median", "mean", "std"]).round(3)
