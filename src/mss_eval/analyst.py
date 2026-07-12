"""Analyst-facing fidelity metrics — what a musicologist actually reads off a stem.

BSS-Eval (SDR/SIR/SAR) measures *signal reconstruction*. An analyst never listens
for "SDR"; they read timing, dynamics, and stem identity. These metrics quantify what
separation does to each, computing the property on the estimate and, where a fidelity
comparison is wanted, on the ground-truth stem too.

Reference-based: MUSDB18 supplies real reference stems, so these are real data, not
proxies. They exist to support/refute §4.7 with quantities SDR hides.

Three dimensions:
  timing    -> onset_fmeasure       (§4.7 transient/attack claims)
  dynamics  -> dynamic_profile      (§4.7 claim A1: the dynamic profile varies a lot;
                                      envelope shape/slope + intensity distribution)
  labeling  -> labeling_precision   (§4.7 claim A2 labeling + row 9: with ground truth,
                                      how well does each model route energy to the right
                                      stem label)
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8
_HOP_S = 0.010          # envelope hop (10 ms)


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else np.asarray(x, dtype=np.float64)


def _lin_env(x: np.ndarray, sr: int, win_ms: float = 20.0) -> np.ndarray:
    """Linear short-term RMS envelope (10 ms hop)."""
    x = _mono(x)
    win = max(1, int(sr * win_ms / 1000))
    hop = max(1, int(sr * _HOP_S * 1000 / 1000))
    return np.array([np.sqrt(np.mean(x[i:i + win] ** 2) + _EPS)
                     for i in range(0, len(x) - win + 1, hop)])


# --- timing -----------------------------------------------------------------

def onset_fmeasure(estimate: np.ndarray, reference: np.ndarray, sr: int,
                   window: float = 0.05) -> dict[str, float]:
    """Onset-detection F-measure of the estimate against the reference stem.

    Detects onsets on both signals and matches within `window` s (default 50 ms).
    A stem can post a healthy SDR yet smear or invent attacks — fatal for rhythmic
    analysis. Returns {f_measure, precision, recall, n_ref, n_est}.
    """
    import librosa
    import mir_eval

    ref_on = librosa.onset.onset_detect(y=_mono(reference), sr=sr, units="time", backtrack=True)
    est_on = librosa.onset.onset_detect(y=_mono(estimate), sr=sr, units="time", backtrack=True)
    if len(ref_on) == 0:
        return {"f_measure": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "n_ref": 0, "n_est": int(len(est_on))}
    f, p, r = mir_eval.onset.f_measure(ref_on, est_on, window=window)
    return {"f_measure": float(f), "precision": float(p), "recall": float(r),
            "n_ref": int(len(ref_on)), "n_est": int(len(est_on))}


# --- dynamics ---------------------------------------------------------------

def crest_factor(x: np.ndarray) -> float:
    """Peak-to-RMS ratio (dB). Transient/attack squashing lowers it."""
    x = _mono(x)
    peak = np.max(np.abs(x)) + _EPS
    rms = np.sqrt(np.mean(x ** 2)) + _EPS
    return float(20.0 * np.log10(peak / rms))


def integrated_loudness(x: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770 integrated loudness (LUFS)."""
    import pyloudnorm as pyln
    audio = x if x.ndim == 2 else x[:, None]
    return float(pyln.Meter(sr).integrated_loudness(audio))


def dynamic_profile(x: np.ndarray, sr: int) -> dict[str, float]:
    """A stem's dynamic profile — envelope shape/slope AND intensity distribution.

    The point of §4.7 claim A1 is that this profile is *model-dependent*: run the same
    source through different separators and these descriptors move a lot. Compute them
    per model estimate and look at the cross-model spread.

    Descriptors:
      level_db      mean intensity level, 10·log10 mean-square (dB)
      variation     dynamic variation — IQR of short-term loudness over ACTIVE frames
                    (dB); the "how much does level swing while sounding" figure (analogous
                    in spirit to, but independent of, the drummer-study measure)
      crest         peak-to-RMS (dB)
      impulsiveness raw-waveform excess kurtosis — transient/spiky vs. sustained; the
                    standard impulsiveness measure, well-defined on the signal itself
      env_skew      asymmetry of the loudness distribution (attack- vs. sustain-heavy)
      env_flux      mean |Δ loudness| per hop (dB) over active frames — envelope motion;
                    smoothing/compression by a model reduces it
      attack_slope  90th-pct positive loudness rise per hop (dB) — attack sharpness
      burst_rate    prominent envelope peaks per second (transient density)

    All spread/motion descriptors use an active-frame gate (within 60 dB of peak) so a
    sparse, bursty stem's silence floor doesn't swamp — or, if gated as outliers, erase —
    the very dynamics being measured.
    """
    from scipy import signal, stats

    xm = _mono(x)
    env = _lin_env(x, sr)
    if len(env) < 4:
        keys = ["level_db", "variation", "crest", "impulsiveness", "env_skew",
                "env_flux", "attack_slope", "burst_rate"]
        return {k: float("nan") for k in keys}
    env_db = 20.0 * np.log10(env + _EPS)

    peak_db = env_db.max()
    active = env_db > peak_db - 60.0                            # "sounding" frames
    adb = env_db[active] if active.sum() > 4 else env_db

    q1, q3 = np.percentile(adb, [25, 75])
    denv = np.diff(env_db)
    denv_active = denv[active[:-1]] if active[:-1].any() else denv
    # burst threshold uses the FULL envelope (silence floor as baseline) so peaks stand out
    med_all = np.median(env_db)
    mad_all = np.median(np.abs(env_db - med_all)) + _EPS
    return {
        "level_db": float(10.0 * np.log10(np.mean(xm ** 2) + _EPS)),
        "variation": float(q3 - q1),
        "crest": crest_factor(x),
        "impulsiveness": float(stats.kurtosis(xm, fisher=True, bias=False)),
        "env_skew": float(stats.skew(adb, bias=False)),
        "env_flux": float(np.mean(np.abs(denv_active))),
        "attack_slope": float(np.percentile(denv[denv > 0], 90)) if np.any(denv > 0) else 0.0,
        "burst_rate": float(len(signal.find_peaks(
            env_db, height=med_all + 3 * 1.4826 * mad_all,
            distance=max(1, int(0.05 / _HOP_S)))[0]) / (len(env) * _HOP_S + _EPS)),
    }


def dynamics_fidelity(estimate: np.ndarray, reference: np.ndarray, sr: int) -> dict[str, float]:
    """Per-descriptor deviation of the estimate's dynamic profile from the true stem.

    Deltas are estimate − reference (keys `d_<descriptor>`). Feeds A1b (per-model bias):
    if the deltas differ *by model* with no common offset, dynamics can't be corrected
    for and are unreliable through separation.
    """
    pe = dynamic_profile(estimate, sr)
    pr = dynamic_profile(reference, sr)
    return {f"d_{k}": pe[k] - pr[k] for k in pe}


# --- labeling (ground-truth stem identity) ----------------------------------

def labeling_precision(estimates: dict[str, np.ndarray],
                       references: dict[str, np.ndarray],
                       sr: int) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """How well each model routes energy to the correctly-labeled stem.

    Each estimated stem is decomposed by least squares onto the TRUE sources
    (MUSDB stems sum to the mixture in the time domain, so this is principled).
    The explained energy splits *additively* across sources via the identity
    aᵀGa = Σᵢ aᵢbᵢ  (G = TᵀT, b = Tᵀe, a = G⁻¹b), so

        precision(ŝ) = share of ŝ's explained energy from the correctly-labeled source.

    Perfect separation (ŝ = true_ŝ) gives precision 1.0 by construction — unlike a
    similarity score, it is NOT confounded by how alike the sources inherently are.
    ~1 means the "drums" stem really is drums; lower means energy leaked from / was
    labelled as another source. Tests §4.7's labelling-policy claim (row 9) and the
    A2 labeling facet. Returns ({stem: precision}, routing[est_stem][true_stem]).
    """
    labels = [s for s in estimates if s in references]
    ref = {s: _mono(references[s]) for s in labels}
    n = min(min(len(v) for v in ref.values()),
            min(len(_mono(estimates[s])) for s in labels))
    T = np.stack([ref[s][:n] for s in labels], axis=1)     # (n_samples, k)
    Ginv = np.linalg.pinv(T.T @ T)                          # (k, k)

    precision, routing = {}, {}
    for e in labels:
        b = T.T @ _mono(estimates[e])[:n]                  # (k,)  Tᵀe
        shares = np.clip((Ginv @ b) * b, 0.0, None)        # additive energy shares
        tot = shares.sum() + _EPS
        routing[e] = {labels[i]: float(shares[i] / tot) for i in range(len(labels))}
        precision[e] = float(shares[labels.index(e)] / tot)
    return precision, routing


# --- convenience ------------------------------------------------------------

def analyst_row(estimate: np.ndarray, reference: np.ndarray, sr: int) -> dict[str, float]:
    """Per (estimate, reference) stem pair: timing + dynamic profile (raw + Δ vs. truth).

    Labeling is model-wide (needs all stems at once) so it's computed separately via
    labeling_precision() in the run script, not here.
      onset_*   timing
      dyn_*     raw dynamic-profile descriptors on the estimate (for A1a spread / A1c ordering)
      d_*       deviation of each descriptor from the true stem (for A1b bias)
    """
    row: dict[str, float] = {}
    row.update({f"onset_{k}": v for k, v in onset_fmeasure(estimate, reference, sr).items()})
    row.update({f"dyn_{k}": v for k, v in dynamic_profile(estimate, sr).items()})
    row.update(dynamics_fidelity(estimate, reference, sr))
    return row
