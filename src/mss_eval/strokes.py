"""Stroke-level shape measures: the Phase 2 definition (`strokes`, legacy) and v3 (`strokes_v3`).

Two short-term RMS envelopes of the mono signal, both on a 64-sample hop (1.45 ms, the
refined onset detector's hop), centred frames, dB = 20 log10(RMS + 1e-12):
  env10  10 ms rectangular RMS  -> peak level, inter-stroke level ("floor"), valley depth,
         accents. A 10 ms frame spans at least one period down to 100 Hz.
  env3   2.9 ms (128-sample) RMS  -> rise time (and, legacy only, decay time).
Digital silence has RMS 0 and therefore reads -240 dB (the 1e-12 floor).

LEGACY (Phase 2, `strokes` / `stroke`; kept unchanged so Phase 2 results reproduce).
Per stroke, from the signal's OWN detected onset t (which lies ~12-15 ms after the physical
onset, because the detected time is the flux peak) to the next REFERENCE onset t_next (the
segment is capped at 1 s):
  peak_db    max env10 in [t, t + 50 ms]; it starts after the physical onset, so it reads low
             for fast decays and absorbs the onset deviation
  floor_db   5th percentile of env10 from the env10 peak frame to the segment end: the bottom
             of the valley between strokes, not a noise floor; -240 dB after digital
             silence
  valley_db  peak_db - floor_db
  rise_ms    10 -> 90 % of the rise of env3 above its minimum in the 20 ms before the env3
             peak (peak searched in [t - 10 ms, t + 50 ms]), linear amplitude, interpolated
             crossings; about 2 ms resolution
  decay_ms   env3 peak -> the last frame >= max(peak - 20 dB, floor + 1 dB) up to AND
             INCLUDING the frame of the next detected onset. That frame lies after the next
             stroke's physical onset, so the next attack is inside the segment and the decay
             is capped by construction: `decay_ms` mostly equals the time to the next
             onset. `decay_capped` flags this.

V3 (`strokes_v3`, Phase 3). Both signals are measured on the SAME segments, defined by the
reference onsets t_k (detected, Phase 2 detector):
  segment k  [s_k, s_k+1) with s_k = t_k - 20 ms (detector latency on synthetic events, audit/l2_t1_onsets.py
             test 1a: +14.6 ms median, P95 19.8 ms), capped at s_k + 1 s; the last segment ends at s_k + 1 s or
             the end of the signal
  peak_db    max env10 in [s_k, min(t_k + 50 ms, s_k+1))  (the cap only binds for inter-onset
             intervals < 70 ms)
  floor_db   the inter-stroke level: 5th percentile of env10 from the peak frame to the
             segment end (exclusive); NaN if < 5 frames
  valley_db  peak_db - floor_db
  rise_ms    as legacy, with the env3 peak searched in the same window as peak_db
  pre_db     minimum env10 in the 20 ms before the env10 peak frame (pre-stroke level)
  jump_db    peak_db - pre_db (per-stroke level jump)
  no decay   (capped by construction in the legacy definition)
Digital silence: `pair_v3` sets floor and valley to NaN in BOTH signals for any stroke whose
floor is below -150 dB in either signal, and flags which signal was silent; pre_db and jump_db
get the same treatment when the pre-stroke level is below -150 dB in either signal.
"""
from __future__ import annotations

import numpy as np

HOP = 64
W10 = 441
W3 = 128
EPS = 1e-12


def _mono(x):
    return x.mean(axis=1) if x.ndim == 2 else np.asarray(x, dtype=np.float64)


def rms_env(x: np.ndarray, win: int, hop: int = HOP) -> np.ndarray:
    """Centred rectangular RMS; frame k is centred on sample k*hop."""
    y = _mono(x).astype(np.float64)
    c = np.concatenate([[0.0], np.cumsum(np.pad(y * y, (win // 2, win - win // 2)))])
    k = np.arange(0, len(y) + 1, hop)
    return np.sqrt(np.maximum((c[k + win] - c[k]) / win, 0.0))


def db(a):
    return 20 * np.log10(np.asarray(a) + EPS)


class Envelopes:
    def __init__(self, x: np.ndarray, sr: int):
        self.sr = sr
        self.e10 = rms_env(x, W10)
        self.e3 = rms_env(x, W3)
        self.e10db = db(self.e10)
        self.e3db = db(self.e3)

    def idx(self, t):
        return int(round(t * self.sr / HOP))


def stroke(env: Envelopes, t: float, t_next: float | None) -> dict:
    sr, n = env.sr, len(env.e10)
    i0 = env.idx(t)
    i_end = n - 1 if t_next is None else env.idx(t_next)
    i_end = min(i_end, i0 + env.idx(1.0), n - 1)
    w50 = env.idx(0.050)
    out = {"peak_db": np.nan, "floor_db": np.nan, "valley_db": np.nan, "rise_ms": np.nan,
           "decay_ms": np.nan, "decay_capped": False}
    if i0 >= n - 2 or i_end - i0 < 3:
        return out
    a, b = i0, min(i0 + w50 + 1, n)
    kp10 = a + int(np.argmax(env.e10[a:b]))
    peak = env.e10db[kp10]
    seg = env.e10db[kp10:i_end]
    floor = np.percentile(seg, 5) if len(seg) >= 5 else np.nan
    out.update(peak_db=peak, floor_db=floor, valley_db=peak - floor)
    # rise on env3 (linear), peak searched in [t - 10 ms, t + 50 ms]
    a3 = max(0, i0 - env.idx(0.010))
    kp = a3 + int(np.argmax(env.e3[a3:b]))
    P = env.e3[kp]
    lo = max(0, kp - env.idx(0.020))
    base = env.e3[lo:kp + 1].min()
    l10, l90 = base + 0.1 * (P - base), base + 0.9 * (P - base)
    e = env.e3
    k90 = lo + int(np.argmax(e[lo:kp + 1] >= l90))
    below = np.where(e[lo:k90 + 1] < l10)[0]
    k10 = lo + (below[-1] + 1 if len(below) else 0)
    # linear interpolation of both crossings (sub-hop times)
    t90 = k90 - (1 - _frac(e, k90, l90)) if k90 > lo else float(k90)
    t10 = k10 - (1 - _frac(e, k10, l10)) if len(below) else float(k10)
    out["rise_ms"] = max(0.0, t90 - t10) * HOP / sr * 1000
    # decay on env3 (dB), upper-hull criterion
    thr = env.e3db[kp] - 20.0
    if np.isfinite(floor):
        thr = max(thr, floor + 1.0)
    after = env.e3db[kp:i_end + 1]
    above = np.where(after >= thr)[0]
    last = above[-1] if len(above) else 0
    out["decay_capped"] = bool(last >= len(after) - 1)
    t_end = float(last)
    if not out["decay_capped"]:                  # interpolate the downward crossing
        y0, y1 = after[last], after[last + 1]
        t_end += (y0 - thr) / (y0 - y1) if y0 != y1 else 0.0
    out["decay_ms"] = t_end * HOP / sr * 1000
    return out


def _frac(e: np.ndarray, k: int, level: float) -> float:
    """Fraction of the way from frame k-1 to k at which e crosses `level` upwards."""
    a, b = e[k - 1], e[k]
    return float(np.clip((level - a) / (b - a), 0.0, 1.0)) if b != a else 1.0


def strokes(x: np.ndarray, sr: int, onsets: np.ndarray, next_onsets: np.ndarray | None = None
            ) -> list[dict]:
    """Stroke measures at `onsets`; segment ends at `next_onsets` (default: the next onset)."""
    env = Envelopes(x, sr)
    onsets = np.asarray(onsets)
    if next_onsets is None:
        next_onsets = np.append(onsets[1:], np.nan)
    res = []
    for t, tn in zip(onsets, next_onsets):
        res.append(stroke(env, float(t), None if not np.isfinite(tn) else float(tn)))
    return res


# ---------------------------------------------------------------------------------- v3
PRE_S = 0.020
PEAK_S = 0.050
CAP_S = 1.0
SILENCE_DB = -150.0
V3_KEYS = ["peak_db", "floor_db", "valley_db", "rise_ms", "pre_db", "jump_db", "peak_t"]


def _rise(e: np.ndarray, kp: int, lo_win: int) -> float:
    """10 -> 90 % rise time (frames) into the env3 peak at kp (legacy definition)."""
    lo = max(0, kp - lo_win)
    P = e[kp]
    base = e[lo:kp + 1].min()
    l10, l90 = base + 0.1 * (P - base), base + 0.9 * (P - base)
    k90 = lo + int(np.argmax(e[lo:kp + 1] >= l90))
    below = np.where(e[lo:k90 + 1] < l10)[0]
    k10 = lo + (below[-1] + 1 if len(below) else 0)
    t90 = k90 - (1 - _frac(e, k90, l90)) if k90 > lo else float(k90)
    t10 = k10 - (1 - _frac(e, k10, l10)) if len(below) else float(k10)
    return max(0.0, t90 - t10)


def strokes_v3(x: np.ndarray, sr: int, ref_onsets: np.ndarray,
               env: "Envelopes | None" = None) -> dict[str, np.ndarray]:
    """v3 stroke measures of signal x on the segments defined by the REFERENCE onsets.

    Returns a dict of arrays (one entry per reference onset): V3_KEYS. Call it with the
    same `ref_onsets` for the reference and for every estimate.
    """
    env = env or Envelopes(x, sr)
    n = len(env.e10)
    t = np.asarray(ref_onsets, float)
    K = len(t)
    out = {k: np.full(K, np.nan) for k in V3_KEYS}
    if K == 0:
        return out
    s = np.maximum(t - PRE_S, 0.0)
    s_next = np.append(s[1:], np.inf)
    end = np.minimum(s_next, s + CAP_S)
    w20 = env.idx(PRE_S)
    for k in range(K):
        i_s = env.idx(s[k])
        i_end = min(env.idx(end[k]) if np.isfinite(end[k]) else n, n)
        i_pk = min(env.idx(t[k] + PEAK_S) + 1, i_end, n)
        if i_s >= n - 2 or i_pk - i_s < 1:
            continue
        kp10 = i_s + int(np.argmax(env.e10[i_s:i_pk]))
        peak = env.e10db[kp10]
        seg = env.e10db[kp10:i_end]
        floor = np.percentile(seg, 5) if len(seg) >= 5 else np.nan
        pre = env.e10db[max(0, kp10 - w20):kp10 + 1].min()
        kp3 = i_s + int(np.argmax(env.e3[i_s:i_pk]))
        out["peak_db"][k] = peak
        out["floor_db"][k] = floor
        out["valley_db"][k] = peak - floor
        out["pre_db"][k] = pre
        out["jump_db"][k] = peak - pre
        out["rise_ms"][k] = _rise(env.e3, kp3, w20) * HOP / sr * 1000
        out["peak_t"][k] = kp10 * HOP / sr
    return out


def pair_v3(ref: dict, est: dict) -> tuple[dict, dict, dict]:
    """Apply the digital-silence rule to a (reference, estimate) pair of strokes_v3 outputs.

    Strokes whose floor is below -150 dB in either signal get floor and valley NaN in both.
    Returns (ref, est, flags) with flags['silent_ref'], flags['silent_est'] (bool arrays).
    """
    ref = {k: v.copy() for k, v in ref.items()}
    est = {k: v.copy() for k, v in est.items()}
    sr_ = np.nan_to_num(ref["floor_db"], nan=0.0) < SILENCE_DB
    se_ = np.nan_to_num(est["floor_db"], nan=0.0) < SILENCE_DB
    bad = sr_ | se_
    # the same rule for the pre-stroke level used in C3 (it can also sit in digital silence)
    pr_ = np.nan_to_num(ref["pre_db"], nan=0.0) < SILENCE_DB
    pe_ = np.nan_to_num(est["pre_db"], nan=0.0) < SILENCE_DB
    badp = pr_ | pe_
    for d in (ref, est):
        d["floor_db"][bad] = np.nan
        d["valley_db"][bad] = np.nan
        d["pre_db"][badp] = np.nan
        d["jump_db"][badp] = np.nan
    return ref, est, {"silent_ref": sr_, "silent_est": se_,
                      "silent_pre_ref": pr_, "silent_pre_est": pe_}
