"""Onset timing at sub-frame resolution (analyses J0/J1).

`analyst.onset_fmeasure` detects onsets with librosa defaults: log-mel spectral flux at
hop 512 (11.6 ms at 44.1 kHz), peaks backtracked to the preceding envelope minimum. Onset
times therefore sit on an 11.6 ms grid, which cannot resolve deviations of 5-20 ms.

`detect_refined` keeps the same feature (n_fft 2048, 128 mels, power_to_db, flux over 11.6 ms,
envelope normalised to [0, 1] over the whole signal) and the same peak-picking time
constants, and changes only the temporal sampling:
  - hop 64 samples (1.45 ms frame period);
  - parabolic interpolation of each picked peak on the onset envelope (sub-frame time);
  - no backtracking: the envelope minimum before a peak is flat and ill-defined at a fine
    hop, while the flux peak is sharp. The event time is the flux peak, which lies a
    roughly constant distance after the backtracked time; that offset cancels in
    estimate - reference differences.
The mel spectrogram is built in blocks, so memory stays bounded at the fine hop; the result
is identical to a single librosa call on the whole signal (same padding, same global dB
reference and top_db clip).
"""
from __future__ import annotations

import numpy as np

N_FFT = 2048
N_MELS = 128
HOP_V1 = 512
HOP_FINE = 64


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else np.asarray(x, dtype=np.float64)


def detect_v1(y: np.ndarray, sr: int) -> np.ndarray:
    """Exactly the v1 pipeline call (analyst.onset_fmeasure): times in s."""
    import librosa
    return librosa.onset.onset_detect(y=_mono(y), sr=sr, units="time", backtrack=True)


def _mel_power_blocked(y: np.ndarray, sr: int, hop: int, n_fft: int = N_FFT,
                       n_mels: int = N_MELS, block_frames: int = 4000) -> np.ndarray:
    """Mel power spectrogram equal to librosa.feature.melspectrogram(center=True), in blocks."""
    import librosa
    import scipy.fft
    y = np.asarray(y, dtype=np.float32)
    n_frames = 1 + len(y) // hop
    yp = np.pad(y, (n_fft // 2, n_fft // 2), mode="constant")   # librosa>=0.10 centre padding
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype(np.float32)
    win = librosa.filters.get_window("hann", n_fft, fftbins=True).astype(np.float32)
    out = np.empty((n_mels, n_frames), dtype=np.float32)
    for f0 in range(0, n_frames, block_frames):
        f1 = min(n_frames, f0 + block_frames)
        seg = yp[f0 * hop: (f1 - 1) * hop + n_fft]
        seg = np.pad(seg, (0, max(0, (f1 - 1 - f0) * hop + n_fft - len(seg))))
        frames = np.lib.stride_tricks.sliding_window_view(seg, n_fft)[::hop][: f1 - f0]
        S = np.abs(scipy.fft.rfft(frames * win, axis=1)) ** 2       # float32 in, complex64 out
        out[:, f0:f1] = fb @ S.T.astype(np.float32)
    return out


def onset_envelope(y: np.ndarray, sr: int, hop: int = HOP_FINE) -> np.ndarray:
    """librosa-default log-mel flux envelope at an arbitrary hop (frame k <-> k*hop/sr).

    The flux lag is held at the v1 value in SECONDS (one 512-sample hop = 11.6 ms), i.e.
    lag = 512 // hop frames: the envelope is the v1 envelope sampled `512 // hop` times more
    densely. A 1-frame lag at a fine hop measures change over ~1.5 ms of a 46 ms window,
    which is dominated by noise and made detection less repeatable in validation.
    """
    import librosa
    M = _mel_power_blocked(_mono(y), sr, hop)
    S = librosa.power_to_db(M)                                     # ref=1, top_db=80 (global)
    return librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop, n_fft=N_FFT,
                                        lag=max(1, HOP_V1 // hop))


def _peak_pick(env: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """librosa.onset.onset_detect peak picking, time constants held fixed in seconds."""
    import librosa
    env = env - env.min()
    env = env / (env.max() + 1e-12)
    kw = dict(pre_max=0.03 * sr // hop, post_max=0.00 * sr // hop + 1,
              pre_avg=0.10 * sr // hop, post_avg=0.10 * sr // hop + 1,
              wait=0.03 * sr // hop, delta=0.07)
    return librosa.util.peak_pick(env, **kw), env


def parabolic(env: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Sub-frame offset in (-0.5, 0.5) of each peak via a parabola through k-1, k, k+1."""
    k = np.asarray(k, dtype=int)
    ok = (k > 0) & (k < len(env) - 1)
    d = np.zeros(len(k))
    a, b, c = env[k[ok] - 1], env[k[ok]], env[k[ok] + 1]
    den = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        d[ok] = np.where(den < 0, 0.5 * (a - c) / den, 0.0)
    return np.clip(d, -0.5, 0.5)


def _to_local_max(env: np.ndarray, k: np.ndarray, look: int) -> np.ndarray:
    """Move each pick forward to the envelope maximum within `look` frames (not past the
    next pick). librosa's peak_pick has no look-ahead (post_max = 1 frame), so at a fine hop
    it fires on the rising edge, where the threshold is first crossed, not at the peak."""
    out = k.copy()
    for n, kk in enumerate(k):
        stop = min(len(env), kk + look + 1, k[n + 1] if n + 1 < len(k) else len(env))
        out[n] = kk + int(np.argmax(env[kk:stop]))
    return out


def _rise_crossing(env: np.ndarray, k: np.ndarray, level: float = 0.5,
                   look: int = 0) -> np.ndarray:
    """Sub-frame time (in frames) at which the rise into each peak crosses `level` x the
    peak height above the preceding minimum (linear interpolation)."""
    t = np.empty(len(k))
    for n, kk in enumerate(k):
        lo = max(0, (k[n - 1] if n else 0), kk - look)
        seg = env[lo:kk + 1]
        i0 = lo + int(np.argmin(seg))
        thr = env[i0] + level * (env[kk] - env[i0])
        j = kk
        while j > i0 and env[j - 1] >= thr:
            j -= 1
        if j == i0:
            t[n] = float(j)
        else:
            a, b = env[j - 1], env[j]
            t[n] = j - 1 + (thr - a) / (b - a + 1e-12)
    return t


def detect_refined(y: np.ndarray, sr: int, hop: int = HOP_FINE, mode: str = "peak",
                   look_s: float = 0.03, min_sep_s: float | None = None) -> np.ndarray:
    """Onset times (s) at hop `hop`.

    Events are those librosa's peak picking finds (same time constants as v1); each event's
    time is then refined:
      mode='peak'  parabolic-interpolated maximum of the flux envelope within `look_s`
                   after the pick (the instant of fastest log-energy rise);
      mode='rise'  the 50 % point of the envelope's rise into that maximum, interpolated.

    min_sep_s: if set, refined events closer than this are merged, keeping the one with the
    larger envelope maximum. The picker's own `wait` (30 ms) spaces PICKS, but on a rise that
    lasts longer than 30 ms it fires twice, and moving both picks forward to the envelope
    maximum then lands them on the same event (duplicate within a few ms). Phase 1 (J1) used
    min_sep_s=None; Phase 2 uses `detect` below (min_sep_s = 0.03, the picker's wait).
    """
    env = onset_envelope(y, sr, hop)
    k, envn = _peak_pick(env, sr, hop)
    k = np.asarray(k, dtype=int)
    if len(k) == 0:
        return np.array([])
    kp = _to_local_max(envn, k, int(round(look_s * sr / hop)))
    if mode == "peak":
        t = kp + parabolic(envn, kp)
    elif mode == "rise":
        t = _rise_crossing(envn, kp, look=int(round(look_s * sr / hop)))
    else:
        raise ValueError(mode)
    if min_sep_s is not None:
        sep = min_sep_s * sr / hop
        order = np.argsort(t)
        t, h = t[order], envn[kp[order]]
        keep_t, keep_h = [t[0]], [h[0]]
        for ti, hi in zip(t[1:], h[1:]):
            if ti - keep_t[-1] < sep:
                if hi > keep_h[-1]:
                    keep_t[-1], keep_h[-1] = ti, hi
            else:
                keep_t.append(ti)
                keep_h.append(hi)
        t = np.asarray(keep_t)
    return np.unique(t) * hop / sr


MIN_SEP_S = 0.03


def detect(y: np.ndarray, sr: int) -> np.ndarray:
    """Phase 2 refined detector: detect_refined with the 30 ms post-refinement separation."""
    return detect_refined(y, sr, min_sep_s=MIN_SEP_S)


def match_pairs(ref: np.ndarray, est: np.ndarray, window: float = 0.05
                ) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matching within +/-window, closest pairs first (greedy on |Δt|).

    Used for signed deviations: mir_eval's matching maximises the NUMBER of pairs but not
    their closeness, which can pair an event with a farther neighbour. Returns index arrays
    (i_ref, i_est) sorted by reference time.
    """
    ref, est = np.asarray(ref), np.asarray(est)
    if len(ref) == 0 or len(est) == 0:
        return np.array([], int), np.array([], int)
    # candidate pairs via searchsorted (events are sorted)
    cand = []
    lo = np.searchsorted(est, ref - window, side="left")
    hi = np.searchsorted(est, ref + window, side="right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        for j in range(a, b):
            cand.append((abs(est[j] - ref[i]), i, j))
    cand.sort()
    used_r, used_e, pr, pe = set(), set(), [], []
    for _, i, j in cand:
        if i in used_r or j in used_e:
            continue
        used_r.add(i); used_e.add(j); pr.append(i); pe.append(j)
    order = np.argsort(pr)
    return np.asarray(pr, int)[order], np.asarray(pe, int)[order]


# --- Phase 3 variants (C2) ---------------------------------------------------------

E_WIN = 441          # 10 ms Hann window for the broadband energy envelope
E_SEARCH_S = 0.030   # search +/- 30 ms around each Phase 2 event


def energy_envelope(y: np.ndarray, sr: int, hop: int = HOP_FINE, win: int = E_WIN) -> np.ndarray:
    """Broadband energy envelope: Hann-weighted mean square of the mono signal over `win`
    samples, centred on sample k*hop (frame k), hop 64 (1.45 ms). For a step onset its first
    difference peaks at the step (the window centre)."""
    x = _mono(y).astype(np.float64) ** 2
    w = np.hanning(win + 2)[1:-1]
    w /= w.sum()
    from scipy.signal import oaconvolve
    e = oaconvolve(np.pad(x, (win // 2, win - win // 2 - 1)), w[::-1], mode="valid")
    return np.maximum(e[::hop], 0.0)


def detect_energy(y: np.ndarray, sr: int, search_s: float = E_SEARCH_S,
                  min_sep_s: float = MIN_SEP_S, events: np.ndarray | None = None) -> np.ndarray:
    """C2(b): which events exist is decided by the Phase 2 detector (`detect`); each event is
    then re-timed on a different onset function, the first difference of the broadband
    energy envelope (`energy_envelope`, 10 ms Hann, 1.45 ms hop):
      time = parabolic-interpolated maximum of diff(E) within +/- search_s of the Phase 2
             time, the search never crossing the midpoint to a neighbouring event;
      events that land < min_sep_s apart are merged (keeping the larger diff(E) peak).
    diff(E) at frame k is attributed to the midpoint between frames k and k+1.
    `events` may pass precomputed Phase 2 event times (identical result, saves time)."""
    ev = detect(y, sr) if events is None else np.asarray(events, float)
    if len(ev) == 0:
        return ev
    E = energy_envelope(y, sr)
    dE = np.diff(E)
    hop = HOP_FINE
    k_ev = ev * sr / hop
    mids = np.concatenate([[-np.inf], (k_ev[1:] + k_ev[:-1]) / 2, [np.inf]])
    w = search_s * sr / hop
    out_t, out_h = [], []
    for i, k in enumerate(k_ev):
        lo = int(np.ceil(max(k - w, mids[i], 0)))
        hi = int(np.floor(min(k + w, mids[i + 1], len(dE) - 1)))
        if hi < lo:
            continue
        j = lo + int(np.argmax(dE[lo:hi + 1]))
        frac = 0.0
        if 0 < j < len(dE) - 1:
            a, b, c = dE[j - 1], dE[j], dE[j + 1]
            den = a - 2 * b + c
            if den < 0:
                frac = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        out_t.append(j + 0.5 + frac)
        out_h.append(dE[j])
    t = np.asarray(out_t)
    h = np.asarray(out_h)
    o = np.argsort(t)
    t, h = t[o], h[o]
    sep = min_sep_s * sr / hop
    kt, kh = [t[0]], [h[0]]
    for ti, hi in zip(t[1:], h[1:]):
        if ti - kt[-1] < sep:
            if hi > kh[-1]:
                kt[-1], kh[-1] = ti, hi
        else:
            kt.append(ti)
            kh.append(hi)
    return np.unique(np.asarray(kt)) * hop / sr


def match_hungarian(ref: np.ndarray, est: np.ndarray, window: float = 0.05
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost maximum-cardinality one-to-one matching within +/-window.

    Among all matchings with the largest number of pairs, the one with the smallest total
    |est - ref| (Hungarian algorithm, pairs outside the window forbidden). Solved per
    connected component of the 'within window' graph. Returns (i_ref, i_est) sorted by ref.
    """
    from scipy.optimize import linear_sum_assignment
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    if len(ref) == 0 or len(est) == 0:
        return np.array([], int), np.array([], int)
    lo = np.searchsorted(est, ref - window, side="left")
    hi = np.searchsorted(est, ref + window, side="right")
    # components: union-find over refs and ests (est index offset by len(ref))
    parent = list(range(len(ref) + len(est)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, (a, b) in enumerate(zip(lo, hi)):
        for j in range(a, b):
            ra, rb = find(i), find(len(ref) + j)
            if ra != rb:
                parent[ra] = rb
    comps: dict[int, tuple[list, list]] = {}
    for i in range(len(ref)):
        if hi[i] > lo[i]:
            comps.setdefault(find(i), ([], []))[0].append(i)
    for j in range(len(est)):
        r = find(len(ref) + j)
        if r in comps:
            comps[r][1].append(j)
    BIG = 1e3
    pr, pe = [], []
    for ri, ej in comps.values():
        ri, ej = np.asarray(ri), np.asarray(ej)
        D = np.abs(ref[ri][:, None] - est[ej][None, :])
        C = np.where(D <= window, D - BIG, 0.0)
        a, b = linear_sum_assignment(C)
        ok = D[a, b] <= window
        pr.extend(ri[a[ok]]); pe.extend(ej[b[ok]])
    o = np.argsort(pr)
    return np.asarray(pr, int)[o], np.asarray(pe, int)[o]


# --- Phase 4 variant (D2): the peak-picking threshold as a parameter ------------------

DELTA = 0.07          # _peak_pick's fixed threshold above the local mean (normalised envelope)
PICK = dict(pre_max_s=0.03, post_max_s=0.00, pre_avg_s=0.10, post_avg_s=0.10, wait_s=0.03)


def _peak_pick_delta(env: np.ndarray, sr: int, hop: int, delta: float = DELTA) -> tuple[np.ndarray, np.ndarray]:
    """Copy of `_peak_pick` with `delta` as a parameter; delta = 0.07 is `_peak_pick`.

    The envelope is normalised to [0, 1] over the whole signal, so `delta` is a fraction of the
    signal's own envelope range: a change of that range moves the effective threshold."""
    import librosa
    env = env - env.min()
    env = env / (env.max() + 1e-12)
    kw = dict(pre_max=0.03 * sr // hop, post_max=0.00 * sr // hop + 1,
              pre_avg=0.10 * sr // hop, post_avg=0.10 * sr // hop + 1,
              wait=0.03 * sr // hop, delta=delta)
    return librosa.util.peak_pick(env, **kw), env


def _refine_peak(envn: np.ndarray, k: np.ndarray, sr: int, hop: int = HOP_FINE,
                 look_s: float = 0.03, min_sep_s: float | None = MIN_SEP_S) -> np.ndarray:
    """The refinement of `detect_refined(mode='peak')` applied to given picks `k` (a copy of
    its code): local maximum within look_s, parabolic peak, merge closer than min_sep_s."""
    k = np.asarray(k, dtype=int)
    if len(k) == 0:
        return np.array([])
    kp = _to_local_max(envn, k, int(round(look_s * sr / hop)))
    t = kp + parabolic(envn, kp)
    if min_sep_s is not None:
        sep = min_sep_s * sr / hop
        order = np.argsort(t)
        t, h = t[order], envn[kp[order]]
        keep_t, keep_h = [t[0]], [h[0]]
        for ti, hi in zip(t[1:], h[1:]):
            if ti - keep_t[-1] < sep:
                if hi > keep_h[-1]:
                    keep_t[-1], keep_h[-1] = ti, hi
            else:
                keep_t.append(ti)
                keep_h.append(hi)
        t = np.asarray(keep_t)
    return np.unique(t) * hop / sr


def detect_delta(y: np.ndarray, sr: int, delta: float = DELTA, env: np.ndarray | None = None
                 ) -> np.ndarray:
    """`detect` (the Phase 2 detector) with the peak-picking threshold `delta` as a parameter.

    delta = 0.07 gives exactly `detect`. `env` may pass a precomputed `onset_envelope(y, sr)`
    (hop 64) to re-pick at several thresholds without recomputing the envelope."""
    if env is None:
        env = onset_envelope(y, sr, HOP_FINE)
    k, envn = _peak_pick_delta(env, sr, HOP_FINE, delta)
    return _refine_peak(envn, k, sr, HOP_FINE)


def pick_margin(envn: np.ndarray, sr: int, hop: int = HOP_FINE) -> np.ndarray:
    """Per frame, the quantity librosa's peak picker compares with `delta`: the normalised
    envelope minus its moving mean over [n - pre_avg, n + post_avg) frames (window clipped at
    the signal ends, as in librosa.util.peak_pick)."""
    pre = int(np.ceil(0.10 * sr // hop))
    post = int(np.ceil(0.10 * sr // hop + 1))
    x = np.asarray(envn, dtype=np.float64)
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - pre)
    hi = np.minimum(n, idx + post)
    return x - (c[hi] - c[lo]) / (hi - lo)
