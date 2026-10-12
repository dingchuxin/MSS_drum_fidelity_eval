"""Synthetic drum-like signals with known ground truth (no project imports).

An event is a carrier (filtered noise or a sine) times an amplitude envelope:
    A(t) = rise(t / attack) for 0 <= t < attack  (0 -> 1),  exp(-(t - attack) / tau) after,
so the envelope peak is at t = attack (seconds after the event start) and the amplitude
falls 20 dB in tau * ln(10) after the peak. Event 'times' are envelope START times.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 44100
KINDS = ("white", "low", "mid", "high")


def carrier(n: int, kind: str, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Unit-RMS carrier: white / low (<150 Hz, kick-like) / mid (200-2k, snare-like) /
    high (>5 kHz, hat-like) filtered noise, or 'sine<f>' for a sine at f Hz."""
    if kind.startswith("sine"):
        f = float(kind[4:])
        x = np.sin(2 * np.pi * f * np.arange(n) / sr)
    else:
        x = rng.standard_normal(n + 4096)
        if kind == "low":
            sos = signal.butter(4, 150, "lowpass", fs=sr, output="sos")
        elif kind == "mid":
            sos = signal.butter(4, [200, 2000], "bandpass", fs=sr, output="sos")
        elif kind == "high":
            sos = signal.butter(4, 5000, "highpass", fs=sr, output="sos")
        else:
            sos = None
        if sos is not None:
            x = signal.sosfilt(sos, x)
        x = x[4096:]                      # drop the filter transient
    return x / np.sqrt(np.mean(x ** 2))


def envelope(n: int, attack_ms: float, tau_ms: float, shape: str = "lin", sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    a = max(attack_ms, 1e-6) / 1000
    u = np.clip(t / a, 0, 1)
    if shape == "lin":
        r = u
    elif shape == "cos":
        r = 0.5 - 0.5 * np.cos(np.pi * u)
    elif shape == "exp":                  # fast-then-slow rise, 1 - e^-5u normalised
        r = (1 - np.exp(-5 * u)) / (1 - np.exp(-5))
    else:
        raise ValueError(shape)
    return np.where(t < a, r, np.exp(-(t - a) / (tau_ms / 1000)))


def event(attack_ms: float, tau_ms: float, kind: str, rng: np.random.Generator,
          dur_s: float | None = None, shape: str = "lin", amp: float = 1.0,
          sr: int = SR) -> np.ndarray:
    if dur_s is None:
        dur_s = attack_ms / 1000 + 8 * tau_ms / 1000       # ~ -70 dB at the end
    n = int(round(dur_s * sr))
    return amp * carrier(n, kind, rng, sr) * envelope(n, attack_ms, tau_ms, shape, sr)


def place(y: np.ndarray, x: np.ndarray, t0: float, sr: int = SR) -> None:
    """Add x into y starting at sample round(t0 * sr) (integer placement)."""
    i = int(round(t0 * sr))
    j = min(len(y), i + len(x))
    if i < len(y):
        y[i:j] += x[: j - i]


def frac_delay(y: np.ndarray, d: float) -> np.ndarray:
    """Delay by d samples (may be fractional) via an FFT phase ramp, zero-padded (no wrap)."""
    n = len(y)
    pad = int(np.ceil(abs(d))) + 64
    Y = np.fft.rfft(np.concatenate([y, np.zeros(pad)]))
    k = np.arange(len(Y))
    Y *= np.exp(-2j * np.pi * k * d / (n + pad))
    return np.fft.irfft(Y, n + pad)[:n]


def drum_track(rng: np.random.Generator, dur_s: float = 30.0, ioi=(0.15, 0.6),
               attack=(0.5, 5.0), tau=(20, 150), kinds=KINDS, level_db=(-12, 0),
               start: float = 0.5, sr: int = SR):
    """A drum-like sequence with random IOIs, spectra, attacks, decays and levels.
    Returns (y, table of events)."""
    y = np.zeros(int(dur_s * sr))
    rows = []
    t = start
    while t < dur_s - 1.0:
        k = kinds[rng.integers(len(kinds))]
        a = float(rng.uniform(*attack))
        tu = float(rng.uniform(*tau))
        g = 10 ** (rng.uniform(*level_db) / 20)
        place(y, event(a, tu, k, rng, amp=0.3 * g, sr=sr), t, sr)
        rows.append({"t": round(t * sr) / sr, "kind": k, "attack_ms": a, "tau_ms": tu, "amp": 0.3 * g})
        t += float(rng.uniform(*ioi))
    return y, rows
