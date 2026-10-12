"""Synthetic signals with analytically known 10-ms RMS levels (no project imports).

Carrier: a 1 kHz sine. At 44.1 kHz one period is 44.1 samples, so the 441-sample (10 ms)
rectangular window spans exactly 10 periods and the mean of sin^2 over any such window is 1/2
(to rounding). Any 10-ms window that lies inside a constant-amplitude section of amplitude A
therefore has RMS A / sqrt(2) exactly, i.e. 20 log10(A) - 3.0103 dB.
"""
from __future__ import annotations

import numpy as np

SR = 44100
F0 = 1000.0
DB_SINE = 20 * np.log10(1 / np.sqrt(2))          # -3.0103 dB: RMS of a unit sine


def sine(n: int, f: float = F0, sr: int = SR) -> np.ndarray:
    return np.sin(2 * np.pi * f * np.arange(n) / sr)


def level(a: float) -> float:
    """10-ms RMS level (dB) of a constant-amplitude section of amplitude a of the 1 kHz sine."""
    return 20 * np.log10(a) + DB_SINE if a > 0 else -240.0


def piecewise(n: int, sections: list[tuple[float, float, float]], sr: int = SR) -> np.ndarray:
    """Amplitude envelope: list of (t_start_s, t_end_s, amplitude); later sections overwrite."""
    a = np.zeros(n)
    for t0, t1, amp in sections:
        a[int(round(t0 * sr)):int(round(t1 * sr))] = amp
    return a


def ramp_decay(n: int, t0: float, attack_ms: float, tau_ms: float, amp: float, sr: int = SR) -> np.ndarray:
    """Linear 0 -> amp ramp over attack_ms starting at t0, then exponential decay (tau_ms)."""
    t = np.arange(n) / sr - t0
    a = attack_ms / 1000
    env = np.where(t < 0, 0.0, np.where(t < a, t / a, np.exp(-(t - a) / (tau_ms / 1000))))
    return amp * env
