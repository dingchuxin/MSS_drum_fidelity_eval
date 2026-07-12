"""Audio I/O and small helpers shared across the pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def load_audio(path: str | Path, target_sr: int = 44100, mono: bool = False) -> tuple[np.ndarray, int]:
    """Load audio as float64 (samples,) or (samples, channels), resampled if needed."""
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    if mono:
        x = x.mean(axis=1)
    if sr != target_sr:
        import librosa
        y = x.T if x.ndim == 2 else x
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        x = y.T if x.ndim == 2 else y
        sr = target_sr
    return x, sr


def load_config(name: str) -> dict:
    """Load a yaml file from config/ by stem name (e.g. 'models', 'datasets')."""
    root = Path(__file__).resolve().parents[2] / "config"
    with open(root / f"{name}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
