"""Regenerate the 3-s WAV excerpts in results/phase1/figure_excerpt/ from the corpus and the
cached stems, so that the WAVs need not be distributed.

The files are those written by scripts/phase1/j6_excerpt.py (same slicing, same PCM_16 writer):
  reference.wav   true drum stem (MUSDB18-HQ test, stereo)
  mixture.wav     sum of the four true stems (drums + bass + other + vocals)
  spleeter.wav, htdemucs.wav, bs_roformer.wav, scnet.wav
                  the cached default-run drum estimates (<MSS_DATA_ROOT>/stems_out/<model>/<track>/)
Excerpt: samples [round(t0 * sr), round(t0 * sr) + 3 s). Track and t0 default to the values in
the output folder's manifest.json (James Elder & Mark M Thompson - The English Actor, 33.0 s).

Existing files are never overwritten unless --overwrite is given; an existing file is instead
compared sample by sample with the regenerated excerpt (as 16-bit integers), and the result is
printed. A new folder (--out) receives all six files.

    MSS_DATA_ROOT=<dir with musdb18hq/ and stems_out/> python scripts/make_excerpt.py
        [--out DIR] [--track NAME] [--t0 SECONDS] [--dur SECONDS] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MSS_DATA_ROOT", ROOT / "data"))
MUSDB_TEST = DATA / "musdb18hq" / "test"
STEMS = DATA / "stems_out"
SR = 44100
MODELS = ["spleeter", "htdemucs", "bs_roformer", "scnet"]
SOURCES = ["drums", "bass", "other", "vocals"]
DEFAULT_OUT = ROOT / "results" / "phase1" / "figure_excerpt"


def safe_name(s: str) -> str:
    return "".join("_" if c in '<>:"/\\|?*' else c for c in s).strip()


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    assert sr == SR, (path, sr)
    return x


def excerpts(track: str, t0: float, dur: float) -> dict[str, np.ndarray]:
    s0 = int(round(t0 * SR))
    seg = slice(s0, s0 + int(round(dur * SR)))
    ref = load(MUSDB_TEST / track / "drums.wav")
    rest = sum(load(MUSDB_TEST / track / f"{s}.wav") for s in SOURCES if s != "drums")
    out = {"reference": ref[seg], "mixture": (ref + rest)[seg]}
    for m in MODELS:
        out[m] = load(STEMS / m / safe_name(track) / "drums.wav")[seg]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--track", default=None, help="default: manifest.json in --out, else the published excerpt")
    ap.add_argument("--t0", type=float, default=None, help="start (s); default as --track")
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--overwrite", action="store_true", help="replace existing WAVs (default: compare only)")
    a = ap.parse_args()
    out = Path(a.out)
    man = out / "manifest.json"
    info = json.load(open(man)) if man.exists() else {}
    track = a.track or info.get("track", "James Elder & Mark M Thompson - The English Actor")
    t0 = a.t0 if a.t0 is not None else float(info.get("t0_s", 33.0))
    print(f"{track}  [{t0:.3f}, {t0 + a.dur:.3f}) s  ->  {out}")
    out.mkdir(parents=True, exist_ok=True)
    all_same = True
    for name, x in excerpts(track, t0, a.dur).items():
        f = out / f"{name}.wav"
        if f.exists() and not a.overwrite:
            import tempfile
            with tempfile.TemporaryDirectory() as td:          # same writer, then read back as int16
                tmp = Path(td) / "x.wav"
                sf.write(tmp, x, SR, subtype="PCM_16")
                new = sf.read(str(tmp), dtype="int16", always_2d=True)[0]
            old, sr = sf.read(str(f), dtype="int16", always_2d=True)
            same = sr == SR and old.shape == new.shape and np.array_equal(old, new)
            nd = int(np.sum(old != new)) if old.shape == new.shape else -1
            all_same &= same
            print(f"  {name}.wav exists: {'identical' if same else f'DIFFERS ({nd} samples)'} (not overwritten)")
        else:
            sf.write(f, x, SR, subtype="PCM_16")
            print(f"  {name}.wav written")
    if not all_same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
