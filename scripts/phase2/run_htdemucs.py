"""In-process HT-Demucs runner (main .venv): audio-separator 0.44.2, checkpoint 955717e8.

Two settings:
  b1       reproducible: the model is built exactly as DemucsSeparator.separate() builds it
           (get_demucs_model + demucs_segments('Default') + eval), and demix_demucs() is
           called directly on the float32 input. That keeps audio-separator's whole-input
           mean/std normalisation and shifts=2 / overlap=0.25 / segments on, but
             - Python's `random` (the only RNG apply_model uses for shift offsets) is seeded
               with the job seed immediately before each input, and
             - the output is NOT passed through spec_utils.normalize (the 0.9 peak cap is
               applied only in write_audio, which is bypassed). normalization_threshold
               cannot turn the cap off: it must lie in (0, 1].
  default  audio-separator's untouched default path: Separator().separate(file), unseeded
           shifts (Python `random` reseeded from OS entropy first), 0.9 peak cap on. The
           input is written as FLOAT WAV so audio-separator writes FLOAT output (it copies
           the input subtype); the input values are identical to the 16-bit originals.

Jobs file: JSON lines {"id", "track", "win", "keep", "seed", "setting", "out", "tags"}
(same conventions as run_msst.py).     python run_htdemucs.py <jobs.jsonl>
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _p2 as P  # noqa: E402

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

MODEL_FILE = "htdemucs.yaml"
DRUMS_IDX = 1          # demix_demucs swaps sources[[0, 1]] -> order bass, drums, other, vocals
STEM_IDX = {"bass": 0, "drums": 1, "other": 2, "vocals": 3}


def build_b1():
    from audio_separator.separator import Separator
    from audio_separator.separator.uvr_lib_v5.demucs.apply import demucs_segments
    from audio_separator.separator.uvr_lib_v5.demucs.pretrained import get_model
    sep = Separator(output_dir=tempfile.gettempdir(), output_format="WAV", log_level=40)
    sep.load_model(MODEL_FILE)
    inst = sep.model_instance
    m = get_model(name=os.path.splitext(os.path.basename(inst.model_path))[0],
                  repo=Path(os.path.dirname(inst.model_path)))
    m = demucs_segments(inst.segment_size, m)
    m.to(inst.torch_device).eval()
    inst.demucs_model_instance = m
    return sep, inst


def run_default(sep, mix: np.ndarray) -> np.ndarray:
    """Untouched default path through a file; returns (4, 2, n) in STEM_IDX order."""
    random.seed()                                    # OS entropy: audio-separator default
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mixture.wav"
        sf.write(str(p), mix.T, 44100, subtype="FLOAT")
        sep.output_dir = tmp
        sep.model_instance.output_dir = tmp
        files = sep.separate(str(p))
        out = np.zeros((4, 2, mix.shape[1]), dtype=np.float32)
        for f in files:
            f = Path(f) if Path(f).is_absolute() else Path(tmp) / f
            name = f.stem.lower()
            for s, k in STEM_IDX.items():
                if f"({s})" in name:
                    x, _ = sf.read(str(f), dtype="float32", always_2d=True)
                    # audio-separator maps a 32-bit input to 32-bit *integer* PCM output;
                    # the 0.9 peak cap precludes clipping, precision is ~ -190 dB
                    run_default.subtype = sf.info(str(f)).subtype
                    out[k] = x.T[:, : mix.shape[1]]
    return out


def main():
    jobs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    jobs = [j for j in jobs if not _done(j)]
    print(f"{len(jobs)} jobs to run", flush=True)
    sep, inst = build_b1()
    b1_model = inst.demucs_model_instance
    base_meta = P.run_metadata({"runner": "run_htdemucs.py", "checkpoint": "955717e8-8726e21a.th",
                                "shifts": inst.shifts, "overlap": inst.overlap,
                                "segment_size": inst.segment_size,
                                "segments_enabled": inst.segments_enabled})
    cache = {}
    for i, j in enumerate(jobs, 1):
        if j["track"] not in cache:
            cache.clear()
            mix, _ = librosa.load(str(P.MUSDB_TEST / j["track"] / "mixture.wav"), sr=44100, mono=False)
            cache[j["track"]] = mix
        mix = cache[j["track"]]
        s0, s1 = j["win"] if j.get("win") else (0, mix.shape[1])
        x = np.ascontiguousarray(mix[:, s0:s1])
        t = time.time()
        if j["setting"] == "b1":
            random.seed(int(j["seed"]))
            src = inst.demix_demucs(x)                  # (4, 2, n), no peak normalisation
            meta_s = {"seed_python_random": int(j["seed"]), "peak_normalisation": "bypassed"}
        else:
            src = run_default(sep, x)
            meta_s = {"seed_python_random": "unseeded (random.seed() from OS entropy)",
                      "peak_normalisation": "on (audio-separator default, 0.9)",
                      "audio_separator_intermediate_subtype": getattr(run_default, "subtype", "?")}
            # separate() rebuilds and then deletes demucs_model_instance: restore ours
            inst.demucs_model_instance = b1_model
        dt = time.time() - t
        meta = {**base_meta, **{k: j[k] for k in j if k != "out"}, **meta_s,
                "sep_seconds": round(dt, 3), "input_samples": int(s1 - s0), "sr": 44100,
                "subtype": "FLOAT"}
        o = Path(j["out"])
        if j["keep"] == "all":
            o.mkdir(parents=True, exist_ok=True)
            for s, k in STEM_IDX.items():
                sf.write(str(o / f"{s}.wav"), src[k].T, 44100, subtype="FLOAT")
            P.write_json(o / "run.json", meta)
        else:
            a, b = j["keep"]
            o.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(o) + ".wav", src[DRUMS_IDX][:, a - s0: b - s0].T, 44100, subtype="FLOAT")
            meta["kept_span_track_samples"] = [a, b]
            P.write_json(Path(str(o) + ".json"), meta)
        print(f"[{i}/{len(jobs)}] {j['id']} {dt:.1f}s", flush=True)


def _done(j) -> bool:
    o = Path(j["out"])
    return (o / "run.json").exists() if j["keep"] == "all" else Path(str(o) + ".json").exists()


if __name__ == "__main__":
    main()
