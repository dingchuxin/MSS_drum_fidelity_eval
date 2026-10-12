"""In-process Spleeter runner (run in .venv-spleeter): spleeter:4stems, MWF off (CLI default).

Mirrors `spleeter separate -p spleeter:4stems`: the waveform is loaded with Spleeter's own
AudioAdapter (ffmpeg, float32, 44.1 kHz) and passed to Separator.separate(); the model is
loaded once. Windows are slices of that waveform. Output is 32-bit float WAV.
MODEL_PATH must point at the directory holding pretrained_models/4stems (no download).

    python run_spleeter.py <jobs.jsonl>        (job format as in run_msst.py)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _p2 as P  # noqa: E402

import soundfile as sf  # noqa: E402
from spleeter.audio.adapter import AudioAdapter  # noqa: E402
from spleeter.separator import Separator  # noqa: E402


def main():
    jobs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    jobs = [j for j in jobs if not _done(j)]
    print(f"{len(jobs)} jobs to run", flush=True)
    sep = Separator("spleeter:4stems", MWF=False)
    adapter = AudioAdapter.default()
    base_meta = P.run_metadata({"runner": "run_spleeter.py", "config": "spleeter:4stems",
                                "MWF": False})
    cache = {}
    for i, j in enumerate(jobs, 1):
        if j["track"] not in cache:
            cache.clear()
            wav, sr = adapter.load(str(P.MUSDB_TEST / j["track"] / "mixture.wav"), sample_rate=44100)
            cache[j["track"]] = wav.astype(np.float32)
        wav = cache[j["track"]]
        s0, s1 = j["win"] if j.get("win") else (0, wav.shape[0])
        t = time.time()
        out = sep.separate(wav[s0:s1])
        dt = time.time() - t
        meta = {**base_meta, **{k: j[k] for k in j if k != "out"}, "sep_seconds": round(dt, 3),
                "input_samples": int(s1 - s0), "sr": 44100, "subtype": "FLOAT"}
        o = Path(j["out"])
        if j["keep"] == "all":
            o.mkdir(parents=True, exist_ok=True)
            for s in ["drums", "bass", "other", "vocals"]:
                sf.write(str(o / f"{s}.wav"), out[s][: s1 - s0], 44100, subtype="FLOAT")
            P.write_json(o / "run.json", meta)
        else:
            a, b = j["keep"]
            o.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(o) + ".wav", out["drums"][a - s0: b - s0], 44100, subtype="FLOAT")
            meta["kept_span_track_samples"] = [a, b]
            P.write_json(Path(str(o) + ".json"), meta)
        print(f"[{i}/{len(jobs)}] {j['id']} {dt:.1f}s", flush=True)


def _done(j) -> bool:
    o = Path(j["out"])
    return (o / "run.json").exists() if j["keep"] == "all" else Path(str(o) + ".json").exists()


if __name__ == "__main__":
    main()
