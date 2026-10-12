"""In-process MSST runner (run in .venv-msst): load each model once, separate many inputs.

Reproduces MSST inference.py @ ccf86c1 for our settings: librosa.load(sr=44100, mono=False)
float32 input, cudnn.benchmark = True, strict checkpoint load, bigshifts_wrapper(bigshifts=1)
-> demix, no input normalisation (none of the configs enables it), no TTA. The only
departures are that inputs may be a window of the track (sliced from the same float32
array) and that outputs are written as 32-bit float WAV.

Jobs file: JSON lines, one job per line:
  {"id": str, "model": key, "track": name, "win": [s0, s1] | null (full track),
   "keep": "all" | [a, b] (track samples; drums only), "overlap": int | null,
   "seed": int, "out": path, "tags": {...}}
"all" writes <out>/<stem>.wav for all four stems; a span writes <out>.wav (drums, track
samples [a, b)). Every output gets a JSON sidecar with settings and run-time versions.

    python run_msst.py <jobs.jsonl>
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _p2 as P  # noqa: E402

sys.path.insert(0, str(P.MSST_DIR))
os.chdir(P.MSST_DIR)

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from utils.model_utils import bigshifts_wrapper, load_start_checkpoint  # noqa: E402
from utils.settings import get_model_from_config  # noqa: E402

W = P.DATA / "weights"
W2 = P.P2DATA / "weights"
MODELS = {
    "bs_roformer": ("bs_roformer", P.MSST_DIR / "configs/config_bs_roformer_384_8_2_485100.yaml",
                    W / "bs_roformer_musdb.ckpt"),
    "scnet": ("scnet", P.MSST_DIR / "configs/config_musdb18_scnet_xl.yaml",
              W / "scnet_xl_musdb.ckpt"),
    "scnet_xl_ihf": ("scnet", W2 / "config_musdb18_scnet_xl_more_wide_v5.yaml",
                     W2 / "model_scnet_ep_36_sdr_10.0891.ckpt"),
    "scnet_masked_xl_ihf": ("scnet_masked", W2 / "config_musdb18_scnet_xl_ihf.yaml",
                            W2 / "model_scnet_masked_ep_111_sdr_9.8286.ckpt"),
}


class _Args:
    lora_checkpoint_loralib = ""
    lora_checkpoint_peft = ""

    def __init__(self, model_type, ckpt):
        self.model_type, self.start_check_point = model_type, str(ckpt)


def load_model(key: str):
    mtype, cfg, ckpt = MODELS[key]
    torch.backends.cudnn.benchmark = True
    model, config = get_model_from_config(mtype, str(cfg))
    sd = torch.load(str(ckpt), weights_only=False, map_location="cpu")
    load_start_checkpoint(_Args(mtype, ckpt), model, sd, type_="inference")
    model = model.to("cuda").eval()
    return model, config, mtype


def main():
    jobs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    jobs = [j for j in jobs if not _done(j)]
    print(f"{len(jobs)} jobs to run", flush=True)
    base_meta = P.run_metadata({"runner": "run_msst.py"})
    jobs.sort(key=lambda j: (j["model"], j.get("overlap") or 0, j["track"]))
    cur, model, config, mtype, mix_cache = None, None, None, None, {}
    for i, j in enumerate(jobs, 1):
        key = (j["model"], j.get("overlap"))
        if key != cur:
            model, config, mtype = load_model(j["model"])
            native = int(config.inference.num_overlap)
            if j.get("overlap"):
                config.inference.num_overlap = int(j["overlap"])
            cur = key
            print(f"loaded {j['model']} (num_overlap {config.inference.num_overlap}, "
                  f"native {native})", flush=True)
        if j["track"] not in mix_cache:
            mix_cache.clear()
            mix, sr = librosa.load(str(P.MUSDB_TEST / j["track"] / "mixture.wav"), sr=44100, mono=False)
            mix_cache[j["track"]] = mix
        mix = mix_cache[j["track"]]
        s0, s1 = j["win"] if j.get("win") else (0, mix.shape[1])
        seed = int(j.get("seed", P.SEED))
        random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
        t = time.time()
        out = bigshifts_wrapper(config, model, mix[:, s0:s1], "cuda", model_type=mtype,
                                pbar=False, bigshifts=1)
        dt = time.time() - t
        meta = {**base_meta, **{k: j[k] for k in j if k != "out"},
                "num_overlap": int(config.inference.num_overlap),
                # the base metadata is captured before load_model() turns this on
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "chunk_size": int(config.audio.chunk_size), "sep_seconds": round(dt, 3),
                "input_samples": int(s1 - s0), "sr": 44100, "subtype": "FLOAT"}
        o = Path(j["out"])
        if j["keep"] == "all":
            o.mkdir(parents=True, exist_ok=True)
            for stem in ["drums", "bass", "other", "vocals"]:
                sf.write(str(o / f"{stem}.wav"), out[stem].T, 44100, subtype="FLOAT")
            P.write_json(o / "run.json", meta)
        else:
            a, b = j["keep"]
            seg = out["drums"][:, a - s0: b - s0]
            o.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(o) + ".wav", seg.T, 44100, subtype="FLOAT")
            meta["kept_span_track_samples"] = [a, b]
            P.write_json(Path(str(o) + ".json"), meta)
        print(f"[{i}/{len(jobs)}] {j['id']} {dt:.1f}s", flush=True)


def _done(j) -> bool:
    o = Path(j["out"])
    return (o / "run.json").exists() if j["keep"] == "all" else Path(str(o) + ".json").exists()


if __name__ == "__main__":
    main()
