"""One-shot setup for the MSST sidecar models (bs_roformer + scnet).

Clones ZFTurbo's Music-Source-Separation-Training into external/MSST and downloads
the two 4-stem MUSDB18 checkpoints (+ their matching release configs) to the paths
config/models.yaml expects. Idempotent — skips anything already present.

Run once after cloning this repo and creating .venv-msst (see docs/environments.md):
    python scripts/setup_msst.py
"""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MSST_REPO = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
REL = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download"

# (url, destination). Configs land in the vendored MSST tree; checkpoints in data/weights.
# Each config/checkpoint pair comes from the SAME MSST release so their architectures match.
ASSETS = [
    (f"{REL}/v1.0.12/config_bs_roformer_384_8_2_485100.yaml",
     ROOT / "external/MSST/configs/config_bs_roformer_384_8_2_485100.yaml"),
    (f"{REL}/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
     ROOT / "data/weights/bs_roformer_musdb.ckpt"),
    (f"{REL}/v1.0.13/config_musdb18_scnet_xl.yaml",
     ROOT / "external/MSST/configs/config_musdb18_scnet_xl.yaml"),
    (f"{REL}/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt",
     ROOT / "data/weights/scnet_xl_musdb.ckpt"),
]


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading -> {dest.relative_to(ROOT)}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    msst = ROOT / "external" / "MSST"
    if not (msst / "inference.py").exists():
        print(f"Cloning MSST -> {msst.relative_to(ROOT)}")
        msst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", MSST_REPO, str(msst)], check=True)
    else:
        print("MSST already cloned; skipping.")

    print("Fetching checkpoints + configs:")
    for url, dest in ASSETS:
        _download(url, dest)
    print("\nDone. bs_roformer + scnet are ready (see config/models.yaml).")


if __name__ == "__main__":
    main()
