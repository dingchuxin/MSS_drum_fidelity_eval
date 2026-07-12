# Environments

The models have mutually incompatible dependency pins, so the project uses **three venvs**.
The main env and the MSST sidecar are **Python 3.12**; the Spleeter sidecar is **Python 3.11**
(Spleeter needs TensorFlow 2.12, which has no 3.12 wheels). Venvs are gitignored and rebuilt
from the requirements files; `uv` manages Python versions and does the installs.

| venv | Python | numpy | holds | runs |
|------|--------|-------|-------|------|
| `.venv` (main) | 3.12 | ≥2 | pipeline, metrics, analysis, `audio-separator` | **HT-Demucs** + all eval/synthesis |
| `.venv-msst` | 3.12 | <2 | ZFTurbo **MSST** inference (torch) | **BS-Roformer + SCNet** → stems ingested |
| `.venv-spleeter` | 3.11 | <2 | `spleeter` (+ TensorFlow 2.12) | **Spleeter** → stems ingested |

**Why three.** `audio-separator` requires `numpy>=2`; Spleeter pins `numpy<2` + TF 2.12; MSST's
bs_roformer/scnet pin an older stack (`numpy<2`, `transformers~=4.35`). No two of these coexist.
Each sidecar writes stems that the main pipeline reads back through the stem cache, so the split
is invisible to the metrics code. `audio-separator` runs HT-Demucs in the main env; **BS-Roformer
and SCNet run via MSST** (audio-separator has no SCNet, and we want ZFTurbo's exact 4-stem MUSDB
checkpoints).

## Setup

Runs happen on a GPU desktop (verified: NVIDIA RTX 4070 SUPER, CUDA 12.x).

```powershell
uv python install 3.11            # 3.12 is typically already present for the main env

# main env — pipeline + audio-separator (HT-Demucs)
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe "audio-separator[cpu]"

# MSST sidecar — bs_roformer + scnet (CUDA torch)
uv venv --python 3.12 .venv-msst
uv pip install --python .venv-msst\Scripts\python.exe torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-msst\Scripts\python.exe -r requirements-msst.txt
python scripts\setup_msst.py       # clones ZFTurbo/MSST + downloads the two 4-stem checkpoints

# Spleeter sidecar (Python 3.11). Windows: Spleeter hard-pins tensorflow-io-gcs-filesystem==0.32.0,
# which has no Windows wheel — override to the last Windows build via overrides-spleeter.txt.
uv venv --python 3.11 .venv-spleeter
uv pip install --python .venv-spleeter\Scripts\python.exe --override overrides-spleeter.txt -r requirements-spleeter.txt
```

**GPU notes.** The torch models (bs_roformer, scnet via MSST) use CUDA automatically once CUDA
torch is installed — this is where the speed-up lands. HT-Demucs via audio-separator also uses
torch (GPU if CUDA torch is in the main env, CPU otherwise — acceptable for a baseline).
**Spleeter is CPU-only on Windows** (TensorFlow dropped native-Windows GPU after 2.10); it is
light and fast enough that this does not matter.

## Model checkpoints (MSST)

`scripts/setup_msst.py` is idempotent: it clones ZFTurbo's MSST into `external/MSST/` and
downloads the matched config + checkpoint pairs to the paths `config/models.yaml` expects.
Config and checkpoint come from the **same MSST release** so their architectures match.

| model | config | checkpoint | source (MUSDB18, 4-stem) |
|-------|--------|-----------|--------------------------|
| bs_roformer | `configs/config_bs_roformer_384_8_2_485100.yaml` | `data/weights/bs_roformer_musdb.ckpt` | MSST v1.0.12 — drums 11.6 dB |
| scnet | `configs/config_musdb18_scnet_xl.yaml` | `data/weights/scnet_xl_musdb.ckpt` | MSST v1.0.13, SCNet-XL — drums 11.5 dB |
