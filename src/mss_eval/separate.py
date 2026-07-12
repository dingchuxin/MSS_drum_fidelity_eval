"""Model runners: turn a mixture into a dict of stems, with provenance.

Three runner kinds mirror config/models.yaml:
  builtin  -> runs in THIS venv, in-process (htdemucs via audio-separator).
  sidecar  -> runs in a separate venv via its CLI, because deps conflict with ours
              (spleeter in .venv-spleeter; bs_roformer/scnet in .venv-msst). We write
              the mixture to a temp file, invoke the sidecar's CLI, read the stems back.
  external -> stems produced out-of-band by a GUI tool and *ingested* from a folder
              (spectralayers). No automation.

All stem dicts are keyed by canonical names {drums, bass, vocals, other}. Every result
carries a provenance dict {model, version, params, input_range, sr} so the thesis can
report exactly how each number was produced — the reproducibility requirement §4.7 lacked.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import load_audio

_ROOT = Path(__file__).resolve().parents[2]
# substrings (lowercased) that identify a canonical stem in an output filename
_STEM_KEYS = ("drums", "bass", "vocals", "other", "piano", "guitar")


def _canon_stem(name: str) -> str:
    """Map a model's output filename/label to a canonical stem key."""
    n = name.lower()
    for s in _STEM_KEYS:
        if s in n:
            return s
    if "instrumental" in n or n.startswith("no_") or n.startswith("no "):
        return "other"
    return n


@dataclass
class Separation:
    stems: dict[str, np.ndarray]
    provenance: dict = field(default_factory=dict)


def separate(mixture: np.ndarray, sr: int, model_cfg: dict,
             input_range: tuple[float, float] | None = None,
             cache_dir: "Path | str | None" = None) -> Separation:
    """Dispatch to the right runner based on model_cfg['runner'].

    If `cache_dir` is given, separated stems are persisted there as <stem>.wav and
    reused on later runs. Separation is the expensive step (esp. Spleeter on CPU);
    metrics are cheap, so caching makes re-scoring instant and runs resumable.
    """
    runner = model_cfg.get("runner", "builtin")
    name = model_cfg["name"]
    prov = {
        "model": name,
        "version": model_cfg.get("version", "unknown"),
        "params": model_cfg.get("params", {}),
        "input_range": list(input_range) if input_range else None,
        "sr": sr,
    }
    expected = model_cfg.get("stems") or ["drums", "bass", "vocals", "other"]
    if cache_dir is not None:
        cached = _load_cached_stems(Path(cache_dir), expected, sr)
        if cached is not None:
            prov["cached"] = True
            return Separation(stems=cached, provenance=prov)

    engine = model_cfg.get("engine")
    if runner == "external":
        stems = _ingest_external(model_cfg, sr)
    elif runner == "sidecar":
        stems = _run_sidecar(mixture, sr, model_cfg)
    elif engine == "audio_separator":
        stems = _run_audio_separator(mixture, sr, model_cfg)
    elif engine == "demucs":
        stems = _run_demucs(mixture, sr, model_cfg)
    elif engine == "spleeter":
        stems = _run_spleeter(mixture, sr, model_cfg)
    elif engine == "openunmix":
        stems = _run_openunmix(mixture, sr, model_cfg)
    else:
        raise ValueError(
            f"model '{name}': runner='{runner}', engine='{engine}' — no matching runner. "
            "Set runner: builtin|sidecar|external in config/models.yaml."
        )
    if cache_dir is not None:
        _write_cached_stems(Path(cache_dir), stems, sr)
    return Separation(stems=stems, provenance=prov)


def _load_cached_stems(cache_dir: Path, expected: list[str], sr: int):
    """Return cached stems if every expected <stem>.wav exists in cache_dir, else None."""
    paths = {s: cache_dir / f"{s}.wav" for s in expected}
    if not all(p.exists() for p in paths.values()):
        return None
    return {s: load_audio(p, target_sr=sr)[0] for s, p in paths.items()}


def _write_cached_stems(cache_dir: Path, stems: dict[str, np.ndarray], sr: int) -> None:
    import soundfile as sf
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stem_name, x in stems.items():
        sf.write(str(cache_dir / f"{stem_name}.wav"), x if x.ndim == 2 else x[:, None], sr)


def _run_sidecar(mixture: np.ndarray, sr: int, model_cfg: dict) -> dict[str, np.ndarray]:
    """Separate in another venv via its CLI (deps conflict with the main env).

    Spleeter (.venv-spleeter) and the MSST models (.venv-msst) both round-trip through
    temp files. NOTE: first-run validation needed on the desktop — the two CLI shapes
    below match spleeter's CLI and MSST's inference.py, but exact output-file naming is
    confirmed only at runtime; `_canon_stem` maps whatever stem names appear.
    """
    import soundfile as sf

    venv = _ROOT / model_cfg["venv"]
    pyexe = venv / "Scripts" / "python.exe"
    if not pyexe.exists():
        pyexe = venv / "bin" / "python"
    if not pyexe.exists():
        raise FileNotFoundError(
            f"sidecar venv missing for '{model_cfg['name']}': {venv}. "
            "Create it per docs/environments.md."
        )

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        indir, outdir = tmp / "in", tmp / "out"
        indir.mkdir(); outdir.mkdir()
        mix_path = indir / "mixture.wav"
        sf.write(str(mix_path), mixture if mixture.ndim == 2 else mixture[:, None], sr)

        if model_cfg.get("model_type"):          # MSST (bs_roformer / scnet)
            cmd = [str(pyexe), str(_ROOT / "external" / "MSST" / "inference.py"),
                   "--model_type", model_cfg["model_type"],
                   "--config_path", str(_ROOT / model_cfg["config"]),
                   "--start_check_point", str(_ROOT / model_cfg["checkpoint"]),
                   "--input_folder", str(indir), "--store_dir", str(outdir)]
            cwd = str(_ROOT / "external" / "MSST")   # inference.py imports from its own dir
        else:                                    # spleeter
            cmd = [str(pyexe), "-m", "spleeter", "separate",
                   "-p", model_cfg.get("config", "spleeter:4stems"),
                   "-o", str(outdir), str(mix_path)]
            cwd = str(_ROOT)

        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"sidecar '{model_cfg['name']}' failed (exit {proc.returncode}).\n"
                f"cmd: {' '.join(cmd)}\nstderr tail:\n{proc.stderr[-1500:]}"
            )

        stems: dict[str, np.ndarray] = {}
        for wavf in sorted(outdir.rglob("*.wav")):
            x, _ = load_audio(wavf, target_sr=sr)
            stems[_canon_stem(wavf.stem)] = x
        if not stems:
            raise RuntimeError(f"sidecar '{model_cfg['name']}' produced no .wav in {outdir}.")
    return stems


def _ingest_external(model_cfg: dict, sr: int) -> dict[str, np.ndarray]:
    """Load hand-exported stems from model_cfg['ingest_dir'] (mvsep/vbsplit/etc.)."""
    d = Path(model_cfg["ingest_dir"])
    if not d.exists():
        raise FileNotFoundError(
            f"External model '{model_cfg['name']}' expects exported stems in {d}. "
            "Run the tool out-of-band and drop <stem>.wav files there."
        )
    stems = {}
    for wav in sorted(d.glob("*.wav")):
        x, _ = load_audio(wav, target_sr=sr)
        stems[wav.stem.lower()] = x
    if not stems:
        raise FileNotFoundError(f"No .wav stems found in {d}.")
    return stems


def _run_demucs(mixture: np.ndarray, sr: int, model_cfg: dict) -> dict[str, np.ndarray]:
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as e:  # pragma: no cover
        raise ImportError("demucs not installed. `pip install demucs`.") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_cfg.get("model", "htdemucs"))
    model.eval()
    wav = torch.as_tensor(np.atleast_2d(mixture.T), dtype=torch.float32)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, progress=False)[0]
    sources = sources.cpu() * ref.std() + ref.mean()
    return {_canon_stem(name): sources[i].numpy().T for i, name in enumerate(model.sources)}


def _run_spleeter(mixture: np.ndarray, sr: int, model_cfg: dict) -> dict[str, np.ndarray]:
    try:
        from spleeter.separator import Separator
    except ImportError as e:  # pragma: no cover
        raise ImportError("spleeter not installed. `pip install spleeter`.") from e
    sep = Separator(model_cfg.get("config", "spleeter:4stems"))
    wav = np.atleast_2d(mixture)
    if wav.shape[0] < wav.shape[1]:
        wav = wav.T  # spleeter wants (samples, channels)
    pred = sep.separate(wav)
    return {k.lower(): v for k, v in pred.items()}


def _run_openunmix(mixture: np.ndarray, sr: int, model_cfg: dict) -> dict[str, np.ndarray]:
    """Open-Unmix (umxl / umxhq) — the light MUSDB baseline, CPU by default."""
    try:
        import torch
        from openunmix import predict
    except ImportError as e:  # pragma: no cover
        raise ImportError("openunmix not installed. `pip install openunmix`.") from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav = torch.as_tensor(np.atleast_2d(mixture.T), dtype=torch.float32)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    estimates = predict.separate(
        audio=wav, rate=sr, model_str_or_path=model_cfg.get("model", "umxl"), device=device)
    # estimates: {name: tensor (1, chan, samples)}
    return {_canon_stem(name): est[0].cpu().numpy().T for name, est in estimates.items()}


def _run_audio_separator(mixture: np.ndarray, sr: int, model_cfg: dict) -> dict[str, np.ndarray]:
    """MDX-Net / UVR models via audio-separator (ONNX, CPU-capable).

    Scriptable stand-in for the MVSEP-MDX23 lineage. audio-separator is file-based,
    so we round-trip through a temp wav and read back the exported stems.
    """
    try:
        from audio_separator.separator import Separator
    except ImportError as e:  # pragma: no cover
        raise ImportError("audio-separator not installed. `pip install audio-separator`.") from e
    import tempfile
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mix_path = tmp / "mixture.wav"
        wav = mixture if mixture.ndim == 2 else mixture[:, None]
        sf.write(str(mix_path), wav, sr)
        sep = Separator(output_dir=str(tmp), output_format="WAV")
        sep.load_model(model_cfg["model"])
        out_files = sep.separate(str(mix_path))
        stems = {}
        for f in out_files:
            p = tmp / f if not Path(f).is_absolute() else Path(f)
            x, _ = load_audio(p, target_sr=sr)
            # audio-separator names files like "mixture_(Drums)_htdemucs.wav" — map by
            # the stem keyword in the name, not the whole filename.
            stems[_canon_stem(p.stem)] = x
    return stems
