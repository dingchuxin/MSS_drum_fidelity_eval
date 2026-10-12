"""Benchmark experiment: run each model over MUSDB18 and, for every stem, record
BOTH the developer-style reference metrics (SDR/SIR/SAR + SI-SDR) AND the
analyst-facing fidelity metrics (onset timing, loudness/dynamics, timbre).

The analyst metrics are reference-based measures of what separation does to transients
and dynamics, quantities a plain SDR number hides.

Usage:
    python scripts/run_benchmark.py --models spleeter htdemucs bs_roformer scnet

Requires MUSDB18-HQ under data/musdb18hq (gitignored) and the metric deps. Stems are cached in
data/stems_out/<model>/<track>/; output: results/tables/benchmark_metrics.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mss_eval import analyst, metrics, report  # noqa: E402
from mss_eval.io import load_config  # noqa: E402
from mss_eval.separate import separate  # noqa: E402


def iter_musdb(cfg, sample_n=None, seed=0):
    import musdb
    root = Path(__file__).resolve().parents[1] / cfg["benchmark"]["root"]   # repo-root-relative
    db = musdb.DB(root=str(root), subsets=cfg["benchmark"]["subset"], is_wav=True)
    tracks = list(db)
    if sample_n:
        rng = np.random.default_rng(seed)
        tracks = [tracks[i] for i in rng.choice(len(tracks), size=min(sample_n, len(tracks)), replace=False)]
    for t in tracks:
        refs = {name: t.targets[name].audio for name in cfg["benchmark"]["reference_stems"]
                if name in t.targets}
        yield t.name, t.audio, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["htdemucs"])
    ap.add_argument("--sample-n", type=int, default=None)
    args = ap.parse_args()

    _models = load_config("models")["models"]           # name-keyed dict
    models_cfg = {name: {**cfg, "name": name} for name, cfg in _models.items()}
    data_cfg = load_config("datasets")
    sr = load_config("models").get("defaults", {}).get("sample_rate", 44100)

    stems_root = Path(__file__).resolve().parents[1] / "data" / "stems_out"
    rows = []
    for model_name in args.models:
        mcfg = models_cfg[model_name]
        for track_name, mixture, refs in iter_musdb(data_cfg, args.sample_n):
            cache_dir = stems_root / model_name / _safe_name(track_name)
            sep = separate(mixture, sr, mcfg, cache_dir=cache_dir)
            common = [s for s in refs if s in sep.stems]
            est = {s: sep.stems[s] for s in common}
            ref = {s: refs[s] for s in common}
            # developer-style reference metrics
            bss = metrics.bss_eval(est, ref, sr)
            # labeling precision is model-wide (needs all stems together)
            lab, _routing = analyst.labeling_precision(est, ref, sr)
            for stem in common:
                row = {"model": model_name, "track": track_name, "stem": stem,
                       "si_sdr": metrics.si_sdr(est[stem], ref[stem]), **bss[stem]}
                # analyst-facing fidelity: timing + dynamic profile (raw + Δ)
                row.update(analyst.analyst_row(est[stem], ref[stem], sr))
                row["labeling_precision"] = lab.get(stem, float("nan"))
                rows.append(row)
        # crash-resilient checkpoint: write the cumulative table after each model,
        # so a failure late in the run (e.g. Spleeter's slow CPU pass) keeps prior results
        if rows:
            report.write_table(metrics.summarize(rows), "benchmark_metrics")
            print(f"  [checkpoint] {model_name} complete — {len(rows)} rows so far", flush=True)

    if not rows:
        print("No results — check that MUSDB18 is present under data/.")
        return
    summary = metrics.summarize(rows)
    report.write_table(summary, "benchmark_metrics")
    print(f"Done: {len(rows)} rows across {len(args.models)} models. See results/tables/.")


def _safe_name(s: str) -> str:
    """Filesystem-safe folder name for a track (drop chars invalid on Windows)."""
    return "".join("_" if c in '<>:"/\\|?*' else c for c in s).strip()


if __name__ == "__main__":
    main()
