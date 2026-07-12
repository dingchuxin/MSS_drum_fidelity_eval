"""A2 experiment driver: sweep input length/context for a fixed target region R and
measure how the separation of R drifts.

    python scripts/run_input_length.py --models htdemucs --mode length
    python scripts/run_input_length.py --models htdemucs --mode phrase   # needs boundaries

length mode runs today on MUSDB18 (no annotation). phrase mode is the thesis design and
needs phrase boundaries — their SOURCE is still to be decided (annotate a MUSDB subset /
auto-detect downbeats); until then it looks for an optional boundaries file per track and
skips tracks without one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mss_eval import report, sweep  # noqa: E402
from mss_eval.io import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_phrase_boundaries(track_id: str) -> list[float] | None:
    """Phrase boundaries (seconds), one per line, from data/phrases/<track>.txt if present.

    This is the pluggable hook: whatever we decide tomorrow (manual annotation export or
    an auto-detector) just has to drop times into this file. The sweep core is agnostic.
    """
    f = ROOT / "data" / "phrases" / f"{track_id}.txt"
    if not f.exists():
        return None
    return [float(x) for x in f.read_text().split() if x.strip()]


def _safe_name(s: str) -> str:
    """Filesystem-safe folder name for a track (must match run_benchmark's cache paths)."""
    return "".join("_" if c in '<>:"/\\|?*' else c for c in s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["htdemucs"])
    ap.add_argument("--mode", choices=["length", "phrase"], default="length")
    ap.add_argument("--sample-n", type=int, default=2)
    args = ap.parse_args()

    models_cfg = {n: {**c, "name": n} for n, c in load_config("models")["models"].items()}
    data_cfg = load_config("datasets")
    exp = data_cfg["input_length_experiment"]
    sr = load_config("models").get("defaults", {}).get("sample_rate", 44100)

    # benchmark source: MUSDB tracks give us reference stems for R
    import musdb
    root = ROOT / data_cfg["benchmark"]["root"]
    db = musdb.DB(root=str(root), subsets=data_cfg["benchmark"]["subset"], is_wav=True)
    tracks = list(db)[: args.sample_n]

    rows = []
    for t in tracks:
        track_len = t.audio.shape[0] / sr
        region = tuple(exp.get("region_default", [30.0, 35.0]))
        if region[1] > track_len:                       # keep R inside short tracks
            region = (max(0.0, track_len - 6), max(2.0, track_len - 1))

        if args.mode == "phrase":
            bounds = load_phrase_boundaries(t.name)
            if not bounds:
                print(f"skip {t.name}: no phrase boundaries (data/phrases/{t.name}.txt)")
                continue
            spec = {"mode": "phrase", "levels": exp["phrase_levels"], "boundaries": bounds}
        else:
            spec = {"mode": "length", "levels": exp["length_levels"]}

        windows = sweep.build_windows(region, track_len, spec)
        refs = {n: t.targets[n].audio for n in data_cfg["benchmark"]["reference_stems"]
                if n in t.targets}
        for model_name in args.models:
            cdir = ROOT / "data" / "stems_out" / model_name / _safe_name(t.name)
            r = sweep.sweep_region(t.audio, sr, models_cfg[model_name], region, windows, refs,
                                   cache_dir=cdir)
            for row in r:
                row["track"] = t.name
            rows.extend(r)

    if not rows:
        print("No results — need MUSDB18 present (length mode) or phrase files (phrase mode).")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    report.write_table(df.round(3), f"input_length_{args.mode}",
                       f"A2 input-length sweep ({args.mode} mode)")
    print(f"Done: {len(rows)} rows. See results/tables/input_length_{args.mode}.md")


if __name__ == "__main__":
    main()
