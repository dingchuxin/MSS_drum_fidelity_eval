"""Per-track benchmark table of the default runs (version 1 analyst row), from the cached stems.

For every cached default-run drum stem (data/stems_out/<model>/<track>/drums.wav) against the
true drum stem: SI-SDR and the version 1 analyst row (envelope descriptors and the onset
F-measure with librosa's default detection). BSS Eval SDR/SIR/SAR per track are computed by
scripts/phase1/j3_bss_pertrack.py.

    python scripts/synthesize.py        (skips the build if the table exists)
Output: results/tables/benchmark_raw.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mss_eval import analyst, metrics                       # noqa: E402
from mss_eval.io import load_audio, load_config            # noqa: E402

FOCUS = ["drums"]                                          # rhythm study
RAW = ROOT / "results" / "tables" / "benchmark_raw.csv"


def _safe_name(s: str) -> str:
    return "".join("_" if c in '<>:"/\\|?*' else c for c in s).strip()


def build_raw() -> pd.DataFrame:
    import musdb
    cfg = load_config("datasets")
    sr = load_config("models").get("defaults", {}).get("sample_rate", 44100)
    root = ROOT / cfg["benchmark"]["root"]
    db = musdb.DB(root=str(root), subsets=cfg["benchmark"]["subset"], is_wav=True)
    stems_root = ROOT / "data" / "stems_out"
    models = sorted(d.name for d in stems_root.iterdir() if d.is_dir())
    print(f"models: {models} | focus stems: {FOCUS}", flush=True)
    rows = []
    for ti, t in enumerate(db, 1):
        refs = {s: t.targets[s].audio for s in FOCUS if s in t.targets}
        for model in models:
            cdir = stems_root / model / _safe_name(t.name)
            for s in FOCUS:
                p = cdir / f"{s}.wav"
                if not p.exists() or s not in refs:
                    continue
                est = load_audio(p, target_sr=sr)[0]
                row = {"model": model, "track": t.name, "stem": s,
                       "si_sdr": metrics.si_sdr(est, refs[s])}
                row.update(analyst.analyst_row(est, refs[s], sr))
                rows.append(row)
        print(f"  [{ti:2d}/50] {t.name[:44]}", flush=True)
    df = pd.DataFrame(rows)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW, index=False)
    print(f"raw table: {len(df)} rows -> {RAW}", flush=True)
    return df


def main():
    if RAW.exists():
        print(f"{RAW} exists; delete it to rebuild it from the cached stems")
        return
    build_raw()


if __name__ == "__main__":
    main()
