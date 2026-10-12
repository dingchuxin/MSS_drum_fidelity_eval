"""B3 ("how the file was cut"): choose passage R per track (before any separation), then
write the job files for every condition.

    python b3_setup.py manifest            -> results/phase2/b3_manifest.csv
    python b3_setup.py jobs [--tracks i,j] -> data/phase2/jobs/b3_{msst,htdemucs,spleeter}.jsonl

R (5 s) per track: candidate starts on a 0.25 s grid with start >= 24 s and end <= length - 30 s;
>= 8 reference drum onsets inside R (timing.detect on the true stem); drum RMS in R above the track's median drum RMS
(median over all 5 s windows on the same grid); most onsets wins, ties -> earliest.

Conditions (window = [w0, w1) in track samples; stored span = R +/- 1 s, drums only):
  F1  o_k = k * 11.9/12 s, k = 0..11: w0 = R0 - 12 s - o_k, length 42 s
  F2  at o = 0: w0 shifted later by round(h/3), round(2h/3) samples, h = the model's STFT hop
      (Spleeter 1024, HT-Demucs 1024, BS-Roformer 441, SCNet-XL 1024)
  F3  R alone (5 s; stored span = R) and 10 s / 20 s windows with R centred
  rep o = 0 run a second time
  HT-Demucs: B1 settings everywhere (seed = 20260925 + track index for every B1 run, so the
      repeat must be bit-identical); plus o = 0 twice with audio-separator defaults
  swap BS-Roformer at num_overlap 4, SCNet-XL at num_overlap 2, all 12 F1 positions
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd

import _p2 as P
import measure as M

SR = P.SR
HOPS = {"spleeter": 1024, "htdemucs": 1024, "bs_roformer": 441, "scnet": 1024}
OFFS = [round(k * 11.9 / 12 * SR) for k in range(12)]
B3 = P.P2DATA / "b3"


def manifest():
    rows = []
    for ti, t in enumerate(P.tracks()):
        ref = P.mono(P.ref_stem(t))
        on = M.ref_info(t, False)["onsets"]           # Phase 2 detector (timing.detect)
        n = len(ref)
        grid = np.arange(0, n - 5 * SR + 1, SR // 4)
        rms = np.array([np.sqrt(np.mean(ref[s:s + 5 * SR] ** 2)) for s in grid])
        med = np.median(rms)
        best = None
        for s, r in zip(grid, rms):
            if s < 24 * SR or s + 5 * SR > n - 30 * SR or r <= med:
                continue
            k = int(((on >= s / SR) & (on < (s + 5 * SR) / SR)).sum())
            if k >= 8 and (best is None or k > best[1]):
                best = (int(s), k, r)
        if best is None:
            rows.append({"track_index": ti, "track": t, "R0": np.nan, "R1": np.nan, "n_onsets": 0,
                         "qualifies": False, "track_len_s": n / SR})
            continue
        s, k, r = best
        rows.append({"track_index": ti, "track": t, "R0": s, "R1": s + 5 * SR, "R0_s": s / SR,
                     "n_onsets": k, "rms_R": r, "median_rms_track": med, "qualifies": True,
                     "track_len_s": n / SR})
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT2 / "b3_manifest.csv", index=False, float_format="%.6g")
    print(df.to_string())
    print("qualifying:", int(df.qualifies.sum()))


def conditions(model: str, R0: int, R1: int):
    """Yield (cond_id, win, keep, extra) for one model and one passage."""
    keep = [R0 - SR, R1 + SR]
    for k, o in enumerate(OFFS):
        w0 = R0 - 12 * SR - o
        yield f"f1_o{k:02d}", [w0, w0 + 42 * SR], keep, {"factor": "F1", "o_index": k, "o_samples": o}
    h = HOPS[model]
    for q in (1, 2):
        sh = round(h * q / 3)
        w0 = R0 - 12 * SR + sh
        yield f"f2_s{q}", [w0, w0 + 42 * SR], keep, {"factor": "F2", "subframe_shift_samples": sh,
                                                      "hop": h}
    yield "f3_L05", [R0, R1], [R0, R1], {"factor": "F3", "length_s": 5}
    for L in (10, 20):
        w0 = R0 - (L - 5) * SR // 2
        yield f"f3_L{L:02d}", [w0, w0 + L * SR], keep, {"factor": "F3", "length_s": L}
    w0 = R0 - 12 * SR
    yield "rep_o00", [w0, w0 + 42 * SR], keep, {"factor": "repeat", "o_index": 0, "o_samples": 0}


def jobs(sel: list[int] | None):
    man = pd.read_csv(P.OUT2 / "b3_manifest.csv")
    man = man[man.qualifies]
    if sel is not None:
        man = man[man.track_index.isin(sel)]
    msst, htd, spl = [], [], []
    for _, r in man.iterrows():
        t, ti, R0, R1 = r.track, int(r.track_index), int(r.R0), int(r.R1)
        for m in ["spleeter", "htdemucs", "bs_roformer", "scnet"]:
            for cid, win, keep, ex in conditions(m, R0, R1):
                j = {"id": f"{m}|{t}|{cid}", "model": m, "track": t, "win": win, "keep": keep,
                     "seed": P.SEED + ti, "tags": {**ex, "cond": cid, "R": [R0, R1]},
                     "out": str(B3 / m / P.safe_name(t) / cid)}
                if m == "htdemucs":
                    htd.append({**j, "setting": "b1"})
                elif m == "spleeter":
                    spl.append(j)
                else:
                    msst.append(j)
            if m == "htdemucs":
                w0 = R0 - 12 * SR
                for rep in (1, 2):
                    htd.append({"id": f"htdemucs|{t}|default_o00_r{rep}", "model": "htdemucs", "track": t,
                                "win": [w0, w0 + 42 * SR], "keep": [R0 - SR, R1 + SR], "seed": None,
                                "setting": "default",
                                "tags": {"factor": "htd_default", "cond": f"default_o00_r{rep}", "R": [R0, R1]},
                                "out": str(B3 / m / P.safe_name(t) / f"default_o00_r{rep}")})
            if m in ("bs_roformer", "scnet"):
                ov = 4 if m == "bs_roformer" else 2
                for k, o in enumerate(OFFS):
                    w0 = R0 - 12 * SR - o
                    msst.append({"id": f"{m}|{t}|swap_o{k:02d}", "model": m, "track": t,
                                 "win": [w0, w0 + 42 * SR], "keep": [R0 - SR, R1 + SR],
                                 "overlap": ov, "seed": P.SEED + ti,
                                 "tags": {"factor": "overlap_swap", "cond": f"swap_o{k:02d}", "o_index": k,
                                          "o_samples": o, "R": [R0, R1], "num_overlap": ov},
                                 "out": str(B3 / m / P.safe_name(t) / f"swap_o{k:02d}")})
    J = P.P2DATA / "jobs"
    J.mkdir(parents=True, exist_ok=True)
    for name, js in [("msst", msst), ("htdemucs", htd), ("spleeter", spl)]:
        with open(J / f"b3_{name}.jsonl", "w", encoding="utf-8") as f:
            for j in js:
                f.write(json.dumps(j) + "\n")
        print(f"b3_{name}: {len(js)} jobs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--tracks", default=None)
    a = ap.parse_args()
    if a.what == "manifest":
        manifest()
    else:
        jobs([int(x) for x in a.tracks.split(",")] if a.tracks else None)
