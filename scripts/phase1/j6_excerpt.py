"""J6: export one 3 s drum excerpt for a figure (reference + four estimates).

Track choice: the per-track contrast is delta = |d_attack_slope|(BS-Roformer) -
|d_attack_slope|(SCNet-XL) from results/tables/benchmark_raw.csv. Its median sign is the
"typical direction". Candidate tracks are those with delta of that sign, ordered by closeness
to the median delta (so the example is typical, not extreme).

Passage choice within a track (3.0 s windows, 0.25 s steps, refined reference onsets):
  - moderate density: 2-5 onsets/s;  clearly separated: min inter-onset interval >= 90 ms;
  - drums prominent: reference drum RMS in the window >= the track's median window RMS and
    drum/rest-of-mix energy ratio >= its track median;
  - the local attack-slope contrast has the typical sign.
Among qualifying windows the one whose local delta is closest to the track-level delta wins;
the first candidate track with a qualifying window is used.

Output: results/phase1/figure_excerpt/manifest.json. The excerpt's audio is written by
scripts/make_excerpt.py, which reads this manifest.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _common as C
from mss_eval import analyst
from mss_eval import timing as T

WIN_S, STEP_S = 3.0, 0.25
OUT = C.OUT / "figure_excerpt"


def slope(x):
    return analyst.dynamic_profile(x, C.SR)["attack_slope"]


def choose_window(track: str, sign: float, delta_track: float):
    ref = C.ref_stem(track)
    rest = sum(C.ref_stem(track, s) for s in C.SOURCES if s != "drums")
    est = {m: C.est_stem(m, track) for m in C.MODELS}
    on = T.detect_refined(ref, C.SR)
    n = len(ref)
    w, st = int(WIN_S * C.SR), int(STEP_S * C.SR)
    starts = np.arange(0, n - w, st)
    rms = np.array([np.sqrt(np.mean(C.mono(ref[s:s + w]) ** 2)) for s in starts])
    ratio = np.array([np.sum(ref[s:s + w] ** 2) / (np.sum(rest[s:s + w] ** 2) + 1e-12) for s in starts])
    cands = []
    for i, s in enumerate(starts):
        t0, t1 = s / C.SR, (s + w) / C.SR
        o = on[(on >= t0 + 0.05) & (on < t1 - 0.05)]
        if not (2 * WIN_S <= len(o) <= 5 * WIN_S):
            continue
        if np.min(np.diff(o)) < 0.090:
            continue
        if rms[i] < np.median(rms) or ratio[i] < np.median(ratio):
            continue
        seg = slice(s, s + w)
        sr_ = slope(ref[seg])
        loc = abs(slope(est["bs_roformer"][seg]) - sr_) - abs(slope(est["scnet"][seg]) - sr_)
        if np.sign(loc) != sign:
            continue
        cands.append((abs(loc - delta_track), s, loc, len(o)))
    if not cands:
        return None
    cands.sort()
    _, s, loc, k = cands[0]
    return s, s + w, loc, k, ref, rest, est


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(C.ROOT / "results" / "tables" / "benchmark_raw.csv")
    piv = raw.pivot(index="track", columns="model", values="d_attack_slope").abs()
    delta = piv["bs_roformer"] - piv["scnet"]
    med = delta.median()
    sign = np.sign(med)
    cand = delta[np.sign(delta) == sign]
    cand = cand.iloc[np.argsort(np.abs(cand - med).to_numpy())]
    print(f"median delta |d attack| (BS - SCNet) = {med:.3f} dB/hop; "
          f"{(np.sign(delta) == sign).sum()}/50 tracks share its sign")
    for track, dt in cand.items():
        got = choose_window(track, sign, dt)
        print(f"  {track}: delta {dt:+.3f} (pctile {100 * (delta < dt).mean():.0f}) -> "
              f"{'window found' if got else 'no qualifying window'}", flush=True)
        if got:
            break
    s0, s1, loc, k, ref, rest, est = got
    t0, t1 = s0 / C.SR, s1 / C.SR
    seg = slice(s0, s1)

    sig = {"reference": ref[seg], **{m: est[m][seg] for m in C.MODELS}}
    ons = {}
    for name, x in sig.items():
        o = T.detect_refined(x, C.SR)                   # on the excerpt itself
        ons[name] = o[(o >= 0.02) & (o <= WIN_S - 0.03)]   # drop excerpt-boundary artefacts
    sl = {k: slope(v) for k, v in sig.items()}
    man = {"track": track, "t0_s": round(t0, 3), "t1_s": round(t1, 3),
           "track_delta_abs_d_attack_bs_minus_scnet": round(float(dt), 4),
           "track_delta_percentile": round(float(100 * (delta < dt).mean()), 1),
           "median_delta_all_tracks": round(float(med), 4),
           "excerpt_delta_abs_d_attack_bs_minus_scnet": round(float(loc), 4),
           "excerpt_attack_slope_db_per_hop": {k: round(float(v), 3) for k, v in sl.items()},
           "n_reference_onsets": int(len(ons["reference"])),
           "onset_detector": "refined (hop 64, parabolic peak), run on the excerpt",
           "envelope": "analyst._lin_env: 20 ms RMS, 10 ms hop, mono mean"}
    json.dump(man, open(OUT / "manifest.json", "w"), indent=2)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
