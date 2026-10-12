"""Per-track measurement of a drum-stem set against the true drum stems (50 tracks).

    python measure.py <label> [--bss] [--bandlimit] [--jobs 2]

Labels (stem roots):
  v1_<model>           cached v1 stems (16-bit), <model> in spleeter|htdemucs|bs_roformer|scnet
  <name>               data/phase2/stems/<name>/<track>/{drums,bass,other,vocals}.wav (float)

For each track: refined onsets (timing.detect: the Phase 1 detector plus a 30 ms
post-refinement minimum separation) of the estimate and the reference, closest-first
matching at +/-50 ms, F/P/R at the J1 tolerances, the analyst row (J2 dyn_/d_), A3 stroke
measures for every matched stroke (estimate at its own onset, segment ended at the next
reference onset), SI-SDR, and optionally drums BSS-Eval v4 (--bss; needs all four stems).

--bandlimit applies Spleeter's band limit (STFT n_fft 4096 / hop 1024, Hann; bins >= 1024,
i.e. >= 11.025 kHz, zeroed; inverse STFT) to the reference AND the estimate, and recomputes
the reference onsets and strokes on the band-limited reference (A5).

Outputs: data/phase2/measure/<label>[_bl]/<track>.pkl and
         results/phase2/measure/<label>[_bl]_by_track.csv
"""
from __future__ import annotations

import argparse
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _p2 as P  # noqa: E402
from mss_eval import analyst, metrics, strokes as SK, timing as T  # noqa: E402

J1 = P.ROOT / "data" / "phase1_cache" / "j1"
MEAS = P.P2DATA / "measure"
TOLS = [5, 10, 20, 30, 50]
THR = [5, 8, 10, 14, 20]
SMEAS = ["peak_db", "floor_db", "valley_db", "rise_ms", "decay_ms"]


def stem_dir(label: str, track: str):
    if label.startswith("v1_"):
        return P.STEMS / label[3:] / P.safe_name(track)
    return P.P2DATA / "stems" / label / P.safe_name(track)


def bandlimit(x: np.ndarray) -> np.ndarray:
    from scipy.signal import istft, stft
    x2 = x if x.ndim == 2 else x[:, None]
    out = np.empty_like(x2, dtype=np.float64)
    for c in range(x2.shape[1]):
        _, _, Z = stft(x2[:, c], nperseg=4096, noverlap=4096 - 1024, window="hann",
                       boundary="zeros", padded=True)
        Z[1024:, :] = 0
        _, y = istft(Z, nperseg=4096, noverlap=4096 - 1024, window="hann", boundary=True)
        out[:, c] = y[: len(x2)]
    return out if x.ndim == 2 else out[:, 0]


def ref_info(track: str, bl: bool) -> dict:
    f = MEAS / ("_ref_bl" if bl else "_ref") / f"{P.safe_name(track)}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    ref = P.ref_stem(track)
    if bl:
        ref = bandlimit(ref)
    on = T.detect(ref, P.SR)
    st = SK.strokes(ref, P.SR, on)
    info = {"onsets": on, "strokes": pd.DataFrame(st),
            "level_db": analyst.dynamic_profile(ref, P.SR)["level_db"]}
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(info, open(f, "wb"))
    return info


def one(args):
    label, track, do_bss, bl = args
    tag = label + ("_bl" if bl else "")
    f = MEAS / tag / f"{P.safe_name(track)}.pkl"
    if f.exists():
        return track
    d = stem_dir(label, track)
    est = P.load(d / "drums.wav")
    ref = P.ref_stem(track)
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    if bl:
        est, ref = bandlimit(est), bandlimit(ref)
    ri = ref_info(track, bl)
    ron = ri["onsets"]
    eon = T.detect(est, P.SR)
    ir, ie = T.match_pairs(ron, eon, 0.05)
    import mir_eval
    prf = {t: mir_eval.onset.f_measure(ron, eon, window=t / 1000) for t in TOLS}
    nxt = np.append(ron[1:], np.nan)
    est_st = pd.DataFrame(SK.strokes(est, P.SR, eon[ie], nxt[ir]))
    res = {"track": track, "label": tag, "ref_onsets": ron, "est_onsets": eon, "i_ref": ir,
           "i_est": ie, "prf": prf, "est_strokes": est_st,
           "analyst": analyst.analyst_row(est, ref, P.SR), "si_sdr": metrics.si_sdr(est, ref)}
    if do_bss:
        import sys
        sys.path.insert(0, str(P.ROOT / "scripts" / "phase1"))
        from j3_bss_pertrack import drums_bss
        refs = {s: P.ref_stem(track, s) for s in P.SOURCES}
        ests = {s: P.load(d / f"{s}.wav") for s in P.SOURCES}
        res["bss"] = drums_bss(refs, ests)
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    return track


def by_track(tag: str, bl: bool) -> pd.DataFrame:
    rows = []
    for t in P.tracks():
        r = pickle.load(open(MEAS / tag / f"{P.safe_name(t)}.pkl", "rb"))
        ri = ref_info(t, bl)
        dev = (r["est_onsets"][r["i_est"]] - r["ref_onsets"][r["i_ref"]]) * 1000
        row = {"label": tag, "track": t, "n_ref": len(r["ref_onsets"]), "n_est": len(r["est_onsets"]),
               "n_pairs": len(dev), "si_sdr": r["si_sdr"]}
        row.update({f"f_{k}ms": v[0] for k, v in r["prf"].items()})
        a = np.abs(dev)
        row.update({"median_dev_ms": np.median(dev), "iqr_dev_ms": np.subtract(*np.percentile(dev, [75, 25])),
                    "median_abs_dev_ms": np.median(a)})
        row.update({f"share_gt_{k}ms": np.mean(a > k) for k in THR})
        row.update({k: v for k, v in r["analyst"].items() if k.startswith(("d_", "dyn_"))
                    and "burst_rate" not in k})
        rs = ri["strokes"].iloc[r["i_ref"]].reset_index(drop=True)
        es = r["est_strokes"]
        for m in SMEAS:
            row[f"ref_{m}"] = np.nanmedian(ri["strokes"][m])          # all reference strokes
            row[f"d_stroke_{m}"] = np.nanmedian(es[m].to_numpy() - rs[m].to_numpy())
        row["ref_accent_db"] = np.nanstd(ri["strokes"]["peak_db"], ddof=1)
        row["d_accent_db"] = np.nanstd(es["peak_db"], ddof=1) - np.nanstd(rs["peak_db"], ddof=1)
        row["share_decay_capped_est"] = es["decay_capped"].mean()
        if "bss" in r:
            row.update({k: r["bss"][k] for k in ["sdr", "sir", "sar", "isr"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    (P.OUT2 / "measure").mkdir(parents=True, exist_ok=True)
    df.to_csv(P.OUT2 / "measure" / f"{tag}_by_track.csv", index=False, float_format="%.6g")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--bss", action="store_true")
    ap.add_argument("--bandlimit", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    from multiprocessing import Pool
    jobs = [(a.label, t, a.bss, a.bandlimit) for t in P.tracks()]
    with Pool(a.jobs, maxtasksperchild=5) as p:
        for i, t in enumerate(p.imap_unordered(one, jobs), 1):
            print(f"[{i}/50] {a.label} {t}", flush=True)
    by_track(a.label + ("_bl" if a.bandlimit else ""), a.bandlimit)


if __name__ == "__main__":
    main()
