"""B3 measurement: every stored R-span drum output vs the true drum stem over the same span.

For each run (data/phase2/b3/<model>/<track>/<cond>.wav, span from its JSON sidecar) and for
the full-track outputs cut to the same spans ('full_pm1' = R +/- 1 s, 'full_R' = R only;
v1 cached stems for Spleeter/BS-Roformer/SCNet-XL, B1 for HT-Demucs):
  - refined onsets on the stored span (the +/-1 s is detector context); reference onsets
    detected on the true stem over the same span; reference onsets kept only inside R,
    estimate onsets inside R +/- 50 ms; closest-first matching at +/-50 ms;
  - J2 descriptors (analyst.dynamic_profile) of the estimate and the reference on R;
  - A3 stroke measures for matched strokes in R (segment ends at the next reference onset).

    python b3_measure.py [--tracks i,j] [--jobs 3]
Output: data/phase2/b3_measure/<model>/<track>.pkl (all conditions of that model and track)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

import _p2 as P  # noqa: E402
from mss_eval import analyst, strokes as SK, timing as T  # noqa: E402

B3 = P.P2DATA / "b3"
OUTM = P.P2DATA / "b3_measure"
FULL = {"spleeter": ("v1", "spleeter"), "bs_roformer": ("v1", "bs_roformer"),
        "scnet": ("v1", "scnet"), "htdemucs": ("p2", "htdemucs_b1")}


def full_stem(model, track):
    kind, name = FULL[model]
    if kind == "v1":
        return P.est_stem(name, track)
    x, _ = sf.read(str(P.P2DATA / "stems" / name / P.safe_name(track) / "drums.wav"),
                   dtype="float64", always_2d=True)
    return x


def ref_span(ref, a, b, R0, R1, cache):
    key = (a, b)
    if key not in cache:
        seg = ref[a:b]
        on = T.detect(seg, P.SR) + a / P.SR
        inR = on[(on >= R0 / P.SR) & (on < R1 / P.SR)]
        st = pd.DataFrame(SK.strokes(seg, P.SR, inR - a / P.SR, _next(on, inR) - a / P.SR))
        prof = analyst.dynamic_profile(ref[R0:R1], P.SR)
        cache[key] = {"onsets_all": on, "onsets_R": inR, "strokes": st, "profile": prof}
    return cache[key]


def _next(on_all, on_sel):
    idx = np.searchsorted(on_all, on_sel, side="right")
    return np.array([on_all[i] if i < len(on_all) else np.nan for i in idx])


def measure(est_span, a, b, R0, R1, rinfo):
    on = T.detect(est_span, P.SR) + a / P.SR
    on = on[(on >= R0 / P.SR - 0.05) & (on < R1 / P.SR + 0.05)]
    ron = rinfo["onsets_R"]
    ir, ie = T.match_pairs(ron, on, 0.05)
    est_t = np.full(len(ron), np.nan)
    est_t[ir] = on[ie]
    nxt = _next(rinfo["onsets_all"], ron)
    st = pd.DataFrame(SK.strokes(est_span, P.SR, on[ie] - a / P.SR, nxt[ir] - a / P.SR))
    rs = rinfo["strokes"].iloc[ir].reset_index(drop=True)
    prof = analyst.dynamic_profile(est_span[R0 - a:R1 - a], P.SR)
    out = {"est_time_per_ref_onset": est_t, "n_est_in_R": len(on),
           "profile": prof, "d_profile": {k: prof[k] - rinfo["profile"][k] for k in prof}}
    for m in SK_MEAS:
        out[f"d_stroke_{m}"] = np.nanmedian(st[m].to_numpy() - rs[m].to_numpy()) if len(st) else np.nan
    out["d_accent_db"] = (np.nanstd(st["peak_db"], ddof=1) - np.nanstd(rs["peak_db"], ddof=1)) if len(st) > 2 else np.nan
    return out


SK_MEAS = ["peak_db", "floor_db", "valley_db", "rise_ms", "decay_ms"]


def one(args):
    model, track = args
    f = OUTM / model / f"{P.safe_name(track)}.pkl"
    d = B3 / model / P.safe_name(track)
    runs = sorted(d.glob("*.json"))
    if f.exists():
        old = pickle.load(open(f, "rb"))
        if len(old["runs"]) == len(runs):
            return track
    ref = P.ref_stem(track)
    full = full_stem(model, track)
    cache, res = {}, {}
    R0 = R1 = None
    for jf in runs:
        meta = json.load(open(jf))
        a, b = meta["kept_span_track_samples"]
        R0, R1 = meta["tags"]["R"]
        x, _ = sf.read(str(jf.with_suffix(".wav")), dtype="float64", always_2d=True)
        ri = ref_span(ref, a, b, R0, R1, cache)
        res[meta["tags"]["cond"]] = {"meta": {k: meta[k] for k in ("tags", "sep_seconds", "win", "seed")
                                              if k in meta} | {"setting": meta.get("setting"),
                                                               "num_overlap": meta.get("num_overlap")},
                                     "span": (a, b), **measure(x, a, b, R0, R1, ri)}
    out = {"model": model, "track": track, "runs": res, "R": (R0, R1)}
    if R0 is not None:
        for name, (a, b) in {"full_pm1": (R0 - P.SR, R1 + P.SR), "full_R": (R0, R1)}.items():
            ri = ref_span(ref, a, b, R0, R1, cache)
            out[name] = {"span": (a, b), **measure(full[a:b], a, b, R0, R1, ri)}
        out["ref"] = {f"{a}_{b}": {"onsets_R": v["onsets_R"], "profile": v["profile"],
                                   "strokes_median": v["strokes"].median(numeric_only=True).to_dict(),
                                   "accent_db": float(np.nanstd(v["strokes"]["peak_db"], ddof=1))}
                      for (a, b), v in cache.items()}
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    man = pd.read_csv(P.OUT2 / "b3_manifest.csv")
    sel = man.track.tolist() if not a.tracks else man[man.track_index.isin([int(x) for x in a.tracks.split(",")])].track.tolist()
    jobs = [(m, t) for t in sel for m in ["spleeter", "htdemucs", "bs_roformer", "scnet"]]
    from multiprocessing import Pool
    with Pool(a.jobs) as p:
        for i, t in enumerate(p.imap_unordered(one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {t}", flush=True)


if __name__ == "__main__":
    main()
