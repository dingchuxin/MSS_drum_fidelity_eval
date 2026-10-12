"""B3 shape measures with the v3 definitions (Phase 3, C0.6), for every stored run and for the
full-track outputs cut to the same spans.

Onsets and matching are reused from the Phase 2 B3 caches (data/phase2/b3_measure; same
detector and rule): reference onsets are re-detected on each span only to recover the span's
full onset list (the segment ends); they are checked against the cached R onsets.
Per run: analyst.dynamics_v3 on R (reference gate), strokes_v3 on the reference-onset segments
of R (all reference strokes in R; matched flags from the run), accent contrast.

    python b3_measure3.py [--jobs 2]
Output: data/phase3/v3/b3/<model>/<track>.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import _p3 as P  # noqa: E402
from b3_measure import full_stem  # noqa: E402
from mss_eval import analyst, strokes as SK, timing as T  # noqa: E402

B3 = P.P2DATA / "b3"
P2B3 = P.P2DATA / "b3_measure"
OUT = P.DATA3 / "b3"
MODELS = ["spleeter", "htdemucs", "bs_roformer", "scnet"]


def shape(est_span, ref_span, a, R0, R1, ron_all, inR, matched):
    """v3 shape of one output span (track samples [a, a + len)) on passage R."""
    r0, r1 = R0 - a, R1 - a
    desc = analyst.dynamics_v3(est_span[r0:r1], ref_span[r0:r1], P.SR)
    rs = SK.strokes_v3(ref_span, P.SR, ron_all - a / P.SR)
    es = SK.strokes_v3(est_span, P.SR, ron_all - a / P.SR)
    rs, es, _ = SK.pair_v3(rs, es)
    out = {f"d_{k}": desc[f"d_{k}"] for k in analyst.V3_DESCS}
    out.update({f"ref_{k}": desc[f"ref_{k}"] for k in analyst.V3_DESCS})
    for set_name, sel in (("all", inR), ("matched", inR & matched)):
        for m in P.SMEAS3:
            d = es[m][sel] - rs[m][sel]
            out[f"d_stroke_{m}_{set_name}"] = np.nanmedian(d) if np.isfinite(d).any() else np.nan
            out[f"ref_stroke_{m}_{set_name}"] = np.nanmedian(rs[m][sel]) if np.isfinite(rs[m][sel]).any() else np.nan
        pr, pe = rs["peak_db"][sel], es["peak_db"][sel]
        ok = np.isfinite(pr) & np.isfinite(pe)
        out[f"d_accent_{set_name}"] = np.std(pe[ok], ddof=1) - np.std(pr[ok], ddof=1) if ok.sum() > 2 else np.nan
        out[f"ref_accent_{set_name}"] = np.std(pr[ok], ddof=1) if ok.sum() > 2 else np.nan
    return out


def one(args):
    model, track = args
    f = OUT / model / f"{P.safe_name(track)}.pkl"
    if f.exists():
        return track
    p2 = pickle.load(open(P2B3 / model / f"{P.safe_name(track)}.pkl", "rb"))
    R0, R1 = p2["R"]
    ref = P.ref_stem(track)
    full = full_stem(model, track)
    spans = {}
    res = {"model": model, "track": track, "R": (R0, R1), "runs": {}, "definition_tag": P.TAG}

    def span_info(a, b):
        if (a, b) not in spans:
            seg = ref[a:b]
            on = T.detect(seg, P.SR) + a / P.SR
            inR = (on >= R0 / P.SR) & (on < R1 / P.SR)
            cached = p2["ref"][f"{a}_{b}"]["onsets_R"]
            assert np.allclose(on[inR], cached, atol=1e-6), (track, a, b)
            spans[(a, b)] = (on, inR)
        return spans[(a, b)]

    for cond, run in p2["runs"].items():
        a, b = run["span"]
        on, inR = span_info(a, b)
        matched = np.zeros(len(on), bool)
        matched[np.where(inR)[0]] = np.isfinite(run["est_time_per_ref_onset"])
        x, _ = sf.read(str(B3 / model / P.safe_name(track) / f"{cond}.wav"), dtype="float64", always_2d=True)
        res["runs"][cond] = {"tags": run["meta"]["tags"], "span": (a, b),
                             **shape(x, ref[a:b], a, R0, R1, on, inR, matched)}
    for name in ("full_pm1", "full_R"):
        a, b = p2[name]["span"]
        on, inR = span_info(a, b)
        matched = np.zeros(len(on), bool)
        matched[np.where(inR)[0]] = np.isfinite(p2[name]["est_time_per_ref_onset"])
        res[name] = {"span": (a, b), **shape(full[a:b], ref[a:b], a, R0, R1, on, inR, matched)}
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    return track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    import pandas as pd
    man = pd.read_csv(P.OUT2 / "b3_manifest.csv")
    jobs = [(m, t) for t in man.track for m in MODELS]
    from multiprocessing import Pool
    with Pool(a.jobs, maxtasksperchild=10) as p:
        for i, t in enumerate(p.imap_unordered(one, jobs), 1):
            if i % 20 == 0:
                print(f"[{i}/{len(jobs)}] {t}", flush=True)


if __name__ == "__main__":
    main()
