"""J3 input: per-track BSS-Eval v4 (SDR/ISR/SIR/SAR) for the DRUMS stem, from cached stems.

The v1 benchmark kept only per-model medians (results/tables/benchmark_metrics.csv), so the
paired SIR/SAR tests need the per-track values. This reproduces metrics.bss_eval exactly
(museval v4, 1 s window/hop, 512-tap filters, all four references, no permutation) but only
decomposes the drums estimate: the projection filters are global, so restricting the loop
over `jtrue` to drums changes nothing for drums (checked against the full call with --check).

    python scripts/phase1/j3_bss_pertrack.py [--jobs 6] [--check TRACK]
Output: results/phase1/bss_drums_pertrack.csv
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _common as C  # noqa: E402


def _stack(d):
    n = min(v.shape[0] for v in d.values())
    return np.stack([d[s][:n] for s in C.SOURCES]), n


def drums_bss(refs: dict, ests: dict, sr: int = C.SR) -> dict:
    from museval import metrics as M
    R, n1 = _stack(refs)
    E, n2 = _stack(ests)
    n = min(n1, n2)
    R, E = R[:, :n], E[:, :n]
    j = C.SOURCES.index("drums")
    flen = 512
    G, sf = M._compute_reference_correlations(R, flen)
    Cd = M._compute_projection_filters(G, sf, E[j])
    Cj = M._compute_projection_filters(G[j, j], sf[j], E[j])       # (nchan, flen, nchan)
    framer = M.Framing(sr, sr, n)
    vals = []
    for win in framer:
        if M._any_source_silent(R[:, win]) or M._any_source_silent(E[:, win]):
            vals.append([np.nan] * 4)
            continue
        s_true, e_spat, e_interf, e_artif = M._bss_decomp_mtifilt(R[:, win], E[j, win], j, Cd, Cj)
        vals.append(M._bss_crit(s_true, e_spat, e_interf, e_artif, False))
    v = np.array(vals, dtype=float)       # (nwin, 4) in order sdr, isr, sir, sar
    return {"sdr": float(np.nanmedian(v[:, 0])), "isr": float(np.nanmedian(v[:, 1])),
            "sir": float(np.nanmedian(v[:, 2])), "sar": float(np.nanmedian(v[:, 3])),
            "n_windows": int(len(v)), "n_valid": int(np.isfinite(v[:, 0]).sum())}


def one(args):
    model, track = args
    refs = {s: C.ref_stem(track, s) for s in C.SOURCES}
    ests = {s: C.est_stem(model, track, s) for s in C.SOURCES}
    return {"model": model, "track": track, **drums_bss(refs, ests)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--check", default=None, help="compare with full museval on one track")
    a = ap.parse_args()
    if a.check:
        import time
        from mss_eval import metrics
        refs = {s: C.ref_stem(a.check, s) for s in C.SOURCES}
        ests = {s: C.est_stem("scnet", a.check, s) for s in C.SOURCES}
        t = time.time(); fast = drums_bss(refs, ests); t1 = time.time() - t
        t = time.time(); full = metrics.bss_eval(ests, refs, C.SR)["drums"]; t2 = time.time() - t
        print("drums-only:", fast, f"{t1:.0f}s"); print("full museval:", full, f"{t2:.0f}s")
        return
    from multiprocessing import Pool
    part = C.OUT / "bss_drums_pertrack.partial.csv"        # resumable: museval is slow
    done = pd.read_csv(part) if part.exists() else pd.DataFrame(columns=["model", "track"])
    have = set(zip(done.model, done.track))
    jobs = [(m, t) for t in C.tracks() for m in C.MODELS if (m, t) not in have]
    rows = done.to_dict("records")
    # long tracks need several GB each in museval's FFTs: keep --jobs modest
    with Pool(a.jobs, maxtasksperchild=4) as p:
        for i, r in enumerate(p.imap_unordered(one, jobs), 1):
            rows.append(r)
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"[{i}/{len(jobs)}] {r['model']:12s} {r['track'][:40]:40s} "
                  f"SIR {r['sir']:.2f} SAR {r['sar']:.2f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["model", "track"])
    df.to_csv(C.OUT / "bss_drums_pertrack.csv", index=False)
    part.unlink()
    print(df.groupby("model")[["sdr", "sir", "sar", "isr"]].median().round(3))


if __name__ == "__main__":
    main()
