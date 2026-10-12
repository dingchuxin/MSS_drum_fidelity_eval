"""v3 per-track measurement (Phase 3, C0.1/C0.2) of a drum-stem set against the true stems.

    python measure3.py <label> [--bandlimit] [--jobs 1]

Onsets and matching are reused from the Phase 2 caches (data/phase2/measure/<label>[_bl]):
their definition is unchanged. Everything else is recomputed with the v3 definitions:
  strokes    strokes.strokes_v3 on the reference-onset segments, for ALL reference strokes,
             in both signals; pair_v3 digital-silence rule; matched flag (+/-50 ms) and
             signed deviation per reference stroke
  envelope   analyst.dynamics_v3 (reference gate for both signals)
Per-track caches: data/phase3/v3/measure/<label>[_bl]/<track>.pkl
By-track table:   results/phase3/rerun/measure/<label>[_bl]_by_track.csv
"""
from __future__ import annotations

import argparse
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _p3 as P  # noqa: E402
import measure as M2  # noqa: E402
from mss_eval import analyst, strokes as SK  # noqa: E402

MEAS3 = P.DATA3 / "measure"


def compute(label: str, track: str, bl: bool) -> dict:
    tag = label + ("_bl" if bl else "")
    f = MEAS3 / tag / f"{P.safe_name(track)}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    p2 = pickle.load(open(P.P2MEAS / tag / f"{P.safe_name(track)}.pkl", "rb"))
    est = P.load(M2.stem_dir(label, track) / "drums.wav")
    ref = P.ref_stem(track)
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    if bl:
        est, ref = M2.bandlimit(est), M2.bandlimit(ref)
    ron, eon, ir, ie = p2["ref_onsets"], p2["est_onsets"], p2["i_ref"], p2["i_est"]
    rs = SK.strokes_v3(ref, P.SR, ron)
    es = SK.strokes_v3(est, P.SR, ron)
    rs, es, flags = SK.pair_v3(rs, es)
    matched = np.zeros(len(ron), bool)
    matched[ir] = True
    dev = np.full(len(ron), np.nan)
    dev[ir] = (eon[ie] - ron[ir]) * 1000
    res = {"track": track, "label": tag, "ref_onsets": ron, "matched": matched, "dev_ms": dev,
           "ref": rs, "est": es, **flags, "desc": analyst.dynamics_v3(est, ref, P.SR),
           "si_sdr": p2["si_sdr"], "n_est_onsets": len(eon),
           "ref_level_db": float(10 * np.log10(np.mean(P.mono(ref) ** 2) + 1e-8)),
           "definition_tag": P.TAG}
    if "bss" in p2:
        res["bss"] = p2["bss"]
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    return res


def summarize(r: dict) -> dict:
    row = {"label": r["label"], "track": r["track"], "n_ref": len(r["ref_onsets"]),
           "n_matched": int(r["matched"].sum()), "si_sdr": r["si_sdr"]}
    sil_r, sil_e = r["silent_ref"], r["silent_est"]
    row.update({"n_silent_either": int((sil_r | sil_e).sum()),
                "n_ref_silent_est_not": int((sil_r & ~sil_e).sum()),
                "n_est_silent_ref_not": int((sil_e & ~sil_r).sum())})
    for set_name, sel in (("all", np.ones(len(r["matched"]), bool)), ("matched", r["matched"])):
        for m in P.SMEAS3:
            d = r["est"][m][sel] - r["ref"][m][sel]
            row[f"d_{m}_{set_name}"] = np.nanmedian(d) if np.isfinite(d).any() else np.nan
            rr = r["ref"][m][sel]
            row[f"ref_{m}_{set_name}"] = np.nanmedian(rr) if np.isfinite(rr).any() else np.nan
        pr, pe = r["ref"]["peak_db"][sel], r["est"]["peak_db"][sel]
        ok = np.isfinite(pr) & np.isfinite(pe)
        row[f"ref_accent_{set_name}"] = np.std(pr[ok], ddof=1) if ok.sum() > 2 else np.nan
        row[f"d_accent_{set_name}"] = (np.std(pe[ok], ddof=1) - np.std(pr[ok], ddof=1)) if ok.sum() > 2 else np.nan
        row[f"n_{set_name}"] = int(sel.sum())
    dv = r["dev_ms"][r["matched"]]
    a = np.abs(dv)
    row.update({"median_dev_ms": np.median(dv), "median_abs_dev_ms": np.median(a),
                **{f"share_gt_{k}ms": np.mean(a > k) for k in (5, 8, 10, 14, 20)},
                "share_ref_unmatched": 1 - r["matched"].mean(),
                "share_est_unmatched": 1 - r["matched"].sum() / max(r["n_est_onsets"], 1)})
    row.update(dict(r["desc"]))
    if "bss" in r:
        row.update({k: r["bss"][k] for k in ("sdr", "sir", "sar", "isr")})
    return row


def by_track(tag: str) -> pd.DataFrame:
    res = [pickle.load(open(MEAS3 / tag / f"{P.safe_name(t)}.pkl", "rb")) for t in P.tracks()]
    df = pd.DataFrame([summarize(r) for r in res])
    (P.RERUN / "measure").mkdir(parents=True, exist_ok=True)
    df.to_csv(P.RERUN / "measure" / f"{tag}_by_track.csv", index=False, float_format="%.6g")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--bandlimit", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    tr = P.tracks()
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs, maxtasksperchild=5) as pool:
            for i, _ in enumerate(pool.starmap(compute, [(a.label, t, a.bandlimit) for t in tr]), 1):
                pass
    else:
        for i, t in enumerate(tr, 1):
            compute(a.label, t, a.bandlimit)
            print(f"[{i}/50] {a.label} {t}", flush=True)
    df = by_track(a.label + ("_bl" if a.bandlimit else ""))
    print(df[[c for c in df.columns if c.startswith("d_") and c.endswith("_all")]].median().round(3).to_string())


if __name__ == "__main__":
    main()
