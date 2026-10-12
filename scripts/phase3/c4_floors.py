"""C4 detections: matched-level floors (Phase 2 detector, timing.detect).

Per track and model (Spleeter, BS-Roformer, SCNet-XL: canonical per-track BSS of the v1 stems,
results/phase1/bss_drums_pertrack.csv; HT-Demucs: the B1 BSS):
  (a) leak   true drums + (bass + other + vocals) scaled so that the drum-to-accompaniment
             energy ratio (mono) equals the model's drum SIR on that track
  (b) noise  true drums + white noise at an SNR equal to the model's drum SDR on that track
  (c) leak+noise   (a) + white noise at an SNR equal to the model's drum SAR (re the drums)
Seeds as in Phase 2's floors: numpy default_rng([20260925, track index, condition index]) with
condition indices 10 + 3 x model index + k (the generic floors use 0-4).
The generic floors (white 40/30/20 dB, leakage 30/20 dB) are the Phase 2 floors
(data/phase2/floors), which were computed with the Phase 2 detector (floors.py uses
timing.detect); c2_detect.py re-checks that the regenerated signals reproduce them.

    python c4_floors.py [--jobs 2]   -> data/phase3/v3/c4/<track>.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _p3 as P  # noqa: E402
from mss_eval import timing as T  # noqa: E402

MODELS = ["spleeter", "htdemucs", "bs_roformer", "scnet"]
OUT = P.DATA3 / "c4"


def levels():
    bss = pd.read_csv(P.ROOT / "results" / "phase1" / "bss_drums_pertrack.csv")
    lv = {}
    for m in ("spleeter", "bs_roformer", "scnet"):
        b = bss[bss.model == m].set_index("track")
        lv[m] = b[["sdr", "sir", "sar"]].to_dict("index")
    lv["htdemucs"] = {}
    for t in P.tracks():
        r = pickle.load(open(P.P2MEAS / "htdemucs_b1" / f"{P.safe_name(t)}.pkl", "rb"))
        lv["htdemucs"][t] = {k: r["bss"][k] for k in ("sdr", "sir", "sar")}
    return lv


def scale_to(sig, target, db):
    """Scale `sig` so that 10 log10(E_target / E_sig) = db."""
    return sig * np.sqrt(np.mean(target ** 2) / (np.mean(sig ** 2) + 1e-20) / 10 ** (db / 10))


def one(args):
    ti, t, lv = args
    f = OUT / f"{P.safe_name(t)}.pkl"
    if f.exists():
        return t
    ref = P.mono(P.ref_stem(t))
    acc = sum(P.mono(P.ref_stem(t, s)) for s in P.SOURCES if s != "drums")
    res = {"track": t, "on": {}, "levels": {}}
    for mi, m in enumerate(MODELS):
        L = lv[m][t]
        res["levels"][m] = L
        rng = [np.random.default_rng([P.SEED, ti, 10 + 3 * mi + k]) for k in range(3)]
        leak = ref + scale_to(acc, ref, L["sir"])
        noise_sdr = scale_to(rng[1].standard_normal(len(ref)), ref, L["sdr"])
        noise_sar = scale_to(rng[2].standard_normal(len(ref)), ref, L["sar"])
        sig = {"leak": leak, "noise": ref + noise_sdr, "leak_noise": leak + noise_sar}
        for k, y in sig.items():
            res["on"][(m, k)] = T.detect(y, P.SR)
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    print(t, flush=True)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    lv = levels()
    from multiprocessing import Pool
    with Pool(a.jobs, maxtasksperchild=4) as p:
        list(p.imap_unordered(one, [(i, t, lv) for i, t in enumerate(P.tracks())]))
