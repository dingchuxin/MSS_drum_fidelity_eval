"""Write JSON-lines job files for the GPU/CPU separation runners.

    python make_jobs.py b01     B0 repeat-run check (BS-Roformer, SCNet-XL, Spleeter: two runs of
                                3 tracks; SCNet-XL's first run is stored as 'scnet_float'), and
                                B1 HT-Demucs (seeded, without the peak cap; 2 tracks repeated)
    python make_jobs.py b2      masking check (SCNet XL IHF vs SCNet Masked XL IHF)
Job files land in data/phase2/jobs/. Seeds: SEED + track index (B1); recorded in each sidecar.
"""
from __future__ import annotations

import json
import sys

import _p2 as P

J = P.P2DATA / "jobs"
S = P.P2DATA / "stems"
B0_TRACKS = [0, 17, 34]          # fixed, spread over the sorted test list


def write(name, jobs):
    J.mkdir(parents=True, exist_ok=True)
    with open(J / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{name}: {len(jobs)} jobs -> {J / (name + '.jsonl')}")


def full(model, ti, t, out, seed=None, **kw):
    return {"id": f"{model}|{t}|{out.parent.name}", "model": model, "track": t, "win": None,
            "keep": "all", "seed": P.SEED + ti if seed is None else seed, "out": str(out), **kw}


def main(which):
    T = P.tracks()
    if which == "b01":
        msst = []
        for i in B0_TRACKS:
            t = T[i]
            msst.append(full("scnet", i, t, S / "scnet_float" / P.safe_name(t)))
            msst.append(full("scnet", i, T[i], S / "b0_scnet_run2" / P.safe_name(t)))
            msst.append(full("bs_roformer", i, t, S / "b0_bs_roformer_run1" / P.safe_name(t)))
            msst.append(full("bs_roformer", i, t, S / "b0_bs_roformer_run2" / P.safe_name(t)))
        write("b01_msst", msst)
        spl = []
        for i in B0_TRACKS:
            t = T[i]
            for r in (1, 2):
                spl.append({**full("spleeter", i, t, S / f"b0_spleeter_run{r}" / P.safe_name(t))})
        write("b01_spleeter", spl)
        htd = [{**full("htdemucs", i, t, S / "htdemucs_b1" / P.safe_name(t)), "setting": "b1"}
               for i, t in enumerate(T)]
        for i in B0_TRACKS[:2]:
            t = T[i]
            htd.append({**full("htdemucs", i, t, S / "htdemucs_b1_repeat" / P.safe_name(t)),
                        "setting": "b1"})
        write("b01_htdemucs", htd)
    elif which == "b2":
        jobs = []
        for m in ["scnet_xl_ihf", "scnet_masked_xl_ihf"]:
            jobs += [full(m, i, t, S / m / P.safe_name(t)) for i, t in enumerate(T)]
        write("b2_msst", jobs)


if __name__ == "__main__":
    main(sys.argv[1])
