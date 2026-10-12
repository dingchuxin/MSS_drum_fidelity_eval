"""B2: SCNet XL IHF (direct output) vs SCNet Masked XL IHF (complex mask): configuration check
(chunk size, overlap, normalisation, all model/inference keys). The paired per-track tests on the
50 tracks are computed with the v3 measures by scripts/phase3/c0_rerun.py (step b2).

    python b2_masking.py   -> results/phase2/b2_config_diff.csv
"""
from __future__ import annotations

import pandas as pd
import yaml

import _p2 as P

W2 = P.P2DATA / "weights"
A, B = "scnet_xl_ihf", "scnet_masked_xl_ihf"
CFG = {A: W2 / "config_musdb18_scnet_xl_more_wide_v5.yaml", B: W2 / "config_musdb18_scnet_xl_ihf.yaml"}


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


class _L(yaml.SafeLoader):
    pass


_L.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))


def main():
    ca, cb = (flat(yaml.load(open(CFG[m]), Loader=_L)) for m in (A, B))
    rows = [{"key": k, A: ca.get(k), B: cb.get(k)} for k in sorted(set(ca) | set(cb))
            if str(ca.get(k)) != str(cb.get(k))]
    diff = pd.DataFrame(rows)
    diff.to_csv(P.OUT2 / "b2_config_diff.csv", index=False)
    must = ["audio.chunk_size", "inference.num_overlap", "inference.normalize", "training.normalize",
            "audio.sample_rate", "model.nfft", "model.hop_size", "model.win_size"]
    print("config differences:\n", diff.to_string(index=False))
    print("matched keys:", {k: (ca.get(k), cb.get(k)) for k in must})

if __name__ == "__main__":
    main()
