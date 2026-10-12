"""C2/C3: parameter counts for the non-MSST models (no inference).

  main .venv:       python scripts/phase1/param_counts_other.py demucs <demucs_repo_dir> <mmi_config.yaml>
  .venv-spleeter:   python scripts/phase1/param_counts_other.py spleeter <pretrained_models/4stems>
HT-Demucs weights are Meta's official release (955717e8), loaded with weights_only=False because
demucs pickles the model class; the Demucs3 mmi count is built from MSST's config (no weights).
Spleeter's count sums TF checkpoint variables except Adam slots (includes BatchNorm moving stats).
"""
import sys

if sys.argv[1] == "demucs":
    import collections
    import functools
    from pathlib import Path

    import torch
    import yaml
    torch.load = functools.partial(torch.load, weights_only=False)
    from demucs.hdemucs import HDemucs
    from demucs.pretrained import get_model
    K = ("LocalState", "BLSTM", "CrossTransformerEncoderLayer", "MyTransformerEncoderLayer")
    m = get_model("htdemucs", repo=Path(sys.argv[2])).models[0]
    print("htdemucs", sum(p.numel() for p in m.parameters()), float(m.segment),
          collections.Counter(type(x).__name__ for x in m.modules() if type(x).__name__ in K))
    cfg = yaml.safe_load(open(sys.argv[3]))
    h = HDemucs(sources=cfg["training"]["instruments"], **cfg["hdemucs"])
    print("hdemucs_mmi", sum(p.numel() for p in h.parameters()),
          collections.Counter(type(x).__name__ for x in h.modules() if type(x).__name__ in K))
else:
    import numpy as np
    import tensorflow as tf
    vs = tf.train.list_variables(sys.argv[2])
    n = sum(int(np.prod(s)) for k, s in vs
            if "Adam" not in k and k not in ("beta1_power", "beta2_power", "global_step"))
    print("spleeter 4stems", n)
