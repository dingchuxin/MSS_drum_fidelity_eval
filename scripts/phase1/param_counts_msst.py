"""C2/C3: parameter counts of MSST models, instantiated from their configs (CPU, no inference).

Run in .venv-msst with cwd = the MSST checkout:
    python scripts/phase1/param_counts_msst.py <msst_dir> <cfg_dir> <out_csv>
For the two checkpoints already on disk the count is also taken from the state dict.
"""
import csv
import os
import sys

import torch

msst, cfg_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
weights = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, msst)
os.chdir(msst)
from utils.settings import get_model_from_config  # noqa: E402

SPECS = [  # (label, model_type, config path)
    ("BS-Roformer (existing)", "bs_roformer", f"{msst}/configs/config_bs_roformer_384_8_2_485100.yaml"),
    ("SCNet-XL (existing)", "scnet", f"{msst}/configs/config_musdb18_scnet_xl.yaml"),
    ("BS Conformer (Medium)", "bs_conformer", f"{cfg_dir}/config_musdb18_bs_conformer_infer.yaml"),
    ("SCNet Large", "scnet", f"{cfg_dir}/config_musdb18_scnet_large.yaml"),
    ("SCNet Large (starrytong)", "scnet", f"{cfg_dir}/config_musdb18_scnet_large_starrytong.yaml"),
    ("SCNet Tran Small", "scnet_tran", f"{cfg_dir}/config_musdb18_scnet_tran.yaml"),
    ("SCNet Small", "scnet", f"{cfg_dir}/config_musdb18_scnet.yaml"),
    ("SCNet Masked XL IHF", "scnet_masked", f"{cfg_dir}/config_musdb18_scnet_xl_ihf.yaml"),
    ("SCNet XL IHF", "scnet", f"{cfg_dir}/config_musdb18_scnet_xl_more_wide_v5.yaml"),
]
CKPT = {"BS-Roformer (existing)": "bs_roformer_musdb.ckpt", "SCNet-XL (existing)": "scnet_xl_musdb.ckpt"}

rows = []
for label, mtype, cfg in SPECS:
    model, config = get_model_from_config(mtype, cfg)
    n = sum(p.numel() for p in model.parameters())
    row = {"model": label, "model_type": mtype, "config": os.path.basename(cfg), "params_from_config": n}
    if weights and label in CKPT:
        sd = torch.load(os.path.join(weights, CKPT[label]), map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        row["params_in_checkpoint"] = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        row["load_missing"], row["load_unexpected"] = len(missing), len(unexpected)
    rows.append(row)
    print(row, flush=True)

keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "model_type", "config", "params_from_config",
                                      "params_in_checkpoint", "load_missing", "load_unexpected"])
    w.writeheader()
    w.writerows(rows)
