"""Test 3: envelope descriptors (analyst.dynamic_profile / dynamics_fidelity).

  A  known values: level, crest, impulsiveness, variation (IQR), env_flux on signals whose
     answer is analytic
  B  what drives attack_slope / env_flux / variation: burst trains with the attack time
     varied at a fixed inter-stroke floor, and the floor varied at a fixed attack
  C  pure gain change (-20, -6, +6 dB) on synthetic trains and on 3 real reference drum
     stems: which descriptors move (only level_db should)
  D  d = estimate - reference when the 'estimate' is the reference plus a low-level noise
     floor (-90..-50 dB re full scale) in its silent stretches: the descriptor change caused
     by gating/EPS asymmetry, with no change to any stroke

Outputs: results/audit/l2_t3_*.csv and l2_t3_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _paths as A
import synth as S

A.project()
from mss_eval import analyst  # noqa: E402

SR = S.SR
D = ["level_db", "variation", "crest", "impulsiveness", "env_skew", "env_flux", "attack_slope"]


def prof(x):
    p = analyst.dynamic_profile(x, SR)
    return {k: p[k] for k in D}


def train(att, tau, floor_db=None, ioi=0.25, n=40, kind="white", seed=0, accent=0.0, silence_s=0.0):
    rng = np.random.default_rng(seed)
    y = np.zeros(int((silence_s + 0.3 + ioi * n + 0.5) * SR))
    for i in range(n):
        a = 0.3 * (10 ** (-accent / 20) if i % 2 else 1.0)
        S.place(y, S.event(att, tau, kind, rng, dur_s=ioi + 0.3, amp=a), silence_s + 0.3 + ioi * i)
    if floor_db is not None:
        y = y + 0.3 * 10 ** (floor_db / 20) * S.carrier(len(y), "white", rng)
    return y


def part_a():
    rows = []
    t = np.arange(10 * SR) / SR
    rng = np.random.default_rng(1)
    sig = {"sine_a0.5": (0.5 * np.sin(2 * np.pi * 440 * t), {"level_db": 20 * np.log10(0.5 / np.sqrt(2)), "crest": 20 * np.log10(np.sqrt(2)), "impulsiveness": -1.5}),
           "square_a0.25": (0.25 * np.sign(np.sin(2 * np.pi * 441 * t + 0.1)), {"level_db": 20 * np.log10(0.25), "crest": 0.0, "impulsiveness": -2.0}),
           "gauss_rms0.1": (0.1 * rng.standard_normal(len(t)), {"level_db": -20.0, "impulsiveness": 0.0}),
           "laplace_rms0.1": (0.1 / np.sqrt(2) * rng.laplace(size=len(t)), {"level_db": -20.0, "impulsiveness": 3.0})}
    # stepped level: 0.5 s segments alternating -20 / -32 dB (white noise): IQR -> 12 dB
    x = 0.1 * rng.standard_normal(len(t))
    seg = (np.floor(t / 0.5) % 2).astype(bool)
    x[seg] *= 10 ** (-12 / 20)
    sig["steps_12dB"] = (x, {"variation": 12.0, "level_db": 10 * np.log10(0.5 * 0.01 + 0.5 * 0.01 * 10 ** (-1.2))})
    for name, (x, known) in sig.items():
        p = prof(x)
        for k, v in known.items():
            rows.append({"signal": name, "descriptor": k, "known": v, "measured": p[k], "err": p[k] - v})
        if name == "steps_12dB":
            # env_flux: 20 step transitions/10 s over 999 hops of |12 dB| plus noise ripple
            rows.append({"signal": name, "descriptor": "env_flux", "known": np.nan, "measured": p["env_flux"], "err": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(A.OUT / "l2_t3a_known.csv", index=False, float_format="%.5g")
    print(df.round(4).to_string(index=False))
    return df


def part_b():
    rows = []
    for kind in ("white", "low", "high"):
        for tau in (20, 60):
            for floor_db in (None, -80, -60, -50, -40, -30):
                for att in (0.5, 1, 2, 5, 10, 20):
                    y = train(att, tau, floor_db, kind=kind, seed=11)
                    p = prof(y)
                    rows.append({"kind": kind, "tau_ms": tau,
                                 "floor_db": -999 if floor_db is None else floor_db,
                                 "attack_ms": att, **p})
    df = pd.DataFrame(rows)
    df.to_csv(A.OUT / "l2_t3b_attack_vs_floor.csv", index=False, float_format="%.5g")
    out = {}
    for d in ("attack_slope", "env_flux", "variation", "env_skew", "crest"):
        # sensitivity to attack (0.5 -> 20 ms) at a fixed floor vs to the floor (-30 -> -60) at a fixed attack
        a = df.pivot_table(index=["kind", "tau_ms", "floor_db"], columns="attack_ms", values=d)
        f = df.pivot_table(index=["kind", "tau_ms", "attack_ms"], columns="floor_db", values=d)
        out[d] = {"median_change_attack_0p5_to_5ms": float((a[5] - a[0.5]).median()),
                  "median_change_attack_0p5_to_20ms": float((a[20] - a[0.5]).median()),
                  "median_change_floor_-30_to_-60dB": float((f[-60] - f[-30]).median()),
                  "median_change_floor_-40_to_-50dB": float((f[-50] - f[-40]).median()),
                  "median_change_floor_-60dB_to_digital_silence": float((f[-999] - f[-60]).median())}
    print(json.dumps(out, indent=1))
    piv = df[(df.kind == "white") & (df.tau_ms == 60)].pivot_table(index="floor_db", columns="attack_ms", values="attack_slope")
    print("attack_slope (white, tau 60), rows floor dB (-999 = digital silence), cols attack ms\n", piv.round(2).to_string())
    return out


def part_c():
    rows = []
    sigs = {"train_silence": train(1, 40, None, seed=3, silence_s=2.0),
            "train_floor-60": train(1, 40, -60, seed=3, silence_s=2.0)}
    for t in A.tracks()[:3]:
        import soundfile as sf
        x, _ = sf.read(str(A.MUSDB_TEST / t / "drums.wav"), always_2d=True, dtype="float64")
        sigs[f"ref:{t}"] = x
    for name, x in sigs.items():
        p0 = prof(x)
        for g in (-20, -6, 6):
            p = prof(x * 10 ** (g / 20))
            rows.append({"signal": name, "gain_db": g, **{f"delta_{k}": p[k] - p0[k] for k in D}})
    df = pd.DataFrame(rows)
    df.to_csv(A.OUT / "l2_t3c_gain.csv", index=False, float_format="%.5g")
    print(df.round(4).to_string(index=False))
    return df


def part_d():
    import soundfile as sf
    rows = []
    rng = np.random.default_rng(5)
    for t in A.tracks():
        x, _ = sf.read(str(A.MUSDB_TEST / t / "drums.wav"), always_2d=True, dtype="float64")
        m = x.mean(axis=1)
        zero = np.mean(m == 0)
        if zero < 0.01:
            continue
        pr = prof(x)
        for nl in (-90, -70, -50):
            est = x + 10 ** (nl / 20) * rng.standard_normal(x.shape)
            pe = prof(est)
            rows.append({"track": t, "ref_share_exact_zero": zero, "noise_dbfs": nl,
                         **{f"d_{k}": pe[k] - pr[k] for k in D}})
        print(t, round(zero, 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(A.OUT / "l2_t3d_silence_noise.csv", index=False, float_format="%.5g")
    print(df.groupby("noise_dbfs")[[f"d_{k}" for k in D]].median().round(3).to_string())
    return df


if __name__ == "__main__":
    out = {}
    a = part_a()
    out["A_known"] = a.round(4).to_dict("records")
    out["B_attack_vs_floor"] = part_b()
    c = part_c()
    out["C_gain_max_abs_delta_excl_level"] = {k: float(c[f"delta_{k}"].abs().max()) for k in D if k != "level_db"}
    d = part_d()
    out["D_silence_noise_median_d"] = d.groupby("noise_dbfs")[[f"d_{k}" for k in D]].median().round(4).to_dict()
    out["D_n_tracks_with_ge1pct_digital_silence"] = int(d.track.nunique())
    json.dump(out, open(A.OUT / "l2_t3_summary.json", "w"), indent=1, default=float)
