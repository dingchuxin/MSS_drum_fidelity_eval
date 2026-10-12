"""Shared helpers for Phase 3 (v3 definitions and robustness checks; CPU only).

Caches carry a definition tag: data/phase3/<TAG>/... Phase 1-2 caches are
reused only for quantities whose definition did not change (the Phase 2 onsets and their
matching: timing.detect + greedy closest-first at +/-50 ms).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2"))
from _p2 import *  # noqa: F401,F403,E402
import _p2 as P  # noqa: E402

ROOT = P.ROOT
SR = P.SR
SEED = P.SEED
tracks = P.tracks
safe_name = P.safe_name
load = P.load
mono = P.mono
ref_stem = P.ref_stem
est_stem = P.est_stem
SOURCES = P.SOURCES
P2DATA = P.P2DATA
OUT2 = P.OUT2

TAG = "v3"
DATA3 = ROOT / "data" / "phase3" / TAG
OUT3 = ROOT / "results" / "phase3"
RERUN = OUT3 / "rerun"
for _d in (DATA3, OUT3, RERUN):
    _d.mkdir(parents=True, exist_ok=True)
P2MEAS = P2DATA / "measure"
LABELS = ["v1_spleeter", "v1_htdemucs", "v1_bs_roformer", "v1_scnet", "htdemucs_b1"]
NICE = {"v1_spleeter": "Spleeter", "v1_htdemucs": "HT-Demucs v1", "v1_bs_roformer": "BS-Roformer",
        "v1_scnet": "SCNet-XL", "htdemucs_b1": "HT-Demucs B1",
        "scnet_xl_ihf": "SCNet XL IHF", "scnet_masked_xl_ihf": "SCNet Masked XL IHF"}
SMEAS3 = ["peak_db", "floor_db", "valley_db", "rise_ms", "pre_db", "jump_db"]
V3DESCS = ["level_db", "variation", "crest", "impulsiveness", "env_skew", "env_flux", "attack_slope"]
NB = 10_000
LCF = "Little Chicago's Finest - My Own"


def holm(p):
    """Holm step-down; NaN p-values are not counted in m and stay NaN."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    ps = p[ok]
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(1.0, run)
    out[ok] = adj
    return out


def rank_biserial(d):
    from scipy.stats import rankdata
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if not len(d):
        return float("nan")
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def boot_ci(v, stat, rng, n=NB):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.array([np.nan, np.nan])
    idx = rng.integers(0, len(v), size=(n, len(v)))
    return np.percentile(stat(v[idx], axis=1), [2.5, 97.5])


def wilcoxon_p(x, alternative="two-sided"):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or not np.any(x != 0):
        return float("nan")
    return float(wilcoxon(x, alternative=alternative).pvalue)


# ---------------------------------------------------------------------- shared models
FEATS_P2 = ["env_strength", "peak_rel_db", "log10_ioi_min_ms", "drum_to_rest_db", "share_low", "share_high"]


def design(g, suffix=""):
    """Standardised A1 predictors (as Phase 2) + sim_any (0/1). suffix='_p2' uses the Phase 2
    versions of the recomputed features."""
    import pandas as pd
    X = pd.DataFrame({"env_strength": g.env_strength,
                      "peak_rel_db": g[f"peak_rel_db{suffix}"],
                      "log10_ioi_min_ms": np.log10(g.ioi_min_ms.clip(lower=1)),
                      "drum_to_rest_db": g[f"drum_to_rest_db{suffix}"].clip(-40, 40),
                      "share_low": g[f"share_low{suffix}"], "share_high": g[f"share_high{suffix}"]})
    Z = (X - X.mean()) / X.std()
    Z["sim_any"] = g.sim_any.astype(float).to_numpy()
    return Z


def logit(y, X, names, groups, fe=False):
    """Logistic regression, CR1 standard errors clustered by `groups` (track).
    fe=True adds a dummy per group (track fixed effects); groups whose outcome does not vary
    are dropped (their dummy is not identified). Returns a DataFrame for the named terms."""
    import pandas as pd
    from scipy import stats
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    groups = np.asarray(groups)
    dropped = []
    if fe:
        keep = np.ones(len(y), bool)
        for gname in np.unique(groups):
            m = groups == gname
            if y[m].min() == y[m].max():
                keep &= ~m
                dropped.append(gname)
        y, X, groups = y[keep], X[keep], groups[keep]
        ug = np.unique(groups)
        D = (groups[:, None] == ug[None, :]).astype(float)
        Xf = np.column_stack([X, D])
        terms = list(names)
    else:
        Xf = np.column_stack([np.ones(len(y)), X])
        terms = ["intercept"] + list(names)
    b = np.zeros(Xf.shape[1])
    for _ in range(100):
        eta = np.clip(Xf @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = Xf.T @ (Xf * W[:, None]) + 1e-9 * np.eye(Xf.shape[1])
        step = np.linalg.solve(H, Xf.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(Xf @ b, -30, 30)))
    Hinv = np.linalg.inv(Xf.T @ (Xf * (p * (1 - p))[:, None]) + 1e-9 * np.eye(Xf.shape[1]))
    u = Xf * (y - p)[:, None]
    idx = pd.Series(range(len(y))).groupby(groups).indices
    meat = sum(np.outer(u[i].sum(0), u[i].sum(0)) for i in idx.values())
    G, N, K = len(idx), len(y), Xf.shape[1]
    V = Hinv @ meat @ Hinv * (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.diag(V))
    k = len(terms)
    off = 0 if fe else 0
    out = pd.DataFrame({"term": terms, "coef": b[off:off + k], "se_cluster": se[off:off + k]})
    out["z"] = out.coef / out.se_cluster
    out["p"] = 2 * stats.norm.sf(np.abs(out.z))
    out["odds_ratio"] = np.exp(out.coef)
    out["n_onsets"], out["n_events"], out["n_tracks"] = N, int(y.sum()), G
    out["n_tracks_dropped_no_variation"] = len(dropped)
    return out


def onset_table():
    """One row per (label, reference onset): matched, dev_ms, displaced, v3 stroke changes and
    the A1 features (Phase 3 versions + Phase 2 versions with suffix _p2). Cached."""
    import pickle
    import pandas as pd
    f = DATA3 / "onset_table.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    feats = pickle.load(open(DATA3 / "a1_features_v3.pkl", "rb"))
    rows = []
    for lab in LABELS:
        for t in tracks():
            r = pickle.load(open(DATA3 / "measure" / lab / f"{safe_name(t)}.pkl", "rb"))
            d = {"label": lab, "track": t, "i_ref": np.arange(len(r["ref_onsets"])),
                 "t": r["ref_onsets"], "matched": r["matched"], "dev_ms": r["dev_ms"]}
            for m in SMEAS3:
                d[f"ref_{m}"] = r["ref"][m]
                d[f"d_{m}"] = r["est"][m] - r["ref"][m]
            rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    df = df.merge(feats.drop(columns=["t"]), on=["track", "i_ref"], how="left")
    assert df.env_strength.notna().all()
    df["displaced"] = df.matched & (df.dev_ms.abs() > 10)
    pickle.dump(df, open(f, "wb"))
    return df
