"""C2: timing headlines under two alternative detectors and an alternative matching rule.

Variants (detections from c2_detect.py):
  p2         Phase 2 detector, greedy closest-first matching at +/-50 ms (the Phase 2 baseline)
  rise       detect_refined(mode='rise', min_sep_s=0.03), greedy matching
  energy     detect_energy (Phase 2 events re-timed on the broadband energy onset function),
             greedy matching
  hungarian  Phase 2 detector, minimum-cost maximum-cardinality matching (timing.match_hungarian)
Per model (v1 Spleeter, HT-Demucs, BS-Roformer, SCNet-XL; B1 HT-Demucs) and generic floor:
F at +/-5/10/20/50 ms (mir_eval; detector variants only), median over tracks of the per-track
median |dev| and median dev, per-track median share > 5/8/10/14/20 ms, unmatched reference and
estimate shares; the A1 logit with track fixed effects (C0.5 features; the variant's reference
onsets are mapped to the Phase 2 reference onsets within 30 ms to take their features); the
A1 recurrence table (P(displaced in B | displaced in A) and lift).
A change table flags every Phase 2 headline (the p2 baseline) that changes by > 20 % relative,
or changes sign or rank order, under a variant.

    python c2_analyze.py
Outputs: results/phase3/c2_*.csv
"""
from __future__ import annotations

import pickle

import mir_eval
import numpy as np
import pandas as pd

import _p3 as P
from mss_eval import timing as T

LAB = ["v1_spleeter", "v1_htdemucs", "v1_bs_roformer", "v1_scnet", "htdemucs_b1"]
FLO = ["floor_white_40dB", "floor_white_30dB", "floor_white_20dB", "floor_leak_30dB", "floor_leak_20dB"]
VAR = ["p2", "rise", "energy", "hungarian"]
TOLS = [5, 10, 20, 50]
THR = [5, 8, 10, 14, 20]


def main():
    feats = pickle.load(open(P.DATA3 / "a1_features_v3.pkl", "rb"))
    by_track, onsets = [], []
    rechecks = []
    for t in P.tracks():
        r = pickle.load(open(P.DATA3 / "c2" / f"{P.safe_name(t)}.pkl", "rb"))
        if "floor_recheck" in r:
            rechecks.append((t, r["floor_recheck"]))
        p2ref = r["on"][("p2", "ref")]
        for v in VAR:
            det = "p2" if v == "hungarian" else v
            ref = r["on"][(det, "ref")]
            # map the variant's reference onsets to the Phase 2 reference onsets (features)
            j = np.searchsorted(p2ref, ref)
            best = np.full(len(ref), -1)
            for k, x in enumerate(ref):
                cand = [c for c in (j[k] - 1, j[k]) if 0 <= c < len(p2ref)]
                if cand:
                    c = min(cand, key=lambda c: abs(p2ref[c] - x))
                    if abs(p2ref[c] - x) <= 0.03:
                        best[k] = c
            for sig in LAB + FLO:
                est = r["on"][(det, sig)]
                ir, ie = (T.match_hungarian if v == "hungarian" else T.match_pairs)(ref, est, 0.05)
                dev = (est[ie] - ref[ir]) * 1000
                a = np.abs(dev)
                row = {"variant": v, "signal": sig, "track": t, "n_ref": len(ref), "n_est": len(est),
                       "n_pairs": len(ir), "median_abs_dev_ms": np.median(a) if len(a) else np.nan,
                       "median_dev_ms": np.median(dev) if len(a) else np.nan,
                       **{f"share_gt_{k}ms": np.mean(a > k) if len(a) else np.nan for k in THR},
                       "share_ref_unmatched": 1 - len(ir) / max(len(ref), 1),
                       "share_est_unmatched": 1 - len(ir) / max(len(est), 1)}
                if v != "hungarian":
                    for tol in TOLS:
                        row[f"f_{tol}ms"] = mir_eval.onset.f_measure(ref, est, window=tol / 1000)[0]
                by_track.append(row)
                if sig in LAB:
                    disp = np.zeros(len(ref), bool)
                    mat = np.zeros(len(ref), bool)
                    mat[ir] = True
                    disp[ir] = a > 10
                    onsets.append(pd.DataFrame({"variant": v, "label": sig, "track": t, "k": np.arange(len(ref)),
                                                "p2_index": best, "matched": mat, "displaced": disp}))
        print(t, flush=True)
    bt = pd.DataFrame(by_track)
    bt.to_csv(P.OUT3 / "c2_by_track.csv", index=False, float_format="%.5g")
    cols = [c for c in bt.columns if c.startswith(("f_", "median_", "share_"))]
    summ = bt.groupby(["variant", "signal"])[cols].median().reset_index()
    pooled = bt.groupby(["variant", "signal"]).agg(pooled_share_ref_unmatched=("n_pairs", "sum"), nref=("n_ref", "sum"),
                                                   nest=("n_est", "sum")).reset_index()
    pooled["pooled_share_ref_unmatched"] = 1 - pooled.pooled_share_ref_unmatched / pooled.nref
    summ = summ.merge(pooled[["variant", "signal", "pooled_share_ref_unmatched"]], on=["variant", "signal"])
    summ.to_csv(P.OUT3 / "c2_summary.csv", index=False, float_format="%.5g")
    # logit with track FE, and recurrence
    on = pd.concat(onsets, ignore_index=True)
    f = feats.rename(columns={"i_ref": "p2_index"})
    lg, rec = [], []
    for v in VAR:
        o = on[(on.variant == v) & (on.p2_index >= 0)].merge(f, on=["track", "p2_index"], how="left")
        for lab in LAB:
            g = o[(o.label == lab) & o.matched]
            Z = P.design(g)
            res = P.logit(g.displaced.to_numpy(float), Z.to_numpy(), list(Z.columns), g.track.to_numpy(), fe=True)
            res.insert(0, "label", lab)
            res.insert(0, "variant", v)
            lg.append(res)
        w = on[on.variant == v].pivot_table(index=["track", "k"], columns="label", values="displaced", aggfunc="first")
        mm = on[on.variant == v].pivot_table(index=["track", "k"], columns="label", values="matched", aggfunc="first")
        for a in LAB:
            for b in LAB:
                if a == b:
                    continue
                both = mm[a].astype(bool) & mm[b].astype(bool)
                wa, wb = w.loc[both, a].astype(bool), w.loc[both, b].astype(bool)
                base = wb.mean()
                cond = wb[wa].mean() if wa.any() else np.nan
                rec.append({"variant": v, "A": a, "B": b, "P_B": base, "P_B_given_A": cond,
                            "lift": cond / base if base else np.nan})
    lg = pd.concat(lg, ignore_index=True)
    lg.to_csv(P.OUT3 / "c2_logit_fe.csv", index=False, float_format="%.5g")
    rec = pd.DataFrame(rec)
    rec.to_csv(P.OUT3 / "c2_recurrence.csv", index=False, float_format="%.5g")
    # change table vs the p2 baseline
    rows = []
    base = summ[summ.variant == "p2"].set_index("signal")
    for v in ("rise", "energy", "hungarian"):
        s = summ[summ.variant == v].set_index("signal")
        for c in cols:
            if c not in s or s[c].isna().all():
                continue
            ro = base.loc[LAB, c].rank()
            rn = s.loc[LAB, c].rank()
            for sig in LAB + FLO:
                o, n = base.loc[sig, c], s.loc[sig, c]
                rel = (n - o) / abs(o) if o not in (0, np.nan) and np.isfinite(o) and o != 0 else np.nan
                rows.append({"variant": v, "signal": sig, "measure": c, "p2": o, "variant_value": n,
                             "rel_change": rel, "flag_rel_gt_20pct": bool(abs(rel) > 0.2) if np.isfinite(rel) else False,
                             "flag_sign": bool(np.sign(o) != np.sign(n) and o != 0 and n != 0),
                             "flag_rank": bool(sig in LAB and ro[sig] != rn[sig])})
        for term in ("env_strength", "sim_any", "log10_ioi_min_ms"):
            for lab in LAB:
                o = lg[(lg.variant == "p2") & (lg.label == lab) & (lg.term == term)].iloc[0]
                n = lg[(lg.variant == v) & (lg.label == lab) & (lg.term == term)].iloc[0]
                rows.append({"variant": v, "signal": lab, "measure": f"logit FE OR {term}", "p2": o.odds_ratio,
                             "variant_value": n.odds_ratio, "rel_change": (n.odds_ratio - o.odds_ratio) / o.odds_ratio,
                             "flag_rel_gt_20pct": abs(n.odds_ratio - o.odds_ratio) / o.odds_ratio > 0.2,
                             "flag_sign": bool(np.sign(np.log(o.odds_ratio)) != np.sign(np.log(n.odds_ratio))),
                             "flag_significance": (o.p < 0.05) != (n.p < 0.05), "flag_rank": False})
    ch = pd.DataFrame(rows)
    ch["flagged"] = ch[[c for c in ch.columns if c.startswith("flag_")]].fillna(False).any(axis=1)
    ch.to_csv(P.OUT3 / "c2_change_table.csv", index=False, float_format="%.5g")
    print("floor recheck (regenerated signal == cached Phase 2 floor onsets):", rechecks)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(summ.round(4).to_string(index=False))
        print(lg[lg.term.isin(["env_strength", "sim_any", "log10_ioi_min_ms"])].pivot_table(
            index=["label", "term"], columns="variant", values="odds_ratio").round(3).to_string())
        print(rec.pivot_table(index=["A", "B"], columns="variant", values="lift").round(2).to_string())
        print(ch[ch.flagged].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
