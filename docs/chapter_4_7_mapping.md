# Section 4.7 → quantitative mapping

Each qualitative claim in the thesis's MSS chapter is paired with a measurable quantity.
Everything is computed on **MUSDB18-HQ**, where real reference stems make each number
checkable — the article is benchmark-only, because *support-or-refute* needs ground truth
(the ecological mix-only tier was dropped for exactly this reason).

## §4.7 claim → benchmark metric

| # | Qualitative claim in 4.7 | Quantity | Metric (module) |
|---|--------------------------|----------|-----------------|
| 1 | "Spleeter shows substantial inter-track leakage" | other-source energy in a stem | `sir` — BSS-Eval (metrics) |
| 2 | "conservative in preserving decay / transient energy" | attack preservation vs. true stem | `onset_f_measure`, `dyn_attack_slope` (analyst) |
| 3 | "HT-Demucs occasionally compresses dynamics" | dynamic-profile change vs. true stem | `d_variation`, `d_crest`, `d_env_flux` (analyst) |
| 4 | "denoising introduced a spectral void" | over-suppression / artifact energy | `sar` + energy deficit vs. true stem (metrics) |
| 5 | "onset deviations 10–60 ms; pre-ringing in Spleeter" | timing displacement | `onset_f_measure` (analyst) |
| 6 | "global dynamic characteristics diverge" | dynamic-profile deviation | `d_*` profile (analyst) |
| 7 | Input-duration test (half/one/four-phrase/full) | metric as a function of input length | A2 sweep (sweep) |
| 8 | "SDR did not correlate with perceptual scores" | SDR vs. analyst-metric rank corr | B3 below |
| 9 | Labelling policy differs (bells/tuned metal routing) | energy routed to each stem label | `labeling_precision` + routing (analyst) |

Claim 4 was a King-Crimson-specific moment; on the benchmark it is tested *generally* (does
a model push energy below the true stem's floor), not at that one timestamp.

## Falsifiable hypotheses

The primary objective: **render real data that supports or refutes the two primary §4.7
claims.** Everything runs on MUSDB18-HQ via `scripts/run_benchmark.py` (A1) and
`scripts/run_input_length.py` (A2). Metrics beyond SDR are what actually test the claims.

The analyst dimensions (`src/mss_eval/analyst.py`):
- **timing** — `onset_fmeasure`: onset F of the estimate vs. its true stem (50 ms tol).
- **dynamics = the full dynamic profile**, not just a range. `dynamic_profile()` returns
  envelope shape/slope (`env_flux`, `attack_slope`, `env_skew`) and intensity distribution
  (`level_db`, `variation`, `crest`, `impulsiveness`, `burst_rate`). The claim is that this
  *whole profile* is model-dependent, so we report both the raw profile per model (`dyn_*`)
  and its deviation from truth (`d_*`).
- **labeling = ground-truth routing.** `labeling_precision()` least-squares-decomposes each
  estimated stem onto the true sources; precision is the correctly-labelled source's
  additive share of explained energy (perfect separation = 1.0, gain-invariant).

### Half A — the two primary §4.7 claims (this is the experiment)

**A1. The dynamic profile varies across models → statistically unreliable.**
Compute `dynamic_profile` descriptors on the *true* stem and each model's estimate, per
track × stem.

| Test | Quantity | Supported if | Refuted if |
|------|----------|--------------|-----------|
| A1a cross-model spread | SD across models of each `dyn_*` descriptor | spread ≳ between-track variance (model choice injects real noise) | spread ≪ between-track variance |
| A1b bias vs. truth | `d_*` descriptors (est − true) | biases differ *by model* (no common correction) | all models ≈ 0 / same offset |
| A1c ordering stability | Spearman ρ of track-ranking-by-descriptor between models | low ρ (models disagree on who's "dynamic") | high ρ (ranking survives) |

*If any of A1a/A1c holds, a dynamics statistic computed through separation is confounded
by model choice — the §4.7 claim.* Ground truth makes this a real test, not a proxy.

**A2. Altering input length changes transient / dynamics / labeling.**
Fix a target region **R**; run the model on inputs of increasing length that all contain
R (R alone → R+context → full track); extract R's output from each run; measure on the
common region. Runner: `scripts/run_input_length.py`; core: `src/mss_eval/sweep.py`.

| Facet | Metric (module) | Supported if | Refuted if |
|-------|-----------------|--------------|-----------|
| transient envelope | `onset_f_measure`, `dyn_attack_slope` (analyst) | drifts with input length | invariant |
| dynamics | `dyn_*` / `d_*` profile (analyst) | drifts | invariant |
| labeling | `labeling_precision` + routing matrix (analyst) | routing shifts with length | routing fixed |

Run in two stages, not one:
- **A2-length — ACTIVE NOW.** Growing context in musically-neutral seconds. MUSDB18 has
  **no phrase annotations**, but the R-window design needs none — length is controlled
  directly. Bonus with ground truth: does more context move the estimate *toward* truth
  (context helps) or merely *change* it (instability)?
- **A2-phrase — DEFERRED (further study).** The thesis's phrase-unit ladder (half / one /
  four / full phrase), where each input is a musically *consistent* unit. Run only as a
  follow-up **if A2-length shows an effect worth explaining**; it needs phrase boundaries
  whose source is still TBD. The A2-length-vs-A2-phrase contrast is what would isolate
  *musical coherence* from raw duration.

> **Mechanism note (for the write-up, to be cross-checked against data — not yet claimed).**
> Two structural reasons input content/length *could* reshape the separation, to be
> evidenced concretely in the arXiv paper:
> (1) **attention models (BS-Roformer)** can exploit long-range musical structure within
> the input chunk, so a coherent phrase gives usable context a fragment doesn't — predicts
> a *model-dependent* A2 effect (larger for transformers than for frame-local CNNs);
> (2) **input-statistics normalization** — Demucs normalizes by the input's global
> mean/std (`_run_demucs` in `src/mss_eval/separate.py`), so surrounding content directly
> rescales the target region's **dynamics**, tying A2 back to A1. Confirm both against the
> collected data before asserting.

### Half B — mechanistic "why models differ" (secondary, kept)

Not the headline, but the interesting surrounding structure.

| H | Restated | Metric | Direction / test |
|---|----------|--------|------------------|
| B1 | Interference vs. artifact trade-off differs by model | `sir` vs. `sar` | freq-domain: lower SIR (more bleed); transformer: higher both |
| B2 | Behaviour clusters by domain (freq / hybrid / transformer / CNN) | all metrics | within-domain < between-domain distance |
| B3 | SDR does not track analyst-relevant degradation | rank corr SDR vs. analyst metrics | weak Spearman ρ (SDR ⊥ timing/dynamics) → §4.7 claim 8 (Torcoli 2021) |

## Design notes

- **Determinism:** all model runs use fixed parameters, no random re-init — matches the
  thesis's stated protocol so numbers are reproducible.
- **Provenance:** every result row carries `{model, version, params, input_range, sr}` —
  the thesis explicitly asks separation results to ship with this metadata.
- **Roster:** four models spanning era + architecture (Spleeter / HT-Demucs / BS-Roformer /
  SCNet), all 4-stem so MUSDB references apply; see `config/models.yaml`. The
  transformer-vs-CNN pair is what makes B2 and the A2 mechanism note testable.

## Open questions

- [ ] Which MUSDB18 subset — full test (50 tracks) or a curated N for turnaround?
- [ ] A2-phrase boundary source, *if* A2-length warrants the follow-up: annotate a MUSDB
      subset, auto-detect (downbeat/hypermeter), or leave phrase-based work to later.
