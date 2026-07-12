# MSS Drum Fidelity Evaluation

A reproducible, quantitative companion to the qualitative music source separation (MSS)
evaluation in **§4.7** of *The Construction and Evolution of Rhythmic Interplay in the Music of
King Crimson* (Ding, 2026). It measures what modern separators do to the signal properties
rhythm analysis depends on — timing, dynamics, transient shape, stem identity — on
**MUSDB18-HQ**, where true reference stems make every claim falsifiable.

An accompanying paper, *Beyond SDR: How Music Source Separation Reshapes the Signal Properties
That Carry Musical Timing*, is forthcoming on **arXiv**; the citation and link will be added
here once it is live.

## Headline finding

By its developers' metric (SDR) MSS looks solved to within a tenth of a decibel. The way a
rhythm analyst must *use* it, it is not solved but **shaped**: the *location* of rhythmic
events is robust everywhere we probed, while the *shape* of those events (attack, envelope,
dynamics) is fragile in every direction — and via the perceptual centre (p-centre), shape is
part of perceived timing.

- **SDR predicts timing but not shape (B3).** Onset F-measure tracks SI-SDR (Spearman ρ =
  +0.62), but transient/dynamic distortion does not (|ρ| ≤ 0.29). The ranking inverts: the SDR
  leader (BS-Roformer) distorts drum attacks ~2× more than its SDR-matched CNN counterpart
  (SCNet-XL), which SDR calls its equal (within 0.1 dB).
- **The dynamic profile carries a model-specific bias (A1).** HT-Demucs reshapes drum dynamics
  ~2.3× more than BS-Roformer — a systematic per-model offset, so dynamics measured through
  different separators are not comparable across studies.
- **Input length reshapes transients, not timing (A2).** Growing the context around a fixed
  passage drifts its rendered attack (marginally more for the transformer; paired p = 0.044)
  while onset locations stay put.
- **Models fail differently (B1).** BS-Roformer maximizes interference suppression at the cost
  of more artifacts; SCNet-XL is the most artifact-free — an architecture-linked trade-off.

All tables and figures are in `results/`; the full argument is in the paper.

## Experiments

- **Benchmark (A1, B3, B1)** — 4 models × 50 tracks × 4 stems; developer + analyst metrics.
  → `scripts/run_benchmark.py`, then `scripts/synthesize.py`.
- **Input-length sweep (A2)** — a fixed region R separated under four growing context windows;
  drift of R's descriptors. → `scripts/run_input_length.py`, then `scripts/synthesize_a2.py`.

## Two metric families (both reference-based)

- **`metrics.py` — developer-style:** SDR/SIR/SAR/ISR (BSS-Eval, via `museval`) + SI-SDR. The
  numbers vendors report, recomputed on our tracks as the comparison baseline.
- **`analyst.py` — what a musicologist reads:** onset F-measure (timing), the full *dynamic
  profile* (envelope shape/slope + intensity distribution, raw and Δ vs. truth), and *labeling
  precision* (ground-truth energy routing). These test the claims SDR hides.

## Models

Four separators spanning architecture and era (see `config/models.yaml`), all 4-stem so
MUSDB18 references apply. BS-Roformer and SCNet-XL are a **capability-matched** attention /
no-attention pair (published drums SDRs within 0.1 dB), so A2 isolates architecture, not quality:

| Model | Era / type | Runner |
|-------|-----------|--------|
| Spleeter | 2019 freq-domain CNN (baseline) | sidecar `.venv-spleeter` |
| HT-Demucs | 2022 hybrid (baseline) | main `.venv` (audio-separator) |
| BS-Roformer | 2023 transformer (SOTA) | sidecar `.venv-msst` (MSST) |
| SCNet-XL | 2024 CNN (SOTA) | sidecar `.venv-msst` (MSST) |

## Layout

```
config/          model + dataset registries (yaml)
src/mss_eval/     io, separate (runners + stem cache), metrics, analyst, sweep, report
scripts/          run_benchmark.py, run_input_length.py, synthesize.py, synthesize_a2.py, setup_msst.py
external/MSST/    vendored ZFTurbo inference for BS-Roformer / SCNet (gitignored)
results/          generated tables + figures (tracked); audio is NOT
docs/             chapter_4_7_mapping.md (claims→metrics), environments.md (venvs)
data/             audio, weights, and cached separated stems (gitignored)
```

## Setup

Three venvs — main and MSST on **Python 3.12**, Spleeter on **3.11** — because the models'
dependency pins conflict. Full install (with the Windows Spleeter fix) is in
[`docs/environments.md`](docs/environments.md); `scripts/setup_msst.py` fetches the MSST
checkpoints. Separations are cached to `data/stems_out/` (gitignored), so runs are resumable
and metrics can be re-scored without re-separating.

## Status

**Complete.** The full benchmark and the input-length sweep have been run on MUSDB18-HQ; all
result tables and figures are in `results/`. The claim→metric design is in
[`docs/chapter_4_7_mapping.md`](docs/chapter_4_7_mapping.md). A2-*phrase* (musically-coherent
context) remains as future work, motivated by the A2-*length* result.
