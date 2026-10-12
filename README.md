# Separated drum stems as measuring instruments

Code and per-track results for the article

> Ding, C. (2026). *Separated drum stems as measuring instruments: an error budget of four open separators for microrhythm analysis.* Full citation to be added on publication.

Analysts increasingly measure separated drum stems as if they were the recorded performance. This study measures what such a stem can and cannot be trusted for. Four open separators, Spleeter, HT-Demucs, BS-RoFormer and SCNet-XL, separate the 50 tracks of the MUSDB18-HQ test set. Every measurement is made twice, on the estimated drum stem and on the true drum stem, with the same code and settings. The difference, estimate minus reference, is reported for each track.

## What is measured

- **Timing.** A refined onset detector (`src/mss_eval/timing.py`) is used. It keeps librosa's onset-strength feature, samples it every 1.45 ms and takes the envelope peak, refined by parabolic interpolation, as the event time. Under known shifts of the true stems, its 95th-percentile error is at most 0.20 ms. For each track:
  - the typical deviation, the median absolute deviation of matched onsets;
  - the share of matched onsets displaced by more than 10 ms;
  - the shares of missed and extra onsets;
  - the onset F-measure at tolerances of ±5 to ±50 ms.

  Floors give reference points: the true stem with added noise or accompaniment leakage, at fixed levels and at each model's own SIR, SDR and SAR.
- **Stroke levels and envelope shape.** Stroke measures are read in stroke windows defined by the reference onsets (`src/mss_eval/strokes.py`): stroke peak, inter-stroke level, valley depth, accent contrast, level dependence, pre-stroke level, level jump and rise time. Seven whole-track envelope descriptors are also computed (`src/mss_eval/analyst.py`).
- **Developer metrics.** SDR, SIR and SAR (BSS Eval version 4, via museval) and SI-SDR (`src/mss_eval/metrics.py`), with their correlations with the timing and shape errors, within and across models.
- **How the file is cut (Experiment 2).** A 5-s passage of each track is separated:
  - at twelve positions relative to each model's chunk grid;
  - on its own and within 10-s and 20-s windows;
  - with BS-RoFormer's and SCNet-XL's chunk overlaps swapped;
  - with HT-Demucs's unseeded random shifts.
- **Further analyses:**
  - which strokes are displaced or missed (logistic regressions);
  - the dependence of missed strokes on the detector's threshold;
  - a local timing statistic computed on separated and true stems;
  - a check with two further SCNet checkpoints, one estimating a mask and one the source directly.

## Reproducing the results

### Environments

There are three Python environments, because the models' dependency pins conflict:
- the main environment and the MSST environment (BS-RoFormer, SCNet-XL) on Python 3.12;
- the Spleeter environment on Python 3.11.

[`docs/environments.md`](docs/environments.md) describes the installation, including a fix for Spleeter on Windows (`overrides-spleeter.txt`). `scripts/setup_msst.py` fetches the BS-RoFormer and SCNet-XL checkpoints. The two further SCNet checkpoints of the masking check (MSST releases v1.0.15 and v1.0.17: `model_scnet_ep_36_sdr_10.0891.ckpt` with `config_musdb18_scnet_xl_more_wide_v5.yaml`, and `model_scnet_masked_ep_111_sdr_9.8286.ckpt` with `config_musdb18_scnet_xl_ihf.yaml`) are not fetched by it; they belong in `data/phase2/weights/`. The separations were run on one NVIDIA RTX 4070 SUPER (12 GB); Spleeter runs on the CPU.

### Data

MUSDB18-HQ (Rafii et al., 2019, https://doi.org/10.5281/zenodo.3338373) is available from Zenodo under its own licence. It is not redistributed here.

- **Where the corpus goes.** Place it at `data/musdb18hq/test/<track>/{mixture,drums,bass,other,vocals}.wav` inside the repository. The default runs (step 1 below) read it there and write their stems to `data/stems_out/`.
- **`MSS_DATA_ROOT`.** All later scripts read the corpus and those stems from the folder in the environment variable `MSS_DATA_ROOT`, which defaults to `data/`. Set it only if both live elsewhere.
- **What is written under `data/`** (all ignored by git):
  - further separations and intermediate caches;
  - model checkpoints, in `data/weights/` and `data/phase2/weights/`.

  The MSST code is cloned into `external/MSST/`.
- **No audio.** This repository contains no audio. The audio excerpt shown in Figure 1 of the article is regenerated from the corpus by `scripts/make_excerpt.py`.

### Steps

Run the steps in this order. Each step reads the outputs of the previous ones. Commands are given from the repository root. `<label>` stands for each of:
- `v1_spleeter`, `v1_htdemucs`, `v1_bs_roformer`, `v1_scnet` (the default runs);
- `htdemucs_b1` (the seeded run);
- `scnet_xl_ihf`, `scnet_masked_xl_ihf` (the masking check).

| Step | Command | Writes |
|---|---|---|
| 1. Separate the test set with the four models (HT-Demucs also with seeded shifts and without its peak cap) | `python scripts/run_benchmark.py --models spleeter htdemucs bs_roformer scnet`; then `python scripts/phase2/make_jobs.py b01` and `python scripts/phase2/make_jobs.py b2`, and the job files run by `scripts/phase2/run_htdemucs.py` (main environment), `run_msst.py` (MSST environment) and `run_spleeter.py` (Spleeter environment), e.g. `python scripts/phase2/run_msst.py data/phase2/jobs/b2_msst.jsonl` | `data/stems_out/<model>/<track>/`, `data/phase2/stems/<run>/<track>/` (with a JSON sidecar per run); `results/tables/benchmark_metrics.csv` |
| 2. Detect onsets and compute the timing, stroke and envelope measures | `python scripts/phase1/j1_timing.py`, `python scripts/phase1/j1_summarize.py`, `python scripts/synthesize.py`; `python scripts/phase2/measure.py <label>` (with `--bss` for the last three labels, and again with `--bandlimit` for the four default runs); `python scripts/phase3/measure3.py <label>` (also with `--bandlimit` for the four default runs); `python scripts/phase2/a1_features.py`, `python scripts/phase3/c0_features.py` | `results/phase1/j0_resolution.csv`, `onset_deviation_by_track.csv`; `results/tables/benchmark_raw.csv`; `results/phase2/measure/`, `results/phase3/rerun/measure/` (per-track CSVs); caches under `data/` |
| 3. Developer metrics | `python scripts/phase1/j3_bss_pertrack.py`, `python scripts/phase1/j3_stats.py` | `results/phase1/bss_drums_pertrack.csv`, `stats_tests.csv` |
| 4. Floors (the floors at each model's own levels use the developer metrics of step 3) | `python scripts/phase2/floors.py`, `python scripts/phase3/c4_floors.py`, `python scripts/phase3/c4_analyze.py` | `results/phase3/c4_by_track.csv`, `c4_summary.csv`, `c4_tests.csv` |
| 5. Experiment 2: separations and measures | `python scripts/phase2/b3_setup.py manifest`, `python scripts/phase2/b3_setup.py jobs`; the runners of step 1 on `data/phase2/jobs/b3_{msst,htdemucs,spleeter}.jsonl`; `python scripts/phase2/b3_measure.py`, `python scripts/phase2/b3_analyze.py`, `python scripts/phase3/b3_measure3.py`, `python scripts/phase3/c6_selection.py` | `results/phase2/b3_manifest.csv`, `b3_*_by_track.csv`, `b3_per_onset_drift.csv.gz`, `b3_htdemucs_runs.csv`, `b3_sanity_window_vs_full.csv`; `results/phase3/c6_*.csv` |
| 6. Further analyses (regressions, threshold, analysis statistic, masking check, validation tests) | In this order, each as `python <script>` from the repository root. Phase 1: `scripts/phase1/j6_excerpt.py` (choice of the excerpt); `param_counts_msst.py` and `param_counts_other.py` (see their usage lines). Phase 2: `scripts/phase2/a0_clipping.py`, `a1_a2_onsets.py`, `b0_b1_checks.py b0` and `b0_b1_checks.py b1`, `b2_masking.py`. Validation tests: `audit/l2_t1_onsets.py`, `audit/l2_t3_descriptors.py`, `audit2/t1_strokes.py`. Phase 3: `scripts/phase3/c0_rerun.py all`, `c0_shift_test.py`, `c0_validate_strokes.py`, `c1_c7_onsets.py`, `c2_detect.py`, `c2_analyze.py`, `c2_validate_synth.py`, `c3_coupling.py`, `c5_c8_strokes.py`, then `audit2/l1_checks.py`. Excerpt audio: `scripts/make_excerpt.py`. Phase 4: `scripts/phase4/d1_statistic.py`, `d2_envelopes.py`, `d2_threshold.py`, `d5_excerpt.py`. | `results/phase2/`, `results/phase3/`, `results/phase3/rerun/`, `results/phase4/`, `results/audit/`, `results/audit2/` |
| 7. Values reported in the article and its supplement that no result file stores directly | `python scripts/paper_values_v2.py` | `results/phase5/paper_values_v2.csv`, with a check of each value against the printed one in `paper_values_v2_checks.csv` |
| 8. Figures | `python scripts/make_figures_v2.py` (add `--formats pdf,tiff` for TIFF copies) | `results/phase5/figures/`: `fig3_excerpt.pdf`, `fig2_budget.pdf`, `fig6_sdr.pdf` and `fig7_exp2.pdf` are Figures 1–4; `fig1_design.pdf`, `fig4_timing.pdf` and `fig5_shape.pdf` are Figures S1–S3 |

**Running time.** Steps 1 and 5 need the GPU and take several hours. The run sidecars record the separation time only:
- 0.9 h for the 150 full-track runs of the seeded HT-Demucs run and the two further SCNet checkpoints;
- 4.9 h for the 4,900 runs of Experiment 2.

The default runs were not timed. All later steps run on the CPU from the stored results.

**Seeds and repeatability.**
- **Seed base:** 20260925. It serves:
  - the bootstrap resampling;
  - the synthetic validation signals;
  - the white-noise floors (seeded by base, track index and condition);
  - HT-Demucs's shifts in the seeded run and in Experiment 2.
- **Bit-identical on repeat:** repeat runs of BS-RoFormer, SCNet-XL, Spleeter and the seeded HT-Demucs run were bit-identical within a session (`results/phase2/b0_repeat_runs.csv`, `b1_determinism.csv`). The default HT-Demucs run is not, because its shifts are unseeded.
- **CPU steps** reproduce their stored outputs exactly from the stored results, with one exception. Test 1b of `audit/l2_t1_onsets.py` (sensitivity to attack time) seeds its synthetic events from Python's string hash, which changes between sessions, so its values vary slightly between runs (median slope 0.123–0.128 ms per ms over five runs). `results/audit/l2_t1_summary.json` and `l2_t1b_attack.csv.gz` hold the run reported in the article.

### Where each result comes from

| In the article or supplement | Computed by | From |
|---|---|---|
| Table 1, Table S1 | — | published studies |
| Table 2, Table S2 (models): drum SDR here, parameters | `scripts/phase1/j3_bss_pertrack.py`, `scripts/phase1/param_counts_msst.py`, `param_counts_other.py` | `results/phase1/bss_drums_pertrack.csv`, `c_param_counts.csv` |
| Table 3 (error budget), Tables S4 and S5 (timing, floors) | `scripts/phase2/measure.py`, `scripts/phase3/c4_floors.py`, `c4_analyze.py`, `c0_rerun.py`, `c5_c8_strokes.py`, `scripts/synthesize.py` | `results/phase3/c4_summary.csv`, `c4_tests.csv`, `results/phase2/measure/`, `results/phase1/bss_drums_pertrack.csv`, `results/tables/benchmark_raw.csv`, `results/phase3/rerun/sd_units_v3.csv`, `results/phase3/c5_expansion.csv` |
| Table 4, Table S16 (local timing statistic) | `scripts/phase4/d1_statistic.py` | `results/phase4/d1_summary.csv`, `d1_statistic_by_track.csv`, `d1_tempo_yardstick.csv` |
| Table S3 (detector) | `audit/l2_t1_onsets.py`, `scripts/phase3/c0_shift_test.py`, `c2_validate_synth.py`, `scripts/phase1/j1_summarize.py` | `results/audit/l2_t1_*`, `results/phase3/c0_shift_test.csv`, `c2_validate_summary.json`, `results/phase1/j0_resolution.csv` |
| Table S6 (rise-point marker) and the timing robustness checks | `scripts/phase3/c2_detect.py`, `c2_analyze.py`, `c0_rerun.py` | `results/phase3/c2_summary.csv`, `c2_change_table.csv`, `c2_recurrence.csv`, `results/phase3/rerun/a5_bandlimit_*_v3.csv` |
| Table S7 (pairwise comparisons) | `scripts/paper_values_v2.py` (rows `D3.1`) | `results/phase2/measure/v1_*_by_track.csv` |
| Table S8 (regressions) | `scripts/phase2/a1_a2_onsets.py`, `scripts/phase3/c1_c7_onsets.py`, `c3_coupling.py` | `results/phase3/c1_logit.csv`, `c7_logit.csv`, `c7_peak_unmatched_vs_matched.csv`, `c3_*.csv`, `results/phase2/a1_recurrence.csv`, `a1_substitution.csv` |
| Tables S9 and S10 (threshold) | `scripts/phase4/d2_envelopes.py`, `d2_threshold.py` | `results/phase4/d2_summary.csv`, `d2_missed_classes.csv`, `d2_missed_strokes.csv.gz`, `d2_normalisation_by_track.csv` |
| Table S11 and Figure S3 (shape) | `scripts/phase3/measure3.py`, `c0_rerun.py`, `c5_c8_strokes.py`, `c0_validate_strokes.py`, `audit/l2_t3_descriptors.py`, `audit2/t1_strokes.py`, `audit2/l1_checks.py` | `results/phase3/rerun/sd_units_v3.csv`, `a3_*_v3.csv`, `a1c_rank_agreement_v3.csv`, `results/phase3/c8_definitions.csv`, `c0_v3_validation_summary.csv`, `results/audit/l2_t3_*`, `results/audit2/t1c_rise_summary.csv`, `l1_silent_*.csv` |
| Table S12 (BS-RoFormer and SCNet-XL) and the masking check | `scripts/phase3/c0_rerun.py`, `scripts/paper_values_v2.py` (rows `D3.2`–`D3.4`), `scripts/phase2/b2_masking.py` | `results/phase3/rerun/j3_stats_v3.csv`, `b2_masking_tests_v3.csv`, `results/phase2/b2_config_diff.csv` |
| Table S13 (developer metrics) | `scripts/phase1/j3_bss_pertrack.py`, `j3_stats.py`, `scripts/run_benchmark.py`, `scripts/synthesize.py` | `results/phase1/bss_drums_pertrack.csv`, `stats_tests.csv`, `results/tables/benchmark_metrics.csv`, `benchmark_raw.csv`, `results/phase3/d3_labeling_precision_medians.csv` |
| Table S14 and Figure 3 (SDR against the errors) | `scripts/phase3/c0_rerun.py`, `scripts/paper_values_v2.py` (rows `D3.5`) | `results/phase3/rerun/a4_sdr_vs_fidelity_v3.csv`, `results/phase3/fig_sdr_vs_fidelity_v3.csv` |
| Table S15 and Figure 4 (Experiment 2) | step 5, `scripts/phase3/c0_rerun.py`, `scripts/paper_values_v2.py` (rows `E2b`) | `results/phase2/b3_*`, `results/phase3/c6_*.csv`, `results/phase3/rerun/b3_*_v3.csv` |
| Table S17 (recommendations) | — | the results above |
| Figure 1 (excerpt) | `scripts/phase1/j6_excerpt.py` (choice of the excerpt), `scripts/make_excerpt.py` (audio), `scripts/phase4/d5_excerpt.py` (onsets) | `results/phase1/figure_excerpt/manifest.json`, `results/phase4/d5_excerpt_strokes.csv` |
| Figure 2 (error budget) | `scripts/make_figures_v2.py`, `scripts/paper_values_v2.py` (rows `D3.7`) | `results/phase2/measure/`, `results/phase2/a2_thresholds.csv`, `results/phase3/c4_summary.csv`, `results/phase3/rerun/sd_units_v3.csv` |
| Figures S1 and S2 (design, timing) | `scripts/make_figures_v2.py` | S2: `results/phase2/fig_timing_error_distribution.csv`, `results/phase3/c4_by_track.csv` |
| Figures 1–4 and S1–S3 (drawing) | `scripts/make_figures_v2.py` | the files above |

Values that `paper_values_v2.py` does not compute are read directly from the result files above. In particular:
- the labelling-precision medians in `results/phase3/d3_labeling_precision_medians.csv` are the medians of `labeling_precision` in `results/tables/benchmark_metrics.csv`;
- the published SDRs, training data and software usage figures of the four models come from outside sources cited in the article.

## Layout

```
config/          model and dataset registries
src/mss_eval/    onset detector, stroke measures, envelope descriptors, developer metrics, separation runners
scripts/         separation, measurement and analysis scripts, in stage folders in the order the analyses were run
                 (phase1: detector resolution, per-track BSS Eval and paired statistics, excerpt choice, parameter counts;
                 phase2: separation runners for the further runs, measurements with the refined detector, floors,
                 Experiment 2; phase3: the stroke and descriptor definitions used in the article and the robustness
                 analyses; phase4: the threshold analysis, the local timing statistic and the excerpt);
                 run_benchmark.py, synthesize.py, setup_msst.py, make_excerpt.py, paper_values_v2.py and make_figures_v2.py
results/         per-track results, summaries and figures (CSV, JSON, PDF); no audio
docs/            environments.md
audit/, audit2/  validation tests on synthetic signals with known values (detector, stroke measures,
                 envelope descriptors) and data checks; their outputs are in results/audit and results/audit2
```

## Versions

- `v2.0.0`: the analyses of the article above.
- `v1.0.0`: the first version of the pipeline, superseded by `v2.0.0`. `scripts/run_benchmark.py` and `scripts/synthesize.py`, with `results/tables/benchmark_metrics.csv` and `benchmark_raw.csv`, remain from it because version 2 uses them.

## Licence and citation

The code and results are released under the MIT licence (`LICENSE`). If you use them, please cite the article above and the archived version of this repository (`CITATION.cff`; the archive DOI will be added here after the release).
