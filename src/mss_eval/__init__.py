"""MSS Drum Fidelity Evaluation: quantitative re-working of thesis section 4.7.

Benchmark-only (MUSDB18-HQ, real reference stems), because the article's claims are
support/refute and that needs ground truth. Two metric families, both reference-based:
  metrics  -> developer-style separation metrics (SDR/SIR/SAR/ISR, SI-SDR)
  analyst  -> what a musicologist reads (timing, dynamic profile, labeling precision)

Experiments: run_benchmark (claim A1, dynamics vary across models) and
run_input_length (claim A2, input length alters the separation of a fixed region).
"""

__version__ = "0.0.1"
