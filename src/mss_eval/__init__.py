"""MSS Drum Fidelity Evaluation: measurements of separated drum stems against the true stems.

Benchmark-only (MUSDB18-HQ, real reference stems): every measure needs ground truth.
Two metric families, both reference-based:
  metrics  -> developer-style separation metrics (SDR/SIR/SAR/ISR, SI-SDR)
  analyst  -> what a musicologist reads (timing, dynamic profile, labeling precision)

The default separations and the version 1 benchmark tables come from scripts/run_benchmark.py;
the version 2 analyses are in scripts/phase1 ... scripts/phase4.
"""

__version__ = "0.0.1"
