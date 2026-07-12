"""Reporting layer: metric rows -> tables/figures that drop into the thesis.

Outputs land in results/tables (markdown + csv) and results/figures (png), both
tracked in git so the written chapter cites committed artifacts.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
TABLES = _ROOT / "results" / "tables"
FIGURES = _ROOT / "results" / "figures"


def _ensure():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_table(df, name: str, caption: str = "") -> Path:
    """Persist a dataframe as both csv (data) and md (thesis-ready)."""
    _ensure()
    df.to_csv(TABLES / f"{name}.csv")
    md = df.to_markdown() if hasattr(df, "to_markdown") else str(df)
    text = (f"# {caption}\n\n" if caption else "") + md + "\n"
    (TABLES / f"{name}.md").write_text(text, encoding="utf-8")
    return TABLES / f"{name}.md"


def bar_by_model(df, value: str, name: str, ylabel: str = "") -> Path:
    """Simple grouped bar chart of a metric across models/stems."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _ensure()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = df.pivot_table(index="model", columns="stem", values=value, aggfunc="median")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(ylabel or value)
    ax.set_title(name.replace("_", " "))
    fig.tight_layout()
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
