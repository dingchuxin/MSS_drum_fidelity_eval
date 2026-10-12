"""Reporting layer for the default-run benchmark: metric tables -> results/tables (csv)."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
TABLES = _ROOT / "results" / "tables"


def write_table(df, name: str) -> Path:
    """Persist a dataframe as csv in results/tables."""
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    df.to_csv(path)
    return path
