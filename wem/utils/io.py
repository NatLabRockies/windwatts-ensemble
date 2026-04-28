"""Generic I/O helpers for reading and writing CSV / Parquet tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame, dispatching on file extension."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV or Parquet, dispatching on file extension."""
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
