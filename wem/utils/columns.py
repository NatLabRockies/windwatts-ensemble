"""Helpers for finding and selecting DataFrame columns by flexible name matching."""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from wem.constants import QCOLS


def choose_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first column name from *candidates* that exists in *df*.

    Falls back to case-insensitive matching if no exact match is found.
    Returns ``None`` if nothing matches.
    """
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def find_qcols(df: pd.DataFrame) -> List[str]:
    """Return the list of quantile column names (q000..q100) present in *df*.

    Prefers the canonical ``QCOLS`` ordering.  Falls back to a heuristic
    pattern match if fewer than 50 canonical names are found.
    """
    have = [c for c in QCOLS if c in df.columns]
    if len(have) < 50:
        have = [c for c in df.columns if len(c) == 4 and c[0] == "q" and c[1:].isdigit()]
        have = sorted(have)
    return have
