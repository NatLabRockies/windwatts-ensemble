"""Quantile computation helpers for wind speed arrays and DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wem.constants import QCOLS

# np.trapz was renamed to np.trapezoid in NumPy 2.0
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz


def quantile_block(spd: np.ndarray) -> np.ndarray:
    """Compute 101 percentiles (0..100) along the time axis.

    Parameters
    ----------
    spd : np.ndarray
        Wind speed array of shape ``(T, S)`` where *T* is the number of
        timesteps and *S* is the number of sites.

    Returns
    -------
    np.ndarray
        Percentile values with shape ``(101, S)``.
    """
    qs = np.arange(101)
    return np.nanpercentile(spd, q=qs, axis=0, method="linear")


def mean_from_quantiles(df: pd.DataFrame) -> np.ndarray:
    """Estimate mean wind speed per row from the 101-point quantile CDF.

    Uses the trapezoidal rule to integrate over the quantile columns
    ``q000`` through ``q100`` (spacing ``dx=0.01``).

    Vectorized variant operating on a full DataFrame.  See also:
    - ``wem.analyze.quantile_maps.mean_from_quantiles`` — row-level with NaN handling
    - ``wem.analyze.ml_results.mean_from_quantile_series`` — long-format (qnum, value) pairs

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``q000`` through ``q100``.

    Returns
    -------
    np.ndarray
        1-D float32 array of length ``len(df)`` with the estimated mean
        wind speed for each row.
    """
    qs = df[QCOLS].to_numpy(dtype="float32", copy=False)
    return _trapezoid(qs, dx=0.01, axis=1).astype("float32", copy=False)


def mean_from_quantiles_row(
    row: pd.Series, qcols: list[str]
) -> float | None:
    """Compute trapezoidal-mean wind speed from a single quantile CDF row.

    NaN-aware, row-level variant.  Skips NaN pairs during integration.

    Parameters
    ----------
    row : pd.Series
        Row containing quantile values at the columns listed in *qcols*.
    qcols : list of str
        Ordered quantile column names (e.g. ``["q000", ..., "q100"]``).

    Returns
    -------
    float or None
        Estimated mean wind speed, or ``None`` if fewer than 2 finite values.
    """
    qvals = row[qcols].to_numpy(dtype=float)
    finite = np.isfinite(qvals)
    if finite.sum() < 2:
        return None
    area = 0.0
    for k in range(1, len(qvals)):
        if finite[k] and finite[k - 1]:
            area += 0.5 * (qvals[k - 1] + qvals[k])
    return float(area * 0.01)


def mean_from_quantile_long(
    qnum: np.ndarray, values: np.ndarray
) -> float:
    """Mean wind speed from long-format (qnum, value) pairs via trapezoidal rule.

    NaN-aware variant for long-format data where quantile numbers and
    values are stored in separate arrays.

    Parameters
    ----------
    qnum : array-like
        Quantile numbers (0..100).
    values : array-like
        Corresponding wind speed values.

    Returns
    -------
    float
        Estimated mean wind speed, or ``NaN`` if fewer than 2 finite pairs.
    """
    q = np.asarray(qnum, dtype="float64")
    v = np.asarray(values, dtype="float64")
    good = np.isfinite(q) & np.isfinite(v)
    if good.sum() < 2:
        return np.nan
    idx = np.argsort(q[good])
    vv = v[good][idx]
    area = np.nansum(0.5 * (vv[1:] + vv[:-1]))
    return float(area * 0.01)
