"""Power-law wind-speed interpolation and height-bracketing utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def bracket_for_height(
    z: float, avail_heights: np.ndarray
) -> Tuple[bool, int, int]:
    """Find the bracketing heights for a target height *z* within *avail_heights*.

    Parameters
    ----------
    z : float
        Target height (metres).
    avail_heights : np.ndarray
        Sorted array of available heights (metres).

    Returns
    -------
    tuple[bool, int, int]
        ``(is_exact, h_lo, h_hi)``.  If *z* matches an available height
        exactly (within 1e-6), ``is_exact`` is ``True`` and ``h_lo == h_hi``.
        Otherwise, the pair brackets *z* (clamped to the edge pair when *z*
        falls outside the range).
    """
    avail = avail_heights
    # exact if close to any level (tolerate float noise)
    exact_mask = np.isclose(z, avail, rtol=0, atol=1e-6)
    if exact_mask.any():
        h = int(avail[exact_mask][0])
        return True, h, h
    # bracket
    lo_idx = np.searchsorted(avail, z, side="right") - 1
    hi_idx = np.searchsorted(avail, z, side="left")
    lo_idx = np.clip(lo_idx, 0, len(avail) - 2)  # ensure a valid pair
    hi_idx = np.clip(hi_idx, lo_idx + 1, len(avail) - 1)
    return False, int(avail[lo_idx]), int(avail[hi_idx])


def power_law_interp(H: float, hv: List[Tuple[float, float]]) -> float:
    """Interpolate/extrapolate wind speed U(H) using a power law between two heights.

    Parameters
    ----------
    H : float
        Target height (metres).
    hv : list[tuple[float, float]]
        List of ``(height, value)`` pairs with non-NaN values.

    Returns
    -------
    float
        Interpolated/extrapolated wind speed at height *H*.

    Strategy
    --------
    - If *H* equals any height, return that value directly.
    - If inside the range, bracket with nearest below and above.
    - If outside the range, use the two nearest on that side for extrapolation.
    - If only one value available, return that value as a fallback.
    - Otherwise, return ``NaN``.
    """
    hv = [(float(h), float(u)) for h, u in hv if np.isfinite(h) and np.isfinite(u)]
    if not hv:
        return np.nan
    # exact match?
    for h, u in hv:
        if np.isclose(H, h, rtol=0, atol=1e-6):
            return u

    hv_sorted = sorted(hv, key=lambda t: t[0])
    hs = [h for h, _ in hv_sorted]
    us = [u for _, u in hv_sorted]
    H = float(H)

    # inside range
    if hs[0] < H < hs[-1]:
        # find bracket
        for i in range(len(hs) - 1):
            h1, h2 = hs[i], hs[i + 1]
            if h1 <= H <= h2 and np.isfinite(us[i]) and np.isfinite(us[i + 1]):
                u1, u2 = us[i], us[i + 1]
                if u1 <= 0 or u2 <= 0 or h1 <= 0 or h2 <= 0:
                    # fallback linear if weird
                    t = (H - h1) / (h2 - h1)
                    return (1 - t) * u1 + t * u2
                alpha = np.log(u2 / u1) / np.log(h2 / h1)
                return u1 * (H / h1) ** alpha

    # below min -> use first two
    if H <= hs[0] and len(hs) >= 2 and np.isfinite(us[0]) and np.isfinite(us[1]):
        h1, h2 = hs[0], hs[1]
        u1, u2 = us[0], us[1]
        if u1 > 0 and u2 > 0 and h1 > 0 and h2 > 0:
            alpha = np.log(u2 / u1) / np.log(h2 / h1)
            return u1 * (H / h1) ** alpha
        return u1  # fallback

    # above max -> use last two
    if H >= hs[-1] and len(hs) >= 2 and np.isfinite(us[-2]) and np.isfinite(us[-1]):
        h1, h2 = hs[-2], hs[-1]
        u1, u2 = us[-2], us[-1]
        if u1 > 0 and u2 > 0 and h1 > 0 and h2 > 0:
            alpha = np.log(u2 / u1) / np.log(h2 / h1)
            return u1 * (H / h1) ** alpha
        return u2  # fallback

    # only one value available
    if len(hs) == 1 and np.isfinite(us[0]):
        return us[0]

    return np.nan


def fit_power_law_alpha(
    heights: np.ndarray, speeds: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Fit ln(U) = a + alpha * ln(z) on positive finite pairs.

    Parameters
    ----------
    heights : np.ndarray
        Array of heights (metres).
    speeds : np.ndarray
        Array of wind speeds (m/s) corresponding to *heights*.

    Returns
    -------
    tuple[float, float] or None
        ``(A, alpha)`` where ``U = A * z^alpha``, or ``None`` if fewer than
        two valid (positive, finite) data points are available.
    """
    z = np.asarray(heights, dtype="float64")
    u = np.asarray(speeds, dtype="float64")
    good = np.isfinite(z) & np.isfinite(u) & (z > 0) & (u > 0)
    if good.sum() < 2:
        return None
    lnz = np.log(z[good])
    lnu = np.log(u[good])
    alpha, a = np.polyfit(lnz, lnu, 1)  # lnu ~ a + alpha*lnz
    A = float(np.exp(a))
    return (A, float(alpha))
