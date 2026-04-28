"""Tests for wem.constants."""

import re

from wem.constants import (
    HEIGHTS,
    HRRR_HEIGHTS,
    KNOT_TO_MS,
    QCOLS,
    WTK_HEIGHTS,
    WTKLED_HEIGHTS,
)


def test_qcols_length():
    assert len(QCOLS) == 101


def test_qcols_format():
    pattern = re.compile(r"^q\d{3}$")
    for c in QCOLS:
        assert pattern.match(c), f"Bad qcol format: {c}"


def test_qcols_sequential():
    expected = [f"q{i:03d}" for i in range(101)]
    assert QCOLS == expected


def test_qcols_first_last():
    assert QCOLS[0] == "q000"
    assert QCOLS[-1] == "q100"


def test_heights():
    assert HEIGHTS == [30, 40, 50, 60, 80, 100]


def test_knot_to_ms():
    assert abs(KNOT_TO_MS - 0.514444) < 1e-6


def test_wtk_heights():
    assert len(WTK_HEIGHTS) == 9
    assert all(WTK_HEIGHTS[i] <= WTK_HEIGHTS[i + 1] for i in range(len(WTK_HEIGHTS) - 1))
    for h in [10, 40, 100, 200]:
        assert h in WTK_HEIGHTS


def test_hrrr_heights():
    assert len(HRRR_HEIGHTS) == 15
    assert 10 in HRRR_HEIGHTS
    assert 1000 in HRRR_HEIGHTS


def test_wtkled_equals_hrrr():
    assert list(WTKLED_HEIGHTS) == list(HRRR_HEIGHTS)
