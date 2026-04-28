"""Tests for wem.utils.io."""

from pathlib import Path

import pandas as pd
import pytest

from wem.utils.io import read_table, write_table

_has_pyarrow = True
try:
    import pyarrow
except ImportError:
    _has_pyarrow = False


class TestReadWriteCSV:
    def test_csv_roundtrip(self, tmp_path):
        p = tmp_path / "test.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        write_table(df, p)
        result = read_table(p)
        pd.testing.assert_frame_equal(df, result)

    def test_no_index_in_output(self, tmp_path):
        p = tmp_path / "test.csv"
        df = pd.DataFrame({"x": [1]})
        write_table(df, p)
        text = p.read_text()
        assert "Unnamed" not in text

    def test_nonexistent_raises(self, tmp_path):
        p = tmp_path / "nope.csv"
        with pytest.raises(Exception):
            read_table(p)


@pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not available")
class TestReadWriteParquet:
    def test_parquet_roundtrip(self, tmp_path):
        p = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_table(df, p)
        result = read_table(p)
        pd.testing.assert_frame_equal(df, result)


class TestExtensionDispatch:
    def test_csv_extension(self, tmp_path):
        p = tmp_path / "data.csv"
        df = pd.DataFrame({"x": [1]})
        write_table(df, p)
        assert p.exists()
        assert p.suffix == ".csv"

    @pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not available")
    def test_parquet_extension(self, tmp_path):
        p = tmp_path / "data.parquet"
        df = pd.DataFrame({"x": [1]})
        write_table(df, p)
        assert p.exists()
        assert p.suffix == ".parquet"
