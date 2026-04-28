"""Tests for wem.grid.merge_grid."""

import numpy as np
import pandas as pd
import pytest

from wem.grid.merge_grid import (
    RX_FILE,
    SOURCE_LABEL,
    detect_qcols,
    find_input_files,
    read_one,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QCOLS_101 = [f"q{i:03d}" for i in range(101)]


def _make_quantile_csv(path, prefix="hrrr", height=60, n_rows=3,
                       extra_qcols=None, add_height_m=True):
    """Write a synthetic *_quantiles_*m.csv file."""
    fname = f"{prefix}_quantiles_{height}m.csv"
    fp = path / fname
    rows = []
    for i in range(n_rows):
        row = {
            "grid_id": f"G{i:03d}",
            "lat": 40.0 + i * 0.25,
            "lon": -100.0 + i * 0.25,
        }
        if add_height_m:
            row["height_m"] = float(height)
        for qi, qc in enumerate(QCOLS_101):
            row[qc] = round(qi * 0.1, 2)
        if extra_qcols:
            for qc in extra_qcols:
                row[qc] = 99.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(fp, index=False)
    return fp


# ---------------------------------------------------------------------------
# TestFindInputFiles
# ---------------------------------------------------------------------------


class TestFindInputFiles:
    """Tests for find_input_files()."""

    def test_matching_filenames_returned_sorted(self, tmp_path):
        _make_quantile_csv(tmp_path, "wtk", 100)
        _make_quantile_csv(tmp_path, "hrrr", 60)
        _make_quantile_csv(tmp_path, "wtk_led", 80)
        files = find_input_files(tmp_path)
        assert len(files) == 3
        # Should be sorted by name
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_non_matching_filenames_filtered(self, tmp_path):
        _make_quantile_csv(tmp_path, "hrrr", 60)
        # Create a file that doesn't match the pattern
        (tmp_path / "random_file.csv").write_text("a,b\n1,2\n")
        (tmp_path / "era5_quantiles_60m.csv").write_text("a,b\n1,2\n")
        files = find_input_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "hrrr_quantiles_60m.csv"

    def test_empty_directory(self, tmp_path):
        files = find_input_files(tmp_path)
        assert files == []


# ---------------------------------------------------------------------------
# TestDetectQcols
# ---------------------------------------------------------------------------


class TestDetectQcols:
    """Tests for detect_qcols()."""

    def test_standard_q000_q100_detected(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "hrrr", 30)
        qcols = detect_qcols(fp)
        assert qcols == QCOLS_101

    def test_non_q_columns_excluded(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "hrrr", 30)
        qcols = detect_qcols(fp)
        assert "grid_id" not in qcols
        assert "lat" not in qcols
        assert "height_m" not in qcols

    def test_sorted_by_numeric_value(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "hrrr", 30)
        qcols = detect_qcols(fp)
        nums = [int(c[1:]) for c in qcols]
        assert nums == sorted(nums)


# ---------------------------------------------------------------------------
# TestReadOne
# ---------------------------------------------------------------------------


class TestReadOne:
    """Tests for read_one()."""

    def test_source_label_hrrr(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "hrrr", 60)
        df = read_one(fp, source_col="source")
        assert (df["source"] == "HRRR").all()

    def test_source_label_wtk(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "wtk", 60)
        df = read_one(fp, source_col="source")
        assert (df["source"] == "WTK").all()

    def test_source_label_wtk_led(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "wtk_led", 60)
        df = read_one(fp, source_col="source")
        assert (df["source"] == "WTK-LED").all()

    def test_height_parsed_from_filename(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "hrrr", 80)
        df = read_one(fp, source_col="source")
        assert (df["height"] == 80).all()

    def test_height_m_filled_when_missing(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "hrrr", 100, add_height_m=False)
        df = read_one(fp, source_col="source")
        assert (df["height_m"] == 100.0).all()

    def test_column_ordering(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "hrrr", 60)
        df = read_one(fp, source_col="source")
        expected_prefix = ["grid_id", "lat", "lon", "source", "height_m", "height"]
        assert df.columns.tolist()[:6] == expected_prefix
        # Remaining columns should all be q-cols
        assert all(c.startswith("q") for c in df.columns[6:])

    def test_nan_rows_dropped(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "hrrr", 60, n_rows=3)
        # Corrupt one row's lat to NaN
        raw = pd.read_csv(fp)
        raw.loc[1, "lat"] = np.nan
        raw.to_csv(fp, index=False)
        df = read_one(fp, source_col="source")
        assert len(df) == 2

    def test_custom_source_col_name(self, tmp_path):
        fp = _make_quantile_csv(tmp_path, "hrrr", 60)
        df = read_one(fp, source_col="dataset")
        assert "dataset" in df.columns
        assert "source" not in df.columns


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() end-to-end merge."""

    def test_merge_two_csvs(self, tmp_path, monkeypatch):
        _make_quantile_csv(tmp_path, "hrrr", 60, n_rows=2)
        _make_quantile_csv(tmp_path, "wtk", 60, n_rows=3)
        out = tmp_path / "merged.csv"

        monkeypatch.setattr(
            "sys.argv",
            ["merge_grid", "--in-dir", str(tmp_path), "--out-file", str(out)],
        )
        main()

        result = pd.read_csv(out)
        assert len(result) == 5  # 2 + 3
        assert set(result["source"].unique()) == {"HRRR", "WTK"}

    def test_overwrite_flag(self, tmp_path, monkeypatch):
        _make_quantile_csv(tmp_path, "hrrr", 60, n_rows=2)
        out = tmp_path / "merged.csv"
        out.write_text("dummy")

        # Without --overwrite should fail
        monkeypatch.setattr(
            "sys.argv",
            ["merge_grid", "--in-dir", str(tmp_path), "--out-file", str(out)],
        )
        with pytest.raises(SystemExit):
            main()

        # With --overwrite should succeed
        monkeypatch.setattr(
            "sys.argv",
            ["merge_grid", "--in-dir", str(tmp_path), "--out-file", str(out),
             "--overwrite"],
        )
        main()
        result = pd.read_csv(out)
        assert len(result) == 2

    def test_missing_input_dir_exits(self, tmp_path, monkeypatch):
        out = tmp_path / "merged.csv"
        monkeypatch.setattr(
            "sys.argv",
            ["merge_grid", "--in-dir", str(tmp_path / "nonexistent"),
             "--out-file", str(out)],
        )
        with pytest.raises(SystemExit):
            main()

    def test_empty_dir_exits(self, tmp_path, monkeypatch):
        out = tmp_path / "merged.csv"
        monkeypatch.setattr(
            "sys.argv",
            ["merge_grid", "--in-dir", str(tmp_path), "--out-file", str(out)],
        )
        with pytest.raises(SystemExit):
            main()

    def test_source_col_custom_name(self, tmp_path, monkeypatch):
        _make_quantile_csv(tmp_path, "hrrr", 60, n_rows=2)
        out = tmp_path / "merged.csv"

        monkeypatch.setattr(
            "sys.argv",
            ["merge_grid", "--in-dir", str(tmp_path), "--out-file", str(out),
             "--source-col", "dataset"],
        )
        main()

        result = pd.read_csv(out)
        assert "dataset" in result.columns
        assert "source" not in result.columns

    def test_three_sources_merged(self, tmp_path, monkeypatch):
        _make_quantile_csv(tmp_path, "hrrr", 60, n_rows=2)
        _make_quantile_csv(tmp_path, "wtk", 60, n_rows=2)
        _make_quantile_csv(tmp_path, "wtk_led", 60, n_rows=2)
        out = tmp_path / "merged.csv"

        monkeypatch.setattr(
            "sys.argv",
            ["merge_grid", "--in-dir", str(tmp_path), "--out-file", str(out)],
        )
        main()

        result = pd.read_csv(out)
        assert len(result) == 6
        assert set(result["source"].unique()) == {"HRRR", "WTK", "WTK-LED"}
