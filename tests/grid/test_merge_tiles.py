"""Tests for wem.grid.merge_tiles."""

import numpy as np
import pandas as pd
import pytest

from wem.constants import QCOLS
from wem.grid.merge_tiles import KEEP, list_input_files, main, read_tile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tile_csv(path, grid_ids=("G001", "G002"), heights=(60, 100),
                   extra_cols=None, drop_qcols=None):
    """Write a synthetic tile CSV and return the path."""
    rows = []
    for gid in grid_ids:
        for h in heights:
            row = {"grid_id": gid, "lat": 40.0, "lon": -100.0, "height_m": h}
            for i, qc in enumerate(QCOLS):
                if drop_qcols and qc in drop_qcols:
                    continue
                row[qc] = round(i * 0.1, 2)
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _make_tile_parquet(path, grid_ids=("G001",), heights=(60,)):
    """Write a synthetic tile Parquet file and return the path."""
    rows = []
    for gid in grid_ids:
        for h in heights:
            row = {"grid_id": gid, "lat": 40.0, "lon": -100.0, "height_m": h}
            for i, qc in enumerate(QCOLS):
                row[qc] = round(i * 0.1, 2)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# list_input_files
# ---------------------------------------------------------------------------

class TestListInputFiles:
    def test_finds_csv(self, tmp_path):
        (tmp_path / "tile_001.csv").write_text("a,b\n1,2\n")
        (tmp_path / "tile_002.csv").write_text("a,b\n3,4\n")
        (tmp_path / "notes.txt").write_text("ignore me")
        result = list_input_files(tmp_path)
        assert len(result) == 2
        assert all(f.suffix == ".csv" for f in result)

    def test_finds_parquet(self, tmp_path):
        _make_tile_parquet(tmp_path / "tile_001.parquet")
        _make_tile_parquet(tmp_path / "tile_002.pq")
        result = list_input_files(tmp_path)
        assert len(result) == 2

    def test_empty_dir(self, tmp_path):
        assert list_input_files(tmp_path) == []


# ---------------------------------------------------------------------------
# read_tile
# ---------------------------------------------------------------------------

class TestReadTile:
    def test_csv(self, tmp_path):
        fp = _make_tile_csv(tmp_path / "tile.csv")
        df = read_tile(fp)
        assert list(df.columns) == KEEP
        assert len(df) == 4  # 2 grids × 2 heights
        assert df["grid_id"].dtype == "string"

    def test_parquet(self, tmp_path):
        fp = _make_tile_parquet(tmp_path / "tile.parquet")
        df = read_tile(fp)
        assert list(df.columns) == KEEP
        assert len(df) == 1

    def test_fills_missing_qcols(self, tmp_path):
        fp = _make_tile_csv(tmp_path / "tile.csv", drop_qcols={"q099", "q100"})
        df = read_tile(fp)
        assert "q099" in df.columns
        assert "q100" in df.columns
        assert df["q099"].isna().all()
        assert df["q100"].isna().all()

    def test_drops_extra_columns(self, tmp_path):
        fp = _make_tile_csv(
            tmp_path / "tile.csv",
            extra_cols={"dataset": "wtk", "processed_utc": "2025-01-01"},
        )
        df = read_tile(fp)
        assert "dataset" not in df.columns
        assert "processed_utc" not in df.columns

    def test_missing_required_col(self, tmp_path):
        fp = tmp_path / "bad.csv"
        pd.DataFrame({"lat": [1], "lon": [2], "height_m": [60]}).to_csv(fp, index=False)
        with pytest.raises(ValueError, match="missing required columns.*grid_id"):
            read_tile(fp)

    def test_drops_nan_coords(self, tmp_path):
        fp = tmp_path / "tile.csv"
        rows = [
            {"grid_id": "G1", "lat": 40.0, "lon": -100.0, "height_m": 60},
            {"grid_id": "G2", "lat": np.nan, "lon": -100.0, "height_m": 60},
            {"grid_id": "G3", "lat": 40.0, "lon": np.nan, "height_m": 60},
            {"grid_id": "G4", "lat": 40.0, "lon": -100.0, "height_m": np.nan},
        ]
        for row in rows:
            for qc in QCOLS:
                row[qc] = 0.0
        pd.DataFrame(rows).to_csv(fp, index=False)
        df = read_tile(fp)
        assert len(df) == 1
        assert df["grid_id"].iloc[0] == "G1"


# ---------------------------------------------------------------------------
# main (end-to-end CLI)
# ---------------------------------------------------------------------------

class TestMergeTilesMain:
    def _run(self, args: list[str], monkeypatch):
        """Run main() with the given CLI args."""
        monkeypatch.setattr("sys.argv", ["wem-merge-tiles"] + args)
        main()

    def test_basic(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "tiles"
        in_dir.mkdir()
        out_dir = tmp_path / "merged"
        _make_tile_csv(in_dir / "tile_001.csv", grid_ids=("G1",), heights=(60, 100))
        _make_tile_csv(in_dir / "tile_002.csv", grid_ids=("G2",), heights=(60, 100))
        _make_tile_csv(in_dir / "tile_003.csv", grid_ids=("G3",), heights=(60, 100))

        self._run([
            "--in-dir", str(in_dir),
            "--out-dir", str(out_dir),
            "--prefix", "test",
        ], monkeypatch)

        f60 = out_dir / "test_quantiles_60m.csv"
        f100 = out_dir / "test_quantiles_100m.csv"
        assert f60.exists()
        assert f100.exists()
        df60 = pd.read_csv(f60)
        df100 = pd.read_csv(f100)
        assert len(df60) == 3  # G1, G2, G3
        assert len(df100) == 3

    def test_height_filter(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "tiles"
        in_dir.mkdir()
        out_dir = tmp_path / "merged"
        _make_tile_csv(in_dir / "tile.csv", heights=(30, 60, 100))

        self._run([
            "--in-dir", str(in_dir),
            "--out-dir", str(out_dir),
            "--prefix", "test",
            "--heights", "60,100",
        ], monkeypatch)

        assert not (out_dir / "test_quantiles_30m.csv").exists()
        assert (out_dir / "test_quantiles_60m.csv").exists()
        assert (out_dir / "test_quantiles_100m.csv").exists()

    def test_dedupe(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "tiles"
        in_dir.mkdir()
        out_dir = tmp_path / "merged"
        # Two tiles with the same grid_id and height — duplicates
        _make_tile_csv(in_dir / "tile_001.csv", grid_ids=("G1",), heights=(60,))
        _make_tile_csv(in_dir / "tile_002.csv", grid_ids=("G1",), heights=(60,))

        self._run([
            "--in-dir", str(in_dir),
            "--out-dir", str(out_dir),
            "--prefix", "test",
            "--dedupe",
        ], monkeypatch)

        df = pd.read_csv(out_dir / "test_quantiles_60m.csv")
        # Dedupe happens per-tile-block, not globally — each tile dedupes
        # within its own block, but cross-tile duplicates may remain
        # (matches the original script behavior)
        assert len(df) == 2

    def test_overwrite(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "tiles"
        in_dir.mkdir()
        out_dir = tmp_path / "merged"
        out_dir.mkdir()
        _make_tile_csv(in_dir / "tile.csv", grid_ids=("G1",), heights=(60,))

        # First run
        self._run([
            "--in-dir", str(in_dir),
            "--out-dir", str(out_dir),
            "--prefix", "test",
        ], monkeypatch)
        df1 = pd.read_csv(out_dir / "test_quantiles_60m.csv")

        # Second run without overwrite — appends
        self._run([
            "--in-dir", str(in_dir),
            "--out-dir", str(out_dir),
            "--prefix", "test",
        ], monkeypatch)
        df2 = pd.read_csv(out_dir / "test_quantiles_60m.csv")
        assert len(df2) == 2 * len(df1)

        # Third run with overwrite — replaces
        self._run([
            "--in-dir", str(in_dir),
            "--out-dir", str(out_dir),
            "--prefix", "test",
            "--overwrite",
        ], monkeypatch)
        df3 = pd.read_csv(out_dir / "test_quantiles_60m.csv")
        assert len(df3) == len(df1)

    def test_prefix(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "tiles"
        in_dir.mkdir()
        out_dir = tmp_path / "merged"
        _make_tile_csv(in_dir / "tile.csv", heights=(60,))

        self._run([
            "--in-dir", str(in_dir),
            "--out-dir", str(out_dir),
            "--prefix", "hrrr",
        ], monkeypatch)

        assert (out_dir / "hrrr_quantiles_60m.csv").exists()

    def test_no_input_files(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "empty"
        in_dir.mkdir()
        out_dir = tmp_path / "merged"

        with pytest.raises(SystemExit, match="No input files"):
            self._run([
                "--in-dir", str(in_dir),
                "--out-dir", str(out_dir),
                "--prefix", "test",
            ], monkeypatch)
