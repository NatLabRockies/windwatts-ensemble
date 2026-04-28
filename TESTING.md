# WEM Testing

This document describes the testing strategy for the WindWatts Ensemble Model (WEM) package. Testing is organized into two tiers: a comprehensive **unit test suite** covering individual functions and modules, and an **end-to-end (E2E) regression test** that exercises the complete pipeline from data assembly through grid-wide inference.

## Unit Tests

The unit suite covers pipeline modules with synthetic data and optional-dependency skips. Use the active project environment from `source scripts/activate-wem` for production validation.

### Running

```bash
# All tests
pytest tests/

# Single module
pytest tests/assemble/
pytest tests/train/

# Skip slow or dependency-gated tests
pytest -m "not slow"

# Verbose
pytest -v tests/
```

### Configuration

Defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: Long-running tests",
    "requires_cartopy: Needs cartopy + shapely",
    "requires_rasterio: Needs rasterio",
    "requires_rex: Needs NREL-rex for HDF5",
    "requires_xgboost: Needs xgboost",
    "requires_network: Needs internet",
]
```

Tests marked with `requires_*` are automatically skipped when the dependency is unavailable or broken in the active environment (handled by `conftest.py`).

### Shared Fixtures (`conftest.py`)

| Fixture | Description |
|---------|-------------|
| `qcols` | List of 101 quantile column names (`q000`..`q100`) |
| `rng` | Deterministic NumPy random generator (seed 42) |
| `synthetic_quantile_row` | Single-row DataFrame with `linspace(0, 10, 101)` |
| `synthetic_quantile_df` | Five-row DataFrame with different linear ramps |
| `synthetic_sites_df` | Ten synthetic CONUS sites with lat/lon/elev/name |
| `synthetic_long_training_df` | Long-format training data (3 stations x 2 heights x 101 qnums) |
| `synthetic_wide_quantile_df` | Wide-format DataFrame (3 sites + q000..q100) |

### Coverage by Module

| Module | File | Tests | What is Validated |
|--------|------|------:|-------------------|
| **acquire** | `test_process_isd.py` | 18 | `days_in_year`, `parse_speed_ms` (valid/NaN/malformed), `expected_per_day`, `years_with_enough_data` (pass/fail/boundary), `dequantize_ceil` (calm/nonzero/NaN/seeds), `make_quantiles` (shape/range/monotonic) |
| | `test_gold_standard_obs.py` | 13 | Gold standard observation parsing and quantile extraction |
| | `test_aggregate_asos.py` | 10 | ASOS observation aggregation and filtering |
| **assemble** | `test_merge_cohorts.py` | 8 | `to_long` shape, qnum range 0-100, meta columns, flexible column names, insufficient q-columns error, numeric coercion, NaN key dropping |
| | `test_build_neighbors.py` | 7 | Neighbor list construction within 10km radius |
| | `test_obs_wind_avg.py` | 9 | `mean_from_qnum_block` and `first_nonnull` helpers |
| **train** | `test_loocv_xgb.py` | 8 | `_interp_gwa_row` GWA interpolation across heights (tests moved to utils) |
| | `test_final_xgb.py` | 10 | `balance_indices` class balancing (via utils), training metrics computation |
| **analyze** | `test_ml_results.py` | 11 | Analysis output generation, site metrics, bias calculations |
| | `test_extended_metrics.py` | 6 | Extended per-dataset error metrics |
| | `test_row_metrics.py` | 6 | Row-level prediction metrics |
| | `test_error_diffs.py` | 6 | Error difference plots |
| | `test_site_cdfs.py` | 6 | Per-site CDF comparison plots |
| | `test_feature_importance.py` | 6 | Feature importance visualization |
| | `test_qc_outliers.py` | 6 | QC outlier filtering analysis |
| | `test_nn_lookup.py` | 7 | Nearest-neighbor site lookup |
| | `test_grid_means.py` | 4 | Grid-scale mean wind speed predictions |
| | `test_quantile_maps.py` | 22 | Quantile comparison maps |
| **grid** | `test_fill_missing.py` | 8 | `process_df` NaN/inf filling behavior |
| | `test_prepare_inference.py` | 11 | Wide-to-long melting and dataset pivoting |
| | `test_pivot_predictions.py` | 4 | Long-to-wide prediction pivoting |
| | `test_build_grid.py` | 8 | ERA5 grid ID construction |
| | `test_merge_tiles.py` | 15 | Per-tile output merging |
| | `test_merge_grid.py` | 20 | Grid quantile source merging |
| **experiment** | `test_helpers.py` | 41 | Experiment shared helper functions |
| | `test_analyze_sweep.py` | 12 | Feature sweep analysis |
| **maps** | `test_app_format.py` | 23 | Web application formatting |
| | `test_diff_maps.py` | 6 | Difference map generation |
| **utils** | `test_ml.py` | 22 | XGBoost helpers, feature selection, monotonic constraints |
| | `test_power_law.py` | 22 | Wind power law interpolation/extrapolation |
| | `test_spatial.py` | 19 | Coordinate transforms, haversine distance, Web Mercator |
| | `test_sites.py` | 24 | `load_sites` (flexible column names, dedup, NaN filtering), `already_done`, `normalize_obs_type` |
| | `test_columns.py` | 14 | Column name normalization and mapping |
| | `test_plotting.py` | 18 | Plot generation helpers |
| | `test_elevation.py` | 9 | Elevation sampling and 3DEP helpers |
| | `test_quantiles.py` | 20 | Quantile computation and validation |
| | `test_wind.py` | 9 | Wind speed unit conversion and de-quantization |
| | `test_io.py` | 6 | CSV/Parquet read/write roundtrips, extension dispatch |
| | `test_raster.py` | 5 | GeoTIFF raster sampling |
| | `test_logging.py` | 3 | Timestamped log formatting |
| | `test_constants.py` | 9 | Constants validation |

---

## End-to-End Regression Test

The E2E test exercises every pipeline stage that can run locally, using real data from the original workflow. It serves as a **regression baseline**: after it passes, individual stages can be refactored with confidence that the outputs remain identical.

### What it Tests

16 pipeline stages covering data assembly (Stages 3A-3E), ML training and evaluation (Stages 4A-4B, 5), full-grid inference (Stages 6H-6L), and app output formatting (Stage 7C). Stages that require HPC access (grid data extraction from ERA5/HRRR/WTK on Kestrel) are skipped; the test starts from pre-built grid intermediates for those stages.

**Note:** The Ozark MO Gold Standard site correction (formerly Stage 3F / `patch_ozark`) has been applied directly to the source data files. The separate patch step is no longer needed.

### Running

```bash
cd /path/to/wem

# Run everything (~20 minutes):
bash tests/e2e/run_all.sh

# Run a single stage:
bash tests/e2e/07_train_loocv.sh

# Re-run a failed stage after fixing:
bash tests/e2e/09_analyze.sh
```

Each script is self-contained: it sources `helpers.sh` for shared constants and validation functions, runs one pipeline command, validates outputs, and exits non-zero on failure. The master `run_all.sh` calls each script sequentially, stopping on first failure.

### Prerequisites

- `source scripts/activate-wem` or another environment with WEM installed
- `bash tests/e2e/verify_artifacts.sh` passes before running the suite
- Pipeline data populated under `data/` (large artifacts are gitignored and must be restored or symlinked locally):
  - `data/quantiles/` — 12 quantile CSVs (6 ASOS + 6 GS)
  - `data/gwa/` — 4 GWA raster TIFs
  - `data/grid/` — grid inference inputs
  - `data/reference/` — regression test reference files, including ignored `loocv/` and `grid/` references
- ~15 GB free disk for grid stage outputs

The required file list is tracked in `tests/e2e/required_artifacts.txt`; the verifier reports exact missing paths without modifying data.

### Validation Methods

The test suite uses four validation functions defined in `helpers.sh`:

| Function | Purpose | Example |
|----------|---------|---------|
| `check_file FILE` | File exists and is non-empty | `check_file "$OUT/model.json"` |
| `check_rows FILE N` | Line count equals expected (including header) | `check_rows "$OUT/data.csv" 221090` |
| `check_cols FILE "c1,c2"` | Named columns present in CSV header | `check_cols "$OUT/data.csv" "elevation_m,slope_deg"` |
| `diff_csv NEW REF "cols" [tol] [keys]` | Max absolute difference < tolerance after merging on key columns | `diff_csv "$OUT/data.csv" "$REF/data.csv" "ws_avg" "1e-6" "station_id,height_m"` |

The `diff_csv` function loads both CSVs in Python, performs an inner merge on key columns, and asserts `max(abs(new - ref)) < tolerance` for each specified numeric column.

### Stage Results

All stages produce outputs **numerically identical** to the reference files from the original workflow (max absolute difference = 0.00e+00 across all comparisons).

#### Stage 00: Setup (0s)

Creates the working directory `data/e2e/` and symlinks all input data from source directories. Verifies all symlinks resolve.

#### Stage 01: Merge Cohorts — Stage 3A (3s)

Merges 12 quantile CSVs (6 ASOS + 6 Gold Standard) into a single long-format training table.

| Validation | Expected | Result |
|------------|----------|--------|
| Row count | 221,090 (221,089 data + header) | PASS |
| Columns | `station_id, height_m, qnum, observation, era5, hrrr, wtk, wtk_led_conus, wtk_led_climate, observation_type` | PASS |

#### Stage 02: Add Topography — Stage 3B (2s)

Merges elevation, slope, and aspect from the reference topo file by `station_id`. This ensures deterministic regression testing — the live 3DEP API is tested by the production pipeline, not the e2e suite.

| Validation | Expected | Result |
|------------|----------|--------|
| Row count | 221,090 | PASS |
| Columns | `elevation_m, slope_deg, aspect_deg` | PASS |

#### Stage 03: Build Neighbors — Stage 3C (3s)

Computes LOOCV spatial exclusion lists: for each Gold Standard site, identifies all sites within 10 km.

| Validation | Expected | Result |
|------------|----------|--------|
| Row count | 221,090 | PASS |
| Columns | `neighbors_10km_site_ids, neighbors_10km_count` | PASS |

Result: 285 GS stations, 137 with at least one neighbor within 10 km.

#### Stage 04: Obs Wind Speed Average — Stage 3D (2s)

Computes observed mean wind speed per site-height from the quantile values.

| Validation | Expected | Result |
|------------|----------|--------|
| Columns | `station_id, height_m, ws_avg` | PASS |
| `ws_avg` vs reference (tol=1e-6, keys=`station_id,height_m`) | max diff < 1e-6 | PASS (0.00e+00) |

Output: 2,189 site-height combinations.

#### Stage 05: Add GWA — Stage 3E (6s)

Samples Global Wind Atlas rasters at 10/50/100/150 m and interpolates to each site's measurement height using a power-law profile.

| Validation | Expected | Result |
|------------|----------|--------|
| Columns | `gwa_10, gwa_50, gwa_100, gwa_150, gwa_interp` | PASS |
| `gwa_interp` vs reference (tol=1e-6, keys=`station_id,height_m`) | max diff < 1e-6 | PASS (0.00e+00) |

#### Stage 07: Train LOOCV — Stage 4A (551s)

Trains 285 XGBoost models using leave-one-out cross-validation by Gold Standard site, with 10 km spatial exclusion and ASOS downsampling for class balance.

| Validation | Expected | Result |
|------------|----------|--------|
| Column | `pred_observation` | PASS |
| `pred_observation` vs reference (tol=1e-6) | max diff < 1e-6 | PASS (0.00e+00) |

Performance metrics (identical to original workflow):
- **RMSE = 0.8713 m/s**
- **MAE = 0.5239 m/s**
- N = 35,047 GS predictions across 285 folds

Features used (9): `qnum, hrrr, wtk, wtk_led_conus, lat, lon, height_m, elevation_m, gwa_interp`

#### Stage 08: Train Final Model — Stage 4B (10s)

Trains one XGBoost model on all balanced data (34,643 GS + 34,643 downsampled ASOS = 69,286 rows) for grid-wide inference.

| Validation | Expected | Result |
|------------|----------|--------|
| `xgb_model.json` exists | non-empty | PASS |
| `feature_names.json` exists | non-empty | PASS |
| `metrics_training.json` exists | non-empty | PASS |

In-sample training metrics: RMSE = 0.840 m/s, MAE = 0.178 m/s (N = 221,089).

#### Stage 09: Analyze — Stage 5 (7s)

Generates analysis outputs: bias boxplots, scatter plots, site-level metrics, and comparison maps.

| Validation | Expected | Result |
|------------|----------|--------|
| `bias_boxplots.png` exists | non-empty | PASS |
| `abs_bias_boxplots.png` exists | non-empty | PASS |
| `mean_pred_observation` vs reference (tol=1e-6, keys=`station_id,height_m`) | max diff < 1e-6 | PASS (0.00e+00) |
| `bias_pred_observation` vs reference (tol=1e-6, keys=`station_id,height_m`) | max diff < 1e-6 | PASS (0.00e+00) |

Mean absolute bias over 342 GS site-height combinations:

| Dataset | Mean |Bias| (m/s) | Median |Bias| (m/s) |
|---------|----------------------|------------------------|
| **WEM** | **0.436** | **0.360** |
| GWA | 0.629 | 0.483 |
| WTK | 0.691 | 0.621 |
| HRRR | 0.732 | 0.662 |
| ERA5 | 0.990 | 0.740 |
| WTK-LED Climate | 1.093 | 0.912 |
| WTK-LED CONUS | 1.207 | 1.009 |

#### Stage 10: Fill Missing — Stage 6H (65s)

Replaces NaN and infinite values with 0 in the merged grid quantile table.

| Validation | Expected | Result |
|------------|----------|--------|
| `elevation_m` vs reference (tol=1e-6, keys=`grid_id,height_m`) | max diff < 1e-6 | PASS (0.00e+00) |

Input: 1.43 GB wide-format CSV. Output: 1.44 GB.

#### Stage 11: Prepare Inference — Stage 6I (144s)

Transforms the wide-format grid table (q000-q100 columns per dataset) into long format suitable for XGBoost inference.

| Validation | Expected | Result |
|------------|----------|--------|
| `hrrr, wtk, wtk_led_conus` vs reference (tol=1e-6, keys=`lat,lon,height_m,qnum`) | max diff < 1e-6 | PASS (0.00e+00) |

Output: 25,679,250 rows (long format).

#### Stage 12: Add Grid GWA — Stage 6J (134s)

Samples Global Wind Atlas rasters for all 42,375 unique grid locations across 6 heights and interpolates to each height.

| Validation | Expected | Result |
|------------|----------|--------|
| `gwa_interp` vs reference (tol=1e-6, keys=`lat,lon,height_m,qnum`) | max diff < 1e-6 | PASS (0.00e+00) |

42,375 unique coordinates x 6 heights = 254,250 unique (lat, lon, height) triples.

#### Stage 13: Infer — Stage 6K (143s)

Runs the final XGBoost model on the full inference table (25.7M rows) in batches of 500K.

| Validation | Expected | Result |
|------------|----------|--------|
| `pred_observation` column exists | present | PASS |
| Output file | non-empty | PASS |

Output: 25,679,250 predictions.

#### Stage 14: Pivot Predictions — Stage 6L (35s)

Pivots the long-format predictions back to wide format (one row per grid site-height, q000-q100 columns) and merges ERA5 location metadata.

| Validation | Expected | Result |
|------------|----------|--------|
| Output file | non-empty | PASS |

Output: 254,250 rows (site-height combinations).

#### Stage 15: App Format — Stage 7C (<30s)

Converts a subset (first 1,000 rows) of the Stage 14 wide-format predictions into per-location gzipped CSV files with wind speed profiles at 6 heights (30-100m), suitable for web app delivery.

| Validation | Expected | Result |
|------------|----------|--------|
| `location_index.csv.gz` exists | non-empty | PASS |
| Location `.csv.gz` files created | at least 1 | PASS |
| Columns | `probability, windspeed_30m, ..., windspeed_100m` | PASS |
| Row count per file | 102 (header + 101 quantiles) | PASS |

### Timing Summary

| Stage | Description | Time |
|-------|-------------|-----:|
| 00 | Setup | 0s |
| 01 | Merge Cohorts | 2s |
| 02 | Add Topography | 2s |
| 03 | Build Neighbors | 2s |
| 04 | Obs Wind Speed Avg | 1s |
| 05 | Add GWA | 4s |
| 06 | Merge Tiles | <1s |
| 07 | Train LOOCV | 551s |
| 08 | Train Final | 10s |
| 09 | Analyze | 5s |
| 10 | Fill Missing | 69s |
| 11 | Prepare Inference | 162s |
| 12 | Add Grid GWA | 134s |
| 13 | Infer | 147s |
| 14 | Pivot Predictions | 38s |
| 15 | App Format | <30s |
| | **Total** | **~20 min** |

### Stages Not Covered

The following stages require HPC (Kestrel) or external data sources not available locally:

| Stage | Description | Reason Skipped |
|-------|-------------|----------------|
| 1-2 | Raw data acquisition (ISD download, ASOS processing) | Requires NOAA API access + long download times |
| 6A | Build ERA5 grid | Needs ERA5 grid quantiles source |
| 6B-E | Grid data extraction (ERA5, HRRR, WTK, WTK-LED) | Requires HPC with HDF5 datasets |
| 6F | Merge grid datasets | Needs per-height grid CSVs from extraction |
| 6G | Add grid elevation | Needs 3DEP API for ~42K points (~30 min) |
| 7A-B | Maps (mean wind, difference) | Needs cartopy rendering + per-height grid data |

Grid stages 6H-6L use pre-built intermediates from the original workflow as their starting point.
