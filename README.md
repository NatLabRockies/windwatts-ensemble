# WEM — WindWatts Ensemble Model

Machine learning bias-correction system that synthesizes multiple wind resource datasets to predict wind speed distributions across the continental United States.

## Overview

Numerical wind resource models — ERA5, HRRR, WTK (Wind Toolkit), and WTK-LED — each carry systematic biases when predicting site-level wind speeds. No single model is uniformly best across all locations, heights, and parts of the wind speed distribution.

WEM addresses this by training an XGBoost model on co-located wind observations from two station networks: high-quality Gold Standard (GS) sites and the much larger ASOS/ISD network. The model learns to correct biases across all input datasets simultaneously, producing an ensemble prediction that outperforms any individual wind resource model.

The pipeline outputs 101-point quantile wind speed CDFs — not just mean wind speeds — at six hub heights (30, 40, 50, 60, 80, 100 m) for ~42,000 ERA5 grid points across CONUS. Model performance is validated through leave-one-out cross-validation on Gold Standard sites with a 10 km spatial exclusion radius.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Quantile CDFs** | Wind speeds stored as 101 percentiles (q000–q100) representing the full distribution, not just a mean |
| **Two cohorts** | Gold Standard (285 station-height combinations across ~60 research-grade towers) and ASOS/ISD (~900+ stations, de-quantized from integer knots) |
| **Long format** | Training data is one row per (station_id, height_m, qnum) — not one row per station |
| **Hub heights** | Predictions at 30, 40, 50, 60, 80, 100 m above ground |
| **Wind resource models** | ERA5, HRRR, WTK, WTK-LED CONUS, WTK-LED Climate |
| **LOOCV** | Leave-one-out cross-validation by GS site, excluding all stations within 10 km |

## Package Structure

```
wem/
├── pyproject.toml          # Package config, 53 CLI entry points
├── PIPELINE.md             # Detailed run-order with all commands
├── README.md               # This file
├── wem/
│   ├── constants.py        # QCOLS, HEIGHTS, unit conversions
│   ├── acquire/            # Stage 1: Download & process observations
│   ├── extract/            # Stage 2: Extract wind resource model quantiles
│   ├── assemble/           # Stage 3: Merge, enrich, and prepare training data
│   ├── train/              # Stage 4: XGBoost LOOCV & final model
│   ├── analyze/            # Stage 5: Error metrics & diagnostic plots
│   ├── grid/               # Stage 6: Full-grid inference pipeline
│   ├── maps/               # Stage 7: Visualization & output formatting
│   ├── utils/              # Shared utilities (12 modules)
│   └── dev/                # Development & analysis scripts (not in main pipeline)
├── data/                   # Pipeline data (quantiles, GWA, grid, raw, references)
├── hpc/                    # Kestrel HPC infrastructure (Slurm wrappers, tile runners, grid data)
└── tests/                  # Unit and e2e regression tests
```

## Installation

Requires Python >= 3.10.

```bash
# Project venv used for local development/audit validation
uv venv --python 3.11 .venv
source scripts/activate-wem
uv pip install -e ".[viz,hpc,hpo]" pytest

# With visualization support (matplotlib, cartopy, shapely)
pip install -e ".[viz]"

# With HPC/HDF5 support (NREL-rex — for WTK/HRRR extraction on Kestrel)
pip install -e ".[hpc]"

# Full development install (all optional deps + pytest)
pip install -e ".[dev]"
```

### Core Dependencies

pandas, numpy, xarray, xgboost, scikit-learn, h5py, pyarrow, requests, pyproj, scipy, tqdm, rasterio, joblib, matplotlib

### Optional Dependency Groups

| Group | Packages | Purpose |
|-------|----------|---------|
| `viz` | cartopy, shapely | Map generation (cartopy projections and shapely geometries) |
| `hpc` | NREL-rex | HDF5 wind resource data access (Kestrel supercomputer) |
| `hpo` | optuna | Hyperparameter optimization |
| `dev` | All of the above + pytest | Full development environment |

## Pipeline Overview

The package exposes 48 production-stage entry points across 7 sequential stages, plus 5 experiment entry points. See [`PIPELINE.md`](PIPELINE.md) for the complete command reference.

```
┌─────────────┐    ┌────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Observations │───▶│ Extraction │───▶│ Assembly │───▶│ Training │───▶│ Analysis  │
│  (4 scripts) │    │(10 scripts)│    │(5 scripts)│    │(2 scripts)│    │(11 scripts)│
└─────────────┘    └────────────┘    └──────────┘    └────┬─────┘    └──────────┘
                                                          │
                                                          ▼
                                     ┌────────────────┐    ┌──────────────┐
                                     │ Grid Inference │───▶│ Maps & Output│
                                     │  (13 scripts)  │    │  (3 scripts) │
                                     └────────────────┘    └──────────────┘
```

### Stage 1: Observations (4 scripts)

Download ASOS/ISD hourly wind observations from NOAA (2007–2024), process raw data to 101-point quantile CDFs with ASOS de-quantization (integer knots to continuous distributions), and aggregate into a site matrix. Separately, convert Gold Standard time series to quantiles with month-balance filtering.

### Stage 2: Wind Resource Extraction (10 scripts)

Extract ERA5, HRRR, WTK, and WTK-LED quantiles at all observation locations. Two parallel tracks (GS and ASOS cohorts), each with 5 datasets. ERA5 must run first per cohort because it defines the retained site list; the remaining 4 sources can run in parallel.

### Stage 3: Dataset Assembly (5 scripts)

Merge GS and ASOS cohorts into a single long-format training table. Enrich with USGS 3DEP topography (elevation, slope, aspect), build 10 km neighbor exclusion lists for LOOCV, compute per-site mean wind speeds via quantile integration, and add Global Wind Atlas interpolated features.

### Stage 4: ML Training (2 scripts)

**LOOCV** (`wem-train-loocv`): Train XGBoost with leave-one-out cross-validation by Gold Standard site. Each fold excludes the target GS site plus all stations within 10 km. ASOS sites are downsampled to balance the GS contribution.

**Final model** (`wem-train-final`): Train on all data (no holdout) with the same balanced downsampling. Produces the model used for grid-wide inference. Outputs the trained model, feature importance, in-sample metrics, and metadata.

### Stage 5: Analysis (11 scripts)

Evaluate LOOCV predictions on Gold Standard sites. Compute per-site RMSE, MAE, bias, and R-squared. Generate bias maps, absolute bias maps, delta maps (ML vs. each wind resource model), boxplots, parity plots, feature importance visualizations, site CDFs, quantile comparison maps, and extended row-level metrics.

### Stage 6: Grid Inference (13 scripts)

Build the ERA5 grid point list (~42,000 points). Extract all wind resource quantiles at grid locations. Merge per-tile outputs into per-height CSVs. Add elevation and GWA features. Run the final XGBoost model. Pivot predictions from long format to one row per (grid point, height) with q000–q100 columns.

### Stage 7: Maps & Output (3 scripts)

Generate mean wind speed maps for all sources, difference maps showing where the ML ensemble diverges from individual models, and format predictions for web application delivery.

## Quick Start

Assuming training data has already been assembled (Stages 1–3 complete):

```bash
# Train LOOCV model (evaluation)
wem-train-loocv --infile data/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outfile data/models/loocv/ml_results.csv \
    --balance_strategy downsample --n_jobs_outer 12 \
    --gwa-file data/training/site_height_ws_avg_with_gwa.csv \
    --gwa-col gwa_interp --include-gwa

# Train final model (for inference)
wem-train-final --infile data/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --out-dir data/models/final/ --n_jobs_model 8 \
    --gwa-file data/training/site_height_ws_avg_with_gwa.csv \
    --gwa-col gwa_interp --include-gwa

# Analyze LOOCV results
wem-analyze --infile data/models/loocv/ml_results.csv \
    --outdir data/output/analysis/ --conus \
    --gwa data/training/site_height_ws_avg_with_gwa.csv

# Run inference on full grid
wem-infer --in data/grid/inference_table_with_gwa.csv \
    --model-dir data/models/final/ \
    --out data/grid/predictions_fullgrid.csv

# Pivot to wide format (one row per grid point + height)
wem-pivot-predictions --in data/grid/predictions_fullgrid.csv \
    --era5 data/grid/era5_grid.csv \
    --out data/grid/site_quantiles_predicted.csv --decimals 2
```

For the complete pipeline from raw data, see [`PIPELINE.md`](PIPELINE.md).

## ML Model Details

### Algorithm

XGBoost regression with MAE loss (`reg:absoluteerror`). Monotonic constraint (+1) on the `qnum` feature ensures that predicted wind speeds increase with quantile index, maintaining physical consistency of the output CDF.

### Features

| Feature | Description | Constraint |
|---------|-------------|------------|
| `qnum` | Quantile index (0–100) | Monotonic +1 |
| `hrrr` | HRRR wind speed quantile (m/s) | — |
| `wtk` | WTK wind speed quantile (m/s) | — |
| `wtk_led_conus` | WTK-LED CONUS wind speed quantile (m/s) | — |
| `lat` | Latitude (degrees) | — |
| `lon` | Longitude (degrees) | — |
| `height_m` | Hub height above ground (m) | — |
| `elevation_m` | Surface elevation (m, USGS 3DEP) | — |
| `gwa_interp` | Global Wind Atlas mean wind speed at hub height (m/s) | — |

Additional optional features (`slope_deg`, `aspect_sin`, `aspect_cos`) are available via `--aux_features` but are not used in the default production configuration. `era5` and `wtk_led_climate` are excluded from the production feature set; see METHODS.md §6 for rationale.

### Optimized Hyperparameters

```
learning_rate     0.022
max_depth         20
min_child_weight  4.28
subsample         0.61
colsample_bytree  0.98
n_estimators      500
tree_method       hist
```

### Training Strategy

- **LOOCV**: Leave-one-out by GS site; each fold excludes the target site and all stations within a 10 km radius
- **Balance**: Downsample ASOS indices to match GS count per fold (deterministic seed per station)
- **Final model**: Train on all data with the same downsampling strategy; no holdout
- **Target**: `observation` — observed wind speed (m/s) at each quantile

## Data Conventions

### Quantile Representation

Wind speeds are stored as 101 percentiles per station-height combination, not as time series. Columns `q000` through `q100` represent the 0th through 100th percentiles of the wind speed distribution. Mean wind speed is recovered by trapezoidal integration of the quantile function.

### Long Format Schema

The training table uses one row per (station_id, height_m, qnum):

```
station_id  height_m  qnum  observation  era5  hrrr  wtk  wtk_led_conus  wtk_led_climate  observation_type
S001        60        0     0.00         0.12  0.08  0.05 0.10           0.09             ASOS
S001        60        1     0.15         0.18  0.14  0.11 0.16           0.14             ASOS
...
S001        60        100   18.50        16.20 17.80 15.90 17.10         16.50            ASOS
```

### ASOS De-quantization

Raw ASOS wind speeds are quantized to integer knots. The pipeline reconstructs continuous distributions:
- **Calm** (0 kt): Sample from U(0, 2] kt
- **Reported k kt** (k >= 3): Sample from U(k-1, k] kt
- All values converted to m/s (1 kt = 0.514444 m/s)

### ERA5 Site Filter

ERA5 extraction defines the authoritative site list for each cohort. Only station-height combinations present in the ERA5 output proceed through the rest of the pipeline.

### Restart Safety

Most scripts check for existing output files and skip already-processed items. Interrupted runs can be resumed without reprocessing completed work.

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `NCEI_TOKEN` | NOAA NCEI API token for ISD data downloads | Required for Stage 1 |
| `ERA5_BATCH_SIZE` | Batch size for vectorized ERA5 interpolation | 200 |
| `ERA5_TIME_CHUNK` | Dask time-chunk size for ERA5 processing | 8928 |
| `WEM_CONDA_ENV` | Kestrel conda environment used by HPC wrappers | `/scratch/kmenear/windwatts/env` |

## CLI Reference

All 53 entry points, grouped by pipeline stage:

### Stage 1: Observations

| Command | Description |
|---------|-------------|
| `wem-gs-obs` | Convert gold standard time series to 101-point quantiles |
| `wem-download-isd` | Download ASOS/ISD hourly wind observations from NOAA |
| `wem-process-isd` | Process raw ISD files to per-station quantile CSVs |
| `wem-aggregate-asos` | Aggregate station quantile CSVs into a single matrix |

### Stage 2: Wind Resource Extraction

| Command | Description |
|---------|-------------|
| `wem-gs-era5` | Extract ERA5 quantiles for Gold Standard sites |
| `wem-gs-wtk` | Extract WTK quantiles for Gold Standard sites |
| `wem-gs-wtkled-conus` | Extract WTK-LED CONUS quantiles for Gold Standard sites |
| `wem-gs-wtkled-climate` | Extract WTK-LED Climate quantiles for Gold Standard sites |
| `wem-gs-hrrr` | Extract HRRR quantiles for Gold Standard sites |
| `wem-asos-era5` | Extract ERA5 quantiles for ASOS sites |
| `wem-asos-wtk` | Extract WTK quantiles for ASOS sites |
| `wem-asos-wtkled-conus` | Extract WTK-LED CONUS quantiles for ASOS sites |
| `wem-asos-wtkled-climate` | Extract WTK-LED Climate quantiles for ASOS sites |
| `wem-asos-hrrr` | Extract HRRR quantiles for ASOS sites |

### Stage 3: Dataset Assembly

| Command | Description |
|---------|-------------|
| `wem-merge-cohorts` | Merge GS and ASOS cohorts into one long-format table |
| `wem-add-topo` | Add USGS 3DEP elevation, slope, and aspect |
| `wem-build-neighbors` | Build 10 km neighbor exclusion lists for LOOCV |
| `wem-obs-wsavg` | Compute per-site mean wind speed from quantile CDFs |
| `wem-add-gwa` | Add Global Wind Atlas interpolated wind speeds |

### Stage 4: Training

| Command | Description |
|---------|-------------|
| `wem-train-loocv` | LOOCV XGBoost training for evaluation |
| `wem-train-final` | Final XGBoost model on all data for inference |

### Stage 5: Analysis

| Command | Description |
|---------|-------------|
| `wem-analyze` | Compute error metrics and generate diagnostic plots |
| `wem-analyze-extended` | Extended per-dataset error metrics |
| `wem-row-metrics` | Row-level prediction metrics |
| `wem-error-diffs` | Error difference plots (ML vs. each model) |
| `wem-site-cdfs` | Per-site CDF comparison plots |
| `wem-viz-fi` | Feature importance visualization |
| `wem-qc-filter` | QC outlier filtering analysis |
| `wem-nn-lookup` | Nearest-neighbor site lookup |
| `wem-grid-means` | Grid-scale mean wind speed predictions |
| `wem-quantile-maps` | Pre-ML quantile comparison maps |
| `wem-interannual` | Interannual Gold Standard wind-speed analysis |

### Stage 6: Grid Inference

| Command | Description |
|---------|-------------|
| `wem-build-grid` | Build ERA5 grid point list |
| `wem-grid-era5` | Extract ERA5 quantiles for all grid points |
| `wem-grid-hrrr` | Extract HRRR quantiles for all grid points |
| `wem-grid-wtk` | Extract WTK quantiles for all grid points |
| `wem-grid-wtkled` | Extract WTK-LED quantiles for all grid points |
| `wem-merge-tiles` | Merge per-tile outputs into per-height CSVs |
| `wem-merge-grid` | Merge all grid quantile sources |
| `wem-grid-elev` | Add USGS 3DEP elevation to grid points |
| `wem-fill-missing` | Fill missing elevation values |
| `wem-prepare-inference` | Align grid features with training schema |
| `wem-grid-gwa` | Add GWA features to inference table |
| `wem-infer` | Run XGBoost inference on full grid |
| `wem-pivot-predictions` | Pivot predictions to wide format (one row per grid point) |

### Stage 7: Maps & Output

| Command | Description |
|---------|-------------|
| `wem-maps-mean` | Generate mean wind speed maps for all sources |
| `wem-maps-diff` | Generate difference maps (ML vs. wind resource models) |
| `wem-app-format` | Format predictions for web application |

### Experiment Infrastructure

| Command | Description |
|---------|-------------|
| `wem-exp-hpo` | Optuna hyperparameter optimization |
| `wem-exp-param-sweep` | n_estimators and max_depth parameter sweeps |
| `wem-exp-feature-sweep` | Wind and auxiliary feature sweep experiments |
| `wem-exp-analyze-sweep` | Analyze feature sweep results |
| `wem-experiment` | Unified experiment runner |

## Output Files

The pipeline produces:

| File | Description |
|------|-------------|
| `data/models/final/xgb_model.json` | Trained XGBoost model (native format) |
| `data/models/final/feature_names.json` | Ordered feature list used by the model |
| `data/models/final/feature_importance.csv` | Feature importance (gain, weight, cover) |
| `data/models/final/metadata.json` | Training metadata (features, hyperparams, counts, versions) |
| `data/models/final/metrics_training.json` | In-sample RMSE, MAE, counts by observation type |
| `data/models/loocv/ml_results.csv` | LOOCV predictions (long format with `pred_observation`) |
| `data/grid/site_quantiles_predicted.csv` | Full-grid predictions (one row per grid point + height, q000–q100) |
| `data/output/analysis/site_metrics_gs.csv` | Per-GS-site RMSE, MAE, bias, R-squared |
| `data/output/analysis/*.png` | Bias maps, absolute bias maps, boxplots, parity plots |
| `data/output/maps/*.png` | Mean wind and difference maps |
| `data/output/app/` | Web-application-ready formatted output |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Skip tests requiring optional dependencies
pytest tests/ -v -m "not requires_cartopy and not requires_rasterio and not requires_rex and not requires_xgboost"
```

The unit suite uses synthetic data and does not require network access, large production data, or HPC access. The e2e suite uses large local artifacts that are intentionally ignored by git; run `bash tests/e2e/verify_artifacts.sh` before `bash tests/e2e/run_all.sh`.

### Test Markers

| Marker | Skips when missing |
|--------|--------------------|
| `requires_cartopy` | cartopy + shapely |
| `requires_rasterio` | rasterio |
| `requires_rex` | NREL-rex (HDF5 access) |
| `requires_xgboost` | xgboost |
| `requires_network` | Internet access |

Tests marked with optional-dependency markers are automatically skipped when the corresponding package is unavailable or broken in the active environment.

## HPC (Kestrel)

The `hpc/` directory contains all Kestrel supercomputer infrastructure for running the wind resource extraction scripts (Stages 2 and 6) at scale. All Slurm wrappers and parallel tile runners invoke WEM CLI commands directly (e.g., `wem-asos-hrrr`, `wem-gs-wtk`, `wem-grid-wtkled`).

HPC wrappers activate `${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}`. Set `WEM_CONDA_ENV` before submission to use a different Kestrel environment.

```
hpc/
├── slurm/extract/      # 8 Slurm wrappers for site extraction (ASOS + GS cohorts)
├── slurm/grid/         # 5 Slurm wrappers for grid extraction (single-job + tiled)
├── grid/               # Parallel tile runners, debug tools, grid config (era5_grid.csv)
└── grid_data/          # Extraction output: 450 tiles × 3 datasets + merged per-height CSVs
```

See [`hpc/HPC_INVENTORY.md`](hpc/HPC_INVENTORY.md) for the full directory inventory, wrapper-to-CLI-command mapping, and grid data details.

## Utility Modules

The `wem/utils/` package provides 12 shared modules used across the pipeline:

| Module | Key Functions | Purpose |
|--------|---------------|---------|
| `columns` | `choose_col`, `find_qcols` | Case-insensitive column matching, quantile column detection |
| `quantiles` | `quantile_block`, `mean_from_quantiles` | Compute 101-point CDFs, integrate to mean wind speed |
| `spatial` | `to_xy_lcc`, `to_webmercator`, `idw_weights_from_dd`, `pairwise_haversine_km` | Coordinate projections, IDW interpolation, distance matrices |
| `power_law` | `bracket_for_height`, `power_law_interp`, `fit_power_law_alpha` | Power-law wind speed extrapolation across heights |
| `wind` | `uv_from_ws_wd`, `gather_unique` | Wind vector decomposition, neighbor index deduplication |
| `ml` | `pick_present`, `make_features`, `build_neighbor_map`, `balance_indices`, `fold_seed` | Feature engineering, LOOCV neighbor maps, cohort balancing |
| `sites` | `load_sites`, `already_done`, `normalize_obs_type` | Site metadata I/O, restart safety, observation type normalization |
| `io` | `read_table`, `write_table` | CSV/Parquet dispatch based on file extension |
| `elevation` | `identify_point_3857`, `sample_elevation_points` | USGS 3DEP elevation queries with resolution fallback |
| `raster` | `sample_raster_points` | GeoTIFF point sampling (used for GWA rasters) |
| `plotting` | `make_custom_cmap`, `make_diff_cmap`, `wrap_lon180`, `mask_points_to_us`, `robust_limits`, `symmetric_bias_limit`, `setup_cartopy_axes` | Colormaps, longitude wrapping, CONUS masking, robust limits, cartopy basemap setup |
| `logging` | `log` | Timestamped `[HH:MM:SS]` logging to stderr |
