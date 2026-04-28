# WEM Methodology

## 1. Introduction

Individual wind resource models — ERA5, HRRR, WTK, WTK-LED — carry systematic biases that vary by location, height, and wind regime. No single model is uniformly best across CONUS. The Wind Ensemble Model (WEM) addresses this by learning a mapping from multiple model predictions and auxiliary site characteristics to observed wind speeds, correcting biases across the full wind speed distribution. Rather than selecting a single "best" model or applying uniform correction factors, WEM trains an ensemble machine learning model that adapts its bias correction to the local context defined by the input features. The result is a set of corrected wind speed quantiles at every ERA5 grid point, suitable for wind resource assessment at hub heights from 30 to 100 meters.

## 2. Wind Speed Representation

### Why Quantile CDFs

Wind speed distributions are non-Gaussian: bounded below by zero, typically right-skewed, and sometimes bimodal (e.g., sites with diurnal or seasonal wind regime shifts). Summary statistics such as mean wind speed or Weibull shape and scale parameters lose distributional information that matters for resource assessment. WEM instead represents each wind speed distribution as a full empirical CDF stored as 101 percentiles (q000 through q100) at each station-height combination.

Quantile computation uses `np.nanpercentile(spd, q=range(101))` with linear interpolation over the available time series at each site. This produces a vector **Q** = [Q(0), Q(0.01), ..., Q(1.00)] where Q(p) is the wind speed at cumulative probability p.

### Mean Recovery

The mean wind speed is recoverable from quantiles via trapezoidal integration of the quantile function:

> **μ = ∫₀¹ Q(p) dp ≈ Σᵢ 0.5 · (Q(pᵢ₋₁) + Q(pᵢ)) · Δp**

with Δp = 0.01 and 101 quantile values, yielding 100 trapezoids. This allows WEM to predict full distributions while retaining the ability to compute mean wind speeds for downstream applications.

The quantile representation enables per-quantile bias correction — the model can learn that a given dataset overpredicts calms (low quantiles) while underpredicting extremes (high quantiles), or vice versa.

**Reference:** `wem/utils/quantiles.py` (quantile_block, mean_from_quantiles)

## 3. Observation Data

WEM draws on two complementary station networks that together provide broad spatial coverage and high-quality validation targets.

### ASOS/ISD

Approximately 900+ Automated Surface Observing System stations from NOAA's Integrated Surface Database, reporting at 10 m height over the period 2007–2024. These airport weather stations provide dense spatial coverage across CONUS but have a known limitation: wind observations are quantized to integer knots. WEM applies dequantization via a ceil-model approach before computing quantiles: calm reports (0 kt) are replaced by samples from U(0, 2] kt, and non-calm reports of k kt (k ≥ 3) are replaced by samples from U(k−1, k] kt. This reconstructs continuous distributions from the quantized observations.

### Gold Standard

285 station-height combinations across approximately 60 research-grade meteorological towers from the NREL wind resource validation dataset, with measurements at multiple heights spanning 10–120 m. These sites have higher-quality instrumentation and multi-height profiles, making them the primary evaluation ground truth for model validation.

The two cohorts are distinguished by an `observation_type` field carried through every pipeline stage, ensuring that training, validation, and analysis can stratify by data source.

**Reference:** `wem/acquire/` (process_isd.py, gold_standard_obs.py)

## 4. Wind Resource Datasets

Five gridded wind resource datasets provide the input features for bias correction:

| Dataset | Source | Resolution | Period | Heights |
|---------|--------|------------|--------|---------|
| ERA5 | ECMWF | 0.25° (~28 km) | 2007–2024 | 10 m, 100 m → power-law interpolation |
| HRRR | NOAA | 3 km | 2015–2022 | Multiple native levels |
| WTK | NREL | ~2 km | 2007–2013 | Multiple hub heights |
| WTK-LED CONUS | NREL | ~2 km | 2018–2020 | Multiple hub heights |
| WTK-LED Climate | NREL | ~2 km | 2007–2020 | Multiple hub heights |

In addition, the Global Wind Atlas (GWA, DTU, ~250 m resolution, climatological mean at 10, 50, 100, 150 m) serves as an auxiliary feature rather than a primary wind resource input.

ERA5 serves a dual role: it provides wind speed quantiles as a potential feature, and its grid defines the spatial domain. Only site-height combinations present in ERA5 are retained, making ERA5 coverage the binding constraint on the training table. Quantiles are extracted from each model's time series at observation station locations, then merged into a single training record per station-height-quantile combination.

**Reference:** `wem/extract/` (10 modules: 2 per dataset × 2 cohorts)

## 5. Feature Engineering & Data Assembly

### Long-Format Representation

The merged training table stores one row per (station_id, height_m, qnum) combination. Each row contains:

- **Target:** `observation` — the observed wind speed at this quantile index
- **Wind resource features:** `era5`, `hrrr`, `wtk`, `wtk_led_conus`, `wtk_led_climate` — each model's wind speed at the same quantile index
- **`qnum`** (0–100) — the quantile index, treated as both a structural element (defining the CDF position) and a model input feature (allowing quantile-dependent corrections)

This long-format design means each station-height combination contributes 101 rows (one per quantile), and the model learns a single function that maps (quantile index, model predictions, auxiliary features) → corrected wind speed. The alternative — training 101 separate per-quantile models — would sacrifice the ability to share information across quantiles and require monotonicity enforcement as a post-processing step.

### Auxiliary Features

Beyond wind resource predictions, WEM incorporates site-level features:

- **Latitude, longitude** — spatial anchoring for regional bias patterns
- **`height_m`** — measurement/prediction height above ground
- **`elevation_m`** — terrain elevation from USGS 3DEP at 10 m resolution
- **`gwa_interp`** — Global Wind Atlas climatological mean wind speed, interpolated to the station height via power-law fit

### GWA Integration

GWA provides an independent estimate of mean wind speed at ~250 m resolution across four reference heights (10, 50, 100, 150 m). To obtain a value at an arbitrary station height, WEM fits a power-law profile ln(U) = ln(A) + α·ln(z) across the four GWA heights and evaluates at the target height. GWA is treated as an optional feature: XGBoost's native missing-value handling routes samples without GWA coverage through alternative tree splits.

**Reference:** `wem/assemble/` (merge_cohorts.py, add_topography.py, add_gwa.py), `wem/utils/power_law.py`

## 6. Model Formulation

### Algorithm

WEM uses XGBoost gradient-boosted trees with `reg:absoluteerror` (MAE) loss. MAE is preferred over MSE for robustness to wind speed outliers — extreme quantiles can have large residuals, and L1 loss prevents these from dominating the gradient signal.

### Monotonic Constraint

A +1 monotonic constraint on `qnum` ensures that model predictions are non-decreasing with quantile index. This physically enforces valid CDFs: the predicted wind speed at the 90th percentile must be greater than or equal to the predicted wind speed at the 50th percentile. All other features are left unconstrained.

### Hyperparameters

Hyperparameters were selected via Optuna TPE (Tree-structured Parzen Estimator) optimization:

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.022 |
| max_depth | 20 |
| n_estimators | 500 |
| subsample | 0.61 |
| colsample_bytree | 0.98 |
| min_child_weight | 4.28 |

The combination of deep trees (max_depth = 20) with a low learning rate (0.022) allows complex feature interactions — such as location-dependent, height-dependent, and quantile-dependent bias corrections — while limiting overfitting through the regularizing effect of averaging many weak learners across 500 boosting rounds.

### Feature Set

The production configuration uses 9 features:

- **Always included:** `qnum`
- **Wind resources:** `hrrr`, `wtk`, `wtk_led_conus`
- **Auxiliary:** `lat`, `lon`, `height_m`, `elevation_m`, `gwa_interp`

### Excluded Wind Resources

ERA5 and WTK-LED Climate are excluded from the production feature set based on an exhaustive wind feature sweep evaluating all 31 non-empty subsets of the five wind resource datasets via LOOCV RMSE. Two factors drive their exclusion:

1. **Collinearity.** ERA5 defines the spatial domain — every training row lies on the ERA5 grid by construction — making its quantiles highly collinear with the remaining sources and contributing near-zero marginal predictive gain. WTK-LED Climate is strongly correlated with WTK-LED CONUS (same underlying model, overlapping temporal period).

2. **Inference availability.** Neither ERA5 nor WTK-LED Climate columns are present in the full-grid inference table (which contains only `wtk`, `hrrr`, `wtk_led_conus`). Including them in training would produce a model inapplicable to grid-wide prediction.

### Balance Downsampling

ASOS stations outnumber Gold Standard stations approximately 15:1. To prevent airport-station characteristics from dominating the learned corrections, training data is balanced by randomly downsampling the majority class (ASOS) to match the minority class (GS) count. In LOOCV, each fold uses a different random subsample seeded deterministically per fold (see Section 7).

**Reference:** `wem/train/loocv_xgb.py`, `wem/train/final_xgb.py`, `wem/constants.py` (DEFAULT_XGB_PARAMS), `wem/utils/ml.py` (balance_indices)

## 7. Validation Strategy

### Leave-One-Out Cross-Validation with Spatial Exclusion

Model evaluation is performed exclusively on Gold Standard sites. For each of the 285 GS station-height combinations:

1. Remove the target station's data from the training set
2. Remove all stations (both ASOS and GS) within a 10 km radius of the target
3. Balance the remaining data via majority-class downsampling
4. Train a model and predict the held-out station's 101 quantiles

### Spatial Exclusion Rationale

Wind resource characteristics exhibit spatial autocorrelation — nearby stations share similar wind climates, terrain exposure, and model biases. Without spatial exclusion, a model could achieve low validation error by memorizing local patterns from neighboring stations rather than learning generalizable bias-correction relationships. The 10 km exclusion radius provides a conservative buffer ensuring the model is tested on its ability to correct biases at locations where no nearby observations exist, which is precisely the production use case (grid inference at arbitrary CONUS locations). Neighbor distances are computed via pairwise Haversine formula.

### Deterministic Reproducibility

Each LOOCV fold uses a seed derived from `SHA1(station_id + str(base_seed))`, truncated to 32 bits. This guarantees identical balance-downsampling draws regardless of fold execution order or parallelism level.

### Performance Results

LOOCV performance on Gold Standard sites (N = 35,047 predictions across 285 folds):

| Metric | Value |
|--------|-------|
| RMSE | 0.87 m/s |
| MAE | 0.52 m/s |

Comparison of mean absolute bias at Gold Standard site-height combinations:

| Dataset | Mean |Bias| (m/s) | Median |Bias| (m/s) |
|---------|----------------------|------------------------|
| **WEM** | **0.44** | **0.36** |
| GWA | 0.63 | 0.48 |
| WTK | 0.69 | 0.62 |
| HRRR | 0.73 | 0.66 |
| ERA5 | 0.99 | 0.74 |
| WTK-LED Climate | 1.09 | 0.91 |
| WTK-LED CONUS | 1.21 | 1.01 |

WEM reduces mean absolute bias by 30% relative to the best-performing individual dataset (GWA) and by over 50% relative to ERA5. Notably, WEM outperforms every input dataset, including those it does not use as features (ERA5, WTK-LED Climate), confirming that the ensemble learns genuine bias-correction structure rather than simply averaging its inputs.

**Reference:** `wem/assemble/build_neighbors.py`, `wem/train/loocv_xgb.py`, `wem/utils/ml.py` (build_neighbor_map, fold_seed)

## 8. Grid Inference & Output

### Production Model

A single XGBoost model is trained on all balanced data (no holdout) using the same architecture, hyperparameters, and feature set validated through LOOCV. The balanced training set contains 69,286 rows (34,643 GS + 34,643 downsampled ASOS).

### Grid Coverage

The inference domain spans 42,375 ERA5 grid points across CONUS at 6 hub heights (30, 40, 50, 60, 80, 100 m), yielding 254,250 site-height combinations × 101 quantiles = 25.7 million predictions.

### Vertical Interpolation

Where model predictions are not available at all six target heights, power-law extrapolation fills gaps:

> **U(z) = U(z_ref) · (z / z_ref)^α**

where α is derived from bracketing heights when two or more predictions are available, or defaults to 1/7 (the neutral-stability approximation) when only a single height is available. The shear exponent is computed as α = ln(U₂/U₁) / ln(h₂/h₁) from the nearest available height pair.

### Output Format

Final outputs are per-location gzipped CSV files with columns: `probability, windspeed_30m, ..., windspeed_100m`. The probability column runs from 0.00 to 1.00 in steps of 0.01 (101 rows per file). An `enforce_monotonic()` post-processing step applies `np.maximum.accumulate()` along each height column to guarantee valid CDFs after vertical interpolation, correcting any minor non-monotonicity introduced by the power-law step.

**Reference:** `wem/grid/` (14 modules), `wem/maps/app_format.py`, `wem/utils/power_law.py`
