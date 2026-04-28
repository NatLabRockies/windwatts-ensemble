"""Shared constants for the WEM pipeline."""

from __future__ import annotations

import numpy as np

# 101-point quantile column names (q000 through q100)
QCOLS = [f"q{p:03d}" for p in range(101)]

# Target heights in meters for grid-level inference
HEIGHTS = [30, 40, 50, 60, 80, 100]

# Wind speed conversion factor
KNOT_TO_MS = 0.514444

# Available heights in each wind resource dataset (meters)
WTK_HEIGHTS = np.array([10, 40, 60, 80, 100, 120, 140, 160, 200], dtype=int)
HRRR_HEIGHTS = np.array([10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 500, 1000], dtype=int)
WTKLED_HEIGHTS = np.array([10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 500, 1000], dtype=int)

# Display labels for dataset columns (used in plots and tables)
DATASET_LABELS: dict[str, str] = {
    "era5": "ERA5",
    "wtk": "WTK",
    "hrrr": "HRRR",
    "wtk_led_conus": "WTK-LED CONUS",
    "wtk_led_climate": "WTK-LED Climate",
    "bchrrr": "BC-HRRR",
    "gwa": "GWA",
    "gwa_interp": "GWA",
    "pred_observation": "WEM",
}

# LaTeX macro names for dataset columns (used in paper tables)
DATASET_LATEX: dict[str, str] = {
    "era5": r"\era",
    "wtk": r"\wtk",
    "wtk_led_climate": r"\wtkledclim",
    "wtk_led_conus": r"\wtkledconus",
    "hrrr": r"\hrrr",
    "bchrrr": r"\bchrrr",
    "gwa": r"\gwa",
    "pred_observation": r"\Method{}~(final)",
}

# Pretty display names for feature columns (used in feature importance plots)
FEATURE_DISPLAY_MAP: dict[str, str] = {
    "wtk_led_conus": "WTK-LED CONUS",
    "wtk_led_climate": "WTK-LED Climate",
    "gwa_interp": "GWA",
    "height_m": "Height (m)",
    "elevation_m": "Elevation (m)",
    "lat": "Latitude",
    "lon": "Longitude",
    "hrrr": "HRRR",
    "wtk": "WTK",
    "era5": "ERA5",
    "qnum": "Quantile index (qnum)",
    "slope_deg": "Slope (deg)",
    "aspect_deg": "Aspect (deg)",
}

# Wind resource feature names used in feature-sweep experiments
WIND_FEATURES: list[str] = ["era5", "hrrr", "wtk", "wtk_led_conus", "wtk_led_climate"]

# Auxiliary feature group tokens used in aux-sweep experiments
AUX_GROUPS: list[str] = ["latlon", "height", "elevation", "slope", "aspect"]

# Maps CLI wind-source flag names to the column prefix in the training table
WIND_FEATURE_MAP: dict[str, str] = {
    "era5": "era5",
    "hrrr": "hrrr",
    "wtk": "wtk",
    "wtk_led_conus": "wtk_led_conus",
    "wtk_led_climate": "wtk_led_climate",
}

# Maps auxiliary feature group names to their constituent column names
AUX_FEATURE_MAP: dict[str, list[str]] = {
    "latlon": ["lat", "lon"],
    "height": ["height_m"],
    "elevation": ["elevation_m"],
    "slope": ["slope_deg"],
    "aspect": ["aspect_sin", "aspect_cos"],
}

# Optimized XGBoost hyperparameters (Optuna HPO result)
DEFAULT_XGB_PARAMS: dict[str, float | int] = {
    "learning_rate": 0.02216030268952961,
    "max_depth": 20,
    "min_child_weight": 4.2832509812996635,
    "subsample": 0.6098353951742953,
    "colsample_bytree": 0.9761640794652597,
    "n_estimators": 500,
}
