# dev/ — Development Scripts (Archived)

All development scripts have been promoted to production modules or removed
from the clean WEM repository. This directory retains only `__init__.py` and
this README as a migration map.

## Tier 1: Promoted to `wem/analyze/` (8 modules)

| Old file | New module | CLI entry point |
|----------|-----------|-----------------|
| `get_analysis_metrics.py` | `wem.analyze.extended_metrics` | `wem-analyze-extended` |
| `row_level_metrics.py` | `wem.analyze.row_metrics` | `wem-row-metrics` |
| `qc_filter_outliers.py` | `wem.analyze.qc_outliers` | `wem-qc-filter` |
| `viz_feature_importance.py` | `wem.analyze.feature_importance` | `wem-viz-fi` |
| `plot_error_diffs.py` | `wem.analyze.error_diffs` | `wem-error-diffs` |
| `make_gs_site_cdfs.py` + `make_ml_results_cdf.py` | `wem.analyze.site_cdfs` | `wem-site-cdfs` |
| `make_means_predictions.py` | `wem.analyze.grid_means` | `wem-grid-means` |
| `nn_lookup_sites.py` | `wem.analyze.nn_lookup` | `wem-nn-lookup` |

## Tier 2: Promoted to `wem/experiment/` (4 modules + helpers)

| Old files | New module | CLI entry point |
|-----------|-----------|-----------------|
| `optimize_hyperparams.py` | `wem.experiment.optuna_hpo` | `wem-exp-hpo` |
| `optimize_n_estimators.py` + `optimize_max_depth.py` | `wem.experiment.param_sweep` | `wem-exp-param-sweep` |
| `sweep_wind_features.py` + `sweep_aux_features.py` | `wem.experiment.feature_sweep` | `wem-exp-feature-sweep` |
| `analyze_feature_sweeps.py` + `analyze_aux_sweeps.py` | `wem.experiment.analyze_sweep` | `wem-exp-analyze-sweep` |

## Tier 3: Promoted or archived

| Old file | Disposition |
|----------|-------------|
| `create_maps.py` + `analyze_datasets_gwa.py` | Merged into `wem.analyze.quantile_maps` (`wem-quantile-maps`) |
| `final_graphics.py` | Removed from clean WEM history (publication one-off, requires external asset) |
| `_migrate_helper.py` | Deleted (migration completed March 2025) |

## Shared helpers extracted

Three functions duplicated across the Tier 3 scripts and `ml_results.py` were
extracted to `wem/utils/plotting.py`:

- `robust_limits(series_list, trim)` — trimmed color-scale limits
- `symmetric_bias_limit(bias_series, trim)` — symmetric bias limit
- `setup_cartopy_axes(conus, ne_res, figsize, dpi)` — standard CONUS basemap

## See also

- `wem/scripts/fix_ozark_source_data.py` — deleted (fixes baked into source CSVs)
