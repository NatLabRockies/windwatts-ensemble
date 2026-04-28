# WEM Pipeline Run Order

The WindWatts Ensemble Model package exposes 48 production-stage CLI entry
points across 7 stages, plus 5 experiment entry points. This runbook uses the
canonical CLI flags implemented by the package; run `wem-<command> --help` for
full option details.

## Stage 1: Observations

```bash
# 1. Convert gold standard time series to 101-point quantiles
wem-gs-obs --pkl data/raw/gold_standard_timeseries.pkl \
    --out data/quantiles/gs/gold_standard_quantiles.csv \
    --filtered-out-csv data/quantiles/gs/gold_standard_filtered_out.csv

# 2. Download ASOS/ISD hourly wind observations from NOAA
wem-download-isd --out-dir data/raw/noaa_isd_csv \
    --meta-out data/raw/us_wind_station_metadata_2007_2024.csv

# 3. Process raw ISD files to per-station quantile CSVs
wem-process-isd --raw-dir data/raw/noaa_isd_csv \
    --proc-dir data/quantiles/asos/processed \
    --meta-csv data/raw/us_wind_station_metadata_2007_2024.csv

# 4. Aggregate station quantile CSVs into one ASOS matrix
wem-aggregate-asos --proc-dir data/quantiles/asos/processed \
    --out-csv data/quantiles/asos/all_sites_quantiles_2007_2024.csv
```

## Stage 2: Wind Resource Extraction

ERA5 runs first for each cohort because it defines the retained site list. The
remaining sources can run in parallel after ERA5 completes.

```bash
# Gold Standard cohort
wem-gs-era5 --sites data/quantiles/gs/gold_standard_quantiles.csv \
    --era5-dir /datasets/ERA5/conus \
    --out data/quantiles/gs/era5_quantiles_gold_standard_2007_2024.csv

wem-gs-wtk --sites data/quantiles/gs/era5_quantiles_gold_standard_2007_2024.csv \
    --data-dir /datasets/WIND/conus/v1.0.0 \
    --out data/quantiles/gs/wtk_quantiles_gold_standard_2007_2013.csv

wem-gs-wtkled-conus --sites data/quantiles/gs/era5_quantiles_gold_standard_2007_2024.csv \
    --data-dir /datasets/WIND/conus/v2.0.0 \
    --out data/quantiles/gs/wtk_led_conus_quantiles_gold_standard_2018_2020.csv

wem-gs-wtkled-climate --sites data/quantiles/gs/era5_quantiles_gold_standard_2007_2024.csv \
    --data-dir /datasets/WIND/ANL_4km_north_america \
    --out data/quantiles/gs/wtk_led_climate_quantiles_gold_standard_2007_2020.csv

wem-gs-hrrr --sites data/quantiles/gs/era5_quantiles_gold_standard_2007_2024.csv \
    --data-dir /datasets/WIND/HRRR \
    --out data/quantiles/gs/hrrr_quantiles_gold_standard_2015_2022.csv

# ASOS cohort
wem-asos-era5 --sites data/quantiles/asos/all_sites_quantiles_2007_2024.csv \
    --era5-dir /datasets/ERA5/conus \
    --out data/quantiles/asos/era5_quantiles_2007_2024.csv

wem-asos-wtk --sites data/quantiles/asos/era5_quantiles_2007_2024.csv \
    --data-dir /datasets/WIND/conus/v1.0.0 \
    --out data/quantiles/asos/wtk_quantiles_2007_2013.csv

wem-asos-wtkled-conus --sites data/quantiles/asos/era5_quantiles_2007_2024.csv \
    --out data/quantiles/asos/wtk_led_conus_quantiles_2018_2020.csv \
    --files /datasets/WIND/conus/v2.0.0/2018/conus_2018_10m.h5 \
            /datasets/WIND/conus/v2.0.0/2019/conus_2019_10m.h5 \
            /datasets/WIND/conus/v2.0.0/2020/conus_2020_10m.h5

wem-asos-wtkled-climate --sites data/quantiles/asos/era5_quantiles_2007_2024.csv \
    --data-dir /datasets/WIND/ANL_4km_north_america \
    --out data/quantiles/asos/wtk_led_climate_quantiles_2007_2020.csv

wem-asos-hrrr --sites data/quantiles/asos/era5_quantiles_2007_2024.csv \
    --data-dir /datasets/WIND/HRRR \
    --out data/quantiles/asos/hrrr_quantiles_2015_2022.csv
```

## Stage 3: Dataset Assembly

```bash
wem-merge-cohorts \
    --obs_asos data/quantiles/asos/all_sites_quantiles_2007_2024.csv \
    --era5_asos data/quantiles/asos/era5_quantiles_2007_2024.csv \
    --hrrr_asos data/quantiles/asos/hrrr_quantiles_2015_2022.csv \
    --wtk_asos data/quantiles/asos/wtk_quantiles_2007_2013.csv \
    --ledc_asos data/quantiles/asos/wtk_led_conus_quantiles_2018_2020.csv \
    --ledclim_asos data/quantiles/asos/wtk_led_climate_quantiles_2007_2020.csv \
    --obs_gs data/quantiles/gs/gold_standard_quantiles.csv \
    --era5_gs data/quantiles/gs/era5_quantiles_gold_standard_2007_2024.csv \
    --hrrr_gs data/quantiles/gs/hrrr_quantiles_gold_standard_2015_2022.csv \
    --wtk_gs data/quantiles/gs/wtk_quantiles_gold_standard_2007_2013.csv \
    --ledc_gs data/quantiles/gs/wtk_led_conus_quantiles_gold_standard_2018_2020.csv \
    --ledclim_gs data/quantiles/gs/wtk_led_climate_quantiles_gold_standard_2007_2020.csv \
    --out data/training/combined_quantiles_long.csv

wem-add-topo --in data/training/combined_quantiles_long.csv \
    --out data/training/combined_quantiles_long_with_topo.csv

wem-build-neighbors --infile data/training/combined_quantiles_long_with_topo.csv \
    --outfile data/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --radius-km 10

wem-obs-wsavg --in data/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --out data/training/site_height_ws_avg.csv

wem-add-gwa --in data/training/site_height_ws_avg.csv \
    --out data/training/site_height_ws_avg_with_gwa.csv \
    --gwa10 data/gwa/GWA_wind-speed_10m.tif \
    --gwa50 data/gwa/GWA_wind-speed_50m.tif \
    --gwa100 data/gwa/GWA_wind-speed_100m.tif \
    --gwa150 data/gwa/GWA_wind-speed_150m.tif
```

## Stage 4: ML Training

```bash
wem-train-loocv --infile data/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outfile data/models/loocv/ml_results.csv \
    --balance_strategy downsample --n_jobs_outer 12 --n_jobs_model 1 \
    --gwa-file data/training/site_height_ws_avg_with_gwa.csv --gwa-col gwa_interp --include-gwa

wem-train-final --infile data/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --out-dir data/models/final/ \
    --n_jobs_model 8 --shap false \
    --gwa-file data/training/site_height_ws_avg_with_gwa.csv --gwa-col gwa_interp --include-gwa
```

## Stage 5: Analysis

```bash
wem-analyze --infile data/models/loocv/ml_results.csv \
    --outdir data/output/analysis/ --conus \
    --gwa data/training/site_height_ws_avg_with_gwa.csv

wem-analyze-extended --infile data/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --gwa data/training/site_height_ws_avg_with_gwa.csv \
    --outdir data/output/analysis_extended/

wem-row-metrics --infile data/models/loocv/ml_results.csv \
    --outdir data/output/analysis_row/ --subset GS

wem-error-diffs data/models/loocv/ml_results.csv \
    --outdir data/output/analysis_error_diffs/

wem-site-cdfs --csv data/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outdir data/output/gs_cdf_curves/

wem-site-cdfs --csv data/models/loocv/ml_results.csv \
    --outdir data/output/ml_cdf_curves/ \
    --ml-col pred_observation --gs-col is_gs --gs-value 1

wem-viz-fi --in data/models/final/feature_importance.csv \
    --out-dir data/output/feature_importance/

wem-qc-filter --infile data/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outfile data/training/combined_quantiles_long_filtered.csv \
    --summary_out data/output/qc/asos_station_summary.csv \
    --plots_dir data/output/qc/plots/

wem-grid-means --in data/grid/site_quantiles_predicted.csv \
    --out data/grid/site_mean_winds.csv

wem-nn-lookup --grid data/grid/site_mean_winds.csv \
    --sites data/training/site_height_ws_avg_with_gwa.csv \
    --out data/output/sites_with_pred.csv

wem-quantile-maps \
    --obs data/quantiles/asos/all_sites_quantiles_2007_2024.csv \
    --era5 data/quantiles/asos/era5_quantiles_2007_2024.csv \
    --wtk data/quantiles/asos/wtk_quantiles_2007_2013.csv \
    --hrrr data/quantiles/asos/hrrr_quantiles_2015_2022.csv \
    --ledc data/quantiles/asos/wtk_led_conus_quantiles_2018_2020.csv \
    --ledclim data/quantiles/asos/wtk_led_climate_quantiles_2007_2020.csv \
    --gs-obs data/quantiles/gs/gold_standard_quantiles.csv \
    --gs-era5 data/quantiles/gs/era5_quantiles_gold_standard_2007_2024.csv \
    --gs-wtk data/quantiles/gs/wtk_quantiles_gold_standard_2007_2013.csv \
    --gs-hrrr data/quantiles/gs/hrrr_quantiles_gold_standard_2015_2022.csv \
    --gs-ledc data/quantiles/gs/wtk_led_conus_quantiles_gold_standard_2018_2020.csv \
    --gs-ledclim data/quantiles/gs/wtk_led_climate_quantiles_gold_standard_2007_2020.csv \
    --gwa data/training/site_height_ws_avg_with_gwa.csv \
    --gs-gwa data/training/site_height_ws_avg_with_gwa.csv \
    --outdir data/output/quantile_maps

wem-interannual --infile data/raw/gold_standard_timeseries.pkl \
    --outdir data/output/interannual/
```

## Stage 6: Grid Inference

```bash
# Extract ERA5 grid quantiles first; these define the inference grid.
wem-grid-era5 --era5-dir /datasets/ERA5/conus \
    --out-pattern 'data/grid/merged/all/era5_quantiles_{height_m:.0f}m.csv'

wem-build-grid --in data/grid/merged/all/era5_quantiles_100m.csv \
    --out data/grid/era5_grid.csv

wem-grid-hrrr --era5-grid data/grid/era5_grid.csv \
    --data-dir /datasets/WIND/HRRR \
    --out-dir data/grid/tiles/hrrr/

wem-grid-wtk --era5-grid data/grid/era5_grid.csv \
    --wtk-dir /datasets/WIND/conus/v1.0.0 \
    --out-dir data/grid/tiles/wtk/

wem-grid-wtkled --era5-grid data/grid/era5_grid.csv \
    --data-dir /datasets/WIND/conus/v2.0.0 \
    --out-dir data/grid/tiles/wtkled/

wem-merge-tiles --in-dir data/grid/tiles/wtk/ \
    --out-dir data/grid/merged/all/ --prefix wtk --dedupe
wem-merge-tiles --in-dir data/grid/tiles/hrrr/ \
    --out-dir data/grid/merged/all/ --prefix hrrr --dedupe
wem-merge-tiles --in-dir data/grid/tiles/wtkled/ \
    --out-dir data/grid/merged/all/ --prefix wtk_led --dedupe

wem-merge-grid --in-dir data/grid/merged/all/ \
    --out-file data/grid/merged_quantiles_all.csv

wem-grid-elev --in data/grid/merged_quantiles_all.csv \
    --out data/grid/merged_quantiles_all_with_elev.csv \
    --workers 12 --timeout 15

wem-fill-missing --in data/grid/merged_quantiles_all_with_elev.csv \
    --out data/grid/merged_quantiles_all_with_elev_filled.csv

wem-prepare-inference --in data/grid/merged_quantiles_all_with_elev_filled.csv \
    --out data/grid/inference_table.csv

wem-grid-gwa --in data/grid/inference_table.csv \
    --out data/grid/inference_table_with_gwa.csv \
    --gwa10 data/gwa/GWA_wind-speed_10m.tif \
    --gwa50 data/gwa/GWA_wind-speed_50m.tif \
    --gwa100 data/gwa/GWA_wind-speed_100m.tif \
    --gwa150 data/gwa/GWA_wind-speed_150m.tif

wem-infer --in data/grid/inference_table_with_gwa.csv \
    --model-dir data/models/final/ \
    --out data/grid/predictions_fullgrid.csv

wem-pivot-predictions --in data/grid/predictions_fullgrid.csv \
    --era5 data/grid/era5_grid.csv \
    --out data/grid/site_quantiles_predicted.csv --decimals 2
```

## Stage 7: Maps & Output

```bash
wem-maps-mean --in-dir data/grid/merged/all/ \
    --predictions data/grid/site_quantiles_predicted.csv \
    --gwa data/grid/inference_table_with_gwa.csv \
    --out-dir data/output/maps/ --global-scale --extent -125 -66 24 50

wem-maps-diff --in-dir data/grid/merged/all/ \
    --predictions data/grid/site_quantiles_predicted.csv \
    --gwa data/grid/inference_table_with_gwa.csv \
    --out-dir data/output/maps/ --diff-limit 3 --coord-precision 2

wem-app-format --in data/grid/site_quantiles_predicted.csv \
    --out-dir data/output/app/ --make-index
```

## Experiment Entry Points

Experiment commands are not required for the production pipeline:

```bash
wem-exp-hpo --help
wem-exp-param-sweep --help
wem-exp-feature-sweep --help
wem-exp-analyze-sweep --help
wem-experiment --help
```
