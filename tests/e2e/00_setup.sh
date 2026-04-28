#!/usr/bin/env bash
# 00_setup.sh — Create directories and symlink input data for e2e tests.
source "$(dirname "$0")/helpers.sh"

stage_start "00 Setup"

bash "$(dirname "$0")/verify_artifacts.sh"

# ---- Clean generated outputs from prior runs ----
rm -rf "$E2E_DIR/training" "$E2E_DIR/models" "$E2E_DIR/output"
rm -f "$E2E_DIR/grid/merged_quantiles_all_with_elev_filled.csv"
rm -f "$E2E_DIR/grid/inference_table.csv"
rm -f "$E2E_DIR/grid/inference_table_with_gwa.csv"
rm -f "$E2E_DIR/grid/predictions_fullgrid.csv"
rm -f "$E2E_DIR/grid/site_quantiles_predicted.csv"
rm -f "$E2E_DIR/grid/app_format_input.csv"

# ---- Create directory structure ----
mkdir -p "$E2E_DIR"/{quantiles/asos,quantiles/gs,training,models/loocv,models/final,output/analysis,grid,grid/tiles/test,grid/merged/test}

# ---- Symlink ASOS quantile CSVs ----
ln -sf "$DATA_DIR/quantiles/asos/all_sites_quantiles_2007_2024.csv"      "$E2E_DIR/quantiles/asos/obs.csv"
ln -sf "$DATA_DIR/quantiles/asos/era5_quantiles_2007_2024.csv"           "$E2E_DIR/quantiles/asos/era5.csv"
ln -sf "$DATA_DIR/quantiles/asos/hrrr_quantiles_2015_2022.csv"           "$E2E_DIR/quantiles/asos/hrrr.csv"
ln -sf "$DATA_DIR/quantiles/asos/wtk_quantiles_2007_2013.csv"            "$E2E_DIR/quantiles/asos/wtk.csv"
ln -sf "$DATA_DIR/quantiles/asos/wtk_led_conus_quantiles_2018_2020.csv"  "$E2E_DIR/quantiles/asos/ledc.csv"
ln -sf "$DATA_DIR/quantiles/asos/wtk_led_climate_quantiles_2007_2020.csv" "$E2E_DIR/quantiles/asos/ledclim.csv"

# ---- Symlink Gold Standard quantile CSVs ----
ln -sf "$DATA_DIR/quantiles/gs/gold_standard_quantiles.csv"                          "$E2E_DIR/quantiles/gs/obs.csv"
ln -sf "$DATA_DIR/quantiles/gs/era5_quantiles_gold_standard_2007_2024.csv"           "$E2E_DIR/quantiles/gs/era5.csv"
ln -sf "$DATA_DIR/quantiles/gs/hrrr_quantiles_gold_standard_2015_2022.csv"           "$E2E_DIR/quantiles/gs/hrrr.csv"
ln -sf "$DATA_DIR/quantiles/gs/wtk_quantiles_gold_standard_2007_2013.csv"            "$E2E_DIR/quantiles/gs/wtk.csv"
ln -sf "$DATA_DIR/quantiles/gs/wtk_led_conus_quantiles_gold_standard_2018_2020.csv"  "$E2E_DIR/quantiles/gs/ledc.csv"
ln -sf "$DATA_DIR/quantiles/gs/wtk_led_climate_quantiles_gold_standard_2007_2020.csv" "$E2E_DIR/quantiles/gs/ledclim.csv"

# ---- Symlink GWA TIF files ----
ln -sf "$DATA_DIR/gwa/GWA_wind-speed_10m.tif"  "$E2E_DIR/GWA_wind-speed_10m.tif"
ln -sf "$DATA_DIR/gwa/GWA_wind-speed_50m.tif"  "$E2E_DIR/GWA_wind-speed_50m.tif"
ln -sf "$DATA_DIR/gwa/GWA_wind-speed_100m.tif" "$E2E_DIR/GWA_wind-speed_100m.tif"
ln -sf "$DATA_DIR/gwa/GWA_wind-speed_150m.tif" "$E2E_DIR/GWA_wind-speed_150m.tif"

# ---- Symlink grid intermediate (input for Stage 6H) ----
ln -sf "$DATA_DIR/grid/merged_quantiles_all_with_elev.csv" "$E2E_DIR/grid/merged_quantiles_all_with_elev.csv"

# ---- Symlink ERA5 location data (needed for pivot-predictions) ----
ln -sf "$DATA_DIR/grid/era5_location_data.csv" "$E2E_DIR/grid/era5_location_data.csv"

# ---- Verify symlinks ----
echo "Verifying symlinks..."
for f in \
    "$E2E_DIR/quantiles/asos/obs.csv" \
    "$E2E_DIR/quantiles/asos/era5.csv" \
    "$E2E_DIR/quantiles/asos/hrrr.csv" \
    "$E2E_DIR/quantiles/asos/wtk.csv" \
    "$E2E_DIR/quantiles/asos/ledc.csv" \
    "$E2E_DIR/quantiles/asos/ledclim.csv" \
    "$E2E_DIR/quantiles/gs/obs.csv" \
    "$E2E_DIR/quantiles/gs/era5.csv" \
    "$E2E_DIR/quantiles/gs/hrrr.csv" \
    "$E2E_DIR/quantiles/gs/wtk.csv" \
    "$E2E_DIR/quantiles/gs/ledc.csv" \
    "$E2E_DIR/quantiles/gs/ledclim.csv" \
    "$E2E_DIR/GWA_wind-speed_10m.tif" \
    "$E2E_DIR/grid/merged_quantiles_all_with_elev.csv" \
    "$E2E_DIR/grid/era5_location_data.csv" \
; do
    if [[ ! -e "$f" ]]; then
        echo -e "${RED}[FAIL]${NC} Broken symlink: $f"
        exit 1
    fi
done
echo "  All symlinks OK"

stage_end
