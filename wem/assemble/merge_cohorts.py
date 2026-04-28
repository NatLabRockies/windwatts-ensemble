#!/usr/bin/env python3
"""
Combine ASOS + Gold-Standard quantiles into one long table:

One row per (station_id, height_m, qnum) with:
  station_id, name, lat, lon, elev_m, height_m, qnum,
  observation, era5, hrrr, wtk, wtk_led_conus, wtk_led_climate, observation_type

Cohort logic:
- Build two independent cohorts: ASOS and Gold Standard.
- Within each cohort, keep ONLY site-heights present in that cohort's ERA5 file
  (so all datasets align to ERA5 coverage).
- There is no overlap of sites between cohorts; we simply concat both.

Inputs (defaults can be overridden via CLI):
  ASOS:
    --obs_asos     all_sites_quantiles_2007_2024.csv
    --era5_asos    era5_quantiles_2007_2024.csv
    --hrrr_asos    hrrr_quantiles_2015_2022.csv
    --wtk_asos     wtk_quantiles_2007_2013.csv
    --ledc_asos    wtk_led_conus_quantiles_2018_2020.csv
    --ledclim_asos wtk_led_climate_quantiles_2007_2020.csv

  Gold Standard:
    --obs_gs       gold_standard_quantiles.csv
    --era5_gs      era5_quantiles_gold_standard_2007_2024.csv
    --hrrr_gs      hrrr_quantiles_gold_standard_2015_2022.csv
    --wtk_gs       wtk_quantiles_gold_standard_2007_2013.csv
    --ledc_gs      wtk_led_conus_quantiles_gold_standard_2018_2020.csv
    --ledclim_gs   wtk_led_climate_quantiles_gold_standard_2007_2020.csv

Output:
  combined_quantiles_long.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from wem.utils.logging import log
from wem.utils.columns import choose_col, find_qcols


# -------------- helpers --------------
def to_long(df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a dataset-wide quantile table to long + meta.

    Returns:
      long: ['station_id','height_m','qnum','value']
      meta: ['station_id','height_m','name','lat','lon','elev_m'] (unique)
    """
    idc   = choose_col(df, ["station_id","site_id","STATION","id"])
    namec = choose_col(df, ["name","NAME","station_name","site_name","STATION NAME"])
    latc  = choose_col(df, ["lat","LAT","Latitude"])
    lonc  = choose_col(df, ["lon","LON","Longitude"])
    elvc  = choose_col(df, ["elev_m","elevation_m","elevation_meters","elevation"])
    if "height_m" not in df.columns:
        df["height_m"] = 10
    if "elev_m" not in df.columns:
        df["elev_m"] = np.nan
    hc    = choose_col(df, ["height_m","height","z"])
    if not (idc and latc and lonc and hc):
        raise ValueError(f"{label}: required columns not found (need station_id, lat, lon, height_m).")

    qcols = find_qcols(df)
    if len(qcols) < 50:
        raise ValueError(f"{label}: expected q000..q100; found {len(qcols)}.")

    dd = df.rename(columns={
        idc: "station_id",
        (namec or idc): "name",
        latc: "lat",
        lonc: "lon",
        (elvc or "elev_m"): "elev_m",
        hc: "height_m",
    })[["station_id","name","lat","lon","elev_m","height_m"] + qcols].copy()

    # Types
    dd["station_id"] = dd["station_id"].astype(str)
    dd["height_m"]   = pd.to_numeric(dd["height_m"], errors="coerce")
    dd["lat"]        = pd.to_numeric(dd["lat"], errors="coerce")
    dd["lon"]        = pd.to_numeric(dd["lon"], errors="coerce")
    if "elev_m" in dd:
        dd["elev_m"] = pd.to_numeric(dd["elev_m"], errors="coerce")
    dd = dd.dropna(subset=["station_id","height_m","lat","lon"]).reset_index(drop=True)

    # Long
    long = dd.melt(id_vars=["station_id","height_m"], value_vars=qcols,
                   var_name="qstr", value_name="value")
    long["qnum"] = long["qstr"].str[1:].astype(int)
    long = long.drop(columns=["qstr"])
    long = long[["station_id","height_m","qnum","value"]]

    # Meta unique
    meta = dd[["station_id","height_m","name","lat","lon","elev_m"]].drop_duplicates(["station_id","height_m"])

    return long, meta

def safe_read(path: Path, label: str) -> Optional[pd.DataFrame]:
    if path is None or not Path(path).exists():
        log(f"[WARN] Missing {label}: {path}")
        return None
    log(f"[INFO] Reading {label}: {path}")
    return pd.read_csv(path)


# -------------- cohort builder --------------
def build_cohort(
    label: str,
    era5_path: Path,
    obs_path: Path | None,
    hrrr_path: Path | None,
    wtk_path: Path | None,
    ledc_path: Path | None,
    ledclim_path: Path | None,
) -> pd.DataFrame:
    """
    Build one cohort (ASOS or GoldStandard) restricted to site-heights in the
    cohort's ERA5 file. Returns long table with dataset columns + meta +
    observation_type column set to label.
    """
    # Load ERA5 first (acts as key filter)
    df_e5 = safe_read(era5_path, f"{label} ERA5")
    if df_e5 is None:
        log(f"[ERROR] {label}: ERA5 is required.")
        return pd.DataFrame()

    long_e5, meta_e5 = to_long(df_e5, f"{label} ERA5")
    keys = long_e5[["station_id","height_m","qnum"]].drop_duplicates()

    # Observation (cohort-specific)
    obs_long = None
    if obs_path:
        df_obs = safe_read(obs_path, f"{label} observations")
        if df_obs is not None:
            obs_long, _ = to_long(df_obs, f"{label} observations")
            obs_long = keys.merge(obs_long, on=["station_id","height_m","qnum"], how="left")  # align to ERA5 keys

    # Each model
    def load_model(p: Path | None, name: str) -> Optional[pd.DataFrame]:
        if p is None: return None
        df = safe_read(p, f"{label} {name}")
        if df is None: return None
        l, _ = to_long(df, f"{label} {name}")
        return keys.merge(l, on=["station_id","height_m","qnum"], how="left")

    long_hrrr    = load_model(hrrr_path,    "HRRR")
    long_wtk     = load_model(wtk_path,     "WTK")
    long_ledc    = load_model(ledc_path,    "WTK-LED CONUS")
    long_ledclim = load_model(ledclim_path, "WTK-LED Climate")

    # Assemble cohort table from ERA5 keys
    out = keys.copy()
    out = out.merge(long_e5.rename(columns={"value":"era5"}), on=["station_id","height_m","qnum"], how="left")

    if obs_long is not None:    out = out.merge(obs_long.rename(columns={"value":"observation"}), on=["station_id","height_m","qnum"], how="left")
    if long_hrrr is not None:   out = out.merge(long_hrrr.rename(columns={"value":"hrrr"}),       on=["station_id","height_m","qnum"], how="left")
    if long_wtk is not None:    out = out.merge(long_wtk.rename(columns={"value":"wtk"}),         on=["station_id","height_m","qnum"], how="left")
    if long_ledc is not None:   out = out.merge(long_ledc.rename(columns={"value":"wtk_led_conus"}), on=["station_id","height_m","qnum"], how="left")
    if long_ledclim is not None:out = out.merge(long_ledclim.rename(columns={"value":"wtk_led_climate"}), on=["station_id","height_m","qnum"], how="left")

    # Attach metadata (from ERA5)
    out = out.merge(meta_e5, on=["station_id","height_m"], how="left")

    # Add cohort label
    out["observation_type"] = label

    # Column order
    cols = ["station_id","name","lat","lon","elev_m","height_m","qnum",
            "observation","era5","hrrr","wtk","wtk_led_conus","wtk_led_climate","observation_type"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    return out.sort_values(["station_id","height_m","qnum"]).reset_index(drop=True)


# -------------- main --------------
def main():
    ap = argparse.ArgumentParser(description="Combine ASOS + Gold-Standard quantiles into one tidy CSV (by cohort).")

    # ASOS cohort (legacy paths)
    ap.add_argument("--obs_asos",     type=Path, default=Path("all_sites_quantiles_2007_2024.csv"))
    ap.add_argument("--era5_asos",    type=Path, default=Path("era5_quantiles_2007_2024.csv"))
    ap.add_argument("--hrrr_asos",    type=Path, default=Path("hrrr_quantiles_2015_2022.csv"))
    ap.add_argument("--wtk_asos",     type=Path, default=Path("wtk_quantiles_2007_2013.csv"))
    ap.add_argument("--ledc_asos",    type=Path, default=Path("wtk_led_conus_quantiles_2018_2020.csv"))
    ap.add_argument("--ledclim_asos", type=Path, default=Path("wtk_led_climate_quantiles_2007_2020.csv"))

    # Gold-Standard cohort
    ap.add_argument("--obs_gs",       type=Path, default=Path("gold_standard_quantiles.csv"))
    ap.add_argument("--era5_gs",      type=Path, default=Path("era5_quantiles_gold_standard_2007_2024.csv"))
    ap.add_argument("--hrrr_gs",      type=Path, default=Path("hrrr_quantiles_gold_standard_2015_2022.csv"))
    ap.add_argument("--wtk_gs",       type=Path, default=Path("wtk_quantiles_gold_standard_2007_2013.csv"))
    ap.add_argument("--ledc_gs",      type=Path, default=Path("wtk_led_conus_quantiles_gold_standard_2018_2020.csv"))
    ap.add_argument("--ledclim_gs",   type=Path, default=Path("wtk_led_climate_quantiles_gold_standard_2007_2020.csv"))

    # Output
    ap.add_argument("--out",          type=Path, default=Path("combined_quantiles_long.csv"))
    args = ap.parse_args()

    # Build cohorts (each restricted to its ERA5 coverage)
    asos = build_cohort(
        label="ASOS",
        era5_path=args.era5_asos,
        obs_path=args.obs_asos,
        hrrr_path=args.hrrr_asos,
        wtk_path=args.wtk_asos,
        ledc_path=args.ledc_asos,
        ledclim_path=args.ledclim_asos,
    )

    gs = build_cohort(
        label="GoldStandard",
        era5_path=args.era5_gs,
        obs_path=args.obs_gs,
        hrrr_path=args.hrrr_gs,
        wtk_path=args.wtk_gs,
        ledc_path=args.ledc_gs,
        ledclim_path=args.ledclim_gs,
    )

    # Concatenate cohorts (non-overlapping by design)
    frames = [df for df in [asos, gs] if df is not None and not df.empty]
    if not frames:
        log("[ERROR] No data loaded; exiting.")
        return

    out = pd.concat(frames, ignore_index=True)

    # Write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    log(f"[DONE] Wrote {len(out):,} rows → {args.out.resolve()}")

if __name__ == "__main__":
    main()
