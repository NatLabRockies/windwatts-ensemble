# HPC Directory Inventory

Operational HPC infrastructure and extraction data for the Kestrel supercomputer. Python extraction scripts have been migrated to the WEM package (`wem-asos-*`, `wem-gs-*`, `wem-grid-*` CLI commands); this directory contains Slurm job wrappers, parallel tile runners, grid config, merge scripts, and extraction output data.

---

## Current Directory Structure (2026-03-05)

```
hpc/
├── HPC_INVENTORY.md
├── slurm/                                   # Slurm/SBATCH job wrappers
│   ├── extract/                             # Site extraction jobs (8 files)
│   │   ├── hrrr_to_quantiles.sh             # ASOS HRRR
│   │   ├── wtk_to_quantiles.sh              # ASOS WTK
│   │   ├── wtk_led_conus_to_quantiles.sh    # ASOS WTK-LED CONUS
│   │   ├── wtk_led_climate_to_quantiles.sh  # ASOS WTK-LED Climate
│   │   ├── goldstandard_hrrr_to_quantiles.sh       # GS HRRR (new, uses wem-gs-hrrr)
│   │   ├── goldstandard_wtk_to_quantiles.sh        # GS WTK
│   │   ├── goldstandard_wtk_led_conus_to_quantiles.sh    # GS WTK-LED CONUS
│   │   └── goldstandard_wtk_led_climate_to_quantiles.sh  # GS WTK-LED Climate
│   └── grid/                                # Grid extraction jobs (5 files)
│       ├── fullgrid_hrrr_to_quantiles.sh    # Grid HRRR (new, uses wem-grid-hrrr)
│       ├── fullgrid_wtk_to_quantiles.sh     # Grid WTK
│       ├── fullgrid_wtkled_conus_to_quantiles.sh  # Grid WTK-LED
│       ├── run_tiles_wtkled.slurm           # WTK-LED Slurm array job
│       └── submit_wtkled_tiles.sbatch       # WTK-LED batch submission
├── grid/                                    # Grid operational tooling (10 files)
│   ├── run_tiles_locally.sh                 # Original parallel tile runner
│   ├── run_tiles_locally_wtk.sh             # WTK parallel runner
│   ├── run_tiles_locally_hrrr.sh            # HRRR parallel runner
│   ├── run_tiles_locally_wtkled.sh          # WTK-LED parallel runner
│   ├── debug_one_tile_wtkled.sh             # Single-tile debug runner
│   ├── split_missing_tiles_wtkled.sh        # Split missing tiles for retry
│   ├── find_missing_tiles.py                # Compare expected vs actual tiles
│   ├── era5_grid.csv                        # ERA5 CONUS grid (42,375 points)
│   ├── tiles.txt                            # Full tile ID list (450 tiles)
│   └── tiles_hrrr.txt                       # HRRR tile ID list (450 tiles)
└── grid_data/                               # Extraction output data (~2.8 GB)
    ├── tiles/
    │   ├── wtk/                             # 450 WTK tile CSVs (final)
    │   ├── hrrr/                            # 450 HRRR tile CSVs (final)
    │   └── wtkled/                          # 450 WTK-LED tile CSVs (final)
    └── merged/
        ├── wtk/                             # 6 per-height CSVs (merged via wem-merge-tiles)
        ├── hrrr/                            # 6 per-height CSVs (merged via wem-merge-tiles)
        └── wtkled/                          # 6 per-height CSVs (merged via wem-merge-tiles)
```

---

## Slurm Job Wrappers (`slurm/`)

Each wrapper provides Kestrel-specific infrastructure: `#SBATCH` directives, `ml conda` + `conda activate`, `srun` invocation, and log routing. The Python extraction logic is handled by WEM CLI commands. Wrappers activate `${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}` so production runs can override the default Kestrel environment without editing scripts.

### Site Extraction (`slurm/extract/`)

| Wrapper | WEM Command | Partition | Time | Account |
|---------|-------------|-----------|------|---------|
| `hrrr_to_quantiles.sh` | `wem-asos-hrrr` | standard | 12h | tap |
| `wtk_to_quantiles.sh` | `wem-asos-wtk` | standard | 12h | tap |
| `wtk_led_conus_to_quantiles.sh` | `wem-asos-wtkled-conus` | standard | 12h | tap |
| `wtk_led_climate_to_quantiles.sh` | `wem-asos-wtkled-climate` | standard | 12h | tap |
| `goldstandard_hrrr_to_quantiles.sh` | `wem-gs-hrrr` | standard | 12h | tap |
| `goldstandard_wtk_to_quantiles.sh` | `wem-gs-wtk` | short | 4h | tap |
| `goldstandard_wtk_led_conus_to_quantiles.sh` | `wem-gs-wtkled-conus` | standard | 12h | tap |
| `goldstandard_wtk_led_climate_to_quantiles.sh` | `wem-gs-wtkled-climate` | short | 4h | tap |

All wrappers call WEM CLI commands directly (e.g., `wem-asos-hrrr`, `wem-gs-wtk`, `wem-grid-wtkled`). No legacy Python script references remain.

### Grid Extraction (`slurm/grid/`)

| Wrapper | WEM Command | Partition | Time | Account |
|---------|-------------|-----------|------|---------|
| `fullgrid_hrrr_to_quantiles.sh` | `wem-grid-hrrr` | standard | 12h | tap |
| `fullgrid_wtk_to_quantiles.sh` | `wem-grid-wtk` | standard | 12h | tap |
| `fullgrid_wtkled_conus_to_quantiles.sh` | `wem-grid-wtkled` | long | 4 days | cscdav |

---

## Grid Operational Tooling (`grid/`)

### Parallel Tile Runners

N-way parallel execution harnesses for running grid extraction across tiles on a single node. Features: configurable parallelism, skip-if-exists, per-tile logs, `.ok`/`.fail` status flags, signal handling.

| Script | Dataset |
|--------|---------|
| `run_tiles_locally_wtk.sh` | WTK |
| `run_tiles_locally_hrrr.sh` | HRRR |
| `run_tiles_locally_wtkled.sh` | WTK-LED |
| `run_tiles_locally.sh` | Original (WTK-LED specific) |

### Utilities

| Script | Purpose |
|--------|---------|
| `find_missing_tiles.py` | Compare expected vs actual tile outputs, write missing tile IDs |
| `split_missing_tiles_wtkled.sh` | Split missing tiles into batch files for parallel retry |
| `debug_one_tile_wtkled.sh` | Single-tile debug runner (expects `salloc`) |

### Config

| File | Contents |
|------|----------|
| `era5_grid.csv` | ERA5 CONUS grid definition (42,375 points: grid_id, lat, lon) |
| `tiles.txt` | Full WTK/WTK-LED tile ID list (450 tiles) |
| `tiles_hrrr.txt` | HRRR tile ID list (450 tiles) |

---

## Grid Data (`grid_data/`)

### Tile Outputs (`grid_data/tiles/`)

450 per-tile CSVs per dataset, produced by the grid extraction scripts.

| Directory | Size | Status |
|-----------|------|--------|
| `tiles/wtk/` | 484 MB | Final |
| `tiles/hrrr/` | 482 MB | Final |
| `tiles/wtkled/` | 483 MB | Final |

### Merged Per-Height Outputs (`grid_data/merged/`)

6 per-height CSVs per dataset, produced by the `wem-merge-tiles` CLI command.

| Directory | Size | Status |
|-----------|------|--------|
| `merged/wtk/` | 453 MB | Final |
| `merged/hrrr/` | 453 MB | Final |
| `merged/wtkled/` | 453 MB | Final |

---

## WEM CLI Command Reference

All Python extraction scripts have been migrated to the WEM package. Install with `pip install -e .` from the wem repo root.

| WEM Command | Dataset | Cohort |
|-------------|---------|--------|
| `wem-asos-hrrr` | HRRR | ASOS (1,842 stations, 10m) |
| `wem-asos-wtk` | WTK | ASOS |
| `wem-asos-wtkled-conus` | WTK-LED CONUS | ASOS |
| `wem-asos-wtkled-climate` | WTK-LED Climate | ASOS |
| `wem-gs-hrrr` | HRRR | Gold Standard (348 site-heights) |
| `wem-gs-wtk` | WTK | Gold Standard |
| `wem-gs-wtkled-conus` | WTK-LED CONUS | Gold Standard |
| `wem-gs-wtkled-climate` | WTK-LED Climate | Gold Standard |
| `wem-grid-hrrr` | HRRR | Full grid (42,375 points) |
| `wem-grid-wtk` | WTK | Full grid |
| `wem-grid-wtkled` | WTK-LED | Full grid |
| `wem-merge-tiles` | All | Merge per-tile outputs into per-height CSVs |
| `wem-merge-grid` | All | Merge per-height CSVs across datasets |

---

## Future Work: Automated Tile Job Management

The current tile extraction workflow requires manual intervention when HPC jobs fail or time out: run `find_missing_tiles.py` to identify gaps, split the missing list, and resubmit. This should be replaced with an automated system that:

- **Tracks completion state** — log each tile as it finishes (e.g., a manifest file or lightweight database), so progress survives job restarts without re-scanning the output directory
- **Iterative resubmission** — automatically generate and submit follow-up jobs for only the remaining tiles, with configurable retry limits and backoff
- **Unified across datasets** — one tool that works for WTK, HRRR, and WTK-LED extraction (the tile geometry is the same across all three)
- **Integration with `wem-merge-tiles`** — once all tiles are confirmed complete, trigger the merge step automatically

This would replace `find_missing_tiles.py`, `split_missing_tiles_wtkled.sh`, and the manual tile retry workflow. The parallel tile runners (`run_tiles_locally_*.sh`) could also be consolidated into a single parameterized runner.

---

## Legacy Data (moved to `asos_data/legacy/hpc/`)

The following were moved out of `wem/` as they are superseded or historical:

- **Python extraction scripts** (12) — superseded by WEM CLI commands above
- **BC-HRRR script + wrapper** — not in main pipeline
- **ASOS quantile CSVs** (6) — 5 duplicates of `data/quantiles/asos/`, 1 BC-HRRR
- **GS quantile CSVs** (5) — pre-Ozark-fix, superseded by corrected `data/quantiles/gs/`
- **Ozark patch CSVs** (5) — fix applied upstream
- **Legacy WTK-LED tiles** (450) — superseded by `grid_data/tiles/wtkled/`
- **Debug artifacts** — incomplete early runs, debug tile outputs, batch split files
- **Execution logs** — Slurm job logs and status flags
- **Missing tile lists** (5 `.txt`) — historical from past runs
