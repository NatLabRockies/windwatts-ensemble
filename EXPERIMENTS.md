# WEM Experiment Framework

## Overview

The experiment framework provides a unified CLI for running and comparing different LOOCV formulations of the WEM bias-correction model.

```bash
wem-experiment <type> [args]        # Run an experiment
wem-experiment compare [args]       # Compare experiment results
```

### Experiment Types

| Type | Format | Features | Models | Description |
|------|--------|----------|--------|-------------|
| `baseline` | Long (1 row per quantile) | 9 | 1 shared | Production formulation with `qnum` monotonic constraint |
| `enriched` | Long (1 row per quantile) | 309 | 1 shared | Baseline + full 101-point CDFs from each wind source |
| `wide` | Wide (1 row per station/height) | 308 | 101 independent | CDF-in, CDF-out with post-hoc monotonicity |
| `convnet` | Wide (1 row per station/height) | 3×101 CDF + 4–5 aux | 1D ConvNet | CDF-to-CDF with architectural monotonicity (softplus+cumsum) |
| `mlp` | Long (1 row per quantile) | 9 | 2-layer MLP | Same features as baseline, PyTorch MLP instead of XGBoost |
| `hybrid` | Long (1 row per quantile) | 9 base / 15 tail | 2 XGBoost | Baseline q0-q94, log-target tail-specialized model q95-q100 |

### Key Differences

| Aspect | Baseline | Enriched | Wide | ConvNet | MLP | Hybrid |
|--------|----------|----------|------|---------|-----|
| **Rows per fold** | ~70,000 | ~70,000 | ~570 | ~570 | ~70,000 | ~70,000 |
| **Features** | qnum + 3 wind + 4 aux + GWA | qnum + 303 CDF + 4 aux + GWA | 303 CDF + 4 aux + GWA | 3×101 CDF (channels) + 4 aux | qnum + 3 wind + 4 aux + GWA | base: 9, tail: 15 (+ CDF context) |
| **Monotonicity** | `qnum` constraint (+1) | `qnum` constraint (+1) | `np.maximum.accumulate` | Architectural (softplus→cumsum) | Learned from data | Base only (tail: none) |
| **Parameter sharing** | Full (single model) | Full (single model) | None (per-quantile models) | Full (single model) | Full (single model) | Two models (base + tail) |

---

## Running Experiments

### Prerequisites

- WEM installed in editable mode: `pip install -e .`
- Training data: `data/e2e/training/combined_quantiles_long_with_topo_loocv_10km.csv`
- GWA data: `data/e2e/training/site_height_ws_avg_with_gwa.csv`

### Baseline

```bash
wem-experiment baseline \
    --infile data/e2e/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outfile data/output/baseline_results.csv \
    --gwa-file data/e2e/training/site_height_ws_avg_with_gwa.csv \
    --include-gwa \
    --n-jobs 12 \
    --overwrite
```

### Enriched (Long + Full CDFs)

```bash
wem-experiment enriched \
    --infile data/e2e/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outfile data/output/enriched_results.csv \
    --gwa-file data/e2e/training/site_height_ws_avg_with_gwa.csv \
    --include-gwa \
    --n-jobs 12 \
    --stations "pauldingl2,150,7" \
    --overwrite
```

### Wide (Multi-Output)

```bash
wem-experiment wide \
    --infile data/e2e/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outfile data/output/wide_results.csv \
    --gwa-file data/e2e/training/site_height_ws_avg_with_gwa.csv \
    --include-gwa \
    --n-jobs 12 \
    --overwrite
```

### ConvNet (1D CNN)

```bash
wem-experiment convnet \
    --infile data/e2e/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outfile data/output/convnet_results.csv \
    --gwa-file data/e2e/training/site_height_ws_avg_with_gwa.csv \
    --include-gwa \
    --epochs 300 --batch-size 32 --lr 1e-3 \
    --patience 30 --device cpu \
    --overwrite
```

### MLP (Long-Format Neural Network)

```bash
wem-experiment mlp \
    --infile data/e2e/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outfile data/output/mlp_results.csv \
    --gwa-file data/e2e/training/site_height_ws_avg_with_gwa.csv \
    --include-gwa \
    --epochs 100 --batch-size 512 --lr 1e-3 \
    --patience 15 --device mps \
    --overwrite
```

### Hybrid (Tail-Specialized)

```bash
wem-experiment hybrid \
    --infile data/e2e/training/combined_quantiles_long_with_topo_loocv_10km.csv \
    --outfile data/output/hybrid_results.csv \
    --gwa-file data/e2e/training/site_height_ws_avg_with_gwa.csv \
    --include-gwa \
    --n-jobs 12 \
    --tail-cutoff 95 \
    --overwrite
```

Hybrid-specific options:

| Option | Default | Description |
|--------|---------|-------------|
| `--tail-cutoff` | 95 | Quantile index cutoff for tail model (q >= cutoff uses tail) |
| `--tail-log-floor` | 1e-6 | Floor for log transform of target |

MLP-specific options:

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 100 | Max training epochs |
| `--batch-size` | 512 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--dropout` | 0.3 | Dropout rate |
| `--patience` | 15 | Early stopping patience (epochs) |
| `--val-frac` | 0.2 | Fraction of training stations for validation |
| `--device` | cpu | PyTorch device (cpu/cuda/mps) |
| `--hidden-dims` | 128 64 | Hidden layer sizes |

ConvNet-specific options:

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 300 | Max training epochs |
| `--batch-size` | 32 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--dropout` | 0.3 | Dropout rate |
| `--patience` | 30 | Early stopping patience (epochs) |
| `--val-frac` | 0.2 | Fraction of training stations for validation |
| `--device` | cpu | PyTorch device (cpu/cuda/mps) |
| `--n-conv-layers` | 3 | Number of conv blocks (2 or 3) |

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--infile` | `combined_quantiles_long_with_topo_loocv_10km.csv` | Input long-format CSV |
| `--outfile` | (required) | Output CSV |
| `--wind-features` | `hrrr,wtk,wtk_led_conus` | Wind resource columns |
| `--gwa-file` | None | GWA CSV for merge |
| `--include-gwa` | False | Include GWA as feature |
| `--n-jobs` | 12 | Parallel LOOCV folds |
| `--seed` | 42 | Random seed |
| `--stations` | None | Comma-separated GS subset |
| `--overwrite` | False | Overwrite existing output |
| `--balance-strategy` | `downsample` | GS/ASOS balancing |

Hyperparameters default to the production Optuna-tuned values from `DEFAULT_XGB_PARAMS`.

---

## Comparing Results

```bash
wem-experiment compare \
    --baseline data/reference/loocv/ml_results.csv \
    --experiments data/output/enriched_results.csv \
    --labels Enriched

# Multiple experiment files (concatenated)
wem-experiment compare \
    --baseline data/reference/loocv/ml_results.csv \
    --experiments data/output/wide_batch*.csv \
    --labels Wide \
    --hybrid-cutoff 90 \
    --top-n 10

# Save per-station comparison
wem-experiment compare \
    --baseline data/reference/loocv/ml_results.csv \
    --experiments data/output/enriched_results.csv \
    --labels Enriched \
    --save-csv data/output/experiments/comparison.csv
```

Reports include:
1. Overall metrics (pooled and mean-of-station)
2. Per-quantile RMSE summary (use `--quantile-detail` for all 101)
3. Per-station RMSE with winner tags
4. Divergence analysis (stations driving metric differences)
5. Optional hybrid analysis (`--hybrid-cutoff`)

---

## Implementation

### Source Code

| File | Description |
|------|-------------|
| `wem/experiment/runner.py` | Unified CLI, experiment orchestration, wide fold worker |
| `wem/experiment/transforms.py` | Data transforms: `enrich_with_cdf`, `pivot_to_wide`, `wide_preds_to_long`, `wide_to_convnet_arrays` |
| `wem/experiment/convnet.py` | 1D ConvNet model, dataset, and fold worker (PyTorch) |
| `wem/experiment/mlp.py` | Tabular MLP model and fold worker (PyTorch) |
| `wem/experiment/compare.py` | Comparison analysis and reporting |
| `tests/experiment/test_runner.py` | 27 tests (feature builders, fold worker, E2E) |
| `tests/experiment/test_convnet.py` | 15 tests (model, dataset, fold worker, transforms) |
| `tests/experiment/test_mlp.py` | 7 tests (model, fold worker) |
| `tests/experiment/test_transforms.py` | 18 tests (enrich, pivot, conversion) |

### Architecture

The framework reuses production code directly:

| Production code | Used by |
|----------------|---------|
| `train/loocv_xgb.py:run_one_fold()` | Baseline + enriched + hybrid (base model) fold worker |
| `utils/ml.py` (make_features, build_neighbor_map, etc.) | All experiment types |
| `constants.py` (DEFAULT_XGB_PARAMS, WIND_FEATURE_MAP) | All experiment types |

Experiment types differ in **data transform** (none / enrich_with_cdf / pivot_to_wide), **feature selection** (raw wind / CDF columns), and **fold worker** (run_one_fold / run_one_fold_wide).

---

## Experiment Results Summary (285 GS stations)

Every alternative formulation was evaluated against the production baseline via full 285-station LOOCV. The baseline (9-feature XGBoost with MAE loss and monotonic `qnum` constraint) remains the best overall formulation. No experiment improved both pooled and mean-of-station RMSE simultaneously; most degraded at least one metric at scale.

### Results Table

| Experiment | Pooled RMSE | vs Base | Mean-Stn RMSE | vs Base | Stn Wins | Key Finding |
|------------|-------------|---------|---------------|---------|----------|-------------|
| **Baseline** | **0.8624** | — | **0.7352** | — | — | Production formulation |
| Enriched (309 feat) | — | worse | — | worse | — | CDF features memorize, don't generalize |
| Wide (101 models) | 0.8292 | -0.2% | 0.7363 | +4.1% | 52/144 (36%) | Worse mean-stn, gap grows with N |
| ConvNet (500ep, mps) | 1.0563 | +12.2% | 1.0378 | +17.3% | 3/10 | Overfits with ~570 wide rows per fold |
| MLP (100ep, mps) | 0.8759 | +1.6% | 0.7668 | +4.3% | 128/285 | Largest variance; big wins and big losses |
| Hybrid q95 (+ CDF) | — | +4.2% | — | +5.3% | 1/10 | Log transform amplifies tail errors |
| Hybrid q100 (+ CDF) | 0.8581 | -0.5% | 0.7364 | +0.2% | 141/285 | Small q100 gain (-1.4%), wash overall |
| Hybrid q100 (no CDF) | 0.8622 | -0.0% | 0.7368 | +0.2% | 136/285 | Same features, log target — no benefit |
| CDF Context (15 feat) | 0.8692 | +0.8% | 0.7418 | +0.9% | 150/285 | More station wins but larger losses |

### Key Conclusions

1. **The baseline is robust.** Its combination of MAE loss, monotonic `qnum` constraint, and 9 well-chosen features is hard to beat. More complex formulations trade gains at some stations for larger losses at others.

2. **CDF features don't generalize in XGBoost.** Whether using all 303 CDF columns (enriched) or just 6 context columns (q50/q90), the additional features give XGBoost more dimensions to memorize station-specific patterns that fail under LOOCV. The CDF context experiment won more stations (150 vs 135) but had worse aggregate metrics because its losses were larger than its gains.

3. **The tail problem is real but unsolvable with this approach.** Residual analysis confirmed q95-q100 has 3-5x worse RMSE than lower quantiles, with q100 (RMSE=5.19) driven by extreme max wind events (observed up to 54.5 m/s vs max prediction ~27.7 m/s). The hybrid experiment tried log-target transformation and CDF context features to specialize for the tail, but `exp()` back-transform amplifies prediction errors, and the training signal at q100 (~570 rows) is too noisy for a separate model to learn from.

4. **Wide format trades mean-station for pooled.** The 101-independent-model approach showed early promise on pooled RMSE but consistently degraded mean-of-station RMSE, with the gap widening as more stations were added (from +1.3% at 72 stations to +4.1% at 144). Without parameter sharing, per-quantile models overfit to the majority pattern.

5. **Neural networks need more data.** The ConvNet (+12.2% pooled RMSE) overfits badly with only ~570 wide-format training rows per fold — insufficient for learning generalizable CDF-to-CDF corrections. The MLP (+1.6% pooled, +4.3% mean-station) has the highest variance of any experiment, with individual station improvements up to -1.15 m/s but losses up to +1.41 m/s. Both models lack the inductive biases (monotonic constraint, MAE robustness) that make XGBoost effective at this scale.

6. **The error floor is set by observation quality.** Max wind (q100) captures rare extreme events that are inherently noisy and poorly represented in the wind resource models. No feature engineering or model architecture can predict a 54.5 m/s observed maximum from wind model inputs that top out around 28 m/s. The baseline's MAE loss already provides natural robustness to these outliers.

---

## Archived: Enriched Experiment Note

The enriched experiment (XGBoost + 303 CDF features) proved to be a dud at scale — initial gains eroded as more diverse stations were added. The 303 CDF features are constant within each station-height group (~570 unique CDF patterns), so XGBoost memorizes rather than generalizes the high-dimensional CDF structure. The ConvNet experiment addresses this by processing CDFs as structured signals (3-channel 1D convolution) rather than unrelated features.

---

## Archived: Wide Experiment Batch History (2026-03-07 to 2026-03-08)

Batches 1–6 were run with the old `wem-exp-multi-output` CLI (now replaced by `wem-experiment wide`).

| Batch | N | Cumul N | Pooled RMSE (base) | Pooled RMSE (wide) | Mean-Stn RMSE (base) | Mean-Stn RMSE (wide) | Wide Wins | Date |
|-------|---|---------|--------------------|--------------------|----------------------|----------------------|-----------|------|
| batch1–3 | 72 | 72 | 0.8885 | 0.8623 (-3.0%) | 0.7396 | 0.7493 (+1.3%) | 37/72 (51%) | 2026-03-07 |
| batch4 | 24 | 96 | 0.8381 | 0.8216 (-2.0%) | 0.7078 | 0.7210 (+1.9%) | 47/96 (49%) | 2026-03-08 |
| batch5 | 24 | 120 | 0.8559 | 0.8419 (-1.6%) | 0.7236 | 0.7442 (+2.9%) | 47/120 (39%) | 2026-03-08 |
| batch6 | 24 | 144 | 0.8307 | 0.8292 (-0.2%) | 0.7076 | 0.7363 (+4.1%) | 52/144 (36%) | 2026-03-08 |

**Conclusion**: Wide format shows marginally better pooled RMSE but worse mean-of-station RMSE, with the gap growing as more stations are added. The baseline remains the preferred production formulation.
