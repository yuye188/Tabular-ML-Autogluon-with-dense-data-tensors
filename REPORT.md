# Project Analysis Report

## Overview

This project explores **dense tabular data representation** for AutoGluon ML experiments, using datasets from [TabArena](https://github.com/TabArena/tabarena_dataset_curation) and [OpenML](https://www.openml.org/).

---

## Scripts Summary

### [`scripts/getDensityTable.py`](scripts/getDensityTable.py)
Core density analysis script. Defines `getDensitiesPlot()` and `trainAutogluonModels()`.

- **`getDensitiesPlot(df, target_feature, density_threshold=0.1)`**: Iteratively drops the highest-cardinality features from a dataset until the *density* (unique rows / Cartesian product of unique values per column) exceeds a threshold. Returns the reduced DataFrame and metadata.
- **`trainAutogluonModels(...)`**: Trains or loads an AutoGluon `TabularPredictor` (XGB or NN_TORCH) on a 70/30 split, returning the evaluation metric (r2 or accuracy).
- Runs a one-off experiment on OpenML dataset `46904` (acoustic data) to validate the approach.

### [`scripts/tabArenaDensityPlots.py`](scripts/tabArenaDensityPlots.py)
Batch pipeline over all TabArena datasets.

- Iterates every dataset via [`TabArenaIterator`](TabArenaIterator.py) (reads TabArena metadata CSV, fetches each dataset from OpenML).
- For each dataset: computes density before/after feature dropping, trains XGB and NN_TORCH on both the original and dense versions, records metrics in `final_table`.
- Produces Plotly histograms of original and final density distributions (log10 scale).

### [`scripts/generateDatasetForAGtests.py`](scripts/generateDatasetForAGtests.py)
Synthetic dataset generator for CPU vs GPU benchmarking.

- Generates a regression dataset (Gaussian output from 2 discrete + 1 categorical + 1 continuous feature).
- Benchmarks AutoGluon `medium_quality` preset across dataset sizes `[100 → 100k rows]`, comparing CPU-only vs CPU+GPU training time, RMSE, and disk usage (full vs deployment-optimised model).

### [`scripts/createPydantic-models.py`](scripts/createPydantic-models.py)
Pydantic v2 data validation exploration.

- Demonstrates `BaseModel` with strict typing, `Literal` enums, `PositiveInt`, `NonNegativeInt`, and `ValidationError` handling.
- Models the UCI Adult Income dataset schema (`Person` model) and validates sample records against it.

---

## Key Concepts

| Concept | Description |
|---|---|
| **Dataset Density** | `unique_rows / ∏(unique_values_per_col)` — measures how "full" the feature space is |
| **Dense Reduction** | Drop highest-cardinality features until density > 0.1 threshold |
| **AutoGluon** | Used for both regression (r2) and classification (accuracy) with XGB and NN_TORCH |
| **TabArena** | Curated benchmark of 51 OpenML tabular datasets |

---

## Setup (uv)

```bash
# Install uv
curl -Lsf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Install dev tools (Jupyter etc.)
uv sync --extra dev

# Run a script
uv run python scripts/tabArenaDensityPlots.py
```

---

## File Structure

```
.
├── notebooks/                  # Original Jupyter notebooks
├── scripts/                    # Converted .py scripts
├── dense_dfs/                  # Pre-computed dense DataFrames (OpenML IDs)
├── dense_dfs_with_transformed_binary_feature/
├── TabArenaIterator.py         # Iterator over TabArena metadata + OpenML fetch
├── pyproject.toml              # uv/hatch project config
└── REPORT.md