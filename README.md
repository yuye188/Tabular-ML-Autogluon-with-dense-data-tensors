# Tabular ML with AutoGluon and Dense Data Tensors

This repository investigates whether reducing tabular datasets to their **dense subspace** — a subset of features where the observed combinations of values cover a high fraction of the theoretical combinatorial space — improves the predictive performance of AutoGluon models (XGBoost and Neural Network).

Datasets are sourced from [OpenML](https://www.openml.org/) via the [TabArena](https://github.com/TabArena/tabarena_dataset_curation) benchmark collection.

---

## Repository Structure

```
.
├── TabArenaIterator.py                        # Iterator utility: streams TabArena datasets from OpenML
├── pydantic_create_model.py                   # Utility: auto-generates Pydantic models from a DataFrame
├── pyproject.toml                             # Project dependencies (uv / pip)
│
├── scripts/
│   ├── getDensityTable.py                     # Main experiment: density reduction + AutoGluon training
│   ├── tabArenaDensityPlots.py                # Density analysis and histogram/scatter plots
│   ├── generateDatasetForAGtests.py           # Synthetic dataset generation + CPU vs GPU benchmarks
│   └── createPydantic-models.py              # Pydantic model generation from OpenML datasets
│
├── notebooks/                                 # Jupyter notebook versions of the scripts above
│   ├── getDensityTable.ipynb
│   ├── tabArenaDensityPlots.ipynb
│   ├── generateDatasetForAGtests.ipynb
│   └── createPydantic-models.ipynb
│
├── dense_dfs/                                 # Dense subspace CSVs (one per dataset, by OpenML ID)
│   └── <dataset_id>.csv
│
├── dense_dfs_with_transformed_binary_feature/ # Dense CSVs after binary-feature pivot transformation
│   └── <dataset_id>.csv
│
├── final_table.csv                            # Aggregated results across all datasets
├── final_table_medium_preset.csv              # Results with AutoGluon medium preset
└── final_table_best_preset.csv               # Results with AutoGluon best preset
```

---

## Key Utility: `TabArenaIterator`

[`TabArenaIterator`](TabArenaIterator.py) is a Python iterator that reads the TabArena metadata CSV and lazily downloads each dataset from OpenML one at a time.

```python
from TabArenaIterator import TabArenaIterator

TABARENA_URL = (
    "https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation"
    "/refs/heads/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv"
)

iterator = TabArenaIterator(TABARENA_URL)
for row, df in iterator:
    print(f"Dataset ID : {row['dataset_id']}")
    print(f"Dataset name: {row['dataset_name']}")
    print(f"Target      : {row['target_feature']}")
    print(f"Problem type: {row['problem_type']}")
    print(df.head())
```

Each iteration yields:
- `row` — a pandas `Series` with metadata (id, name, target feature, problem type, number of classes, …)
- `df` — the full dataset as a `pandas.DataFrame` downloaded from OpenML

---

## Example: Downloading Datasets and Inspecting Features

The script below downloads **three** TabArena datasets, saves each as a CSV under `downloaded_datasets/`, and prints the independent variables (features) and the target variable for each one.

```python
import os
import pandas as pd
from TabArenaIterator import TabArenaIterator

TABARENA_URL = (
    "https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation"
    "/refs/heads/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv"
)

OUTPUT_DIR = "downloaded_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_DATASETS = 3

iterator = TabArenaIterator(TABARENA_URL)
for i, (row, df) in enumerate(iterator):
    if i >= MAX_DATASETS:
        break

    dataset_id   = row["dataset_id"]
    dataset_name = row["dataset_name"]
    target       = row["target_feature"]

    # Independent variables (features)
    features = [col for col in df.columns if col != target]

    print(f"=== Dataset {i+1}: {dataset_name} (ID {dataset_id}) ===")
    print(f"  Target variable : {target}")
    print(f"  Features ({len(features)}): {features}")
    print(f"  Shape           : {df.shape}")
    print()

    # Save to CSV
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_id}.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")
    print()
```

Running this script produces output similar to:

```
=== Dataset 1: airfoil (ID 46904) ===
  Target variable : scaled-sound-pressure
  Features (5): ['frequency', 'attack-angle', 'chord-length', 'free-stream-velocity', 'suction-side-displacement-thickness']
  Shape           : (1503, 6)

  Saved to downloaded_datasets/46904.csv

=== Dataset 2: ...
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Tabular-ML-Autogluon-with-dense-data-tensors.git
cd Tabular-ML-Autogluon-with-dense-data-tensors

# Create and activate a virtual environment (uv recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Install dev extras (Jupyter)
uv pip install -e ".[dev]"
```

---

## Scripts Overview

| Script | Description |
|--------|-------------|
| [`scripts/getDensityTable.py`](scripts/getDensityTable.py) | Core experiment: iterates all TabArena datasets, computes the dense subspace via `getDensitiesPlot`, trains XGBoost and NN_TORCH models on both the original and dense datasets, and saves results to `final_table*.csv` |
| [`scripts/tabArenaDensityPlots.py`](scripts/tabArenaDensityPlots.py) | Computes density statistics across all datasets and generates histogram / scatter plots with Plotly |
| [`scripts/generateDatasetForAGtests.py`](scripts/generateDatasetForAGtests.py) | Generates a synthetic tabular dataset and benchmarks AutoGluon training time and RMSE across CPU and GPU |
| [`scripts/createPydantic-models.py`](scripts/createPydantic-models.py) | Demonstrates automatic Pydantic model generation from OpenML datasets using `pydantic_create_model.py` |

---

## Core Concept: Density Reduction

Given a dataset with features **F₁, F₂, …, Fₙ** and target **y**, the *density* of the feature space is defined as:

```
density = number_of_unique_rows / product_of_cardinalities(F₁, …, Fₙ)
```

The `getDensitiesPlot` function (defined in [`scripts/getDensityTable.py`](scripts/getDensityTable.py:160)) iteratively drops the highest-cardinality feature until the density exceeds a threshold (default `0.1`). The resulting **dense subspace** is then used to train AutoGluon models and compared against models trained on the full feature set.