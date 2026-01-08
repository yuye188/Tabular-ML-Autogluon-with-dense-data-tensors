# Tabular-ML-Autogluon-with-dense-data-tensors

Explorative Analysis of the application of techniques for dense data tensors techniques to tabArena datasets.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yuye188/Tabular-ML-Autogluon-with-dense-data-tensors.git
cd Tabular-ML-Autogluon-with-dense-data-tensors
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies with uv:

```bash
# Install uv if you don't have it
pip install uv

# Install project dependencies
uv pip install -r requirements.txt
```

Alternatively, if you prefer using pip directly:

```bash
pip install -r requirements.txt
```

## Structure of this repository
- `tabArenaDensityPlots.ipynb:` It contains the main process of the research. It displays the results stored in other subfolders, mainly executed by the test script `test scripts/getDensityTable.ipynb` and subsequent analysis.
- `TabArenaIterator.py:` It implements an iterator that processes all TabArena datasets one by one.
- `data folder:` This folder contains processed data from the original TabArena datasets.
- `scripts:` This is the subfolder where the scripts that perform heavy computational steps, such as density calculations and model training, are located.
- `pydanticModelsForTabArenaDatasets:` This is the folder that contains the experiments related to the Pydantic Models for dataset validation purposes.
