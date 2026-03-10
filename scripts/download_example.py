import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # locate TabArenaIterator

import pandas as pd
from TabArenaIterator import TabArenaIterator

TABARENA_URL = (
    "https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation"
    "/refs/heads/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv"
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "downloaded_datasets")
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

# Made with Bob
