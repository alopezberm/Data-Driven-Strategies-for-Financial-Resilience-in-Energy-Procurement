"""
build_modeling_dataset.py

Create the full modeling dataset by applying feature engineering
to the merged interim dataset.

Usage
-----
python -m src.pipeline.build_modeling_dataset
"""

from __future__ import annotations

import pandas as pd

from src.config.paths import MERGED_INTERIM_FILE, MODELING_DATASET_FILE
from src.features.build_feature_matrix import build_feature_matrix


class ModelingDatasetError(Exception):
    """Raised when the modeling dataset cannot be built safely."""


def build_modeling_dataset() -> pd.DataFrame:
    """Load merged interim data, build features, and save the modeling dataset."""
    if not MERGED_INTERIM_FILE.exists():
        raise ModelingDatasetError(
            f"Merged interim file not found: {MERGED_INTERIM_FILE}. Run merge_data.py first."
        )

    df = pd.read_csv(MERGED_INTERIM_FILE)
    feature_df = build_feature_matrix(df, save=False)
    feature_df.to_csv(MODELING_DATASET_FILE, index=False)

    return feature_df


if __name__ == "__main__":
    modeling_df = build_modeling_dataset()
    print("Modeling dataset created successfully.")
    print(f"Shape: {modeling_df.shape}")
    print(f"Saved to: {MODELING_DATASET_FILE}")