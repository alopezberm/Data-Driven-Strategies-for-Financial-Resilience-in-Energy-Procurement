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
from src.utils.logger import get_logger


class ModelingDatasetError(Exception):
    """Raised when the modeling dataset cannot be built safely."""

logger = get_logger(__name__)


def build_modeling_dataset() -> pd.DataFrame:
    """Load merged interim data, build features, and save the modeling dataset."""
    logger.info("Starting modeling dataset build...")

    if not MERGED_INTERIM_FILE.exists():
        raise ModelingDatasetError(
            f"Merged interim file not found: {MERGED_INTERIM_FILE}. Run merge_data.py first."
        )

    logger.info(f"Loading merged data from {MERGED_INTERIM_FILE}")
    df = pd.read_csv(MERGED_INTERIM_FILE)

    logger.info("Building feature matrix...")
    feature_df = build_feature_matrix(df, save=False)

    logger.info(f"Feature matrix shape: {feature_df.shape}")

    MODELING_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(MODELING_DATASET_FILE, index=False)

    logger.info(f"Saved modeling dataset to {MODELING_DATASET_FILE}")

    return feature_df


if __name__ == "__main__":
    modeling_df = build_modeling_dataset()
    logger.info("Modeling dataset created successfully.")
    logger.info(f"Shape: {modeling_df.shape}")
    logger.info(f"Saved to: {MODELING_DATASET_FILE}")