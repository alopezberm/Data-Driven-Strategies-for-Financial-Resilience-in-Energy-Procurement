

"""
build_feature_matrix.py

Orchestrator for project feature engineering.
This module combines all deterministic feature builders into a single,
reproducible feature-matrix construction step.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import MODELING_DATASET_FILE
from src.features.build_future_features import build_future_features
from src.features.build_lag_features import build_lag_features
from src.features.build_rolling_features import build_rolling_features
from src.features.build_time_features import build_time_features


class FeatureMatrixError(Exception):
    """Raised when the feature matrix cannot be built safely."""


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the input dataframe is suitable for feature construction.

    Returns
    -------
    pd.DataFrame
        Sorted copy of the dataframe with a standardized `date` column.
    """
    if df.empty:
        raise FeatureMatrixError("Input dataframe is empty.")

    if "date" not in df.columns:
        raise FeatureMatrixError("Input dataframe must contain a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        invalid_count = int(df["date"].isna().sum())
        raise FeatureMatrixError(
            f"Found {invalid_count} invalid date values while building the feature matrix."
        )

    if df["date"].duplicated().any():
        raise FeatureMatrixError(
            "Input dataframe contains duplicated dates. The feature matrix requires unique chronological rows."
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df



def _validate_output_dataframe(df: pd.DataFrame) -> None:
    """Run final validation checks on the feature matrix."""
    if df.empty:
        raise FeatureMatrixError("The resulting feature matrix is empty.")

    if "date" not in df.columns:
        raise FeatureMatrixError("The resulting feature matrix does not contain a 'date' column.")

    if df["date"].duplicated().any():
        raise FeatureMatrixError("The resulting feature matrix contains duplicated dates.")


# =========================
# Save helper
# =========================

def _save_feature_matrix(df: pd.DataFrame, output_path: Path = MODELING_DATASET_FILE) -> None:
    """Save the feature matrix to the configured processed-data location."""
    df.to_csv(output_path, index=False)


# =========================
# Public API
# =========================

def build_feature_matrix(
    df: pd.DataFrame,
    use_time_features: bool = True,
    use_lag_features: bool = True,
    use_rolling_features: bool = True,
    use_future_features: bool = True,
    save: bool = False,
) -> pd.DataFrame:
    """
    Build the project's full feature matrix by applying modular feature blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, typically the merged interim dataset.
    use_time_features : bool, optional
        Whether to add deterministic calendar/time features.
    use_lag_features : bool, optional
        Whether to add lag-based features.
    use_rolling_features : bool, optional
        Whether to add rolling-window statistics.
    use_future_features : bool, optional
        Whether to add futures-market and term-structure features.
    save : bool, optional
        Whether to save the final feature matrix to
        `data/processed/modeling_dataset.csv`.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with engineered features.
    """
    df = _validate_input_dataframe(df)

    if use_time_features:
        df = build_time_features(df)

    if use_lag_features:
        df = build_lag_features(df)

    if use_rolling_features:
        df = build_rolling_features(df)

    if use_future_features:
        df = build_future_features(df)

    _validate_output_dataframe(df)

    if save:
        _save_feature_matrix(df)

    return df


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "Spot_Price_SPEL": range(40),
            "Future_M1_Price": range(100, 140),
            "Future_M1_OpenInterest": range(1000, 1040),
            "Future_M2_Price": range(200, 240),
            "Future_M2_OpenInterest": range(2000, 2040),
        }
    )

    feature_df = build_feature_matrix(example_df, save=False)
    print(feature_df.head())
    print(f"Feature matrix shape: {feature_df.shape}")