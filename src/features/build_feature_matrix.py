"""
build_feature_matrix.py

Orchestrator for project feature engineering.
This module combines all deterministic feature builders into a single,
reproducible feature-matrix construction step.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.constants import DATE_COLUMN, PRIMARY_FUTURE_COLUMN, PRIMARY_OPEN_INTEREST_COLUMN, SPOT_PRICE_COLUMN
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
        Sorted copy of the dataframe with a standardized date column.
    """
    if df.empty:
        raise FeatureMatrixError("Input dataframe is empty.")

    if DATE_COLUMN not in df.columns:
        raise FeatureMatrixError(
            f"Input dataframe must contain a '{DATE_COLUMN}' column."
        )

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if df[DATE_COLUMN].isna().any():
        invalid_count = int(df[DATE_COLUMN].isna().sum())
        raise FeatureMatrixError(
            f"Found {invalid_count} invalid date values while building the feature matrix."
        )

    if df[DATE_COLUMN].duplicated().any():
        raise FeatureMatrixError(
            "Input dataframe contains duplicated dates. The feature matrix requires unique chronological rows."
        )

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df



def _validate_output_dataframe(df: pd.DataFrame) -> None:
    """Run final validation checks on the feature matrix."""
    if df.empty:
        raise FeatureMatrixError("The resulting feature matrix is empty.")

    if DATE_COLUMN not in df.columns:
        raise FeatureMatrixError(
            f"The resulting feature matrix does not contain a '{DATE_COLUMN}' column."
        )

    if df[DATE_COLUMN].duplicated().any():
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
        Whether to save the final feature matrix to the configured processed-data path.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with engineered features.
    """
    df = _validate_input_dataframe(df)

    # BUSINESS LOGIC

    # 1. Day-1 Backfilling (Initial Holidays in Futures Markets)
    # Resolves initial NaN values to ensure Lags and Rolling Windows 
    # maintain structural integrity and prevent data leakage.
    future_cols = [col for col in df.columns if 'Future' in col]
    df[future_cols] = df[future_cols].bfill()

    # 2. Liquidity Pruning (Removing distal Open Interest due to extreme scarcity)
    # Enhances model robustness by eliminating highly illiquid features 
    # that could introduce noise during market disruptions.
    illiquid_oi_cols = ['Future_M4_OpenInterest', 'Future_M5_OpenInterest', 'Future_M6_OpenInterest']
    df = df.drop(columns=[col for col in illiquid_oi_cols if col in df.columns], errors='ignore')
    
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
            DATE_COLUMN: pd.date_range("2024-01-01", periods=40, freq="D"),
            SPOT_PRICE_COLUMN: range(40),
            PRIMARY_FUTURE_COLUMN: range(100, 140),
            PRIMARY_OPEN_INTEREST_COLUMN: range(1000, 1040),
            "Future_M2_Price": range(200, 240),
            "Future_M2_OpenInterest": range(2000, 2040),
        }
    )

    feature_df = build_feature_matrix(example_df, save=False)
    print(feature_df.head())
    print(f"Feature matrix shape: {feature_df.shape}")