"""
build_rolling_features.py

Feature engineering utilities for rolling-window statistics.
These features are computed using only past information by applying
`shift(1)` before rolling operations, which prevents temporal leakage.
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import DATE_COLUMN, DEFAULT_ROLLING_WINDOWS, PRIMARY_FUTURE_COLUMN, TARGET_COLUMN
from src.config.settings import TrainingSettings, get_default_settings
from src.utils.validation import ValidationError, validate_and_sort_by_date


def get_default_rolling_config(training_settings: TrainingSettings | None = None) -> dict[str, dict[str, list[int]]]:
    """Build the default rolling-feature configuration from project settings."""
    if training_settings is None:
        training_settings = get_default_settings().training

    rolling_windows = (
        list(training_settings.rolling_windows)
        if training_settings.rolling_windows
        else list(DEFAULT_ROLLING_WINDOWS)
    )
    short_and_long_windows = sorted({window for window in [3, *rolling_windows] if window > 0})
    medium_and_long_windows = sorted({window for window in rolling_windows if window >= 7})

    return {
        TARGET_COLUMN: {
            "mean": short_and_long_windows,
            "std": short_and_long_windows,
            "min": medium_and_long_windows,
            "max": medium_and_long_windows,
        },
        PRIMARY_FUTURE_COLUMN: {
            "mean": medium_and_long_windows,
            "std": medium_and_long_windows,
        },
        "Future_M1_OpenInterest": {
            "mean": medium_and_long_windows,
            "std": medium_and_long_windows,
        },
    }


SUPPORTED_OPERATIONS = {"mean", "std", "min", "max"}


class RollingFeaturesError(Exception):
    """Raised when rolling features cannot be created safely."""


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort a time-series input dataframe."""
    try:
        return validate_and_sort_by_date(df, df_name="rolling features input")
    except ValidationError as exc:
        raise RollingFeaturesError(str(exc)) from exc



def _validate_rolling_config(rolling_config: dict[str, dict[str, list[int]]]) -> None:
    """Validate rolling configuration structure and values."""
    if not rolling_config:
        raise RollingFeaturesError("rolling_config cannot be empty.")

    for column, operations in rolling_config.items():
        if not isinstance(column, str):
            raise RollingFeaturesError(
                "All rolling_config keys must be column names (strings)."
            )

        if not isinstance(operations, dict) or not operations:
            raise RollingFeaturesError(
                f"Rolling operations for column '{column}' must be a non-empty dictionary."
            )

        for operation, windows in operations.items():
            if operation not in SUPPORTED_OPERATIONS:
                raise RollingFeaturesError(
                    f"Unsupported rolling operation '{operation}' for column '{column}'. "
                    f"Supported operations: {sorted(SUPPORTED_OPERATIONS)}"
                )

            if not isinstance(windows, list) or not windows:
                raise RollingFeaturesError(
                    f"Window list for operation '{operation}' in column '{column}' must be a non-empty list."
                )

            if len(set(windows)) != len(windows):
                raise RollingFeaturesError(
                    f"Window list for operation '{operation}' in column '{column}' contains duplicated values."
                )

            if any((not isinstance(window, int) or window <= 0) for window in windows):
                raise RollingFeaturesError(
                    f"All rolling windows for operation '{operation}' in column '{column}' must be positive integers."
                )


# =========================
# Core feature builder
# =========================

def _apply_rolling_operation(series: pd.Series, operation: str, window: int) -> pd.Series:
    """Apply one supported rolling operation using only past information."""
    shifted = series.shift(1)
    rolling_obj = shifted.rolling(window=window)

    if operation == "mean":
        return rolling_obj.mean()
    if operation == "std":
        return rolling_obj.std()
    if operation == "min":
        return rolling_obj.min()
    if operation == "max":
        return rolling_obj.max()

    raise RollingFeaturesError(f"Unsupported rolling operation encountered: {operation}")



def _create_rolling_features(
    df: pd.DataFrame,
    rolling_config: dict[str, dict[str, list[int]]],
) -> pd.DataFrame:
    """
    Create rolling-window features for selected columns.

    Missing source columns are skipped intentionally to keep the function robust
    to evolving datasets.
    """
    df = df.copy()

    for column, operations in rolling_config.items():
        if column not in df.columns:
            continue

        for operation, windows in operations.items():
            for window in sorted(set(windows)):
                feature_name = f"{column}_rolling_{operation}_{window}"
                df[feature_name] = _apply_rolling_operation(df[column], operation, window)

    return df


# =========================
# Public API
# =========================

def build_rolling_features(
    df: pd.DataFrame,
    rolling_config: dict[str, dict[str, list[int]]] | None = None,
) -> pd.DataFrame:
    """
    Add rolling-window features to a time-ordered dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the configured date column and time-dependent variables.
    rolling_config : dict[str, dict[str, list[int]]] | None, optional
        Dictionary mapping column names to rolling operations and their windows.
        If None, a default configuration tailored to the energy procurement
        project is used.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with rolling features.
    """
    df = _validate_input_dataframe(df)
    rolling_config = get_default_rolling_config() if rolling_config is None else rolling_config
    _validate_rolling_config(rolling_config)

    df = _create_rolling_features(df, rolling_config)
    return df


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2024-01-01", periods=40, freq="D"),
            TARGET_COLUMN: range(40),
            PRIMARY_FUTURE_COLUMN: range(100, 140),
            "Future_M1_OpenInterest": range(1000, 1040),
        }
    )

    transformed_df = build_rolling_features(example_df)
    print(transformed_df.tail())