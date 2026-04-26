"""
_simulation_utils.py

Internal helpers shared across backtesting simulation modules.
Not part of the public backtesting API.
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import DEFAULT_DAILY_VOLUME


class SimulationUtilsError(Exception):
    """Raised when a shared simulation utility fails."""


def ensure_volume_column(
    df: pd.DataFrame,
    volume_column: str,
    default_daily_volume: float = DEFAULT_DAILY_VOLUME,
) -> pd.DataFrame:
    """
    Ensure a valid daily energy volume column exists in the dataframe.

    If the column is absent it is populated with `default_daily_volume`.
    Raises SimulationUtilsError if any value is non-numeric or negative.
    """
    result_df = df.copy()

    if volume_column not in result_df.columns:
        result_df[volume_column] = default_daily_volume

    result_df[volume_column] = pd.to_numeric(result_df[volume_column], errors="coerce")

    if result_df[volume_column].isna().any():
        raise SimulationUtilsError(
            f"Column '{volume_column}' contains invalid or missing volumes."
        )

    if (result_df[volume_column] < 0).any():
        raise SimulationUtilsError(
            f"Column '{volume_column}' contains negative volumes."
        )

    return result_df
