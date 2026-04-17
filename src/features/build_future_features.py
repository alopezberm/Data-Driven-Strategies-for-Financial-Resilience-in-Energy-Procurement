"""
build_future_features.py

Feature engineering utilities for futures-market variables.
These features focus on economically meaningful relationships between
spot prices, futures prices, and open interest, helping connect
forecasting with procurement and hedging decisions.
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import (
    DATE_COLUMN,
    FUTURE_PRICE_COLUMNS,
    OPEN_INTEREST_COLUMNS,
    PRIMARY_FUTURE_COLUMN,
    PRIMARY_OPEN_INTEREST_COLUMN,
    SPOT_PRICE_COLUMN,
)


class FutureFeaturesError(Exception):
    """Raised when futures-market features cannot be created safely."""


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the input dataframe is suitable for futures feature generation.

    Returns
    -------
    pd.DataFrame
        Sorted copy of the dataframe.
    """
    if df.empty:
        raise FutureFeaturesError("Input dataframe is empty.")

    if DATE_COLUMN not in df.columns:
        raise FutureFeaturesError(
            f"Input dataframe must contain a '{DATE_COLUMN}' column."
        )

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if df[DATE_COLUMN].isna().any():
        invalid_count = int(df[DATE_COLUMN].isna().sum())
        raise FutureFeaturesError(
            f"Found {invalid_count} invalid date values while building futures features."
        )

    if df[DATE_COLUMN].duplicated().any():
        raise FutureFeaturesError(
            "Input dataframe contains duplicated dates. Futures features require unique chronological rows."
        )

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


# =========================
# Core feature builders
# =========================

def _add_spot_vs_future_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add absolute and relative spreads between spot and futures prices.

    These features capture whether the market is pricing futures above or below
    the current spot level.
    """
    df = df.copy()

    if SPOT_PRICE_COLUMN not in df.columns:
        return df

    future_price_columns = [col for col in FUTURE_PRICE_COLUMNS if col in df.columns]

    for future_col in future_price_columns:
        maturity_label = future_col.replace("_Price", "").lower()
        abs_name = f"spread_spot_vs_{maturity_label}"
        rel_name = f"spread_spot_vs_{maturity_label}_rel"

        df[abs_name] = df[SPOT_PRICE_COLUMN] - df[future_col]
        df[rel_name] = (df[SPOT_PRICE_COLUMN] - df[future_col]) / df[future_col].replace(0, pd.NA)

    return df



def _add_futures_curve_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spreads across futures maturities to capture curve shape.

    These features help identify term-structure signals such as contango or
    backwardation.
    """
    df = df.copy()

    ordered_pairs = [
        ("Future_M1_Price", "Future_M2_Price"),
        ("Future_M2_Price", "Future_M3_Price"),
        ("Future_M3_Price", "Future_M4_Price"),
        ("Future_M4_Price", "Future_M5_Price"),
        ("Future_M5_Price", "Future_M6_Price"),
        ("Future_M1_Price", "Future_M3_Price"),
        ("Future_M1_Price", "Future_M6_Price"),
    ]

    for left_col, right_col in ordered_pairs:
        if left_col in df.columns and right_col in df.columns:
            left_label = left_col.replace("_Price", "").lower()
            right_label = right_col.replace("_Price", "").lower()

            abs_name = f"spread_{left_label}_vs_{right_label}"
            rel_name = f"spread_{left_label}_vs_{right_label}_rel"

            df[abs_name] = df[left_col] - df[right_col]
            df[rel_name] = (df[left_col] - df[right_col]) / df[right_col].replace(0, pd.NA)

    return df



def _add_open_interest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add open-interest level, total, ratio, and change features.

    Open interest may proxy market participation, liquidity, or positioning.
    """
    df = df.copy()

    available_oi_columns = [col for col in OPEN_INTEREST_COLUMNS if col in df.columns]

    if available_oi_columns:
        df["open_interest_total"] = df[available_oi_columns].sum(axis=1, min_count=1)

    ordered_oi_pairs = [
        ("Future_M1_OpenInterest", "Future_M2_OpenInterest"),
        ("Future_M1_OpenInterest", "Future_M3_OpenInterest"),
    ]

    for left_col, right_col in ordered_oi_pairs:
        if left_col in df.columns and right_col in df.columns:
            left_label = left_col.replace("_OpenInterest", "").lower()
            right_label = right_col.replace("_OpenInterest", "").lower()
            ratio_name = f"oi_ratio_{left_label}_vs_{right_label}"
            df[ratio_name] = df[left_col] / df[right_col].replace(0, pd.NA)

    for oi_col in available_oi_columns:
        label = oi_col.replace("_OpenInterest", "").lower()
        df[f"{label}_oi_change_1d"] = df[oi_col].diff(1)
        df[f"{label}_oi_change_7d"] = df[oi_col].diff(7)
        df[f"{label}_oi_pct_change_1d"] = df[oi_col].pct_change(1)
        df[f"{label}_oi_pct_change_7d"] = df[oi_col].pct_change(7)

    return df



def _add_front_month_premium_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add focused features around the front-month contract.

    These are especially relevant because the project's hedging logic is likely
    to rely mostly on M1 futures.
    """
    df = df.copy()

    if SPOT_PRICE_COLUMN in df.columns and PRIMARY_FUTURE_COLUMN in df.columns:
        df["front_month_premium"] = df[PRIMARY_FUTURE_COLUMN] - df[SPOT_PRICE_COLUMN]
        df["front_month_premium_rel"] = (
            df[PRIMARY_FUTURE_COLUMN] - df[SPOT_PRICE_COLUMN]
        ) / df[SPOT_PRICE_COLUMN].replace(0, pd.NA)

    return df


# =========================
# Public API
# =========================

def build_future_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add futures-market and term-structure features to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the configured date column and market variables.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with futures-related features.
    """
    df = _validate_input_dataframe(df)
    df = _add_spot_vs_future_spreads(df)
    df = _add_futures_curve_spreads(df)
    df = _add_open_interest_features(df)
    df = _add_front_month_premium_features(df)

    return df


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2024-01-01", periods=10, freq="D"),
            SPOT_PRICE_COLUMN: [50, 55, 52, 60, 70, 65, 62, 58, 61, 63],
            PRIMARY_FUTURE_COLUMN: [54, 56, 55, 59, 66, 64, 63, 60, 62, 64],
            "Future_M2_Price": [55, 57, 56, 60, 67, 65, 64, 61, 63, 65],
            "Future_M3_Price": [56, 58, 57, 61, 68, 66, 65, 62, 64, 66],
            PRIMARY_OPEN_INTEREST_COLUMN: [1000, 1010, 1020, 1030, 1015, 1005, 995, 980, 990, 1000],
            "Future_M2_OpenInterest": [900, 905, 910, 920, 915, 910, 900, 890, 895, 905],
            "Future_M3_OpenInterest": [400, 410, 420, 430, 425, 420, 418, 415, 417, 419],
        }
    )

    transformed_df = build_future_features(example_df)
    print(transformed_df.head())