

"""
test_feature_engineering.py

Unit tests for the feature-engineering layer:
- build_time_features
- build_lag_features
- build_rolling_features
- build_future_features
- build_feature_matrix

These tests are intentionally lightweight and use a small synthetic dataframe.
They are also slightly defensive in how they resolve function names, so they
remain usable if the implementation uses `add_*` or `build_*` naming.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import build_feature_matrix as build_feature_matrix_module
from src.features import build_future_features as build_future_features_module
from src.features import build_lag_features as build_lag_features_module
from src.features import build_rolling_features as build_rolling_features_module
from src.features import build_time_features as build_time_features_module


# =========================
# Helpers
# =========================


def _resolve_callable(module, candidate_names: list[str]):
    """Return the first callable found in a module from a list of candidate names."""
    for name in candidate_names:
        func = getattr(module, name, None)
        if callable(func):
            return func
    raise AttributeError(
        f"None of the expected callables were found in module {module.__name__}: {candidate_names}"
    )


BUILD_TIME_FEATURES = _resolve_callable(
    build_time_features_module,
    ["build_time_features", "add_time_features"],
)
BUILD_LAG_FEATURES = _resolve_callable(
    build_lag_features_module,
    ["build_lag_features", "add_lag_features"],
)
BUILD_ROLLING_FEATURES = _resolve_callable(
    build_rolling_features_module,
    ["build_rolling_features", "add_rolling_features"],
)
BUILD_FUTURE_FEATURES = _resolve_callable(
    build_future_features_module,
    ["build_future_features", "add_future_features"],
)
BUILD_FEATURE_MATRIX = _resolve_callable(
    build_feature_matrix_module,
    ["build_feature_matrix"],
)


@pytest.fixture
def base_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "Spot_Price_SPEL": [50 + i for i in range(40)],
            "Future_M1_Price": [52 + i for i in range(40)],
            "Future_M2_Price": [53 + i for i in range(40)],
            "Future_M1_OpenInterest": [1000 + 2 * i for i in range(40)],
            "Future_M2_OpenInterest": [1100 + 2 * i for i in range(40)],
            "temperature_2m_mean": [10 + (i % 5) for i in range(40)],
            "wind_speed_10m_max": [20 + (i % 3) for i in range(40)],
            "Is_national_holiday": [1 if i in [0, 10] else 0 for i in range(40)],
        }
    )


# =========================
# build_time_features tests
# =========================


def test_build_time_features_adds_calendar_columns(base_feature_df: pd.DataFrame) -> None:
    featured_df = BUILD_TIME_FEATURES(base_feature_df)

    assert not featured_df.empty
    assert len(featured_df) == len(base_feature_df)
    assert "date" in featured_df.columns
    assert pd.api.types.is_datetime64_any_dtype(featured_df["date"])

    expected_any = {
        "day_of_week",
        "month",
        "year",
        "is_weekend",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
    }
    assert any(column in featured_df.columns for column in expected_any)


# =========================
# build_lag_features tests
# =========================


def test_build_lag_features_creates_lagged_columns(base_feature_df: pd.DataFrame) -> None:
    featured_df = BUILD_LAG_FEATURES(base_feature_df)

    lag_columns = [column for column in featured_df.columns if "_lag_" in column]

    assert not featured_df.empty
    assert len(featured_df) == len(base_feature_df)
    assert len(lag_columns) > 0
    assert any("Spot_Price_SPEL" in column for column in lag_columns)


# =========================
# build_rolling_features tests
# =========================


def test_build_rolling_features_creates_rolling_columns(base_feature_df: pd.DataFrame) -> None:
    featured_df = BUILD_ROLLING_FEATURES(base_feature_df)

    rolling_columns = [column for column in featured_df.columns if "_rolling_" in column]

    assert not featured_df.empty
    assert len(featured_df) == len(base_feature_df)
    assert len(rolling_columns) > 0
    assert any("Spot_Price_SPEL" in column for column in rolling_columns)


# =========================
# build_future_features tests
# =========================


def test_build_future_features_creates_market_structure_features(base_feature_df: pd.DataFrame) -> None:
    featured_df = BUILD_FUTURE_FEATURES(base_feature_df)

    derived_columns = [
        column
        for column in featured_df.columns
        if ("spread" in column.lower())
        or ("premium" in column.lower())
        or ("open_interest_total" == column)
        or ("oi_ratio" in column.lower())
    ]

    assert not featured_df.empty
    assert len(featured_df) == len(base_feature_df)
    assert len(derived_columns) > 0


# =========================
# build_feature_matrix tests
# =========================


def test_build_feature_matrix_combines_multiple_feature_families(base_feature_df: pd.DataFrame) -> None:
    featured_df = BUILD_FEATURE_MATRIX(base_feature_df, save=False)

    assert not featured_df.empty
    assert len(featured_df) == len(base_feature_df)
    assert "date" in featured_df.columns
    assert featured_df.shape[1] > base_feature_df.shape[1]

    assert any("_lag_" in column for column in featured_df.columns)
    assert any("_rolling_" in column for column in featured_df.columns)
    assert any(
        ("spread" in column.lower()) or ("premium" in column.lower())
        for column in featured_df.columns
    )
    assert any(
        column in featured_df.columns
        for column in ["day_of_week", "is_weekend", "day_of_week_sin", "month_sin"]
    )


# =========================
# Failure-mode tests
# =========================


def test_build_time_features_raises_when_date_missing() -> None:
    invalid_df = pd.DataFrame(
        {
            "Spot_Price_SPEL": [50, 55, 60],
            "Future_M1_Price": [52, 56, 61],
        }
    )

    with pytest.raises(Exception):
        BUILD_TIME_FEATURES(invalid_df)



def test_build_feature_matrix_raises_when_date_missing() -> None:
    invalid_df = pd.DataFrame(
        {
            "Spot_Price_SPEL": [50, 55, 60],
            "Future_M1_Price": [52, 56, 61],
            "Future_M1_OpenInterest": [1000, 1010, 1020],
        }
    )

    with pytest.raises(Exception):
        BUILD_FEATURE_MATRIX(invalid_df, save=False)