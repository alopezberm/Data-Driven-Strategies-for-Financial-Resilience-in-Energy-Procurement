

"""
test_data_pipeline.py

Unit tests for the core data pipeline modules:
- clean_omip
- clean_weather
- clean_holidays
- merge_data
- split_data

These tests are intentionally lightweight and rely on small synthetic dataframes
so they can run quickly and validate the main expected behavior.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean_holidays import clean_holidays_dataframe
from src.data.clean_omip import clean_omip_dataframe
from src.data.clean_weather import clean_weather_dataframe
from src.data.merge_data import merge_datasets
from src.data.split_data import chronological_train_validation_test_split


# =========================
# Fixtures
# =========================


@pytest.fixture
def raw_omip_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Spot_Price_SPEL": [50.0, 55.0, 60.0],
            "Future_M1_Price": [52.0, 56.0, 61.0],
            "Future_M1_OpenInterest": [1000, 1010, 1020],
        }
    )


@pytest.fixture
def raw_weather_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "temperature_2m_mean": [10.5, 11.0, 9.5],
            "wind_speed_10m_max": [20.0, 22.0, 18.0],
            "Is_weekend": [0, 0, 0],
            "Month": [1, 1, 1],
            "Year": [2024, 2024, 2024],
        }
    )


@pytest.fixture
def raw_holidays_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-12-25"],
            "holiday_name": ["New Year", "Christmas"],
            "Is_national_holiday": [1, 1],
        }
    )


# =========================
# clean_omip tests
# =========================


def test_clean_omip_dataframe_returns_expected_core_columns(raw_omip_df: pd.DataFrame) -> None:
    cleaned_df = clean_omip_dataframe(raw_omip_df)

    assert not cleaned_df.empty
    assert "date" in cleaned_df.columns
    assert "Spot_Price_SPEL" in cleaned_df.columns
    assert "Future_M1_Price" in cleaned_df.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["date"])
    assert cleaned_df["date"].is_monotonic_increasing


# =========================
# clean_weather tests
# =========================


def test_clean_weather_dataframe_parses_date_and_preserves_weather_columns(raw_weather_df: pd.DataFrame) -> None:
    cleaned_df = clean_weather_dataframe(raw_weather_df)

    assert not cleaned_df.empty
    assert "date" in cleaned_df.columns
    assert "temperature_2m_mean" in cleaned_df.columns
    assert "wind_speed_10m_max" in cleaned_df.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["date"])
    assert cleaned_df["date"].is_monotonic_increasing


# =========================
# clean_holidays tests
# =========================


def test_clean_holidays_dataframe_keeps_holiday_flag(raw_holidays_df: pd.DataFrame) -> None:
    cleaned_df = clean_holidays_dataframe(raw_holidays_df)

    assert not cleaned_df.empty
    assert "date" in cleaned_df.columns
    assert "Is_national_holiday" in cleaned_df.columns
    assert "holiday_name" in cleaned_df.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["date"])


# =========================
# merge_data tests
# =========================


def test_merge_datasets_combines_omip_weather_and_holidays(
    raw_omip_df: pd.DataFrame,
    raw_weather_df: pd.DataFrame,
    raw_holidays_df: pd.DataFrame,
) -> None:
    omip_df = clean_omip_dataframe(raw_omip_df)
    weather_df = clean_weather_dataframe(raw_weather_df)
    holidays_df = clean_holidays_dataframe(raw_holidays_df)

    merged_df = merge_datasets(omip_df, weather_df, holidays_df)

    assert not merged_df.empty
    assert len(merged_df) == len(omip_df)
    assert "date" in merged_df.columns
    assert "Spot_Price_SPEL" in merged_df.columns
    assert "temperature_2m_mean" in merged_df.columns
    assert "Is_national_holiday" in merged_df.columns
    assert pd.api.types.is_datetime64_any_dtype(merged_df["date"])
    assert merged_df["date"].is_monotonic_increasing


# =========================
# split_data tests
# =========================


def test_chronological_split_returns_ordered_non_overlapping_partitions() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30, freq="D"),
            "Spot_Price_SPEL": range(30),
        }
    )

    train_df, validation_df, test_df = chronological_train_validation_test_split(
        df,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    assert len(train_df) + len(validation_df) + len(test_df) == len(df)
    assert not train_df.empty
    assert not validation_df.empty
    assert not test_df.empty

    assert train_df["date"].max() < validation_df["date"].min()
    assert validation_df["date"].max() < test_df["date"].min()
    assert train_df["date"].is_monotonic_increasing
    assert validation_df["date"].is_monotonic_increasing
    assert test_df["date"].is_monotonic_increasing


# =========================
# failure-mode tests
# =========================


def test_clean_omip_dataframe_raises_when_date_column_missing() -> None:
    invalid_df = pd.DataFrame(
        {
            "Spot_Price_SPEL": [50.0, 55.0],
            "Future_M1_Price": [52.0, 56.0],
        }
    )

    with pytest.raises(Exception):
        clean_omip_dataframe(invalid_df)



def test_merge_datasets_raises_when_required_date_column_missing() -> None:
    omip_df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "Spot_Price_SPEL": [1, 2]})
    weather_df = pd.DataFrame({"bad_date": pd.date_range("2024-01-01", periods=2), "temperature_2m_mean": [10, 11]})
    holidays_df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "Is_national_holiday": [0, 1]})

    with pytest.raises(Exception):
        merge_datasets(omip_df, weather_df, holidays_df)