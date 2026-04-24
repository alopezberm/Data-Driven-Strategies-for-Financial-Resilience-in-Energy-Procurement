"""
build_feature_dictionary.py

Create a feature dictionary for the modeling dataset.
The dictionary documents each column with a lightweight description,
feature group, and source category.

Usage
-----
python -m src.pipeline.build_feature_dictionary
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import DATE_COLUMN
from src.config.paths import FEATURE_DICTIONARY_FILE, MODELING_DATASET_FILE
from src.utils.logger import get_logger


class FeatureDictionaryError(Exception):
    """Raised when the feature dictionary cannot be built safely."""


logger = get_logger(__name__)


BASE_DESCRIPTIONS = {
    DATE_COLUMN: "Calendar date of the observation.",
    "Spot_Price_SPEL": "Observed daily spot electricity price.",
    "Future_M1_Price": "Observed daily price of the front-month futures contract.",
    "Future_M1_OpenInterest": "Observed daily open interest for the front-month futures contract.",
    "Future_M2_Price": "Observed daily price of the second-month futures contract.",
    "Future_M2_OpenInterest": "Observed daily open interest for the second-month futures contract.",
    "Future_M3_Price": "Observed daily price of the third-month futures contract.",
    "Future_M3_OpenInterest": "Observed daily open interest for the third-month futures contract.",
    "Future_M4_Price": "Observed daily price of the fourth-month futures contract.",
    "Future_M4_OpenInterest": "Observed daily open interest for the fourth-month futures contract.",
    "Future_M5_Price": "Observed daily price of the fifth-month futures contract.",
    "Future_M5_OpenInterest": "Observed daily open interest for the fifth-month futures contract.",
    "Future_M6_Price": "Observed daily price of the sixth-month futures contract.",
    "Future_M6_OpenInterest": "Observed daily open interest for the sixth-month futures contract.",
    "weather_code": "Encoded daily weather condition indicator.",
    "temperature_2m_mean": "Daily mean temperature at 2 meters.",
    "temperature_2m_max": "Daily maximum temperature at 2 meters.",
    "temperature_2m_min": "Daily minimum temperature at 2 meters.",
    "apparent_temperature_mean": "Daily mean apparent temperature.",
    "wind_speed_10m_max": "Daily maximum wind speed at 10 meters.",
    "wind_gusts_10m_max": "Daily maximum wind gust speed at 10 meters.",
    "shortwave_radiation_sum": "Daily accumulated shortwave solar radiation.",
    "precipitation_sum": "Daily total precipitation.",
    "rain_sum": "Daily total rainfall.",
    "snowfall_sum": "Daily total snowfall.",
    "precipitation_hours": "Number of hours with precipitation during the day.",
    "surface_pressure_mean": "Daily mean surface pressure.",
    "et0_fao_evapotranspiration_sum": "Daily reference evapotranspiration.",
    "sunrise": "Daily sunrise timestamp.",
    "sunset": "Daily sunset timestamp.",
    "peninsular_max_temperature": "Maximum temperature observed across the peninsular area.",
    "peninsular_min_temperature": "Minimum temperature observed across the peninsular area.",
    "max_windspeed": "Maximum wind speed indicator.",
    "daily_temperature_range": "Difference between daily maximum and minimum temperature.",
    "rolling_avg_temp": "Rolling average of temperature over time from the raw weather source.",
    "delta_temp_with_previous": "Temperature change relative to the previous day.",
    "std_avg_temperature": "Rolling standard deviation of average temperature from the raw weather source.",
    "Day_of_the_week": "Day of week provided in the raw weather/calendar dataset.",
    "Is_weekend": "Weekend flag provided in the raw weather/calendar dataset.",
    "Month": "Month identifier provided in the raw weather/calendar dataset.",
    "Year": "Year identifier provided in the raw weather/calendar dataset.",
    "Season": "Season identifier provided in the raw weather/calendar dataset.",
    "Is_national_holiday": "National holiday indicator.",
    "holiday_name": "Name of the holiday when available.",
    "units_needed_per_day": "Daily production demand in units.",
    "energy_per_unit_mwh": "Energy required to produce one unit, measured in MWh per unit.",
    "daily_energy_needed_mwh": "Total daily energy demand implied by units and energy intensity.",
    "inventory_holding_cost_per_unit_eur_day": "Inventory holding cost per unit per day in EUR.",
    "max_capacity_units_per_day": "Maximum production capacity in units per day.",
    "max_capacity_mwh_per_day": "Maximum daily energy usage implied by production capacity.",
}


# =========================
# Metadata inference helpers
# =========================

def _infer_feature_group(column_name: str) -> str:
    """Infer a broad feature group from the column name."""
    if column_name == DATE_COLUMN:
        return "identifier"

    if column_name.startswith("Spot_Price"):
        return "spot_market"

    if column_name.startswith("Future_") and "OpenInterest" not in column_name:
        return "futures_market"

    if "OpenInterest" in column_name or column_name.startswith("open_interest") or "_oi_" in column_name:
        return "open_interest"

    if column_name in {
        "weather_code",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_mean",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "shortwave_radiation_sum",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "surface_pressure_mean",
        "et0_fao_evapotranspiration_sum",
        "sunrise",
        "sunset",
        "peninsular_max_temperature",
        "peninsular_min_temperature",
        "max_windspeed",
        "daily_temperature_range",
        "rolling_avg_temp",
        "delta_temp_with_previous",
        "std_avg_temperature",
    }:
        return "weather"

    if column_name in {
        "Day_of_the_week",
        "Is_weekend",
        "Month",
        "Year",
        "Season",
        "Is_national_holiday",
        "holiday_name",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "season",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    }:
        return "calendar_time"

    if "_lag_" in column_name:
        return "lag_feature"

    if "_rolling_" in column_name:
        return "rolling_feature"

    if column_name.startswith("spread_") or "premium" in column_name:
        return "market_structure"

    return "other"



def _infer_source(column_name: str) -> str:
    """Infer a simple source label for the feature."""
    group = _infer_feature_group(column_name)

    source_map = {
        "identifier": "system",
        "spot_market": "omip",
        "futures_market": "omip",
        "open_interest": "omip",
        "weather": "weather",
        "calendar_time": "calendar_or_derived",
        "lag_feature": "derived",
        "rolling_feature": "derived",
        "market_structure": "derived",
        "other": "derived",
    }

    return source_map[group]



def _infer_data_type(series: pd.Series) -> str:
    """Infer a user-friendly data type string from a pandas Series."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "string_or_mixed"



def _infer_description(column_name: str) -> str:
    """Infer a lightweight feature description from column naming patterns."""
    if column_name in BASE_DESCRIPTIONS:
        return BASE_DESCRIPTIONS[column_name]

    if "_lag_" in column_name:
        base_name, lag = column_name.rsplit("_lag_", 1)
        return f"Lagged version of '{base_name}' using {lag} previous time step(s)."

    if "_rolling_" in column_name:
        prefix, operation, window = column_name.rsplit("_", 2)
        base_name = prefix.replace("_rolling", "")
        return (
            f"Rolling {operation} of '{base_name}' computed over a trailing window of {window} period(s), "
            "using past information only."
        )

    if column_name.startswith("spread_spot_vs_") and column_name.endswith("_rel"):
        target_name = column_name.replace("spread_spot_vs_", "").replace("_rel", "")
        return f"Relative spread between the spot price and {target_name}."

    if column_name.startswith("spread_spot_vs_"):
        target_name = column_name.replace("spread_spot_vs_", "")
        return f"Absolute spread between the spot price and {target_name}."

    if column_name.startswith("spread_") and column_name.endswith("_rel"):
        pair_name = column_name.replace("spread_", "").replace("_rel", "")
        return f"Relative spread between futures-market quantities represented by '{pair_name}'."

    if column_name.startswith("spread_"):
        pair_name = column_name.replace("spread_", "")
        return f"Absolute spread between market quantities represented by '{pair_name}'."

    if column_name.startswith("oi_ratio_"):
        return "Ratio between open-interest measures across futures maturities."

    if column_name.endswith("_oi_change_1d"):
        return "One-day absolute change in open interest."

    if column_name.endswith("_oi_change_7d"):
        return "Seven-day absolute change in open interest."

    if column_name.endswith("_oi_pct_change_1d"):
        return "One-day percentage change in open interest."

    if column_name.endswith("_oi_pct_change_7d"):
        return "Seven-day percentage change in open interest."

    if column_name == "open_interest_total":
        return "Total open interest aggregated across available futures maturities."

    if column_name == "front_month_premium":
        return "Absolute premium of the front-month future over the current spot price."

    if column_name == "front_month_premium_rel":
        return "Relative premium of the front-month future over the current spot price."

    return "Derived or project-specific feature used in modeling and decision support."


# =========================
# Main builder
# =========================

def build_feature_dictionary() -> pd.DataFrame:
    """Build and save the feature dictionary from the modeling dataset."""
    logger.info("Starting feature dictionary build...")

    if not MODELING_DATASET_FILE.exists():
        raise FeatureDictionaryError(
            f"Modeling dataset not found: {MODELING_DATASET_FILE}. Run build_modeling_dataset first."
        )

    logger.info(f"Loading modeling dataset from {MODELING_DATASET_FILE}")
    df = pd.read_csv(MODELING_DATASET_FILE)

    rows: list[dict[str, object]] = []
    for column in df.columns:
        rows.append(
            {
                "feature_name": column,
                "description": _infer_description(column),
                "feature_group": _infer_feature_group(column),
                "source": _infer_source(column),
                "data_type": _infer_data_type(df[column]),
                "n_missing": int(df[column].isna().sum()),
                "missing_share": float(df[column].isna().mean()),
            }
        )

    feature_dictionary_df = pd.DataFrame(rows).sort_values(
        ["feature_group", "feature_name"]
    ).reset_index(drop=True)

    logger.info(f"Feature dictionary shape: {feature_dictionary_df.shape}")

    FEATURE_DICTIONARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    feature_dictionary_df.to_csv(FEATURE_DICTIONARY_FILE, index=False)

    logger.info(f"Saved feature dictionary to {FEATURE_DICTIONARY_FILE}")
    return feature_dictionary_df


if __name__ == "__main__":
    dictionary_df = build_feature_dictionary()
    logger.info("Feature dictionary created successfully.")
    logger.info(f"Shape: {dictionary_df.shape}")
    logger.info(f"Saved to: {FEATURE_DICTIONARY_FILE}")
