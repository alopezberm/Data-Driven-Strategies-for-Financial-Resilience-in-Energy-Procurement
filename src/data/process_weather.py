"""
process_weather.py

Batch processing and aggregation engine for historical weather data.
Implements a dual-weighting strategy (population vs. surface area) to
generate a consolidated national dataset for energy market analysis.
"""

from pathlib import Path

import holidays
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


class WeatherProcessingError(Exception):
    """Raised when the weather batch processing pipeline fails."""


# Columns aggregated by population weight (temperature / apparent conditions).
_POP_COLS = [
    "weather_code",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
]

# Columns aggregated by surface-area weight (wind / radiation / precipitation).
_SURF_COLS = [
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "surface_pressure_mean",
    "et0_fao_evapotranspiration_sum",
]

_NON_PENINSULAR_PROVINCES = {
    "Canarias",
    "Las Palmas",
    "Santa Cruz de Tenerife",
    "Baleares",
    "Illes Balears",
    "Ceuta",
    "Melilla",
}


def aggregate_weather_batches(
    raw_weather_dir: Path,
    provinces_info_file: Path,
    output_file: Path,
) -> None:
    """
    Load two raw weather batch files, apply dual-weighting aggregation, engineer
    national-level features, and write the result to `output_file`.

    Parameters
    ----------
    raw_weather_dir : Path
        Directory containing the two raw weather batch CSVs.
    provinces_info_file : Path
        CSV with province-level population and surface-area weight columns.
    output_file : Path
        Destination path for the processed national weather dataset.
    """
    logger.info("Step 1: Loading raw meteorological batches...")

    batch_1_matches = list(raw_weather_dir.glob("*top0-10*.csv"))
    batch_2_matches = list(raw_weather_dir.glob("*top11-51*.csv"))

    if not batch_1_matches or not batch_2_matches:
        raise WeatherProcessingError(
            f"Expected two batch files in {raw_weather_dir} matching "
            "'*top0-10*.csv' and '*top11-51*.csv', but one or both were not found."
        )

    df_raw = pd.concat(
        [pd.read_csv(batch_1_matches[0]), pd.read_csv(batch_2_matches[0])],
        ignore_index=True,
    )

    if "city" in df_raw.columns:
        df_raw["prov"] = df_raw["city"]

    logger.info(f"Step 2: Processing weights from {provinces_info_file.name}...")
    df_weights = pd.read_csv(provinces_info_file, sep=",")

    provinces_map = (
        df_weights[["prov", "%Poblacion", "%Superficie2"]]
        .copy()
        .rename(columns={"%Poblacion": "pop_weight", "%Superficie2": "surf_weight"})
    )

    logger.info("Step 3: Applying dual-weighting strategy...")
    df_merged = pd.merge(df_raw, provinces_map, on="prov", how="left")
    df_merged["date"] = pd.to_datetime(df_merged["date"], utc=True).dt.tz_localize(None)

    for col in _POP_COLS:
        df_merged[f"{col}_w"] = df_merged[col] * df_merged["pop_weight"]
    for col in _SURF_COLS:
        df_merged[f"{col}_w"] = df_merged[col] * df_merged["surf_weight"]

    agg_map = {f"{col}_w": "sum" for col in _POP_COLS + _SURF_COLS}
    agg_map.update({"pop_weight": "sum", "surf_weight": "sum"})
    df_agg = df_merged.groupby("date").agg(agg_map)

    for col in _POP_COLS:
        df_agg[col] = df_agg[f"{col}_w"] / df_agg["pop_weight"]
    for col in _SURF_COLS:
        df_agg[col] = df_agg[f"{col}_w"] / df_agg["surf_weight"]

    # Standard deviation of temperature across provinces — kept for EDA.
    df_agg["std_avg_temperature"] = df_merged.groupby("date")["temperature_2m_mean"].std()

    logger.info("Step 4: Engineering advanced features...")
    mad_data = df_merged[df_merged["prov"].str.contains("Madrid", case=False)].set_index("date")
    df_agg["sunrise"] = mad_data["sunrise"]
    df_agg["sunset"] = mad_data["sunset"]

    pen_data = df_merged[~df_merged["prov"].isin(_NON_PENINSULAR_PROVINCES)]
    df_agg["peninsular_max_temperature"] = pen_data.groupby("date")["temperature_2m_max"].apply(
        lambda x: x.nlargest(3).min()
    )
    df_agg["peninsular_min_temperature"] = pen_data.groupby("date")["temperature_2m_min"].apply(
        lambda x: x.nsmallest(3).max()
    )

    df_agg = df_agg.reset_index()

    es_holidays = holidays.Spain(years=range(2020, 2027))
    df_agg["is_national_holiday"] = df_agg["date"].dt.date.apply(
        lambda x: 1 if x in es_holidays else 0
    )

    # Drop intermediate weighted columns.
    weighted_cols = [col for col in df_agg.columns if col.endswith("_w")]
    df_agg = df_agg.drop(columns=weighted_cols + ["pop_weight", "surf_weight"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(output_file, index=False)
    logger.info(f"Weather aggregation complete. Output: {output_file}")
