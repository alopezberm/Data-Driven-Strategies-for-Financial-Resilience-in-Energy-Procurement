"""
generate_holidays_raw.py

Generate a raw holidays CSV for the project using python-holidays.
By default, this script creates Spanish national holidays from 2020 to 2025.
"""

from __future__ import annotations

import pandas as pd
import holidays

from src.config.paths import HOLIDAYS_RAW_FILE


class HolidaysGenerationError(Exception):
    """Raised when the holidays raw dataset cannot be generated."""


def generate_holidays_raw(
    start_year: int = 2020,
    end_year: int = 2025,
    country_code: str = "ES",
) -> pd.DataFrame:
    """
    Generate a raw holidays dataframe with columns:
    - date
    - is_holiday
    - holiday_name
    """
    if end_year < start_year:
        raise HolidaysGenerationError("end_year must be greater than or equal to start_year.")

    holiday_calendar = holidays.country_holidays(country_code, years=range(start_year, end_year + 1))

    # Build a complete daily calendar so downstream joins never depend on NaN flags.
    date_index = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="D",
    )

    holidays_df = pd.DataFrame({"date": date_index})
    holidays_df["is_holiday"] = holidays_df["date"].dt.date.map(lambda day: 1 if day in holiday_calendar else 0)
    holidays_df["holiday_name"] = holidays_df["date"].dt.date.map(
        lambda day: str(holiday_calendar.get(day)) if day in holiday_calendar else pd.NA
    )

    holidays_df = holidays_df.sort_values("date").reset_index(drop=True)

    HOLIDAYS_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    holidays_df.to_csv(HOLIDAYS_RAW_FILE, index=False)

    return holidays_df


if __name__ == "__main__":
    generated_df = generate_holidays_raw(start_year=2020, end_year=2025, country_code="ES")
    print(f"Generated holidays raw file at: {HOLIDAYS_RAW_FILE}")
    print(f"Rows: {len(generated_df)}")
    print(generated_df.head())