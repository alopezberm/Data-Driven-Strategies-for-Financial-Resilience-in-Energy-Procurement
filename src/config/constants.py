

"""
constants.py

Centralized project constants.
This module stores reusable configuration values that should not be hardcoded
across multiple files.
"""

from __future__ import annotations


# =========================
# Core dataset / modeling constants
# =========================

TARGET_COLUMN = "Spot_Price_SPEL"
DATE_COLUMN = "date"
DEFAULT_FORECAST_HORIZON = 1

DEFAULT_QUANTILES = [0.5, 0.9, 0.95]
DEFAULT_ROLLING_WINDOWS = [7, 14, 28]
DEFAULT_LAG_STEPS = [1, 2, 3, 7, 14, 28]


# =========================
# Column-name conventions
# =========================

SPOT_PRICE_COLUMN = "Spot_Price_SPEL"
FUTURE_PRICE_COLUMNS = [
    "Future_M1_Price",
    "Future_M2_Price",
    "Future_M3_Price",
    "Future_M4_Price",
    "Future_M5_Price",
    "Future_M6_Price",
]

OPEN_INTEREST_COLUMNS = [
    "Future_M1_OpenInterest",
    "Future_M2_OpenInterest",
    "Future_M3_OpenInterest",
    "Future_M4_OpenInterest",
    "Future_M5_OpenInterest",
    "Future_M6_OpenInterest",
]

POLICY_REQUIRED_COLUMNS = [
    "date",
    "Spot_Price_SPEL",
    "Future_M1_Price",
    "q_0.5",
    "q_0.9",
]


# =========================
# Baseline strategy defaults
# =========================

DEFAULT_STATIC_HEDGE_RATIO = 0.7
DEFAULT_DAILY_VOLUME = 1.0


# =========================
# Heuristic policy thresholds
# =========================

DEFAULT_Q50_COLUMN = "q_0.5"
DEFAULT_Q90_COLUMN = "q_0.9"
DEFAULT_SPOT_COLUMN = "Spot_Price_SPEL"
DEFAULT_FUTURE_COLUMN = "Future_M1_Price"
DEFAULT_HOLIDAY_COLUMN = "Is_national_holiday"
DEFAULT_WEEKEND_COLUMN = "is_weekend"

MIN_ABS_RISK_PREMIUM_TO_HEDGE = 8.0
MIN_REL_RISK_PREMIUM_TO_HEDGE = 0.10
MIN_ABS_RISK_PREMIUM_TO_SHIFT = 12.0
MIN_REL_RISK_PREMIUM_TO_SHIFT = 0.15

ALLOW_SHIFT_ON_WEEKENDS = True
ALLOW_SHIFT_ON_HOLIDAYS = True


# =========================
# Action-rule thresholds
# =========================

TAIL_VS_FUTURE_ABS_THRESHOLD = 5.0
TAIL_VS_CENTRAL_ABS_THRESHOLD = 3.0
ALLOW_SHIFT_ON_WEEKENDS_RULE = True


# =========================
# Policy simulation defaults
# =========================

DEFAULT_HEDGE_RATIO_ON_BUY_FUTURE = 1.0
DEFAULT_SHIFT_FRACTION = 1.0
DEFAULT_SHIFT_PENALTY_PER_MWH = 2.0


# =========================
# Backtesting defaults
# =========================

DEFAULT_REFERENCE_STRATEGY = "spot_only"
DEFAULT_EXTREME_COST_QUANTILE = 0.90


# =========================
# Visualization defaults
# =========================

DEFAULT_FIGURE_WIDTH = 12
DEFAULT_FIGURE_HEIGHT = 6
DEFAULT_SMALL_FIGURE_WIDTH = 8
DEFAULT_SMALL_FIGURE_HEIGHT = 5


# =========================
# Reporting / metadata
# =========================

PROJECT_NAME = "Data-Driven Strategies for Financial Resilience in Energy Procurement"
PROJECT_SHORT_NAME = "group17_tailrisk_solutions"
COURSE_NAME = "Advanced Business Analytics"


if __name__ == "__main__":
    print("Project constants loaded successfully.")
    print(f"Default quantiles: {DEFAULT_QUANTILES}")
    print(f"Default hedge ratio: {DEFAULT_STATIC_HEDGE_RATIO}")