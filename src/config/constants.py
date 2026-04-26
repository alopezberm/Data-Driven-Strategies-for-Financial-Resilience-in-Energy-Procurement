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

DATE_COLUMN = "date"
TARGET_COLUMN = "Spot_Price_SPEL"
DEFAULT_FORECAST_HORIZON = 1

DEFAULT_QUANTILES = [0.5, 0.9, 0.95]
DEFAULT_ROLLING_WINDOWS = [7, 14, 28]
DEFAULT_LAG_STEPS = [1, 2, 3, 7, 14, 28]

# Reusable aliases
SPOT_PRICE_COLUMN = TARGET_COLUMN
PRIMARY_FUTURE_COLUMN = "Future_M1_Price"
PRIMARY_OPEN_INTEREST_COLUMN = "Future_M1_OpenInterest"
SECONDARY_FUTURE_COLUMN = "Future_M2_Price"
SECONDARY_OPEN_INTEREST_COLUMN = "Future_M2_OpenInterest"


# =========================
# Column-name conventions
# =========================

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

# Quantile column helpers
Q50_COLUMN = "q_0.5"
Q90_COLUMN = "q_0.9"
Q95_COLUMN = "q_0.95"

# Required columns for policy/backtesting inputs
POLICY_REQUIRED_COLUMNS = [
    DATE_COLUMN,
    SPOT_PRICE_COLUMN,
    PRIMARY_FUTURE_COLUMN,
    Q50_COLUMN,
    Q90_COLUMN,
]


# =========================
# Baseline strategy defaults
# =========================

DEFAULT_STATIC_HEDGE_RATIO = 0.7
DEFAULT_DAILY_VOLUME = 1.0


# =========================
# Heuristic policy thresholds
# =========================

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


# =========================
# Actions & strategies
# =========================

ACTIONS = [
    "do_nothing",
    "buy_m1_future",
    "shift_production",
]

ACTION_DO_NOTHING = ACTIONS[0]
ACTION_BUY_M1_FUTURE = ACTIONS[1]
ACTION_SHIFT_PRODUCTION = ACTIONS[2]

STRATEGY_SPOT_ONLY = "spot_only"
STRATEGY_STATIC_HEDGE = "static_hedge"
STRATEGY_HEURISTIC_POLICY = "heuristic_policy"
STRATEGY_RL_POLICY = "rl_policy"


# =========================
# Policy simulation defaults
# =========================

DEFAULT_HEDGE_RATIO_ON_BUY_FUTURE = 1.0
DEFAULT_SHIFT_FRACTION = 1.0
DEFAULT_SHIFT_PENALTY_PER_MWH = 2.0


# =========================
# Backtesting defaults
# =========================

DEFAULT_REFERENCE_STRATEGY = STRATEGY_SPOT_ONLY
DEFAULT_EXTREME_COST_QUANTILE = 0.90


# =========================
# Output names / folders
# =========================

OUTPUT_FORECASTS_DIRNAME = "forecasts"
OUTPUT_POLICIES_DIRNAME = "policies"
OUTPUT_BACKTESTS_DIRNAME = "backtests"
OUTPUT_FIGURES_DIRNAME = "figures"


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


_EXPECTED_ACTIONS = {"do_nothing", "buy_m1_future", "shift_production"}


def validate_action_catalog() -> None:
    """Raise ValueError if the ACTIONS catalog has drifted from the expected set."""
    if len(ACTIONS) < 3 or set(ACTIONS[:3]) != _EXPECTED_ACTIONS:
        raise ValueError(
            f"ACTIONS catalog is inconsistent. Expected first three entries to be "
            f"{sorted(_EXPECTED_ACTIONS)}, got {ACTIONS[:3]}."
        )


if __name__ == "__main__":
    print("Project constants loaded successfully.")
    print(f"Default quantiles: {DEFAULT_QUANTILES}")
    print(f"Default hedge ratio: {DEFAULT_STATIC_HEDGE_RATIO}")