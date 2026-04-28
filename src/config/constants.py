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
DEFAULT_FORECAST_HORIZONS = [2, 3]

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

# Quantile column helpers — horizon t+2 (default)
Q50_COLUMN = "q_0.5"
Q90_COLUMN = "q_0.9"
Q95_COLUMN = "q_0.95"

# Quantile column helpers — horizon t+3
Q50_H3_COLUMN = "q_0.5_h3"
Q90_H3_COLUMN = "q_0.9_h3"
Q95_H3_COLUMN = "q_0.95_h3"

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

# Thresholds for the extended production actions
MIN_ABS_RISK_PREMIUM_TO_INCREASE = 0.0   # always increase when price is low vs. forecast
MIN_ABS_RISK_PREMIUM_TO_DECREASE = 10.0  # decrease production when tail risk is high
MIN_ABS_RISK_PREMIUM_TO_BUY_M2 = 12.0   # buy M+2 when mid-term tail risk is elevated
MIN_ABS_RISK_PREMIUM_TO_BUY_M3 = 18.0   # buy M+3 only under severe long-term risk

ALLOW_SHIFT_ON_WEEKENDS = True
ALLOW_SHIFT_ON_HOLIDAYS = True


# =========================
# Action-rule thresholds
# =========================

TAIL_VS_FUTURE_ABS_THRESHOLD = 5.0
TAIL_VS_CENTRAL_ABS_THRESHOLD = 3.0


# =========================
# MDP Factory Parameters (Single Source of Truth)
# From Mathematical Formulation — do not change without updating rl_environment.py
# =========================

MDP_P_MAX = 2000              # Maximum production capacity (units/day)
MDP_D = 1000                  # Fixed daily demand dispatched at 09:00 (units)
MDP_I_MAX = 3000              # Maximum warehouse capacity (units)
MDP_E_START = 20.0            # Startup energy cost (MWh) when factory is on
MDP_E_UNIT = 1.0              # Variable energy per unit produced (MWh/unit)
MDP_H = 5.0                   # Inventory holding cost (EUR/unit/day)
MDP_M = 200.0                 # Gross profit margin (EUR/unit sold)
MDP_INITIAL_INVENTORY = 1500  # Starting inventory at episode reset (midpoint of I_max)

# Production action space: 0, 100, 200, ..., 2000 (21 discrete levels)
MDP_PROD_LEVELS: list[int] = list(range(0, MDP_P_MAX + 1, 100))
MDP_PROD_STEP = 100

# Futures block sizes (MWh): 0, 500, or 1000 MWh per tenor (M1, M2, M3)
MDP_BLOCK_SIZES: list[int] = [0, 500, 1000]

# Derived action-space dimensions (kept as constants for reuse across modules)
MDP_N_PROD = 21               # len(range(0, 2001, 100))
MDP_N_BLOCK = 3               # len([0, 500, 1000])
MDP_N_ACTIONS = 567           # MDP_N_PROD * MDP_N_BLOCK**3

# Spot-M1 spread column — required by the RL state space
SPOT_M1_SPREAD_COLUMN = "Spot_M1_Spread"

# =========================
# Legacy factory constants (used by heuristic policy — do not remove)
# =========================

PRODUCTION_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_PRODUCTION_LEVEL = 1.0
PRODUCTION_STEP = 0.1

FACTORY_BASE_LOAD = 0.3
FACTORY_VARIABLE_LOAD = 0.7
FACTORY_INVENTORY_CAPACITY = 100.0
FACTORY_INVENTORY_MIN = 0.0
FACTORY_INITIAL_INVENTORY = 50.0
FACTORY_DEMAND_PER_STEP = 8.0
FACTORY_STORAGE_COST_PER_UNIT = 0.5
FACTORY_STARTUP_ENERGY_COST = 5.0
FACTORY_PRODUCT_PRICE = 150.0
FACTORY_TAKEORPAY_FRACTION = 0.5

MAX_HEDGE_FRACTION_M1 = 1.0
MAX_HEDGE_FRACTION_M2 = 0.5
MAX_HEDGE_FRACTION_M3 = 0.25


# =========================
# Actions & strategies
# =========================

# The first three entries are the original action set and are protected by
# validate_action_catalog(). New actions are appended after position 2.
ACTIONS = [
    "do_nothing",          # 0 — baseline: buy everything on spot
    "buy_m1_future",       # 1 — lock in front-month futures price
    "shift_production",    # 2 — legacy: shift flexible load off-peak (kept for compat)
    "increase_production", # 3 — raise output +10 % (run more when prices are low)
    "decrease_production", # 4 — cut output -10 % (avoid high-cost periods)
    "buy_m2_future",       # 5 — lock in month+2 futures
    "buy_m3_future",       # 6 — lock in month+3 futures
]

ACTION_DO_NOTHING = ACTIONS[0]
ACTION_BUY_M1_FUTURE = ACTIONS[1]
ACTION_SHIFT_PRODUCTION = ACTIONS[2]
ACTION_INCREASE_PRODUCTION = ACTIONS[3]
ACTION_DECREASE_PRODUCTION = ACTIONS[4]
ACTION_BUY_M2_FUTURE = ACTIONS[5]
ACTION_BUY_M3_FUTURE = ACTIONS[6]

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