"""
simulate_baseline.py

Baseline strategy simulators for the energy procurement project.
These strategies provide transparent reference cases against which the DSS
policy can be compared during counterfactual backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.constants import (
    DATE_COLUMN,
    DEFAULT_DAILY_VOLUME,
    DEFAULT_STATIC_HEDGE_RATIO,
    PRIMARY_FUTURE_COLUMN,
    SPOT_PRICE_COLUMN,
    STRATEGY_SPOT_ONLY,
    STRATEGY_STATIC_HEDGE,
)
from src.config.settings import SimulationSettings, TrainingSettings, get_default_settings
from src.backtesting._simulation_utils import SimulationUtilsError, ensure_volume_column
from src.utils.validation import ValidationError, validate_and_sort_by_date


class BaselineSimulationError(Exception):
    """Raised when a baseline strategy cannot be simulated safely."""


DEFAULT_VOLUME_COLUMN = "daily_energy_mwh"

ACTION_BUY_ON_SPOT = "buy_on_spot"
ACTION_STATIC_M1_HEDGE = "static_m1_hedge"


@dataclass
class BaselineSimulationConfig:
    """Configuration for baseline procurement simulations."""

    spot_column: str = SPOT_PRICE_COLUMN
    future_column: str = PRIMARY_FUTURE_COLUMN
    volume_column: str = DEFAULT_VOLUME_COLUMN
    default_daily_volume: float = DEFAULT_DAILY_VOLUME
    hedge_ratio: float = DEFAULT_STATIC_HEDGE_RATIO

    @classmethod
    def from_project_settings(
        cls,
        training_settings: TrainingSettings,
        simulation_settings: SimulationSettings,
    ) -> "BaselineSimulationConfig":
        """Build baseline simulation config from centralized project settings."""
        return cls(
            spot_column=training_settings.target_column,
            future_column=PRIMARY_FUTURE_COLUMN,
            volume_column=DEFAULT_VOLUME_COLUMN,
            default_daily_volume=simulation_settings.default_daily_volume,
            hedge_ratio=training_settings.static_hedge_ratio,
        )


def get_default_baseline_simulation_config() -> BaselineSimulationConfig:
    """Build the default baseline simulation config from project settings."""
    settings = get_default_settings()
    return BaselineSimulationConfig.from_project_settings(
        settings.training,
        settings.simulation,
    )


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(df: pd.DataFrame, config: BaselineSimulationConfig) -> pd.DataFrame:
    """Validate the backtesting input dataframe and standardize its date column."""
    if config.spot_column not in df.columns:
        raise BaselineSimulationError(
            f"Missing required column for baseline simulation: '{config.spot_column}'"
        )
    if not 0 <= config.hedge_ratio <= 1:
        raise BaselineSimulationError("hedge_ratio must be between 0 and 1.")
    try:
        return validate_and_sort_by_date(df, df_name="baseline simulation input")
    except ValidationError as exc:
        raise BaselineSimulationError(str(exc)) from exc



def _ensure_volume_column(df: pd.DataFrame, config: BaselineSimulationConfig) -> pd.DataFrame:
    """Ensure a valid daily energy volume column exists."""
    try:
        return ensure_volume_column(df, config.volume_column, config.default_daily_volume)
    except SimulationUtilsError as exc:
        raise BaselineSimulationError(str(exc)) from exc


# =========================
# Core cost helpers
# =========================

def _build_common_output(df: pd.DataFrame, config: BaselineSimulationConfig, strategy_name: str) -> pd.DataFrame:
    """Create the common output table shared by all baseline strategies."""
    output_df = df.copy()
    output_df["strategy_name"] = strategy_name
    output_df["energy_volume_mwh"] = output_df[config.volume_column]
    output_df["spot_price"] = pd.to_numeric(output_df[config.spot_column], errors="coerce")

    if output_df["spot_price"].isna().any():
        raise BaselineSimulationError("Spot price column contains invalid values.")

    return output_df


# =========================
# Baseline 1: Spot-only strategy
# =========================

def simulate_spot_only_baseline(
    df: pd.DataFrame,
    config: BaselineSimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate a pure spot-market strategy.

    Interpretation:
    - The factory buys its full daily energy need on the spot market.
    - No hedging and no load shifting are applied.
    """
    config = get_default_baseline_simulation_config() if config is None else config

    simulation_df = _validate_input_dataframe(df, config)
    simulation_df = _ensure_volume_column(simulation_df, config)
    simulation_df = _build_common_output(simulation_df, config, STRATEGY_SPOT_ONLY)

    simulation_df["hedged_volume_mwh"] = 0.0
    simulation_df["spot_volume_mwh"] = simulation_df["energy_volume_mwh"]
    simulation_df["future_price"] = pd.NA
    simulation_df["future_cost"] = 0.0
    simulation_df["spot_cost"] = (
        simulation_df["spot_volume_mwh"] * simulation_df["spot_price"]
    )
    simulation_df["total_cost"] = simulation_df["spot_cost"]
    simulation_df["action_taken"] = ACTION_BUY_ON_SPOT

    return simulation_df


# =========================
# Baseline 2: Static hedge strategy
# =========================

def simulate_static_hedge_baseline(
    df: pd.DataFrame,
    config: BaselineSimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate a static hedge strategy using the M1 futures price.

    Interpretation:
    - A fixed share of daily consumption is assumed hedged at the available M1 price.
    - The remaining share is exposed to spot.

    Notes
    -----
    This is a simplified benchmark. It does not model contract roll dates,
    margining, or exact market microstructure. Its purpose is to provide a
    transparent comparison point for the DSS.
    """
    config = get_default_baseline_simulation_config() if config is None else config

    simulation_df = _validate_input_dataframe(df, config)
    simulation_df = _ensure_volume_column(simulation_df, config)

    if config.future_column not in simulation_df.columns:
        raise BaselineSimulationError(
            f"Missing futures column '{config.future_column}' for static hedge simulation."
        )

    simulation_df = _build_common_output(simulation_df, config, STRATEGY_STATIC_HEDGE)
    simulation_df["future_price"] = pd.to_numeric(
        simulation_df[config.future_column], errors="coerce"
    )

    if simulation_df["future_price"].isna().any():
        raise BaselineSimulationError(
            f"Futures column '{config.future_column}' contains invalid or missing values."
        )

    simulation_df["hedged_volume_mwh"] = (
        simulation_df["energy_volume_mwh"] * config.hedge_ratio
    )
    simulation_df["spot_volume_mwh"] = (
        simulation_df["energy_volume_mwh"] - simulation_df["hedged_volume_mwh"]
    )

    simulation_df["future_cost"] = (
        simulation_df["hedged_volume_mwh"] * simulation_df["future_price"]
    )
    simulation_df["spot_cost"] = (
        simulation_df["spot_volume_mwh"] * simulation_df["spot_price"]
    )
    simulation_df["total_cost"] = (
        simulation_df["future_cost"] + simulation_df["spot_cost"]
    )
    simulation_df["action_taken"] = ACTION_STATIC_M1_HEDGE

    return simulation_df


# =========================
# Summary helpers
# =========================

def summarize_simulation(simulation_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact one-row summary from a simulation dataframe."""
    required_columns = {"strategy_name", "total_cost", "energy_volume_mwh"}
    missing_columns = required_columns - set(simulation_df.columns)
    if missing_columns:
        raise BaselineSimulationError(
            f"Missing required columns for simulation summary: {sorted(missing_columns)}"
        )

    strategy_name = simulation_df["strategy_name"].iloc[0]
    total_cost = float(simulation_df["total_cost"].sum())
    total_volume = float(simulation_df["energy_volume_mwh"].sum())
    avg_unit_cost = total_cost / total_volume if total_volume > 0 else pd.NA
    daily_cost_volatility = float(simulation_df["total_cost"].std())

    summary_df = pd.DataFrame(
        {
            "strategy_name": [strategy_name],
            "n_days": [int(len(simulation_df))],
            "total_volume_mwh": [total_volume],
            "total_cost": [total_cost],
            "average_unit_cost": [avg_unit_cost],
            "daily_cost_volatility": [daily_cost_volatility],
        }
    )

    return summary_df


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=5, freq="D"),
            SPOT_PRICE_COLUMN: [80, 120, 95, 70, 110],
            PRIMARY_FUTURE_COLUMN: [85, 90, 92, 88, 91],
            DEFAULT_VOLUME_COLUMN: [10, 10, 10, 10, 10],
        }
    )

    config = get_default_baseline_simulation_config()
    print(config)

    spot_only_df = simulate_spot_only_baseline(example_df, config=config)
    static_hedge_df = simulate_static_hedge_baseline(
        example_df,
        config=config,
    )

    print(spot_only_df[[DATE_COLUMN, "strategy_name", "spot_cost", "total_cost"]])
    print(static_hedge_df[[DATE_COLUMN, "strategy_name", "future_cost", "spot_cost", "total_cost"]])
    print(summarize_simulation(spot_only_df))
    print(summarize_simulation(static_hedge_df))