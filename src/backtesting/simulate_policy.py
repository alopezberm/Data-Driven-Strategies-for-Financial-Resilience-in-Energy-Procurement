"""
simulate_policy.py

Simulation utilities for the heuristic DSS policy.
This module converts recommended actions into counterfactual daily procurement
costs so the policy can be compared against baseline strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.constants import (
    ACTIONS,
    DATE_COLUMN,
    DEFAULT_DAILY_VOLUME,
    DEFAULT_HEDGE_RATIO_ON_BUY_FUTURE,
    DEFAULT_SHIFT_FRACTION,
    DEFAULT_SHIFT_PENALTY_PER_MWH,
    PRIMARY_FUTURE_COLUMN,
    SPOT_PRICE_COLUMN,
    STRATEGY_HEURISTIC_POLICY,
    validate_action_catalog,
)
from src.config.settings import SimulationSettings, TrainingSettings, get_default_settings
from src.backtesting._simulation_utils import SimulationUtilsError, ensure_volume_column
from src.utils.validation import ValidationError, validate_and_sort_by_date


DEFAULT_ACTION_COLUMN = "recommended_action"
DEFAULT_REASON_COLUMN = "decision_reason"
DEFAULT_VOLUME_COLUMN = "daily_energy_mwh"

ACTION_DO_NOTHING = ACTIONS[0]
ACTION_BUY_M1_FUTURE = ACTIONS[1]
ACTION_SHIFT_PRODUCTION = ACTIONS[2]

SUPPORTED_POLICY_ACTIONS = {ACTION_DO_NOTHING, ACTION_BUY_M1_FUTURE, ACTION_SHIFT_PRODUCTION}


class PolicySimulationError(Exception):
    """Raised when the DSS policy cannot be simulated safely."""


@dataclass
class PolicySimulationConfig:
    """Configuration for policy backtesting assumptions."""

    action_column: str = DEFAULT_ACTION_COLUMN
    reason_column: str = DEFAULT_REASON_COLUMN
    spot_column: str = SPOT_PRICE_COLUMN
    future_column: str = PRIMARY_FUTURE_COLUMN
    volume_column: str = DEFAULT_VOLUME_COLUMN
    default_daily_volume: float = DEFAULT_DAILY_VOLUME

    # Economic assumptions
    hedge_ratio_on_buy_future: float = DEFAULT_HEDGE_RATIO_ON_BUY_FUTURE
    shift_fraction: float = DEFAULT_SHIFT_FRACTION
    shift_penalty_per_mwh: float = DEFAULT_SHIFT_PENALTY_PER_MWH

    @classmethod
    def from_project_settings(
        cls,
        training_settings: TrainingSettings,
        simulation_settings: SimulationSettings,
    ) -> "PolicySimulationConfig":
        """Build policy simulation config from centralized project settings."""
        return cls(
            action_column=DEFAULT_ACTION_COLUMN,
            reason_column=DEFAULT_REASON_COLUMN,
            spot_column=training_settings.target_column,
            future_column=PRIMARY_FUTURE_COLUMN,
            volume_column=DEFAULT_VOLUME_COLUMN,
            default_daily_volume=simulation_settings.default_daily_volume,
            hedge_ratio_on_buy_future=simulation_settings.hedge_ratio_on_buy_future,
            shift_fraction=simulation_settings.shift_fraction,
            shift_penalty_per_mwh=simulation_settings.shift_penalty_per_mwh,
        )


def get_default_policy_simulation_config() -> PolicySimulationConfig:
    """Build the default policy simulation config from project settings."""
    settings = get_default_settings()
    return PolicySimulationConfig.from_project_settings(
        settings.training,
        settings.simulation,
    )


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(
    df: pd.DataFrame,
    config: PolicySimulationConfig,
) -> pd.DataFrame:
    """Validate input data and standardize the date column."""
    for col in [config.action_column, config.spot_column]:
        if col not in df.columns:
            raise PolicySimulationError(
                f"Missing required column for policy simulation: '{col}'"
            )
    if not 0 <= config.hedge_ratio_on_buy_future <= 1:
        raise PolicySimulationError("hedge_ratio_on_buy_future must be between 0 and 1.")
    if not 0 <= config.shift_fraction <= 1:
        raise PolicySimulationError("shift_fraction must be between 0 and 1.")
    try:
        validated_df = validate_and_sort_by_date(df, df_name="policy simulation input")
    except ValidationError as exc:
        raise PolicySimulationError(str(exc)) from exc
    invalid_actions = set(validated_df[config.action_column].dropna().unique()) - SUPPORTED_POLICY_ACTIONS
    if invalid_actions:
        raise PolicySimulationError(
            f"Unsupported policy actions found: {sorted(invalid_actions)}"
        )
    return validated_df



def _ensure_volume_column(df: pd.DataFrame, config: PolicySimulationConfig) -> pd.DataFrame:
    """Ensure a valid daily energy volume column exists."""
    try:
        return ensure_volume_column(df, config.volume_column, config.default_daily_volume)
    except SimulationUtilsError as exc:
        raise PolicySimulationError(str(exc)) from exc



def _validate_future_column_if_needed(df: pd.DataFrame, config: PolicySimulationConfig) -> None:
    """Require a futures column if the policy uses buy_m1_future at least once."""
    needs_future = (df[config.action_column] == ACTION_BUY_M1_FUTURE).any()
    if needs_future and config.future_column not in df.columns:
        raise PolicySimulationError(
            f"Policy includes '{ACTION_BUY_M1_FUTURE}' but futures column '{config.future_column}' is missing."
        )


# =========================
# Core row-level simulation
# =========================

def _simulate_policy_row(row: pd.Series, config: PolicySimulationConfig) -> dict[str, object]:
    """Translate one policy action into cost components."""
    action = row[config.action_column]
    spot_price = float(row[config.spot_column])
    volume = float(row[config.volume_column])
    reason = row.get(config.reason_column, pd.NA)

    result = {
        "action_taken": action,
        "decision_reason": reason,
        "energy_volume_mwh": float(volume),
        "spot_price": float(spot_price),
        "future_price": pd.NA,
        "hedged_volume_mwh": 0.0,
        "shifted_volume_mwh": 0.0,
        "spot_volume_mwh": float(volume),
        "future_cost": 0.0,
        "spot_cost": float(volume) * float(spot_price),
        "shift_penalty_cost": 0.0,
        "total_cost": float(volume) * float(spot_price),
    }

    if action == ACTION_DO_NOTHING:
        return result

    if action == ACTION_BUY_M1_FUTURE:
        raw_future = row.get(config.future_column, pd.NA)
        if pd.isna(raw_future):
            raise PolicySimulationError(
                "Encountered 'buy_m1_future' action but the futures price is missing or invalid."
            )
        future_price = float(raw_future)

        hedged_volume = float(volume) * config.hedge_ratio_on_buy_future
        spot_volume = float(volume) - hedged_volume
        future_cost = hedged_volume * float(future_price)
        spot_cost = spot_volume * float(spot_price)
        total_cost = future_cost + spot_cost

        result.update(
            {
                "future_price": float(future_price),
                "hedged_volume_mwh": hedged_volume,
                "spot_volume_mwh": spot_volume,
                "future_cost": future_cost,
                "spot_cost": spot_cost,
                "total_cost": total_cost,
            }
        )
        return result

    if action == ACTION_SHIFT_PRODUCTION:
        shifted_volume = float(volume) * config.shift_fraction
        spot_volume = float(volume) - shifted_volume
        shift_penalty_cost = shifted_volume * config.shift_penalty_per_mwh
        spot_cost = spot_volume * float(spot_price)
        total_cost = spot_cost + shift_penalty_cost

        result.update(
            {
                "shifted_volume_mwh": shifted_volume,
                "spot_volume_mwh": spot_volume,
                "spot_cost": spot_cost,
                "shift_penalty_cost": shift_penalty_cost,
                "total_cost": total_cost,
            }
        )
        return result

    raise PolicySimulationError(f"Unsupported action encountered during simulation: {action}")


# =========================
# Public API
# =========================

def simulate_policy_strategy(
    df: pd.DataFrame,
    config: PolicySimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate the cost consequences of the heuristic DSS policy.

    The returned dataframe preserves the original rows and appends simulated
    cost components for transparent backtesting.
    """
    config = get_default_policy_simulation_config() if config is None else config
    validate_action_catalog()

    simulation_df = _validate_input_dataframe(df, config)
    simulation_df = _ensure_volume_column(simulation_df, config)
    _validate_future_column_if_needed(simulation_df, config)

    row_outputs = simulation_df.apply(
        lambda row: _simulate_policy_row(row, config),
        axis=1,
        result_type="expand",
    )

    output_df = simulation_df.copy()
    output_df["strategy_name"] = STRATEGY_HEURISTIC_POLICY

    for column in row_outputs.columns:
        output_df[column] = row_outputs[column]

    return output_df



def summarize_policy_simulation(simulation_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact one-row summary for a simulated policy run."""
    required_columns = {
        "strategy_name",
        "total_cost",
        "energy_volume_mwh",
        "action_taken",
    }
    missing_columns = required_columns - set(simulation_df.columns)
    if missing_columns:
        raise PolicySimulationError(
            f"Missing required columns for policy simulation summary: {sorted(missing_columns)}"
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
            "n_buy_m1_future_days": [int((simulation_df["action_taken"] == ACTION_BUY_M1_FUTURE).sum())],
            "n_shift_production_days": [int((simulation_df["action_taken"] == ACTION_SHIFT_PRODUCTION).sum())],
            "n_do_nothing_days": [int((simulation_df["action_taken"] == ACTION_DO_NOTHING).sum())],
        }
    )

    return summary_df


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=6, freq="D"),
            SPOT_PRICE_COLUMN: [70, 75, 80, 78, 82, 76],
            PRIMARY_FUTURE_COLUMN: [72, 73, 74, 75, 76, 77],
            DEFAULT_VOLUME_COLUMN: [10, 10, 10, 10, 10, 10],
            "recommended_action": [
                ACTION_DO_NOTHING,
                ACTION_BUY_M1_FUTURE,
                ACTION_BUY_M1_FUTURE,
                ACTION_SHIFT_PRODUCTION,
                ACTION_DO_NOTHING,
                ACTION_SHIFT_PRODUCTION,
            ],
            "decision_reason": [
                "No rule triggered.",
                "Tail risk above hedge price.",
                "Tail risk above hedge price.",
                "Flexible day and high tail risk.",
                "No rule triggered.",
                "Flexible day and high tail risk.",
            ],
        }
    )

    config = get_default_policy_simulation_config()
    print(config)

    simulated_df = simulate_policy_strategy(
        example_df,
        config=config,
    )

    print(
        simulated_df[
            [
                DATE_COLUMN,
                "action_taken",
                "spot_price",
                "future_price",
                "energy_volume_mwh",
                "hedged_volume_mwh",
                "shifted_volume_mwh",
                "spot_cost",
                "future_cost",
                "shift_penalty_cost",
                "total_cost",
            ]
        ]
    )
    print(summarize_policy_simulation(simulated_df))
