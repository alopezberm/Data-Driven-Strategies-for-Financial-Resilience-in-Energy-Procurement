"""
simulate_rl_policy.py

Backtesting simulation for RL-based procurement decisions.

This module mirrors the economic logic used by the heuristic policy simulator,
while taking decisions produced by a trained RL policy. The output is designed
to be fully compatible with the existing strategy-comparison and resilience
modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.constants import (
    ACTION_BUY_M1_FUTURE,
    ACTION_BUY_M2_FUTURE,
    ACTION_BUY_M3_FUTURE,
    ACTION_DECREASE_PRODUCTION,
    ACTION_DO_NOTHING,
    ACTION_INCREASE_PRODUCTION,
    ACTION_SHIFT_PRODUCTION,
    ACTIONS,
    DATE_COLUMN,
    DEFAULT_DAILY_VOLUME,
    DEFAULT_HEDGE_RATIO_ON_BUY_FUTURE,
    DEFAULT_SHIFT_FRACTION,
    DEFAULT_SHIFT_PENALTY_PER_MWH,
    PRIMARY_FUTURE_COLUMN,
    PRODUCTION_STEP,
    SECONDARY_FUTURE_COLUMN,
    SPOT_PRICE_COLUMN,
    STRATEGY_RL_POLICY,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RLPolicySimulationError(Exception):
    """Raised when RL policy simulation cannot be executed safely."""


DEFAULT_ACTION_COLUMN = "recommended_action"
DEFAULT_VOLUME_COLUMN = "daily_energy_mwh"

# All 7 actions are supported; original 3 are the backward-compatible core.
SUPPORTED_RL_ACTIONS = set(ACTIONS)

_M2_FUTURE_COLUMN = SECONDARY_FUTURE_COLUMN   # Future_M2_Price
_M3_FUTURE_COLUMN = "Future_M3_Price"


@dataclass
class RLPolicySimulationConfig:
    """Configuration for simulating RL-driven procurement decisions."""

    action_column: str = DEFAULT_ACTION_COLUMN
    spot_column: str = SPOT_PRICE_COLUMN
    future_column: str = PRIMARY_FUTURE_COLUMN
    volume_column: str = DEFAULT_VOLUME_COLUMN
    hedge_ratio_on_buy_future: float = DEFAULT_HEDGE_RATIO_ON_BUY_FUTURE
    shift_fraction: float = DEFAULT_SHIFT_FRACTION
    shift_penalty_per_mwh: float = DEFAULT_SHIFT_PENALTY_PER_MWH


# =========================
# Validation helpers
# =========================


def _validate_input_dataframe(
    df: pd.DataFrame,
    config: RLPolicySimulationConfig,
) -> pd.DataFrame:
    """Validate RL policy simulation inputs and standardize the date column."""
    if not isinstance(df, pd.DataFrame):
        raise RLPolicySimulationError("Input must be a pandas DataFrame.")
    if df.empty:
        raise RLPolicySimulationError("Input dataframe is empty.")

    required_columns = [
        DATE_COLUMN,
        config.action_column,
        config.spot_column,
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise RLPolicySimulationError(
            f"Missing required columns for RL policy simulation: {missing_columns}"
        )

    result_df = df.copy()
    result_df[DATE_COLUMN] = pd.to_datetime(result_df[DATE_COLUMN], errors="coerce")

    if result_df[DATE_COLUMN].isna().any():
        invalid_count = int(result_df[DATE_COLUMN].isna().sum())
        raise RLPolicySimulationError(
            f"Found {invalid_count} invalid date values in RL policy simulation input."
        )

    if result_df[DATE_COLUMN].duplicated().any():
        raise RLPolicySimulationError("Input dataframe contains duplicated dates.")

    if not 0 <= config.hedge_ratio_on_buy_future <= 1:
        raise RLPolicySimulationError(
            "hedge_ratio_on_buy_future must be between 0 and 1."
        )

    if not 0 <= config.shift_fraction <= 1:
        raise RLPolicySimulationError("shift_fraction must be between 0 and 1.")

    invalid_actions = set(result_df[config.action_column].dropna().unique()) - SUPPORTED_RL_ACTIONS
    if invalid_actions:
        raise RLPolicySimulationError(
            f"Unsupported RL actions found: {sorted(invalid_actions)}"
        )

    numeric_columns = [config.spot_column]
    if config.future_column in result_df.columns:
        numeric_columns.append(config.future_column)
    if config.volume_column in result_df.columns:
        numeric_columns.append(config.volume_column)

    for column in numeric_columns:
        result_df[column] = pd.to_numeric(result_df[column], errors="coerce")
        if result_df[column].isna().any():
            invalid_count = int(result_df[column].isna().sum())
            raise RLPolicySimulationError(
                f"Column '{column}' contains {invalid_count} invalid numeric values."
            )

    return result_df.sort_values(DATE_COLUMN).reset_index(drop=True)



def _validate_future_column_if_needed(
    df: pd.DataFrame,
    config: RLPolicySimulationConfig,
) -> None:
    """Ensure futures prices are available if the RL policy buys futures."""
    needs_future = (df[config.action_column] == ACTION_BUY_M1_FUTURE).any()
    if needs_future and config.future_column not in df.columns:
        raise RLPolicySimulationError(
            f"RL policy includes '{ACTION_BUY_M1_FUTURE}' but futures column '{config.future_column}' is missing."
        )


# =========================
# Row-level simulation logic
# =========================


def _get_energy_volume(row: pd.Series, config: RLPolicySimulationConfig) -> float:
    """Resolve daily energy volume from row data or project default."""
    if config.volume_column in row.index and pd.notna(row[config.volume_column]):
        return float(row[config.volume_column])
    return float(DEFAULT_DAILY_VOLUME)



def _resolve_futures_price(
    row: pd.Series,
    m1_column: str,
    m2_column: str,
    m3_column: str,
    action: str,
) -> float:
    """Return the appropriate futures price for a hedge action, with graceful fallback."""
    m1_raw = row[m1_column] if m1_column in row.index else float("nan")
    m1_price = float(m1_raw) if pd.notna(m1_raw) else float("nan")

    if action == ACTION_BUY_M1_FUTURE:
        return m1_price
    if action == ACTION_BUY_M2_FUTURE:
        raw = row[m2_column] if m2_column in row.index else float("nan")
        return float(raw) if pd.notna(raw) else (m1_price * 1.01 if not pd.isna(m1_price) else float("nan"))
    if action == ACTION_BUY_M3_FUTURE:
        raw_m2 = row[m2_column] if m2_column in row.index else float("nan")
        m2_price = float(raw_m2) if pd.notna(raw_m2) else (m1_price * 1.01 if not pd.isna(m1_price) else float("nan"))
        raw_m3 = row[m3_column] if m3_column in row.index else float("nan")
        return float(raw_m3) if pd.notna(raw_m3) else (m2_price * 1.01 if not pd.isna(m2_price) else float("nan"))
    return m1_price


def _simulate_rl_policy_row(
    row: pd.Series,
    config: RLPolicySimulationConfig,
) -> dict[str, float | str]:
    """Simulate one day of procurement cost under an RL-recommended action.

    Supports all 7 actions:
    - do_nothing             → all energy at spot
    - buy_m1_future          → hedged fraction at M1 price, remainder at spot
    - shift_production       → legacy shift-penalty model
    - increase_production    → volume scaled up by PRODUCTION_STEP, pay spot
    - decrease_production    → volume scaled down by PRODUCTION_STEP, pay spot
    - buy_m2_future          → hedged fraction at M2 price, remainder at spot
    - buy_m3_future          → hedged fraction at M3 price, remainder at spot
    """
    action = row[config.action_column]
    spot_price = float(row[config.spot_column])
    energy_volume_mwh = _get_energy_volume(row, config)

    future_volume_mwh = 0.0
    spot_volume_mwh = energy_volume_mwh
    shifted_volume_mwh = 0.0
    future_cost = 0.0
    spot_cost = spot_price * energy_volume_mwh
    shift_cost = 0.0

    if action == ACTION_DO_NOTHING:
        total_cost = spot_cost

    elif action in {ACTION_BUY_M1_FUTURE, ACTION_BUY_M2_FUTURE, ACTION_BUY_M3_FUTURE}:
        future_price = _resolve_futures_price(
            row, config.future_column, _M2_FUTURE_COLUMN, _M3_FUTURE_COLUMN, action
        )
        future_volume_mwh = energy_volume_mwh * config.hedge_ratio_on_buy_future
        spot_volume_mwh = energy_volume_mwh - future_volume_mwh
        future_cost = future_volume_mwh * future_price
        spot_cost = spot_volume_mwh * spot_price
        total_cost = future_cost + spot_cost

    elif action == ACTION_SHIFT_PRODUCTION:
        shifted_volume_mwh = energy_volume_mwh * config.shift_fraction
        spot_volume_mwh = energy_volume_mwh - shifted_volume_mwh
        spot_cost = spot_volume_mwh * spot_price
        shift_cost = shifted_volume_mwh * config.shift_penalty_per_mwh
        total_cost = spot_cost + shift_cost

    elif action == ACTION_INCREASE_PRODUCTION:
        adjusted_volume = energy_volume_mwh * (1.0 + PRODUCTION_STEP)
        spot_volume_mwh = adjusted_volume
        spot_cost = adjusted_volume * spot_price
        total_cost = spot_cost
        energy_volume_mwh = adjusted_volume

    elif action == ACTION_DECREASE_PRODUCTION:
        adjusted_volume = energy_volume_mwh * (1.0 - PRODUCTION_STEP)
        spot_volume_mwh = adjusted_volume
        spot_cost = adjusted_volume * spot_price
        total_cost = spot_cost
        energy_volume_mwh = adjusted_volume

    else:
        raise RLPolicySimulationError(f"Unsupported RL action encountered: {action}")

    return {
        "action_taken": action,
        "energy_volume_mwh": energy_volume_mwh,
        "future_volume_mwh": future_volume_mwh,
        "spot_volume_mwh": spot_volume_mwh,
        "shifted_volume_mwh": shifted_volume_mwh,
        "future_cost": future_cost,
        "spot_cost": spot_cost,
        "shift_cost": shift_cost,
        "total_cost": total_cost,
    }


# =========================
# Public simulation API
# =========================


def simulate_rl_policy_strategy(
    policy_df: pd.DataFrame,
    config: RLPolicySimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate a trained RL policy over a validation/test decision dataframe.

    Parameters
    ----------
    policy_df : pd.DataFrame
        Dataframe containing dates, market prices, and RL-recommended actions.
    config : RLPolicySimulationConfig | None, optional
        Simulation settings.

    Returns
    -------
    pd.DataFrame
        Daily simulation dataframe compatible with the existing backtesting layer.
    """
    resolved_config = RLPolicySimulationConfig() if config is None else config
    validated_df = _validate_input_dataframe(policy_df, resolved_config)
    _validate_future_column_if_needed(validated_df, resolved_config)

    simulation_rows = []
    for _, row in validated_df.iterrows():
        row_result = _simulate_rl_policy_row(row, resolved_config)
        output_row = row.to_dict()
        output_row.update(row_result)
        output_row["strategy_name"] = STRATEGY_RL_POLICY
        simulation_rows.append(output_row)

    simulation_df = pd.DataFrame(simulation_rows)

    logger.info("RL policy simulation completed successfully.")
    logger.info(f"Simulated rows: {simulation_df.shape[0]}")
    return simulation_df



def summarize_rl_policy_simulation(simulation_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact one-row summary of RL policy simulation results."""
    if not isinstance(simulation_df, pd.DataFrame):
        raise RLPolicySimulationError("simulation_df must be a pandas DataFrame.")
    if simulation_df.empty:
        raise RLPolicySimulationError("simulation_df is empty.")

    summary_df = pd.DataFrame(
        {
            "strategy_name": [STRATEGY_RL_POLICY],
            "n_days": [int(simulation_df.shape[0])],
            "total_volume_mwh": [float(simulation_df["energy_volume_mwh"].sum())],
            "total_future_cost": [float(simulation_df["future_cost"].sum())],
            "total_spot_cost": [float(simulation_df["spot_cost"].sum())],
            "total_shift_cost": [float(simulation_df["shift_cost"].sum())],
            "total_cost": [float(simulation_df["total_cost"].sum())],
            "average_daily_cost": [float(simulation_df["total_cost"].mean())],
            "cost_std": [float(simulation_df["total_cost"].std(ddof=0))],
            "n_buy_m1_future_days": [
                int((simulation_df["action_taken"] == ACTION_BUY_M1_FUTURE).sum())
            ],
            "n_shift_production_days": [
                int((simulation_df["action_taken"] == ACTION_SHIFT_PRODUCTION).sum())
            ],
            "n_do_nothing_days": [
                int((simulation_df["action_taken"] == ACTION_DO_NOTHING).sum())
            ],
        }
    )
    return summary_df


# =========================
# Smoke test
# =========================

if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=6, freq="D"),
            SPOT_PRICE_COLUMN: [70, 75, 80, 78, 82, 76],
            PRIMARY_FUTURE_COLUMN: [72, 73, 74, 75, 76, 77],
            DEFAULT_VOLUME_COLUMN: [10, 10, 10, 10, 10, 10],
            DEFAULT_ACTION_COLUMN: [
                ACTION_DO_NOTHING,
                ACTION_BUY_M1_FUTURE,
                ACTION_BUY_M1_FUTURE,
                ACTION_SHIFT_PRODUCTION,
                ACTION_DO_NOTHING,
                ACTION_SHIFT_PRODUCTION,
            ],
        }
    )

    simulated_df = simulate_rl_policy_strategy(example_df)
    summary_df = summarize_rl_policy_simulation(simulated_df)

    logger.info(f"RL simulation head:\n{simulated_df.head()}")
    logger.info(f"RL simulation summary:\n{summary_df}")
