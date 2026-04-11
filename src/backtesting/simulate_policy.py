"""
simulate_policy.py

Simulation utilities for the heuristic DSS policy.
This module converts recommended actions into counterfactual daily procurement
costs so the policy can be compared against baseline strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_ACTION_COLUMN = "recommended_action"
DEFAULT_REASON_COLUMN = "decision_reason"
DEFAULT_SPOT_COLUMN = "Spot_Price_SPEL"
DEFAULT_FUTURE_COLUMN = "Future_M1_Price"
DEFAULT_VOLUME_COLUMN = "daily_energy_mwh"

SUPPORTED_POLICY_ACTIONS = {
    "do_nothing",
    "buy_m1_future",
    "shift_production",
}


class PolicySimulationError(Exception):
    """Raised when the DSS policy cannot be simulated safely."""


@dataclass
class PolicySimulationConfig:
    """Configuration for policy backtesting assumptions."""

    action_column: str = DEFAULT_ACTION_COLUMN
    reason_column: str = DEFAULT_REASON_COLUMN
    spot_column: str = DEFAULT_SPOT_COLUMN
    future_column: str = DEFAULT_FUTURE_COLUMN
    volume_column: str = DEFAULT_VOLUME_COLUMN
    default_daily_volume: float = 1.0

    # Economic assumptions
    hedge_ratio_on_buy_future: float = 1.0
    shift_fraction: float = 1.0
    shift_penalty_per_mwh: float = 0.0


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(
    df: pd.DataFrame,
    config: PolicySimulationConfig,
) -> pd.DataFrame:
    """Validate input data and standardize the date column."""
    if df.empty:
        raise PolicySimulationError("Input dataframe is empty.")

    required_columns = [
        "date",
        config.action_column,
        config.spot_column,
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise PolicySimulationError(
            f"Missing required columns for policy simulation: {missing_columns}"
        )

    validated_df = df.copy()
    validated_df["date"] = pd.to_datetime(validated_df["date"], errors="coerce")

    if validated_df["date"].isna().any():
        invalid_count = int(validated_df["date"].isna().sum())
        raise PolicySimulationError(
            f"Found {invalid_count} invalid date values in policy simulation input."
        )

    if validated_df["date"].duplicated().any():
        raise PolicySimulationError("Input dataframe contains duplicated dates.")

    if not 0 <= config.hedge_ratio_on_buy_future <= 1:
        raise PolicySimulationError("hedge_ratio_on_buy_future must be between 0 and 1.")

    if not 0 <= config.shift_fraction <= 1:
        raise PolicySimulationError("shift_fraction must be between 0 and 1.")

    invalid_actions = set(validated_df[config.action_column].dropna().unique()) - SUPPORTED_POLICY_ACTIONS
    if invalid_actions:
        raise PolicySimulationError(
            f"Unsupported policy actions found: {sorted(invalid_actions)}"
        )

    return validated_df.sort_values("date").reset_index(drop=True)



def _ensure_volume_column(df: pd.DataFrame, config: PolicySimulationConfig) -> pd.DataFrame:
    """Ensure a valid daily energy volume column exists."""
    result_df = df.copy()

    if config.volume_column not in result_df.columns:
        result_df[config.volume_column] = config.default_daily_volume

    result_df[config.volume_column] = pd.to_numeric(
        result_df[config.volume_column], errors="coerce"
    )

    if result_df[config.volume_column].isna().any():
        raise PolicySimulationError(
            f"Column '{config.volume_column}' contains invalid or missing volumes."
        )

    if (result_df[config.volume_column] < 0).any():
        raise PolicySimulationError(
            f"Column '{config.volume_column}' contains negative volumes."
        )

    return result_df



def _validate_future_column_if_needed(df: pd.DataFrame, config: PolicySimulationConfig) -> None:
    """Require a futures column if the policy uses buy_m1_future at least once."""
    needs_future = (df[config.action_column] == "buy_m1_future").any()
    if needs_future and config.future_column not in df.columns:
        raise PolicySimulationError(
            f"Policy includes 'buy_m1_future' but futures column '{config.future_column}' is missing."
        )


# =========================
# Core row-level simulation
# =========================

def _simulate_policy_row(row: pd.Series, config: PolicySimulationConfig) -> dict[str, object]:
    """Translate one policy action into cost components."""
    action = row[config.action_column]
    spot_price = pd.to_numeric(pd.Series([row[config.spot_column]]), errors="coerce").iloc[0]
    volume = pd.to_numeric(pd.Series([row[config.volume_column]]), errors="coerce").iloc[0]
    reason = row.get(config.reason_column, pd.NA)

    if pd.isna(spot_price):
        raise PolicySimulationError("Encountered invalid spot price during policy simulation.")
    if pd.isna(volume):
        raise PolicySimulationError("Encountered invalid energy volume during policy simulation.")

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

    if action == "do_nothing":
        return result

    if action == "buy_m1_future":
        future_price = pd.to_numeric(pd.Series([row.get(config.future_column, pd.NA)]), errors="coerce").iloc[0]
        if pd.isna(future_price):
            raise PolicySimulationError(
                "Encountered 'buy_m1_future' action but the futures price is missing or invalid."
            )

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

    if action == "shift_production":
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
    config = PolicySimulationConfig() if config is None else config

    simulation_df = _validate_input_dataframe(df, config)
    simulation_df = _ensure_volume_column(simulation_df, config)
    _validate_future_column_if_needed(simulation_df, config)

    row_outputs = simulation_df.apply(
        lambda row: _simulate_policy_row(row, config),
        axis=1,
        result_type="expand",
    )

    output_df = simulation_df.copy()
    output_df["strategy_name"] = "heuristic_policy"

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
            "n_buy_m1_future_days": [int((simulation_df["action_taken"] == "buy_m1_future").sum())],
            "n_shift_production_days": [int((simulation_df["action_taken"] == "shift_production").sum())],
            "n_do_nothing_days": [int((simulation_df["action_taken"] == "do_nothing").sum())],
        }
    )

    return summary_df


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="D"),
            "Spot_Price_SPEL": [70, 75, 80, 78, 82, 76],
            "Future_M1_Price": [72, 73, 74, 75, 76, 77],
            "daily_energy_mwh": [10, 10, 10, 10, 10, 10],
            "recommended_action": [
                "do_nothing",
                "buy_m1_future",
                "buy_m1_future",
                "shift_production",
                "do_nothing",
                "shift_production",
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

    simulated_df = simulate_policy_strategy(
        example_df,
        config=PolicySimulationConfig(
            hedge_ratio_on_buy_future=1.0,
            shift_fraction=1.0,
            shift_penalty_per_mwh=2.0,
        ),
    )

    print(
        simulated_df[
            [
                "date",
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
