"""
simulate_rl_policy.py

Backtesting simulation for RL-based procurement decisions.

Uses the MDP energy-cost formula:
  E_t          = e_start + e_unit * P   if P > 0 else 0
  hedge_cost   = m1 * p_M1 + m2 * p_M2 + m3 * p_M3   (Fixed Hedge Payment)
  spot_cost    = max(0, E_t - m1 - m2 - m3) * spot    (Spot Payment for deficit)
  total_cost   = hedge_cost + spot_cost

Inventory dynamics are tracked for diagnostics but do not affect total_cost,
keeping the metric comparable to spot_only, static_hedge, and heuristic_policy.

Input: decisions_df produced by apply_rl_policy(), merged with market prices.
Required columns: date, production_units, m1_block_mwh, m2_block_mwh,
                  m3_block_mwh, Spot_Price_SPEL, Future_M1_Price.
Optional columns: Future_M2_Price, Future_M3_Price.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.constants import (
    DATE_COLUMN,
    MDP_D,
    MDP_E_START,
    MDP_E_UNIT,
    MDP_I_MAX,
    MDP_INITIAL_INVENTORY,
    PRIMARY_FUTURE_COLUMN,
    SECONDARY_FUTURE_COLUMN,
    SPOT_PRICE_COLUMN,
    STRATEGY_RL_POLICY,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RLPolicySimulationError(Exception):
    """Raised when RL policy simulation cannot be executed safely."""


_M2_FUTURE_COLUMN = SECONDARY_FUTURE_COLUMN
_M3_FUTURE_COLUMN = "Future_M3_Price"

_PROD_COLUMN = "production_units"
_M1_BLOCK_COLUMN = "m1_block_mwh"
_M2_BLOCK_COLUMN = "m2_block_mwh"
_M3_BLOCK_COLUMN = "m3_block_mwh"


@dataclass
class RLPolicySimulationConfig:
    """Configuration for simulating RL-driven procurement decisions."""

    spot_column: str = SPOT_PRICE_COLUMN
    future_m1_column: str = PRIMARY_FUTURE_COLUMN
    future_m2_column: str = _M2_FUTURE_COLUMN
    future_m3_column: str = _M3_FUTURE_COLUMN
    prod_column: str = _PROD_COLUMN
    m1_block_column: str = _M1_BLOCK_COLUMN
    m2_block_column: str = _M2_BLOCK_COLUMN
    m3_block_column: str = _M3_BLOCK_COLUMN
    initial_inventory: float = float(MDP_INITIAL_INVENTORY)


# =========================
# Validation
# =========================


def _validate_input_dataframe(
    df: pd.DataFrame,
    config: RLPolicySimulationConfig,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise RLPolicySimulationError("Input must be a pandas DataFrame.")
    if df.empty:
        raise RLPolicySimulationError("Input dataframe is empty.")

    required = [
        DATE_COLUMN,
        config.prod_column,
        config.m1_block_column,
        config.m2_block_column,
        config.m3_block_column,
        config.spot_column,
        config.future_m1_column,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RLPolicySimulationError(
            f"Missing required columns for RL policy simulation: {missing}"
        )

    result = df.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    if result[DATE_COLUMN].isna().any():
        raise RLPolicySimulationError("Invalid date values in RL policy simulation input.")
    if result[DATE_COLUMN].duplicated().any():
        raise RLPolicySimulationError("Duplicated dates in RL policy simulation input.")

    numeric_cols = [
        config.prod_column,
        config.m1_block_column, config.m2_block_column, config.m3_block_column,
        config.spot_column, config.future_m1_column,
    ]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        if result[col].isna().any():
            raise RLPolicySimulationError(f"Column '{col}' contains invalid numeric values.")

    for opt_col in [config.future_m2_column, config.future_m3_column]:
        if opt_col in result.columns:
            result[opt_col] = pd.to_numeric(result[opt_col], errors="coerce")

    return result.sort_values(DATE_COLUMN).reset_index(drop=True)


# =========================
# Row-level simulation
# =========================


def _simulate_row(
    row: pd.Series,
    config: RLPolicySimulationConfig,
) -> dict:
    """Compute energy procurement cost for one day under the RL compound action."""
    prod = int(row[config.prod_column])
    b_m1 = int(row[config.m1_block_column])
    b_m2 = int(row[config.m2_block_column])
    b_m3 = int(row[config.m3_block_column])

    spot_price = float(row[config.spot_column])
    m1_price = float(row[config.future_m1_column])
    m2_price = float(
        row[config.future_m2_column]
        if config.future_m2_column in row.index and pd.notna(row[config.future_m2_column])
        else m1_price
    )
    m3_price = float(
        row[config.future_m3_column]
        if config.future_m3_column in row.index and pd.notna(row[config.future_m3_column])
        else m2_price
    )

    # Energy required
    e_req = (MDP_E_START + MDP_E_UNIT * prod) if prod > 0 else 0.0

    # Fixed Hedge Payment
    hedge_cost = b_m1 * m1_price + b_m2 * m2_price + b_m3 * m3_price

    # Spot Payment for energy deficit not covered by futures blocks
    total_hedged = float(b_m1 + b_m2 + b_m3)
    deficit = max(0.0, e_req - total_hedged)
    spot_cost = deficit * spot_price

    return {
        "production_units": prod,
        "energy_req_mwh": e_req,
        "energy_volume_mwh": e_req,  # alias required by compare_strategies
        "total_hedged_mwh": total_hedged,
        "deficit_mwh": deficit,
        "hedge_cost": hedge_cost,
        "spot_cost": spot_cost,
        "total_cost": hedge_cost + spot_cost,
    }


# =========================
# Public API
# =========================


def simulate_rl_policy_strategy(
    policy_df: pd.DataFrame,
    config: RLPolicySimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate a trained RL policy over a decision dataframe produced by apply_rl_policy().

    Returns a daily simulation dataframe with total_cost compatible with the
    existing backtesting comparison layer.
    """
    resolved_config = config or RLPolicySimulationConfig()
    validated_df = _validate_input_dataframe(policy_df, resolved_config)

    rows = []
    inventory = resolved_config.initial_inventory

    for _, row in validated_df.iterrows():
        sim = _simulate_row(row, resolved_config)
        prod = sim["production_units"]

        # Track inventory for diagnostics
        inventory = max(0.0, min(float(MDP_I_MAX), inventory + prod - MDP_D))
        sim["inventory_after"] = inventory

        output = row.to_dict()
        output.update(sim)
        output["strategy_name"] = STRATEGY_RL_POLICY
        rows.append(output)

    result = pd.DataFrame(rows)

    logger.info("RL policy simulation completed.")
    logger.info(f"Simulated rows: {result.shape[0]}")
    return result


def summarize_rl_policy_simulation(simulation_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact one-row summary of RL policy simulation results."""
    if not isinstance(simulation_df, pd.DataFrame) or simulation_df.empty:
        raise RLPolicySimulationError("simulation_df must be a non-empty DataFrame.")

    return pd.DataFrame(
        {
            "strategy_name": [STRATEGY_RL_POLICY],
            "n_days": [int(simulation_df.shape[0])],
            "total_hedge_cost": [float(simulation_df["hedge_cost"].sum())],
            "total_spot_cost": [float(simulation_df["spot_cost"].sum())],
            "total_cost": [float(simulation_df["total_cost"].sum())],
            "average_daily_cost": [float(simulation_df["total_cost"].mean())],
            "cost_std": [float(simulation_df["total_cost"].std(ddof=0))],
            "avg_production_units": [float(simulation_df["production_units"].mean())],
            "avg_deficit_mwh": [float(simulation_df["deficit_mwh"].mean())],
        }
    )


# =========================
# Smoke test
# =========================

if __name__ == "__main__":
    from src.config.constants import Q50_COLUMN, Q50_H3_COLUMN, Q90_COLUMN, Q90_H3_COLUMN
    from src.rl.utils_rl import encode_compound_action

    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=4, freq="D"),
            SPOT_PRICE_COLUMN: [70.0, 75.0, 80.0, 78.0],
            PRIMARY_FUTURE_COLUMN: [72.0, 73.0, 74.0, 75.0],
            _M2_FUTURE_COLUMN: [73.0, 74.0, 75.0, 76.0],
            _M3_FUTURE_COLUMN: [74.0, 75.0, 76.0, 77.0],
            _PROD_COLUMN: [1000, 1000, 800, 1200],
            _M1_BLOCK_COLUMN: [500, 1000, 0, 500],
            _M2_BLOCK_COLUMN: [0, 0, 500, 0],
            _M3_BLOCK_COLUMN: [0, 0, 0, 0],
        }
    )

    simulated_df = simulate_rl_policy_strategy(example_df)
    summary_df = summarize_rl_policy_simulation(simulated_df)

    logger.info(f"RL simulation head:\n{simulated_df[['total_cost','production_units','hedge_cost','spot_cost']].head()}")
    logger.info(f"RL simulation summary:\n{summary_df}")
