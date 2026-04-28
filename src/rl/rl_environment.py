"""
rl_environment.py

Reinforcement learning environment implementing the factory procurement MDP.

MDP Parameters (Single Source of Truth — from Mathematical Formulation):
  P_max   = 2000 units/day   maximum production capacity
  D       = 1000 units/day   fixed daily demand dispatched at 09:00
  I_max   = 3000 units       maximum warehouse capacity
  e_start = 20 MWh           startup energy when factory is on
  e_unit  = 1 MWh/unit       variable energy per unit produced
  h       = 5 EUR/unit/day   inventory holding cost
  M       = 200 EUR/unit     gross profit margin

Action Space (Joint Compound, Discrete — 567 total):
  P_{t+1}  ∈ {0, 100, 200, ..., 2000}   21 production levels
  b_m1     ∈ {0, 500, 1000} MWh          3 M1 block sizes
  b_m2     ∈ {0, 500, 1000} MWh          3 M2 block sizes
  b_m3     ∈ {0, 500, 1000} MWh          3 M3 block sizes
  Encoding: action_id = prod_idx * 27 + m1_idx * 9 + m2_idx * 3 + m3_idx

Reward Formula (Take-or-Pay):
  E_t         = e_start + e_unit * P_{t+1}   if P_{t+1} > 0, else 0
  hedge_pay   = b_m1 * p_M1 + b_m2 * p_M2 + b_m3 * p_M3
  deficit     = max(0, E_t - b_m1 - b_m2 - b_m3)
  spot_pay    = deficit * spot_price
  I_{t+1}     = clip(I_t + P_{t+1} - D, 0, I_max)
  R_t         = M * P_{t+1} - h * I_{t+1} - hedge_pay - spot_pay

Required State Columns:
  q_0.5, q_0.9              (t+2 central / tail forecast)
  q_0.5_h3, q_0.9_h3       (t+3 central / tail forecast)
  Future_M1_Price           (M1 futures price)
  Spot_Price_SPEL           (day-ahead spot price)
  Spot_M1_Spread            (auto-computed as Spot - M1 when absent)

Optional State Columns:
  Future_M2_Price, Future_M3_Price  (fall back to M1 price when absent)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.constants import (
    MDP_D,
    MDP_E_START,
    MDP_E_UNIT,
    MDP_H,
    MDP_I_MAX,
    MDP_INITIAL_INVENTORY,
    MDP_M,
    MDP_N_ACTIONS,
    MDP_P_MAX,
    PRIMARY_FUTURE_COLUMN,
    Q50_COLUMN,
    Q50_H3_COLUMN,
    Q90_COLUMN,
    Q90_H3_COLUMN,
    SECONDARY_FUTURE_COLUMN,
    SPOT_M1_SPREAD_COLUMN,
)
from src.rl.utils_rl import decode_compound_action


class RLEnvironmentError(Exception):
    """Raised when the RL environment cannot be initialized or stepped safely."""


_M3_FUTURE_COLUMN = "Future_M3_Price"


@dataclass
class RLEnvironmentConfig:
    """Column-name mappings and initial-state for the RL environment.

    All factory physics (P_max, D, I_max, e_start, e_unit, h, M) are fixed
    by the MDP formulation in constants.py.  This config only controls which
    dataframe columns supply market data and the starting inventory level.
    """

    q50_column: str = Q50_COLUMN
    q90_column: str = Q90_COLUMN
    q50_h3_column: str = Q50_H3_COLUMN
    q90_h3_column: str = Q90_H3_COLUMN
    future_m1_column: str = PRIMARY_FUTURE_COLUMN
    future_m2_column: str = SECONDARY_FUTURE_COLUMN
    future_m3_column: str = _M3_FUTURE_COLUMN
    spot_column: str = "Spot_Price_SPEL"
    spot_m1_spread_column: str = SPOT_M1_SPREAD_COLUMN
    initial_inventory: int = MDP_INITIAL_INVENTORY


class EnergyRLEnvironment:
    """
    Step-based factory procurement environment.

    State  : market forecasts + Spot-M1 spread + inventory level
    Action : integer in [0, 567) encoding (P_{t+1}, b_m1, b_m2, b_m3)
    Reward : R_t = M*P - h*I_{t+1} - hedge_payment - spot_deficit_payment
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: RLEnvironmentConfig | None = None,
    ):
        self.config = config or RLEnvironmentConfig()
        self.df = self._validate_df(df)
        self.current_step: int = 0
        self.done: bool = False
        self.inventory: float = float(self.config.initial_inventory)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise RLEnvironmentError("Input dataframe is empty.")

        c = self.config
        required = [
            c.q50_column,
            c.q90_column,
            c.q50_h3_column,
            c.q90_h3_column,
            c.future_m1_column,
            c.spot_column,
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise RLEnvironmentError(f"Missing required columns: {missing}")

        if df[required].isna().any().any():
            raise RLEnvironmentError(
                "RL environment input contains NaN values in required state columns."
            )

        result = df.copy().reset_index(drop=True)

        # Auto-compute Spot_M1_Spread when absent
        if c.spot_m1_spread_column not in result.columns:
            result[c.spot_m1_spread_column] = (
                result[c.spot_column] - result[c.future_m1_column]
            )

        for col in [c.future_m2_column, c.future_m3_column, c.spot_m1_spread_column]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        return result

    def _validate_action(self, action: int) -> None:
        if not isinstance(action, int) or not (0 <= action < MDP_N_ACTIONS):
            raise RLEnvironmentError(
                f"Invalid action {action!r}. Must be int in [0, {MDP_N_ACTIONS})."
            )

    # ------------------------------------------------------------------
    # Core RL API
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Reset environment to initial state and return the first observation."""
        self.current_step = 0
        self.done = False
        self.inventory = float(self.config.initial_inventory)
        return self._get_state()

    def step(
        self, action: int
    ) -> tuple[dict[str, float], float, bool, dict[str, Any]]:
        """Apply action, compute reward, advance to next step."""
        if self.done:
            raise RLEnvironmentError("Episode finished. Call reset().")
        self._validate_action(action)

        prod, b_m1, b_m2, b_m3 = decode_compound_action(action)

        # Enforce inventory feasibility: don't overfill warehouse
        prod = _feasible_production(prod, self.inventory)

        reward = self._compute_reward(prod, b_m1, b_m2, b_m3)

        # I_{t+1} = clip(I_t + P - D, 0, I_max)
        self.inventory = float(
            max(0.0, min(float(MDP_I_MAX), self.inventory + prod - MDP_D))
        )

        info: dict[str, Any] = {
            "step": self.current_step,
            "production": prod,
            "m1_block_mwh": b_m1,
            "m2_block_mwh": b_m2,
            "m3_block_mwh": b_m3,
            "inventory_after": self.inventory,
        }

        self.current_step += 1
        self.done = self.current_step >= len(self.df)
        next_state: dict[str, float] = {} if self.done else self._get_state()
        return next_state, reward, self.done, info

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_state(self) -> dict[str, float]:
        """Return the current observation as a numeric dictionary."""
        row = self.df.iloc[self.current_step]
        c = self.config

        return {
            "forecast_central": float(row[c.q50_column]),
            "forecast_tail": float(row[c.q90_column]),
            "forecast_central_h3": float(row[c.q50_h3_column]),
            "forecast_tail_h3": float(row[c.q90_h3_column]),
            "m1_price": float(row[c.future_m1_column]),
            "spot_price": float(row[c.spot_column]),
            "spot_m1_spread": float(row[c.spot_m1_spread_column]),
            "inventory": float(self.inventory),
            "inventory_bin": float(self._inventory_bin()),
        }

    def _inventory_bin(self) -> int:
        """Discretize inventory into three coarse bins: 0=low, 1=mid, 2=high."""
        ratio = self.inventory / MDP_I_MAX
        if ratio < 0.33:
            return 0
        if ratio < 0.67:
            return 1
        return 2

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        prod: int,
        b_m1: int,
        b_m2: int,
        b_m3: int,
    ) -> float:
        """
        R_t = M * P_{t+1} - h * I_{t+1} - hedge_payment - spot_payment

        hedge_payment = b_m1*p_M1 + b_m2*p_M2 + b_m3*p_M3  (Fixed Hedge Payment)
        spot_payment  = max(0, E_t - total_hedged) * spot    (Spot deficit)
        E_t           = e_start + e_unit * P  if P > 0  else 0
        I_{t+1}       = clip(I_t + P - D, 0, I_max)
        """
        row = self.df.iloc[self.current_step]
        c = self.config

        spot_price = float(row[c.spot_column])
        m1_price = float(row[c.future_m1_column])
        m2_price = float(
            row[c.future_m2_column]
            if c.future_m2_column in row.index and pd.notna(row[c.future_m2_column])
            else m1_price
        )
        m3_price = float(
            row[c.future_m3_column]
            if c.future_m3_column in row.index and pd.notna(row[c.future_m3_column])
            else m2_price
        )

        # Energy required this step
        e_req = (MDP_E_START + MDP_E_UNIT * prod) if prod > 0 else 0.0

        # Fixed Hedge Payment
        hedge_payment = b_m1 * m1_price + b_m2 * m2_price + b_m3 * m3_price

        # Spot Payment for any energy deficit not covered by futures blocks
        total_hedged = float(b_m1 + b_m2 + b_m3)
        deficit = max(0.0, e_req - total_hedged)
        spot_payment = deficit * spot_price

        # Next-step inventory (used for holding cost)
        i_next = max(0.0, min(float(MDP_I_MAX), self.inventory + prod - MDP_D))

        return float(MDP_M * prod - MDP_H * i_next - hedge_payment - spot_payment)


# ------------------------------------------------------------------
# Private helper
# ------------------------------------------------------------------


def _feasible_production(prod: int, inventory: float) -> int:
    """Cap production to prevent warehouse overflow, then snap to nearest valid level."""
    max_feasible = int(MDP_I_MAX - inventory + MDP_D)
    capped = min(prod, max(0, max_feasible))
    snapped = round(capped / 100) * 100
    return int(max(0, min(MDP_P_MAX, snapped)))


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            Q50_COLUMN: [50.0, 55.0, 60.0],
            Q90_COLUMN: [60.0, 70.0, 80.0],
            Q50_H3_COLUMN: [52.0, 58.0, 63.0],
            Q90_H3_COLUMN: [65.0, 75.0, 85.0],
            PRIMARY_FUTURE_COLUMN: [52.0, 57.0, 63.0],
            "Spot_Price_SPEL": [51.0, 58.0, 66.0],
            SECONDARY_FUTURE_COLUMN: [53.0, 58.0, 64.0],
            _M3_FUTURE_COLUMN: [54.0, 59.0, 65.0],
        }
    )

    env = EnergyRLEnvironment(example_df)
    state = env.reset()
    print("Initial state:", state)

    done = False
    step = 0
    while not done:
        action = 10 * 27 + 1 * 9 + 0 * 3 + 0  # P=1000, M1=500, M2=0, M3=0
        next_state, reward, done, info = env.step(action)
        print(f"Step {step}: reward={reward:.2f}, info={info}")
        step += 1
