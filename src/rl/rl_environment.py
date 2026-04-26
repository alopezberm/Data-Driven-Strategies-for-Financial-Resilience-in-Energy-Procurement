"""
rl_environment.py

Lightweight reinforcement learning environment for energy procurement decisions.
This is NOT a full Gym environment, but a clean, extensible abstraction that
can later be upgraded to Gym / Gymnasium if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.constants import (
    ACTIONS,
    ACTION_BUY_M1_FUTURE,
    ACTION_BUY_M2_FUTURE,
    ACTION_BUY_M3_FUTURE,
    ACTION_DECREASE_PRODUCTION,
    ACTION_DO_NOTHING,
    ACTION_INCREASE_PRODUCTION,
    ACTION_SHIFT_PRODUCTION,
    DEFAULT_PRODUCTION_LEVEL,
    FACTORY_BASE_LOAD,
    FACTORY_VARIABLE_LOAD,
    PRIMARY_FUTURE_COLUMN,
    PRODUCTION_LEVELS,
    PRODUCTION_STEP,
    Q50_COLUMN,
    Q90_COLUMN,
    SECONDARY_FUTURE_COLUMN,
    validate_action_catalog,
)
from src.config.settings import RLSettings, get_default_settings


class RLEnvironmentError(Exception):
    """Raised when the RL environment cannot be initialized or stepped safely."""


@dataclass
class RLEnvironmentConfig:
    q50_column: str = Q50_COLUMN
    q90_column: str = Q90_COLUMN
    future_column: str = PRIMARY_FUTURE_COLUMN
    future_m2_column: str = SECONDARY_FUTURE_COLUMN
    spot_column: str = "Spot_Price_SPEL"
    tail_vs_future_abs_column: str = "tail_vs_future_abs"
    tail_vs_central_abs_column: str = "tail_vs_central_abs"
    weekend_column: str = "is_weekend"
    holiday_column: str = "Is_national_holiday"
    action_column: str = "action"

    # Reward shaping
    risk_aversion: float = 0.0
    hedge_cost_penalty: float = 0.0
    shift_penalty: float = 2.0
    action_penalty: float = 0.0

    # Factory model — energy consumption = base_load + variable_load * production_level
    factory_base_load: float = FACTORY_BASE_LOAD
    factory_variable_load: float = FACTORY_VARIABLE_LOAD
    initial_production_level: float = DEFAULT_PRODUCTION_LEVEL

    # When True the agent may use the 7-action extended catalog;
    # when False only the original 3 actions (0-2) are valid (backward-compat).
    use_extended_actions: bool = False

    @classmethod
    def from_settings(cls, settings: RLSettings) -> "RLEnvironmentConfig":
        """Build environment configuration from centralized RL settings."""
        return cls(
            q50_column=Q50_COLUMN,
            q90_column=Q90_COLUMN,
            future_column=PRIMARY_FUTURE_COLUMN,
            spot_column="Spot_Price_SPEL",
            tail_vs_future_abs_column="tail_vs_future_abs",
            tail_vs_central_abs_column="tail_vs_central_abs",
            weekend_column="is_weekend",
            holiday_column="Is_national_holiday",
            action_column="action",
            risk_aversion=settings.risk_aversion,
            hedge_cost_penalty=settings.hedge_cost_penalty,
            shift_penalty=settings.heuristic_shift_threshold,
            action_penalty=0.0,
        )


def get_default_rl_settings() -> RLSettings:
    """Return default RL settings from the project configuration."""
    return get_default_settings().rl



def _get_config(config: RLEnvironmentConfig | None) -> RLEnvironmentConfig:
    """Resolve an explicit environment config or build one from project settings."""
    if config is not None:
        return config
    return RLEnvironmentConfig.from_settings(get_default_rl_settings())



class EnergyRLEnvironment:
    """
    Simple step-based environment for sequential decision making.

    State = row of dataframe
    Action = discrete (e.g., 0: do nothing, 1: hedge, 2: shift)
    Reward = function of future price vs tail risk
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: RLEnvironmentConfig | None = None,
    ):
        validate_action_catalog()
        self.config = _get_config(config)
        self.df = self._validate_df(df)

        self.current_step = 0
        self.done = False
        # Factory state: production level (fraction of nominal capacity)
        self.production_level: float = self.config.initial_production_level

    # =========================
    # Validation
    # =========================

    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the RL environment input dataframe."""
        if df.empty:
            raise RLEnvironmentError("Input dataframe is empty.")

        required = [
            self.config.q50_column,
            self.config.q90_column,
            self.config.future_column,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RLEnvironmentError(f"Missing required columns: {missing}")

        if df[required].isna().any().any():
            raise RLEnvironmentError(
                "RL environment input contains missing values in required state columns."
            )

        result_df = df.copy()

        optional_numeric_columns = [
            self.config.spot_column,
            self.config.tail_vs_future_abs_column,
            self.config.tail_vs_central_abs_column,
            self.config.weekend_column,
            self.config.holiday_column,
        ]
        for column in optional_numeric_columns:
            if column in result_df.columns:
                result_df[column] = pd.to_numeric(result_df[column], errors="coerce")

        return result_df.reset_index(drop=True)

    def _validate_action(self, action: int) -> None:
        """Validate that the provided encoded action belongs to the supported discrete action space."""
        if self.config.use_extended_actions:
            valid = set(range(len(ACTIONS)))
            if action not in valid:
                raise RLEnvironmentError(
                    f"Unsupported action '{action}'. Extended mode supports 0–{len(ACTIONS)-1}."
                )
        else:
            if action not in {0, 1, 2}:
                raise RLEnvironmentError(
                    f"Unsupported action '{action}'. Expected one of: 0, 1, 2."
                )

    # =========================
    # Core RL API
    # =========================

    def reset(self) -> dict[str, Any]:
        """Reset environment to initial state, including factory production level."""
        self.current_step = 0
        self.done = False
        self.production_level = self.config.initial_production_level
        return self._get_state()

    def step(self, action: int) -> tuple[dict[str, float], float, bool, dict[str, Any]]:
        """Apply one encoded action, observe reward, and transition to the next state."""
        if self.done:
            raise RLEnvironmentError("Episode already finished. Call reset().")
        self._validate_action(action)

        reward = self._compute_reward(action)

        row = self.df.iloc[self.current_step]
        info = {
            "step": self.current_step,
            "action": action,
            "forecast_central": float(row[self.config.q50_column]),
            "forecast_tail": float(row[self.config.q90_column]),
            "current_m1_future": float(row[self.config.future_column]),
        }
        if self.config.spot_column in row.index and pd.notna(row[self.config.spot_column]):
            info["current_spot"] = float(row[self.config.spot_column])

        self.current_step += 1

        if self.current_step >= len(self.df):
            self.done = True
            next_state = {}
        else:
            next_state = self._get_state()

        return next_state, reward, self.done, info

    # =========================
    # State
    # =========================

    def _get_state(self) -> dict[str, float]:
        """Return the current RL state as a compact numeric dictionary."""
        row = self.df.iloc[self.current_step]

        state = {
            "forecast_central": float(row[self.config.q50_column]),
            "forecast_tail": float(row[self.config.q90_column]),
            "current_m1_future": float(row[self.config.future_column]),
        }

        if self.config.spot_column in row.index and pd.notna(row[self.config.spot_column]):
            state["current_spot"] = float(row[self.config.spot_column])

        if (
            self.config.tail_vs_future_abs_column in row.index
            and pd.notna(row[self.config.tail_vs_future_abs_column])
        ):
            state["tail_vs_future_abs"] = float(row[self.config.tail_vs_future_abs_column])
        else:
            state["tail_vs_future_abs"] = state["forecast_tail"] - state["current_m1_future"]

        if (
            self.config.tail_vs_central_abs_column in row.index
            and pd.notna(row[self.config.tail_vs_central_abs_column])
        ):
            state["tail_vs_central_abs"] = float(row[self.config.tail_vs_central_abs_column])
        else:
            state["tail_vs_central_abs"] = state["forecast_tail"] - state["forecast_central"]

        if self.config.weekend_column in row.index and pd.notna(row[self.config.weekend_column]):
            state["is_weekend"] = float(row[self.config.weekend_column])
        else:
            state["is_weekend"] = 0.0

        if self.config.holiday_column in row.index and pd.notna(row[self.config.holiday_column]):
            state["is_holiday"] = float(row[self.config.holiday_column])
        else:
            state["is_holiday"] = 0.0

        # Factory model state components
        state["production_level"] = float(self.production_level)
        state["energy_consumption"] = float(
            self.config.factory_base_load
            + self.config.factory_variable_load * self.production_level
        )

        return state

    # =========================
    # Reward function
    # =========================

    def _compute_reward(self, action: int) -> float:
        """
        Compute a daily procurement-cost-based reward for RL training.

        Cost model
        ----------
        Energy consumed = base_load + variable_load * production_level
        The action determines the unit price paid for that energy:

        Original actions (0–2):
            0 do_nothing      → pay spot price
            1 buy_m1_future   → pay M+1 futures price (+ optional hedge premium)
            2 shift_production → legacy shift penalty model

        Extended actions (3–6, only when use_extended_actions=True):
            3 increase_production → raise output, pay spot for extra energy
            4 decrease_production → cut output, save energy cost
            5 buy_m2_future       → pay M+2 futures price
            6 buy_m3_future       → pay M+3 futures price (slightly higher risk premium)

        Reward = –total_cost  (agent maximises by minimising cost)
        """
        row = self.df.iloc[self.current_step]

        central_forecast = float(row[self.config.q50_column])
        future_price = float(row[self.config.future_column])
        spot_price = (
            float(row[self.config.spot_column])
            if self.config.spot_column in row.index and pd.notna(row[self.config.spot_column])
            else central_forecast
        )

        # Energy consumption scaled by current production level
        energy = (
            self.config.factory_base_load
            + self.config.factory_variable_load * self.production_level
        )

        # --- Original actions ---
        if action == 0:  # do_nothing
            realized_cost = energy * spot_price
        elif action == 1:  # buy_m1_future
            realized_cost = energy * (future_price + self.config.hedge_cost_penalty)
        elif action == 2:  # shift_production (legacy)
            shifted_fraction = 0.5
            realized_cost = energy * (
                (1.0 - shifted_fraction) * spot_price
                + shifted_fraction * self.config.shift_penalty
            )
        # --- Extended actions ---
        elif action == 3:  # increase_production
            new_level = min(self.production_level + PRODUCTION_STEP, PRODUCTION_LEVELS[-1])
            self.production_level = new_level
            new_energy = self.config.factory_base_load + self.config.factory_variable_load * new_level
            realized_cost = new_energy * spot_price
        elif action == 4:  # decrease_production
            new_level = max(self.production_level - PRODUCTION_STEP, PRODUCTION_LEVELS[0])
            self.production_level = new_level
            new_energy = self.config.factory_base_load + self.config.factory_variable_load * new_level
            realized_cost = new_energy * spot_price
        elif action == 5:  # buy_m2_future
            m2_price = (
                float(row[self.config.future_m2_column])
                if self.config.future_m2_column in row.index and pd.notna(row[self.config.future_m2_column])
                else future_price * 1.01  # small liquidity premium if M+2 not available
            )
            realized_cost = energy * (m2_price + self.config.hedge_cost_penalty)
        elif action == 6:  # buy_m3_future
            m2_price = (
                float(row[self.config.future_m2_column])
                if self.config.future_m2_column in row.index and pd.notna(row[self.config.future_m2_column])
                else future_price * 1.01
            )
            m3_price = m2_price * 1.01  # M+3 has slightly wider bid-ask
            realized_cost = energy * (m3_price + self.config.hedge_cost_penalty)
        else:
            realized_cost = energy * spot_price

        if action != 0:
            realized_cost += self.config.action_penalty

        reward = -realized_cost
        return float(reward)


# =========================
# Quick test
# =========================

if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            Q50_COLUMN: [50, 55, 60],
            Q90_COLUMN: [60, 70, 80],
            PRIMARY_FUTURE_COLUMN: [52, 57, 63],
            "Spot_Price_SPEL": [51, 58, 66],
            "tail_vs_future_abs": [8, 13, 17],
            "tail_vs_central_abs": [10, 15, 20],
            "is_weekend": [0, 0, 1],
            "Is_national_holiday": [0, 0, 0],
        }
    )

    env = EnergyRLEnvironment(example_df)

    state = env.reset()
    print("Initial state:", state)

    done = False
    while not done:
        action = 1  # always hedge (dummy policy)
        next_state, reward, done, _ = env.step(action)
        print(f"Reward: {reward:.2f}, Next state: {next_state}")