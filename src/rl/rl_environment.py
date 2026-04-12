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
    ACTION_DO_NOTHING,
    ACTION_SHIFT_PRODUCTION,
    PRIMARY_FUTURE_COLUMN,
    Q50_COLUMN,
    Q90_COLUMN,
)
from src.config.settings import RLSettings, get_default_settings


class RLEnvironmentError(Exception):
    """Raised when the RL environment cannot be initialized or stepped safely."""


@dataclass
class RLEnvironmentConfig:
    q50_column: str = Q50_COLUMN
    q90_column: str = Q90_COLUMN
    future_column: str = PRIMARY_FUTURE_COLUMN
    spot_column: str = "Spot_Price_SPEL"
    tail_vs_future_abs_column: str = "tail_vs_future_abs"
    tail_vs_central_abs_column: str = "tail_vs_central_abs"
    weekend_column: str = "is_weekend"
    holiday_column: str = "Is_national_holiday"
    action_column: str = "action"

    # Reward shaping
    risk_aversion: float = 1.0
    hedge_cost_penalty: float = 0.1
    shift_penalty: float = 0.05
    action_penalty: float = 0.0

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
            shift_penalty=0.05,
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



def _validate_action_catalog() -> None:
    """Validate the centralized action catalog used by the RL environment."""
    expected_actions = {
        ACTION_DO_NOTHING,
        ACTION_BUY_M1_FUTURE,
        ACTION_SHIFT_PRODUCTION,
    }
    if len(ACTIONS) < 3 or set(ACTIONS[:3]) != expected_actions:
        raise RLEnvironmentError(
            "Centralized ACTIONS constant must contain the expected RL action labels in the first three positions."
        )


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
        _validate_action_catalog()
        self.config = _get_config(config)
        self.df = self._validate_df(df)

        self.current_step = 0
        self.done = False

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
        if action not in {0, 1, 2}:
            raise RLEnvironmentError(
                f"Unsupported action '{action}'. Expected one of: 0, 1, 2."
            )

    # =========================
    # Core RL API
    # =========================

    def reset(self) -> dict[str, Any]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.done = False
        return self._get_state()

    def step(self, action: int) -> tuple[dict[str, float], float, bool, dict[str, Any]]:
        """Apply one encoded action, observe reward, and transition to the next state."""
        if self.done:
            raise RLEnvironmentError("Episode already finished. Call reset().")
        self._validate_action(action)

        current_row = self.df.iloc[self.current_step]

        reward = self._compute_reward(current_row, action)

        self.current_step += 1

        if self.current_step >= len(self.df):
            self.done = True

        next_state = self._get_state()

        return next_state, reward, self.done, {}

    # =========================
    # State
    # =========================

    def _get_state(self) -> dict[str, float]:
        """Return the current RL state as a compact numeric dictionary."""
        row = self.df.iloc[self.current_step]

        state = {
            "step": self.current_step,
            "action": action,
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

        return state

    # =========================
    # Reward function
    # =========================

    def _compute_reward(self, action: int) -> float:
        """Compute a cost-based reward aligned with the procurement decision problem."""
        row = self.df.iloc[self.current_step]

        central_forecast = float(row[self.config.q50_column])
        tail_forecast = float(row[self.config.q90_column])
        future_price = float(row[self.config.future_column])
        spot_price = (
            float(row[self.config.spot_column])
            if self.config.spot_column in row.index and pd.notna(row[self.config.spot_column])
            else central_forecast
        )

        base_cost = spot_price
        realized_cost = spot_price

        if action == 1:
            realized_cost = future_price + self.config.hedge_cost_penalty * future_price
        elif action == 2:
            realized_cost = max(0.0, spot_price - 0.5 * (tail_forecast - central_forecast))
            realized_cost += self.config.shift_penalty * spot_price

        tail_risk = max(0.0, tail_forecast - central_forecast)
        reward = -realized_cost - self.config.risk_aversion * tail_risk

        if action != 0:
            reward -= self.config.action_penalty

        reward += max(0.0, base_cost - realized_cost)
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