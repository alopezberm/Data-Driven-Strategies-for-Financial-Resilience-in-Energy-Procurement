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

from src.config.constants import ACTIONS, PRIMARY_FUTURE_COLUMN, Q50_COLUMN, Q90_COLUMN
from src.config.settings import RLSettings, get_default_settings


class RLEnvironmentError(Exception):
    """Raised when the RL environment cannot be initialized or stepped safely."""


ACTION_DO_NOTHING = ACTIONS[0]
ACTION_BUY_M1_FUTURE = ACTIONS[1]
ACTION_SHIFT_PRODUCTION = ACTIONS[2]


@dataclass
class RLEnvironmentConfig:
    q50_column: str = Q50_COLUMN
    q90_column: str = Q90_COLUMN
    future_column: str = PRIMARY_FUTURE_COLUMN
    action_column: str = "action"

    # Reward shaping
    risk_aversion: float = 1.0
    hedge_cost_penalty: float = 0.1

    @classmethod
    def from_settings(cls, settings: RLSettings) -> "RLEnvironmentConfig":
        """Build environment configuration from centralized RL settings."""
        return cls(
            q50_column=Q50_COLUMN,
            q90_column=Q90_COLUMN,
            future_column=PRIMARY_FUTURE_COLUMN,
            action_column="action",
            risk_aversion=settings.risk_aversion,
            hedge_cost_penalty=settings.hedge_cost_penalty,
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

        return df.reset_index(drop=True).copy()

    def _validate_action(self, action: int) -> None:
        """Validate that the provided action belongs to the supported discrete action space."""
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

    def step(self, action: int):
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            0 = do nothing
            1 = hedge (buy future)
            2 = shift production
        """
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

    def _get_state(self) -> dict[str, Any]:
        if self.current_step >= len(self.df):
            return {}

        row = self.df.iloc[self.current_step]

        return {
            "q50": row[self.config.q50_column],
            "q90": row[self.config.q90_column],
            "future_price": row[self.config.future_column],
        }

    # =========================
    # Reward function
    # =========================

    def _compute_reward(self, row: pd.Series, action: int) -> float:
        q50 = float(row[self.config.q50_column])
        q90 = float(row[self.config.q90_column])
        future_price = float(row[self.config.future_column])

        # Risk signal
        tail_risk = q90 - future_price

        # Base reward (negative risk exposure)
        reward = -self.config.risk_aversion * max(tail_risk, 0)

        # Action adjustments
        if action == 1:  # ACTION_BUY_M1_FUTURE
            reward += tail_risk  # hedge protects from tail
            reward -= self.config.hedge_cost_penalty * future_price

        elif action == 2:  # ACTION_SHIFT_PRODUCTION
            reward += 0.5 * tail_risk

        # action == 0 (ACTION_DO_NOTHING) -> no adjustment

        return reward


# =========================
# Quick test
# =========================

if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            Q50_COLUMN: [50, 55, 60],
            Q90_COLUMN: [60, 70, 80],
            PRIMARY_FUTURE_COLUMN: [52, 57, 63],
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