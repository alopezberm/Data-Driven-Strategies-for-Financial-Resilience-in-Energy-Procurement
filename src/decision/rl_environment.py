

"""
rl_environment.py

Lightweight reinforcement learning environment for energy procurement decisions.
This is NOT a full Gym environment, but a clean, extensible abstraction that
can later be upgraded to Gym / Gymnasium if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd

from src.config.constants import DEFAULT_Q50_COLUMN, DEFAULT_Q90_COLUMN, DEFAULT_FUTURE_COLUMN


class RLEnvironmentError(Exception):
    pass


@dataclass
class RLEnvironmentConfig:
    q50_column: str = DEFAULT_Q50_COLUMN
    q90_column: str = DEFAULT_Q90_COLUMN
    future_column: str = DEFAULT_FUTURE_COLUMN
    action_column: str = "action"

    # Reward shaping
    risk_aversion: float = 1.0
    hedge_cost_penalty: float = 0.1


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
        self.config = RLEnvironmentConfig() if config is None else config
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

        return df.reset_index(drop=True).copy()

    # =========================
    # Core RL API
    # =========================

    def reset(self) -> Dict[str, Any]:
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

    def _get_state(self) -> Dict[str, Any]:
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
        if action == 1:  # hedge
            reward += tail_risk  # hedge protects from tail
            reward -= self.config.hedge_cost_penalty * future_price

        elif action == 2:  # shift
            reward += 0.5 * tail_risk

        # action == 0 → no adjustment

        return reward


# =========================
# Quick test
# =========================

if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            "q_0.5": [50, 55, 60],
            "q_0.9": [60, 70, 80],
            "Future_M1_Price": [52, 57, 63],
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