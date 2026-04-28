"""
rl_agent.py

Tabular Q-learning agent for the factory procurement MDP.

Action space: 567 compound actions encoding (P_{t+1}, b_m1, b_m2, b_m3).
State key:    four binned signals — q90_vs_m1, spot_m1_spread,
              q90_h3_vs_m1, inventory_bin.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from src.config.constants import MDP_N_ACTIONS
from src.config.settings import RLSettings, get_default_settings
from src.rl.rl_environment import EnergyRLEnvironment
from src.rl.utils_rl import decode_compound_action


class RLAgentError(Exception):
    """Raised when an RL agent cannot act or train safely."""


# Full compound action space: integers 0 … 566
_FULL_ACTION_SPACE: tuple[int, ...] = tuple(range(MDP_N_ACTIONS))


@dataclass
class RLAgentConfig:
    """Base configuration shared by all lightweight RL agents."""

    action_space: tuple[int, ...] = _FULL_ACTION_SPACE
    random_seed: int = 42
    epsilon: float = 0.25
    learning_rate: float = 0.10
    discount_factor: float = 0.95

    @classmethod
    def from_settings(cls, settings: RLSettings) -> "RLAgentConfig":
        return cls(
            action_space=_FULL_ACTION_SPACE,
            random_seed=42,
            epsilon=settings.epsilon,
            learning_rate=settings.learning_rate,
            discount_factor=settings.discount_factor,
        )


@dataclass
class QLearningAgentConfig(RLAgentConfig):
    """Configuration for the tabular Q-learning agent."""

    epsilon_decay: float = 0.999
    epsilon_min: float = 0.05
    state_bin_step: float = 5.0  # EUR/MWh bin width for continuous state signals
    default_action_id: int = 288  # P=1000|M1=1000|M2=0|M3=0: sensible hedging default

    @classmethod
    def from_settings(cls, settings: RLSettings) -> "QLearningAgentConfig":
        return cls(
            action_space=_FULL_ACTION_SPACE,
            random_seed=42,
            epsilon=settings.epsilon,
            learning_rate=settings.learning_rate,
            discount_factor=settings.discount_factor,
            epsilon_decay=settings.epsilon_decay,
            epsilon_min=settings.epsilon_min,
        )


def get_default_rl_settings() -> RLSettings:
    return get_default_settings().rl


def _validate_encoded_action(action: int, action_space: tuple[int, ...]) -> None:
    if action not in action_space:
        raise RLAgentError(
            f"Unsupported encoded action {action!r}. Must be in [0, {MDP_N_ACTIONS})."
        )


class BaseRLAgent:
    """Base interface for RL-style agents."""

    def __init__(self, config: RLAgentConfig | None = None):
        self.config = config or RLAgentConfig.from_settings(get_default_rl_settings())
        self._rng = random.Random(self.config.random_seed)

    def select_action(self, state: dict[str, Any]) -> int:
        raise NotImplementedError("Child classes must implement select_action(...).")

    def observe(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
    ) -> None:
        return None

    def train(
        self,
        env: EnergyRLEnvironment,
        episodes: int = 1,
    ) -> list[dict[str, float]]:
        if episodes <= 0:
            raise RLAgentError("episodes must be strictly positive.")

        history: list[dict[str, float]] = []

        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            n_steps = 0

            while not done:
                action = self.select_action(state)
                _validate_encoded_action(action, self.config.action_space)
                next_state, reward, done, _ = env.step(action)
                self.observe(state, action, reward, next_state, done)

                total_reward += float(reward)
                n_steps += 1
                state = next_state

            history.append(
                {
                    "episode": float(episode),
                    "total_reward": float(total_reward),
                    "n_steps": float(n_steps),
                }
            )

        return history


class RandomAgent(BaseRLAgent):
    """Baseline agent that samples compound actions uniformly at random."""

    def select_action(self, state: dict[str, Any]) -> int:
        return self._rng.choice(self.config.action_space)


class QLearningAgent(BaseRLAgent):
    """
    Tabular Q-learning agent for the 567-action compound action space.

    State key: four binned signals
      - q90_vs_m1      : forecast_tail - m1_price          (EUR/MWh, 5-EUR bins)
      - spot_m1_spread : spot_price - m1_price             (EUR/MWh, 5-EUR bins)
      - q90_h3_vs_m1   : forecast_tail_h3 - m1_price      (EUR/MWh, 5-EUR bins)
      - inventory_bin  : 0=low / 1=mid / 2=high            (discrete)
    """

    def __init__(self, config: QLearningAgentConfig | None = None):
        self.q_config = (
            QLearningAgentConfig.from_settings(get_default_rl_settings())
            if config is None
            else config
        )
        super().__init__(config=self.q_config)
        self.q_table: dict[tuple[tuple[str, float], ...], dict[int, float]] = {}

    def _state_to_key(self, state: dict[str, Any]) -> tuple[tuple[str, float], ...]:
        """Map raw state dict to a compact, hashable tabular key."""
        if not state:
            return tuple()

        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(state.get(key, default))
            except (TypeError, ValueError) as exc:
                raise RLAgentError(
                    f"State value for '{key}' is not numeric: {state.get(key)}"
                ) from exc

        def _bin(value: float) -> float:
            step = self.q_config.state_bin_step
            return round(value / step) * step

        m1_price = _f("m1_price")
        forecast_tail = _f("forecast_tail")
        forecast_tail_h3 = _f("forecast_tail_h3", forecast_tail)
        spot_price = _f("spot_price", m1_price)

        compact = {
            "q90_vs_m1": _bin(forecast_tail - m1_price),
            "spot_m1_spread": _bin(spot_price - m1_price),
            "q90_h3_vs_m1": _bin(forecast_tail_h3 - m1_price),
            "inventory_bin": round(_f("inventory_bin")),
        }
        return tuple(sorted(compact.items()))

    def _ensure_state(self, key: tuple[tuple[str, float], ...]) -> None:
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in self.config.action_space}
            # Optimistic init: bias toward the sensible default hedge action so
            # unvisited states don't default to the max-key action (566 = over-hedge).
            default_id = self.q_config.default_action_id
            if default_id in self.q_table[key]:
                self.q_table[key][default_id] = 1e-4

    def get_action_values(self, state: dict[str, Any]) -> dict[int, float]:
        key = self._state_to_key(state)
        self._ensure_state(key)
        return dict(self.q_table[key])

    def select_action(self, state: dict[str, Any]) -> int:
        key = self._state_to_key(state)
        self._ensure_state(key)

        if self._rng.random() < self.q_config.epsilon:
            return self._rng.choice(self.config.action_space)

        return int(max(self.q_table[key], key=self.q_table[key].get))

    def observe(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
    ) -> None:
        _validate_encoded_action(action, self.config.action_space)
        key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)

        self._ensure_state(key)
        self._ensure_state(next_key)

        current_q = self.q_table[key][action]
        next_q_max = 0.0 if done else max(self.q_table[next_key].values())

        self.q_table[key][action] = current_q + self.q_config.learning_rate * (
            reward + self.q_config.discount_factor * next_q_max - current_q
        )

        self.q_config.epsilon = max(
            self.q_config.epsilon_min,
            self.q_config.epsilon * self.q_config.epsilon_decay,
        )


def evaluate_agent(
    agent: BaseRLAgent,
    env: EnergyRLEnvironment,
    episodes: int = 1,
) -> list[dict[str, float]]:
    """Convenience wrapper for agent rollout evaluation."""
    return agent.train(env=env, episodes=episodes)


if __name__ == "__main__":
    import pandas as pd

    from src.config.constants import (
        PRIMARY_FUTURE_COLUMN,
        Q50_COLUMN,
        Q50_H3_COLUMN,
        Q90_COLUMN,
        Q90_H3_COLUMN,
        SECONDARY_FUTURE_COLUMN,
    )
    from src.rl.rl_environment import _M3_FUTURE_COLUMN

    example_df = pd.DataFrame(
        {
            Q50_COLUMN: [50.0, 55.0, 60.0, 58.0],
            Q90_COLUMN: [60.0, 68.0, 78.0, 66.0],
            Q50_H3_COLUMN: [52.0, 57.0, 62.0, 59.0],
            Q90_H3_COLUMN: [65.0, 73.0, 83.0, 71.0],
            PRIMARY_FUTURE_COLUMN: [52.0, 57.0, 63.0, 60.0],
            "Spot_Price_SPEL": [51.0, 58.0, 66.0, 61.0],
            SECONDARY_FUTURE_COLUMN: [53.0, 58.0, 64.0, 61.0],
            _M3_FUTURE_COLUMN: [54.0, 59.0, 65.0, 62.0],
        }
    )

    env = EnergyRLEnvironment(example_df)

    print("=== RANDOM AGENT ===")
    random_agent = RandomAgent()
    print(evaluate_agent(random_agent, env, episodes=2))

    print("\n=== Q-LEARNING AGENT ===")
    q_agent = QLearningAgent()
    print(evaluate_agent(q_agent, env, episodes=2))
    print("Q-table states:", len(q_agent.q_table))
