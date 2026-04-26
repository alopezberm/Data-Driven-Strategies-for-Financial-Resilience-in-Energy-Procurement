"""
rl_agent.py

Lightweight reinforcement-learning agent abstractions for the energy
procurement project. This module does not attempt to implement a full RL stack;
instead, it provides clean agent interfaces and simple baseline agents that can
interact with the custom environment defined in `rl_environment.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random

from src.config.constants import (
    ACTIONS,
    ACTION_BUY_M1_FUTURE,
    ACTION_DO_NOTHING,
    ACTION_SHIFT_PRODUCTION,
    validate_action_catalog,
)
from src.config.settings import RLSettings, get_default_settings
from src.rl.rl_environment import EnergyRLEnvironment


class RLAgentError(Exception):
    """Raised when an RL agent cannot act or train safely."""


@dataclass
class RLAgentConfig:
    """Configuration shared by lightweight RL agents."""

    action_space: tuple[int, ...] = (0, 1, 2)
    random_seed: int = 42
    epsilon: float = 0.1
    learning_rate: float = 0.1
    discount_factor: float = 0.95

    @classmethod
    def from_settings(cls, settings: RLSettings) -> "RLAgentConfig":
        """Build an RL agent config from centralized project settings."""
        return cls(
            action_space=(0, 1, 2),
            random_seed=42,
            epsilon=settings.epsilon,
            learning_rate=settings.learning_rate,
            discount_factor=settings.discount_factor,
        )


@dataclass
class QLearningAgentConfig(RLAgentConfig):
    """Configuration for the lightweight tabular Q-learning skeleton."""

    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    state_rounding_digits: int = 0

    @classmethod
    def from_settings(cls, settings: RLSettings) -> "QLearningAgentConfig":
        """Build a Q-learning config from centralized project settings."""
        return cls(
            action_space=(0, 1, 2),
            random_seed=42,
            epsilon=settings.epsilon,
            learning_rate=settings.learning_rate,
            discount_factor=settings.discount_factor,
            epsilon_decay=settings.epsilon_decay,
            epsilon_min=settings.epsilon_min,
            state_rounding_digits=settings.state_rounding_digits,
        )


def get_default_rl_settings() -> RLSettings:
    """Return default RL settings from the project configuration."""
    return get_default_settings().rl



def _get_agent_config(config: RLAgentConfig | None) -> RLAgentConfig:
    """Resolve an explicit RL agent config or build one from project settings."""
    if config is not None:
        return config
    return RLAgentConfig.from_settings(get_default_rl_settings())



def _validate_encoded_action(action: int, action_space: tuple[int, ...]) -> None:
    """Validate that a chosen encoded action belongs to the configured action space."""
    if action not in action_space:
        raise RLAgentError(
            f"Unsupported encoded action '{action}'. Expected one of: {list(action_space)}"
        )


class BaseRLAgent:
    """
    Base interface for RL-style agents.

    Child classes should at minimum implement `select_action(...)`. The default
    `train(...)` method performs rollouts but does not update any policy.
    """

    def __init__(self, config: RLAgentConfig | None = None):
        validate_action_catalog()
        self.config = _get_agent_config(config)
        self._rng = random.Random(self.config.random_seed)

    def select_action(self, state: dict[str, Any]) -> int:
        """Select one action given the current environment state."""
        raise NotImplementedError("Child classes must implement select_action(...).")

    def observe(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
    ) -> None:
        """
        Optional hook for online learning.

        Stateless agents can ignore this method.
        """
        return None

    def train(
        self,
        env: EnergyRLEnvironment,
        episodes: int = 1,
    ) -> list[dict[str, float]]:
        """
        Run training rollouts against the environment.

        Returns
        -------
        list[dict[str, float]]
            One summary dictionary per episode.
        """
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
    """Simple baseline agent that samples actions uniformly at random."""

    def select_action(self, state: dict[str, Any]) -> int:
        if not self.config.action_space:
            raise RLAgentError("Action space is empty.")
        return self._rng.choice(self.config.action_space)


class HeuristicRLAgent(BaseRLAgent):
    """
    Simple rule-based agent for RL-environment benchmarking.

    Policy:
    - hedge when q90 is materially above futures
    - shift when q90 is above q50 but hedge condition is not strong enough
    - otherwise do nothing
    """

    def __init__(
        self,
        config: RLAgentConfig | None = None,
        hedge_threshold: float | None = None,
        shift_threshold: float | None = None,
    ):
        super().__init__(config=config)
        rl_settings = get_default_rl_settings()
        self.hedge_threshold = (
            rl_settings.heuristic_hedge_threshold if hedge_threshold is None else hedge_threshold
        )
        self.shift_threshold = (
            rl_settings.heuristic_shift_threshold if shift_threshold is None else shift_threshold
        )

    def select_action(self, state: dict[str, Any]) -> int:
        required_keys = {"forecast_central", "forecast_tail", "current_m1_future"}
        missing_keys = required_keys - set(state.keys())
        if missing_keys:
            raise RLAgentError(
                f"State is missing required keys for HeuristicRLAgent: {sorted(missing_keys)}"
            )

        forecast_central = float(state["forecast_central"])
        forecast_tail = float(state["forecast_tail"])
        future_price = float(state["current_m1_future"])

        tail_vs_future = float(
            state.get("tail_vs_future_abs", forecast_tail - future_price)
        )
        tail_vs_central = float(
            state.get("tail_vs_central_abs", forecast_tail - forecast_central)
        )

        if tail_vs_future >= self.hedge_threshold:
            return 1
        if tail_vs_central >= self.shift_threshold:
            return 2
        return 0


class QLearningAgent(BaseRLAgent):
    """
    Minimal tabular Q-learning skeleton.

    This implementation is intentionally lightweight and mainly serves as a
    future-ready extension point for the project. It compresses the raw state
    into a smaller set of decision-relevant signals and discretizes them into
    coarse bins before storing Q-values in a Python dictionary.
    """

    def __init__(self, config: QLearningAgentConfig | None = None):
        self.q_config = QLearningAgentConfig.from_settings(get_default_rl_settings()) if config is None else config
        super().__init__(config=self.q_config)
        self.q_table: dict[tuple[tuple[str, float], ...], dict[int, float]] = {}

    def _state_to_key(self, state: dict[str, Any]) -> tuple[tuple[str, float], ...]:
        """
        Convert a raw environment state into a compact tabular-RL key.

        We intentionally keep only the most decision-relevant signals:
        - tail_vs_future_abs
        - tail_vs_central_abs
        - spot_minus_future
        - is_weekend
        - is_holiday

        Continuous values are discretized into coarse bins so that similar days
        map to the same tabular state more often.
        """
        if not state:
            return tuple()

        def _to_float(key: str, value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise RLAgentError(
                    f"State value for key '{key}' is not numeric: {value}"
                ) from exc

        def _bin_value(value: float, step: float) -> float:
            return round(value / step) * step

        forecast_central = _to_float("forecast_central", state.get("forecast_central", 0.0))
        forecast_tail = _to_float("forecast_tail", state.get("forecast_tail", 0.0))
        current_m1_future = _to_float("current_m1_future", state.get("current_m1_future", 0.0))
        current_spot = _to_float("current_spot", state.get("current_spot", forecast_central))

        tail_vs_future_abs = _to_float(
            "tail_vs_future_abs",
            state.get("tail_vs_future_abs", forecast_tail - current_m1_future),
        )
        tail_vs_central_abs = _to_float(
            "tail_vs_central_abs",
            state.get("tail_vs_central_abs", forecast_tail - forecast_central),
        )
        spot_minus_future = current_spot - current_m1_future

        is_weekend = 1.0 if _to_float("is_weekend", state.get("is_weekend", 0.0)) >= 0.5 else 0.0
        is_holiday = 1.0 if _to_float("is_holiday", state.get("is_holiday", 0.0)) >= 0.5 else 0.0

        compact_state = {
            "tail_vs_future_abs": _bin_value(tail_vs_future_abs, 5.0),
            "tail_vs_central_abs": _bin_value(tail_vs_central_abs, 5.0),
            "spot_minus_future": _bin_value(spot_minus_future, 5.0),
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
        }

        return tuple(sorted(compact_state.items()))

    def _ensure_state(self, state_key: tuple[tuple[str, float], ...]) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.config.action_space}

    def get_action_values(self, state: dict[str, Any]) -> dict[int, float]:
        """Return the current Q-values for a given state."""
        state_key = self._state_to_key(state)
        self._ensure_state(state_key)
        return dict(self.q_table[state_key])

    def select_action(self, state: dict[str, Any]) -> int:
        state_key = self._state_to_key(state)
        self._ensure_state(state_key)

        if self._rng.random() < self.q_config.epsilon:
            action = self._rng.choice(self.config.action_space)
            _validate_encoded_action(action, self.config.action_space)
            return action

        action_values = self.q_table[state_key]
        best_action = int(max(action_values, key=action_values.get))
        _validate_encoded_action(best_action, self.config.action_space)
        return best_action

    def observe(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
    ) -> None:
        _validate_encoded_action(action, self.config.action_space)
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        self._ensure_state(state_key)
        self._ensure_state(next_state_key)

        current_q = self.q_table[state_key][action]
        next_q_max = 0.0 if done else max(self.q_table[next_state_key].values())

        updated_q = current_q + self.q_config.learning_rate * (
            reward + self.q_config.discount_factor * next_q_max - current_q
        )
        self.q_table[state_key][action] = updated_q

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

    from src.config.constants import PRIMARY_FUTURE_COLUMN, Q50_COLUMN, Q90_COLUMN

    example_df = pd.DataFrame(
        {
            Q50_COLUMN: [50, 55, 60, 58],
            Q90_COLUMN: [60, 68, 78, 66],
            PRIMARY_FUTURE_COLUMN: [52, 57, 63, 60],
            "Spot_Price_SPEL": [51, 58, 66, 61],
            "tail_vs_future_abs": [8, 11, 15, 6],
            "tail_vs_central_abs": [10, 13, 18, 8],
            "is_weekend": [0, 0, 1, 0],
            "Is_national_holiday": [0, 0, 0, 0],
        }
    )

    env = EnergyRLEnvironment(example_df)
    assert len(ACTIONS) >= 3

    print("=== RANDOM AGENT ===")
    random_agent = RandomAgent()
    print(evaluate_agent(random_agent, env, episodes=2))

    print("\n=== HEURISTIC RL AGENT ===")
    heuristic_agent = HeuristicRLAgent()
    print(evaluate_agent(heuristic_agent, env, episodes=2))

    print("\n=== Q-LEARNING AGENT (SKELETON) ===")
    q_agent = QLearningAgent()
    print(evaluate_agent(q_agent, env, episodes=2))
    print("Q-table size:", len(q_agent.q_table))