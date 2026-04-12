

"""
rl_agent.py

Lightweight reinforcement-learning agent abstractions for the energy
procurement project. This module does not attempt to implement a full RL stack;
instead, it provides clean agent interfaces and simple baseline agents that can
interact with the custom environment defined in `rl_environment.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random

from src.decision.rl_environment import EnergyRLEnvironment


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


class BaseRLAgent:
    """
    Base interface for RL-style agents.

    Child classes should at minimum implement `select_action(...)`. The default
    `train(...)` method performs rollouts but does not update any policy.
    """

    def __init__(self, config: RLAgentConfig | None = None):
        self.config = RLAgentConfig() if config is None else config
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
        n_episodes: int = 1,
    ) -> list[dict[str, float]]:
        """
        Run training rollouts against the environment.

        Returns
        -------
        list[dict[str, float]]
            One summary dictionary per episode.
        """
        if n_episodes <= 0:
            raise RLAgentError("n_episodes must be strictly positive.")

        history: list[dict[str, float]] = []

        for episode in range(n_episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            n_steps = 0

            while not done:
                action = self.select_action(state)
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
        hedge_threshold: float = 5.0,
        shift_threshold: float = 2.0,
    ):
        super().__init__(config=config)
        self.hedge_threshold = hedge_threshold
        self.shift_threshold = shift_threshold

    def select_action(self, state: dict[str, Any]) -> int:
        required_keys = {"q50", "q90", "future_price"}
        missing_keys = required_keys - set(state.keys())
        if missing_keys:
            raise RLAgentError(
                f"State is missing required keys for HeuristicRLAgent: {sorted(missing_keys)}"
            )

        q50 = float(state["q50"])
        q90 = float(state["q90"])
        future_price = float(state["future_price"])

        tail_vs_future = q90 - future_price
        tail_vs_central = q90 - q50

        if tail_vs_future >= self.hedge_threshold:
            return 1  # hedge
        if tail_vs_central >= self.shift_threshold:
            return 2  # shift
        return 0  # do nothing


@dataclass
class QLearningAgentConfig(RLAgentConfig):
    """Configuration for the lightweight tabular Q-learning skeleton."""

    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    state_rounding_digits: int = 1


class QLearningAgent(BaseRLAgent):
    """
    Minimal tabular Q-learning skeleton.

    This implementation is intentionally lightweight and mainly serves as a
    future-ready extension point for the project. It discretizes continuous
    states by rounding values and stores Q-values in a Python dictionary.
    """

    def __init__(self, config: QLearningAgentConfig | None = None):
        self.q_config = QLearningAgentConfig() if config is None else config
        super().__init__(config=self.q_config)
        self.q_table: dict[tuple[tuple[str, float], ...], dict[int, float]] = {}

    def _state_to_key(self, state: dict[str, Any]) -> tuple[tuple[str, float], ...]:
        if not state:
            return tuple()

        key_items: list[tuple[str, float]] = []
        for k, v in sorted(state.items()):
            try:
                value = round(float(v), self.q_config.state_rounding_digits)
            except (TypeError, ValueError) as exc:
                raise RLAgentError(f"State value for key '{k}' is not numeric: {v}") from exc
            key_items.append((k, value))
        return tuple(key_items)

    def _ensure_state(self, state_key: tuple[tuple[str, float], ...]) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.config.action_space}

    def select_action(self, state: dict[str, Any]) -> int:
        state_key = self._state_to_key(state)
        self._ensure_state(state_key)

        if self._rng.random() < self.q_config.epsilon:
            return self._rng.choice(self.config.action_space)

        action_values = self.q_table[state_key]
        best_action = max(action_values, key=action_values.get)
        return int(best_action)

    def observe(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
    ) -> None:
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
    n_episodes: int = 1,
) -> list[dict[str, float]]:
    """Convenience wrapper for agent rollout evaluation."""
    return agent.train(env=env, n_episodes=n_episodes)


if __name__ == "__main__":
    import pandas as pd

    example_df = pd.DataFrame(
        {
            "q_0.5": [50, 55, 60, 58],
            "q_0.9": [60, 68, 78, 66],
            "Future_M1_Price": [52, 57, 63, 60],
        }
    )

    env = EnergyRLEnvironment(example_df)

    print("=== RANDOM AGENT ===")
    random_agent = RandomAgent()
    print(evaluate_agent(random_agent, env, n_episodes=2))

    print("\n=== HEURISTIC RL AGENT ===")
    heuristic_agent = HeuristicRLAgent()
    print(evaluate_agent(heuristic_agent, env, n_episodes=2))

    print("\n=== Q-LEARNING AGENT (SKELETON) ===")
    q_agent = QLearningAgent()
    print(evaluate_agent(q_agent, env, n_episodes=2))
    print("Q-table size:", len(q_agent.q_table))