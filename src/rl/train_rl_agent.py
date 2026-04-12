

"""
train_rl_agent.py

Training utilities for tabular reinforcement-learning agents used in the
project's decision support system.

This module is intentionally lightweight and focused on the current project
setup:
- tabular Q-learning
- episodic training over the policy/backtesting state space
- optional persistence of the learned Q-table
- reward history tracking for diagnostics and comparison
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import RLSettings, get_default_settings
from src.decision.rl_agent import QLearningAgent, QLearningAgentConfig, RLAgentError
from src.rl.rl_environment import EnergyRLEnvironment, RLEnvironmentConfig
from src.rl.utils_rl import (
    RLUtilsError,
    build_episode_rewards_dataframe,
    save_q_table,
    summarize_episode_rewards,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RLTrainingError(Exception):
    """Raised when RL training cannot be executed safely."""


@dataclass
class RLTrainingConfig:
    """Configuration for training a tabular RL agent."""

    episodes: int = 200
    save_q_table: bool = False
    q_table_output_path: str | Path | None = None
    verbose: bool = True

    @classmethod
    def from_settings(cls, settings: RLSettings) -> "RLTrainingConfig":
        """Build training config from centralized project settings."""
        episodes = getattr(settings, "episodes", 200)
        return cls(
            episodes=int(episodes),
            save_q_table=False,
            q_table_output_path=None,
            verbose=True,
        )


@dataclass
class RLTrainingArtifacts:
    """Container for RL training outputs."""

    agent: QLearningAgent
    episode_rewards: list[float]
    rewards_history_df: pd.DataFrame
    rewards_summary_df: pd.DataFrame
    q_table_path: Path | None = None


# =========================
# Helpers
# =========================


def get_default_rl_training_config() -> RLTrainingConfig:
    """Return default RL training configuration from project settings."""
    return RLTrainingConfig.from_settings(get_default_settings().rl)



def _resolve_training_config(
    config: RLTrainingConfig | None,
) -> RLTrainingConfig:
    """Return explicit config or project defaults."""
    return get_default_rl_training_config() if config is None else config



def _validate_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataframe used to build the RL environment."""
    if not isinstance(df, pd.DataFrame):
        raise RLTrainingError("RL training input must be a pandas DataFrame.")
    if df.empty:
        raise RLTrainingError("RL training dataframe is empty.")
    return df.reset_index(drop=True)



def _validate_training_config(config: RLTrainingConfig) -> None:
    """Validate high-level RL training configuration."""
    if config.episodes <= 0:
        raise RLTrainingError("episodes must be a strictly positive integer.")

    if config.save_q_table and config.q_table_output_path is None:
        raise RLTrainingError(
            "q_table_output_path must be provided when save_q_table=True."
        )


# =========================
# Core training API
# =========================


def train_q_learning_agent(
    df: pd.DataFrame,
    agent_config: QLearningAgentConfig | None = None,
    env_config: RLEnvironmentConfig | None = None,
    training_config: RLTrainingConfig | None = None,
) -> RLTrainingArtifacts:
    """
    Train a tabular Q-learning agent on a chronological policy state dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing RL environment inputs.
    agent_config : QLearningAgentConfig | None, optional
        Explicit Q-learning agent configuration.
    env_config : RLEnvironmentConfig | None, optional
        Explicit RL environment configuration.
    training_config : RLTrainingConfig | None, optional
        Explicit RL training configuration.

    Returns
    -------
    RLTrainingArtifacts
        Trained agent together with reward history and optional saved Q-table path.
    """
    training_df = _validate_training_dataframe(df)
    resolved_training_config = _resolve_training_config(training_config)
    _validate_training_config(resolved_training_config)

    q_agent_config = (
        QLearningAgentConfig.from_settings(get_default_settings().rl)
        if agent_config is None
        else agent_config
    )
    resolved_env_config = (
        RLEnvironmentConfig.from_settings(get_default_settings().rl)
        if env_config is None
        else env_config
    )

    logger.info("Starting RL training...")
    logger.info(f"Training dataframe shape: {training_df.shape}")
    logger.info(f"Episodes: {resolved_training_config.episodes}")

    environment = EnergyRLEnvironment(training_df, config=resolved_env_config)
    agent = QLearningAgent(config=q_agent_config)

    try:
        episode_rewards = agent.train(
            env=environment,
            episodes=resolved_training_config.episodes,
        )
        episode_rewards = [
            float(item["total_reward"]) if isinstance(item, dict) else float(item)
            for item in episode_rewards
        ]

    except RLAgentError as exc:
        raise RLTrainingError(f"RL agent training failed: {exc}") from exc
    except Exception as exc:
        raise RLTrainingError(f"Unexpected RL training failure: {exc}") from exc

    if not episode_rewards:
        raise RLTrainingError("RL training produced an empty reward history.")

    try:
        rewards_history_df = build_episode_rewards_dataframe(episode_rewards)
        rewards_summary_df = summarize_episode_rewards(episode_rewards)
    except RLUtilsError as exc:
        raise RLTrainingError(f"Failed to build RL diagnostics: {exc}") from exc

    q_table_path: Path | None = None
    if resolved_training_config.save_q_table:
        try:
            q_table_path = save_q_table(
                agent.q_table,
                resolved_training_config.q_table_output_path,
            )
        except RLUtilsError as exc:
            raise RLTrainingError(f"Failed to save Q-table: {exc}") from exc

    if resolved_training_config.verbose:
        logger.info("RL training completed successfully.")
        logger.info(f"Q-table states learned: {len(agent.q_table)}")
        logger.info(f"Last episode reward: {episode_rewards[-1]:.4f}")
        logger.info(f"Reward summary:\n{rewards_summary_df}")

    return RLTrainingArtifacts(
        agent=agent,
        episode_rewards=episode_rewards,
        rewards_history_df=rewards_history_df,
        rewards_summary_df=rewards_summary_df,
        q_table_path=q_table_path,
    )


# =========================
# Smoke test
# =========================

if __name__ == "__main__":
    from src.config.constants import PRIMARY_FUTURE_COLUMN, Q50_COLUMN, Q90_COLUMN

    example_df = pd.DataFrame(
        {
            Q50_COLUMN: [50, 55, 60, 58, 62, 64],
            Q90_COLUMN: [60, 68, 78, 66, 80, 82],
            PRIMARY_FUTURE_COLUMN: [52, 57, 63, 60, 65, 67],
            "Spot_Price_SPEL": [51, 58, 66, 61, 69, 70],
            "tail_vs_future_abs": [8, 11, 15, 6, 15, 15],
            "tail_vs_central_abs": [10, 13, 18, 8, 18, 18],
            "is_weekend": [0, 0, 1, 0, 0, 1],
            "Is_national_holiday": [0, 0, 0, 0, 0, 0],
        }
    )

    artifacts = train_q_learning_agent(example_df)
    logger.info(f"Trained Q-table size: {len(artifacts.agent.q_table)}")
    logger.info(f"Rewards history head:\n{artifacts.rewards_history_df.head()}")