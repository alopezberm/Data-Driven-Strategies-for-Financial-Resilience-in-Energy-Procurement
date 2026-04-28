

"""
utils_rl.py

Utility helpers for reinforcement-learning workflows in the project.

This module keeps RL-specific helpers separate from the generic utilities in
`src/utils/`. The functions here are focused on:
- state preprocessing / discretization
- action label encoding / decoding
- lightweight artifact persistence for trained tabular agents
- simple training diagnostics summaries
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.constants import (
    ACTION_BUY_M1_FUTURE,
    ACTION_BUY_M2_FUTURE,
    ACTION_BUY_M3_FUTURE,
    ACTION_DECREASE_PRODUCTION,
    ACTION_DO_NOTHING,
    ACTION_INCREASE_PRODUCTION,
    ACTION_SHIFT_PRODUCTION,
    MDP_BLOCK_SIZES,
    MDP_N_ACTIONS,
    MDP_N_BLOCK,
    MDP_PROD_LEVELS,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RLUtilsError(Exception):
    """Raised when an RL utility function fails."""


# =========================
# Compound action encoding (MDP joint action space)
# =========================
#
# Each action_id encodes a tuple (production_units, m1_mwh, m2_mwh, m3_mwh):
#   production_units ∈ {0, 100, ..., 2000}   →  21 values (index 0-20)
#   m1_mwh / m2_mwh / m3_mwh ∈ {0, 500}     →  2 values each (50% hedge cap)
#
# Encoding:  action_id = prod_idx * 8 + m1_idx * 4 + m2_idx * 2 + m3_idx
# Total:     21 * 2^3 = 168 actions  (MDP_N_ACTIONS)

_N_HEDGE = MDP_N_BLOCK ** 3   # 8


def encode_compound_action(
    prod_units: int,
    m1_mwh: int,
    m2_mwh: int,
    m3_mwh: int,
) -> int:
    """Encode (production_units, m1_mwh, m2_mwh, m3_mwh) → action_id in [0, 168)."""
    try:
        pi = MDP_PROD_LEVELS.index(prod_units)
        b1 = MDP_BLOCK_SIZES.index(m1_mwh)
        b2 = MDP_BLOCK_SIZES.index(m2_mwh)
        b3 = MDP_BLOCK_SIZES.index(m3_mwh)
    except ValueError as exc:
        raise RLUtilsError(
            f"Invalid compound action components: prod={prod_units}, "
            f"m1={m1_mwh}, m2={m2_mwh}, m3={m3_mwh}. {exc}"
        ) from exc
    return pi * _N_HEDGE + b1 * 4 + b2 * 2 + b3


def decode_compound_action(action_id: int) -> tuple[int, int, int, int]:
    """Decode action_id in [0, 168) → (production_units, m1_mwh, m2_mwh, m3_mwh)."""
    if not (0 <= action_id < MDP_N_ACTIONS):
        raise RLUtilsError(
            f"action_id {action_id} is out of range [0, {MDP_N_ACTIONS})."
        )
    pi = action_id // _N_HEDGE
    rem = action_id % _N_HEDGE
    b1 = rem // 4
    rem = rem % 4
    b2 = rem // 2
    b3 = rem % 2
    return (MDP_PROD_LEVELS[pi], MDP_BLOCK_SIZES[b1], MDP_BLOCK_SIZES[b2], MDP_BLOCK_SIZES[b3])


def compound_action_label(action_id: int) -> str:
    """Return a human-readable label for a compound action id."""
    prod, m1, m2, m3 = decode_compound_action(action_id)
    return f"P={prod}|M1={m1}|M2={m2}|M3={m3}"


# =========================
# Legacy action encoding helpers (used by heuristic pipeline — do not remove)
# =========================

ACTION_LABEL_TO_ID = {
    ACTION_DO_NOTHING: 0,
    ACTION_BUY_M1_FUTURE: 1,
    ACTION_SHIFT_PRODUCTION: 2,
    ACTION_INCREASE_PRODUCTION: 3,
    ACTION_DECREASE_PRODUCTION: 4,
    ACTION_BUY_M2_FUTURE: 5,
    ACTION_BUY_M3_FUTURE: 6,
}

ACTION_ID_TO_LABEL = {value: key for key, value in ACTION_LABEL_TO_ID.items()}



def encode_action_label(action_label: str) -> int:
    """Convert a project action label into the encoded integer used by RL agents."""
    if action_label not in ACTION_LABEL_TO_ID:
        raise RLUtilsError(
            f"Unknown action label '{action_label}'. Expected one of: {list(ACTION_LABEL_TO_ID.keys())}"
        )
    return ACTION_LABEL_TO_ID[action_label]



def decode_action_id(action_id: int) -> str:
    """Convert an encoded RL action id into the project action label."""
    if action_id not in ACTION_ID_TO_LABEL:
        raise RLUtilsError(
            f"Unknown action id '{action_id}'. Expected one of: {list(ACTION_ID_TO_LABEL.keys())}"
        )
    return ACTION_ID_TO_LABEL[action_id]


# =========================
# State preprocessing
# =========================


def round_state_values(
    state: dict[str, Any],
    digits: int = 2,
) -> dict[str, float | int | str]:
    """
    Round numeric values in a state dictionary while preserving non-numeric values.

    Parameters
    ----------
    state : dict[str, Any]
        State representation.
    digits : int, optional
        Number of decimals used for numeric rounding.

    Returns
    -------
    dict[str, float | int | str]
        Rounded state dictionary.
    """
    if not isinstance(state, dict):
        raise RLUtilsError("State must be provided as a dictionary.")

    rounded_state: dict[str, float | int | str] = {}
    for key, value in state.items():
        if isinstance(value, bool):
            rounded_state[key] = int(value)
        elif isinstance(value, (int, float)):
            rounded_state[key] = round(float(value), digits)
        else:
            rounded_state[key] = value
    return rounded_state



def state_dict_to_key(
    state: dict[str, Any],
    digits: int = 2,
) -> tuple[tuple[str, float | int | str], ...]:
    """
    Convert a state dictionary into a deterministic hashable key for tabular RL.

    Parameters
    ----------
    state : dict[str, Any]
        State representation.
    digits : int, optional
        Number of decimals used for numeric rounding.

    Returns
    -------
    tuple[tuple[str, float | int | str], ...]
        Sorted tuple representation of the state.
    """
    rounded_state = round_state_values(state, digits=digits)
    return tuple(sorted(rounded_state.items(), key=lambda item: item[0]))


# =========================
# Q-table persistence
# =========================


def save_q_table(q_table: dict[Any, Any], file_path: str | Path) -> Path:
    """
    Save a Q-table to disk using pickle.

    Parameters
    ----------
    q_table : dict[Any, Any]
        Tabular RL Q-table.
    file_path : str | Path
        Destination file path.

    Returns
    -------
    Path
        Path to the saved file.
    """
    if not isinstance(q_table, dict):
        raise RLUtilsError("q_table must be a dictionary.")

    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file_obj:
        pickle.dump(q_table, file_obj)

    logger.info(f"Saved Q-table to {output_path}")
    return output_path



def load_q_table(file_path: str | Path) -> dict[Any, Any]:
    """
    Load a Q-table from disk.

    Parameters
    ----------
    file_path : str | Path
        Source file path.

    Returns
    -------
    dict[Any, Any]
        Loaded Q-table.
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise RLUtilsError(f"Q-table file not found: {input_path}")

    with input_path.open("rb") as file_obj:
        q_table = pickle.load(file_obj)

    if not isinstance(q_table, dict):
        raise RLUtilsError("Loaded Q-table is not a dictionary.")

    logger.info(f"Loaded Q-table from {input_path}")
    return q_table


# =========================
# Diagnostics helpers
# =========================


def summarize_episode_rewards(rewards: list[float]) -> pd.DataFrame:
    """
    Build a compact summary dataframe for episode rewards.

    Parameters
    ----------
    rewards : list[float]
        Episode-level total rewards.

    Returns
    -------
    pd.DataFrame
        One-row summary dataframe.
    """
    if not rewards:
        raise RLUtilsError("Rewards list cannot be empty.")

    rewards_series = pd.Series(rewards, dtype=float)
    return pd.DataFrame(
        {
            "n_episodes": [int(rewards_series.shape[0])],
            "reward_mean": [float(rewards_series.mean())],
            "reward_std": [float(rewards_series.std(ddof=0))],
            "reward_min": [float(rewards_series.min())],
            "reward_max": [float(rewards_series.max())],
            "reward_last": [float(rewards_series.iloc[-1])],
            "reward_rolling_mean_last_10": [
                float(rewards_series.tail(min(10, len(rewards_series))).mean())
            ],
        }
    )



def build_episode_rewards_dataframe(rewards: list[float]) -> pd.DataFrame:
    """
    Build an episode-level reward history dataframe.

    Parameters
    ----------
    rewards : list[float]
        Episode-level total rewards.

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per episode.
    """
    if not rewards:
        raise RLUtilsError("Rewards list cannot be empty.")

    rewards_df = pd.DataFrame(
        {
            "episode": list(range(1, len(rewards) + 1)),
            "total_reward": [float(reward) for reward in rewards],
        }
    )
    rewards_df["rolling_mean_10"] = rewards_df["total_reward"].rolling(
        window=10,
        min_periods=1,
    ).mean()
    return rewards_df


# =========================
# Smoke test
# =========================

if __name__ == "__main__":
    example_state = {
        "forecast_central": 52.3456,
        "forecast_tail": 61.9876,
        "current_m1_future": 55.2222,
        "is_weekend": 0,
    }

    rounded_state = round_state_values(example_state, digits=2)
    state_key = state_dict_to_key(example_state, digits=2)
    action_id = encode_action_label(ACTION_BUY_M1_FUTURE)
    action_label = decode_action_id(action_id)
    reward_summary = summarize_episode_rewards([1.0, 2.5, 0.5, 3.0])

    logger.info(f"Rounded state: {rounded_state}")
    logger.info(f"State key: {state_key}")
    logger.info(f"Encoded action id: {action_id}")
    logger.info(f"Decoded action label: {action_label}")
    logger.info(f"Reward summary:\n{reward_summary}")