"""
heuristic_policy.py

Transparent rule-based decision policy for the energy procurement DSS.
This module converts forecast and market signals into daily recommended actions.

The policy is intentionally simple and interpretable so it can serve as:
1. A strong baseline before RL
2. A business-readable decision engine
3. A benchmark for backtesting and stakeholder discussion
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SUPPORTED_ACTIONS = {
    "do_nothing",
    "buy_m1_future",
    "shift_production",
}

DEFAULT_Q50_COLUMN = "q_0.5"
DEFAULT_Q90_COLUMN = "q_0.9"
DEFAULT_SPOT_COLUMN = "Spot_Price_SPEL"
DEFAULT_FUTURE_COLUMN = "Future_M1_Price"
DEFAULT_HOLIDAY_COLUMN = "Is_national_holiday"
DEFAULT_WEEKEND_COLUMN = "is_weekend"


class HeuristicPolicyError(Exception):
    """Raised when the heuristic decision policy cannot be applied safely."""


@dataclass
class PolicyConfig:
    """Configuration container for the heuristic policy rules."""

    q50_column: str = DEFAULT_Q50_COLUMN
    q90_column: str = DEFAULT_Q90_COLUMN
    spot_column: str = DEFAULT_SPOT_COLUMN
    future_column: str = DEFAULT_FUTURE_COLUMN
    holiday_column: str = DEFAULT_HOLIDAY_COLUMN
    weekend_column: str = DEFAULT_WEEKEND_COLUMN

    # Risk thresholds
    min_abs_risk_premium_to_hedge: float = 8.0
    min_rel_risk_premium_to_hedge: float = 0.10

    # Production-shift thresholds
    min_abs_risk_premium_to_shift: float = 12.0
    min_rel_risk_premium_to_shift: float = 0.15

    # Operational flexibility settings
    allow_shift_on_weekends: bool = True
    allow_shift_on_holidays: bool = True


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the basic structure of the policy input dataframe."""
    if df.empty:
        raise HeuristicPolicyError("Input dataframe is empty.")

    if "date" not in df.columns:
        raise HeuristicPolicyError("Input dataframe must contain a 'date' column.")

    validated_df = df.copy()
    validated_df["date"] = pd.to_datetime(validated_df["date"], errors="coerce")

    if validated_df["date"].isna().any():
        invalid_count = int(validated_df["date"].isna().sum())
        raise HeuristicPolicyError(
            f"Found {invalid_count} invalid date values in policy input dataframe."
        )

    if validated_df["date"].duplicated().any():
        raise HeuristicPolicyError("Policy input dataframe contains duplicated dates.")

    return validated_df.sort_values("date").reset_index(drop=True)



def _validate_required_columns(df: pd.DataFrame, config: PolicyConfig) -> None:
    """Ensure all columns required by the policy are present."""
    required_columns = [
        config.q50_column,
        config.q90_column,
        config.spot_column,
        config.future_column,
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise HeuristicPolicyError(
            f"Missing required policy input columns: {missing_columns}"
        )


# =========================
# Signal engineering
# =========================

def _build_policy_signals(df: pd.DataFrame, config: PolicyConfig) -> pd.DataFrame:
    """
    Build interpretable policy signals from forecasts and market data.

    Main signal idea:
    - q50 = central forecast for tomorrow's spot price
    - q90 = upper-tail risk forecast for tomorrow's spot price
    - future price = hedge alternative available today
    """
    policy_df = df.copy()

    policy_df["forecast_central"] = policy_df[config.q50_column]
    policy_df["forecast_tail"] = policy_df[config.q90_column]
    policy_df["current_spot"] = policy_df[config.spot_column]
    policy_df["current_m1_future"] = policy_df[config.future_column]

    # Tail-risk premium over the hedge price
    policy_df["tail_vs_future_abs"] = (
        policy_df["forecast_tail"] - policy_df["current_m1_future"]
    )
    policy_df["tail_vs_future_rel"] = (
        policy_df["forecast_tail"] - policy_df["current_m1_future"]
    ) / policy_df["current_m1_future"].replace(0, pd.NA)

    # Upside risk over the central forecast
    policy_df["tail_vs_central_abs"] = (
        policy_df["forecast_tail"] - policy_df["forecast_central"]
    )
    policy_df["tail_vs_central_rel"] = (
        policy_df["forecast_tail"] - policy_df["forecast_central"]
    ) / policy_df["forecast_central"].replace(0, pd.NA)

    # Simple production flexibility markers
    if config.weekend_column in policy_df.columns:
        policy_df["is_flexible_day_weekend"] = (
            pd.to_numeric(policy_df[config.weekend_column], errors="coerce")
            .fillna(0)
            .round()
            .astype("Int64")
        )
    else:
        policy_df["is_flexible_day_weekend"] = 0

    if config.holiday_column in policy_df.columns:
        policy_df["is_flexible_day_holiday"] = (
            pd.to_numeric(policy_df[config.holiday_column], errors="coerce")
            .fillna(0)
            .round()
            .astype("Int64")
        )
    else:
        policy_df["is_flexible_day_holiday"] = 0

    policy_df["is_flexible_day"] = 0
    if config.allow_shift_on_weekends:
        policy_df["is_flexible_day"] = policy_df["is_flexible_day"] + policy_df["is_flexible_day_weekend"].fillna(0)
    if config.allow_shift_on_holidays:
        policy_df["is_flexible_day"] = policy_df["is_flexible_day"] + policy_df["is_flexible_day_holiday"].fillna(0)

    policy_df["is_flexible_day"] = (policy_df["is_flexible_day"] > 0).astype("Int64")

    return policy_df


# =========================
# Row-wise policy logic
# =========================

def _decide_action_for_row(row: pd.Series, config: PolicyConfig) -> tuple[str, str]:
    """Apply transparent business rules to one row of signals."""
    tail_vs_future_abs = row.get("tail_vs_future_abs")
    tail_vs_future_rel = row.get("tail_vs_future_rel")
    tail_vs_central_abs = row.get("tail_vs_central_abs")
    tail_vs_central_rel = row.get("tail_vs_central_rel")
    is_flexible_day = row.get("is_flexible_day", 0)

    # If key risk signals are missing, default to no action.
    if pd.isna(tail_vs_future_abs) or pd.isna(tail_vs_future_rel):
        return "do_nothing", "Missing forecast or futures information."

    # Rule 1: hedge when the upper-tail forecast is materially above the current M1 future price.
    if (
        tail_vs_future_abs >= config.min_abs_risk_premium_to_hedge
        and tail_vs_future_rel >= config.min_rel_risk_premium_to_hedge
    ):
        return (
            "buy_m1_future",
            (
                "Tail-risk forecast is materially above the current M1 futures price, "
                "so locking in costs is preferred."
            ),
        )

    # Rule 2: shift production when risk is very high and the day is operationally flexible.
    if (
        not pd.isna(tail_vs_central_abs)
        and not pd.isna(tail_vs_central_rel)
        and tail_vs_central_abs >= config.min_abs_risk_premium_to_shift
        and tail_vs_central_rel >= config.min_rel_risk_premium_to_shift
        and int(is_flexible_day) == 1
    ):
        return (
            "shift_production",
            (
                "Upper-tail risk is substantially above the central forecast and the day is flexible, "
                "so postponing production is recommended."
            ),
        )

    return "do_nothing", "No rule was triggered: keep the current procurement plan."


# =========================
# Public API
# =========================

def apply_heuristic_policy(
    df: pd.DataFrame,
    config: PolicyConfig | None = None,
) -> pd.DataFrame:
    """
    Apply the heuristic DSS policy to a dataframe of forecasts and market inputs.

    The output includes:
    - engineered policy signals
    - recommended action
    - human-readable explanation
    """
    config = PolicyConfig() if config is None else config

    policy_df = _validate_input_dataframe(df)
    _validate_required_columns(policy_df, config)
    policy_df = _build_policy_signals(policy_df, config)

    decisions = policy_df.apply(
        lambda row: _decide_action_for_row(row, config),
        axis=1,
    )

    policy_df["recommended_action"] = [decision[0] for decision in decisions]
    policy_df["decision_reason"] = [decision[1] for decision in decisions]

    invalid_actions = set(policy_df["recommended_action"].dropna().unique()) - SUPPORTED_ACTIONS
    if invalid_actions:
        raise HeuristicPolicyError(
            f"Policy generated unsupported actions: {sorted(invalid_actions)}"
        )

    return policy_df



def summarize_policy_actions(policy_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact frequency table of recommended actions."""
    if "recommended_action" not in policy_df.columns:
        raise HeuristicPolicyError(
            "Policy dataframe must contain a 'recommended_action' column."
        )

    summary_df = (
        policy_df["recommended_action"]
        .value_counts(dropna=False)
        .rename_axis("recommended_action")
        .reset_index(name="n_days")
    )

    total_days = summary_df["n_days"].sum()
    summary_df["share"] = summary_df["n_days"] / total_days

    return summary_df


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="D"),
            "Spot_Price_SPEL": [70, 75, 80, 78, 82, 76],
            "Future_M1_Price": [72, 73, 74, 75, 76, 77],
            "q_0.5": [74, 76, 79, 80, 81, 78],
            "q_0.9": [85, 88, 92, 86, 95, 84],
            "is_weekend": [0, 0, 0, 1, 1, 0],
            "Is_national_holiday": [0, 0, 0, 0, 0, 1],
        }
    )

    policy_output = apply_heuristic_policy(example_df)
    action_summary = summarize_policy_actions(policy_output)

    print(policy_output[[
        "date",
        "forecast_central",
        "forecast_tail",
        "current_m1_future",
        "tail_vs_future_abs",
        "tail_vs_future_rel",
        "recommended_action",
        "decision_reason",
    ]])
    print(action_summary)
