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

from src.config.constants import (
    ACTIONS,
    ALLOW_SHIFT_ON_HOLIDAYS,
    ALLOW_SHIFT_ON_WEEKENDS,
    DATE_COLUMN,
    MIN_ABS_RISK_PREMIUM_TO_HEDGE,
    MIN_ABS_RISK_PREMIUM_TO_SHIFT,
    MIN_REL_RISK_PREMIUM_TO_HEDGE,
    MIN_REL_RISK_PREMIUM_TO_SHIFT,
    POLICY_REQUIRED_COLUMNS,
    PRIMARY_FUTURE_COLUMN,
    Q50_COLUMN,
    Q90_COLUMN,
    SPOT_PRICE_COLUMN,
)
from src.config.settings import PolicySettings, get_default_settings
from src.decision.action_rules import ActionRuleConfig, apply_action_rules


HOLIDAY_COLUMN = "Is_national_holiday"
WEEKEND_COLUMN = "is_weekend"

SUPPORTED_ACTIONS = set(ACTIONS)


class HeuristicPolicyError(Exception):
    """Raised when the heuristic decision policy cannot be applied safely."""



@dataclass
class PolicyConfig:
    """Configuration container for the heuristic policy."""

    q50_column: str = Q50_COLUMN
    q90_column: str = Q90_COLUMN
    spot_column: str = SPOT_PRICE_COLUMN
    future_column: str = PRIMARY_FUTURE_COLUMN
    holiday_column: str = HOLIDAY_COLUMN
    weekend_column: str = WEEKEND_COLUMN

    # Risk thresholds
    min_abs_risk_premium_to_hedge: float = MIN_ABS_RISK_PREMIUM_TO_HEDGE
    min_rel_risk_premium_to_hedge: float = MIN_REL_RISK_PREMIUM_TO_HEDGE

    # Production-shift thresholds
    min_abs_risk_premium_to_shift: float = MIN_ABS_RISK_PREMIUM_TO_SHIFT
    min_rel_risk_premium_to_shift: float = MIN_REL_RISK_PREMIUM_TO_SHIFT

    # Operational flexibility settings
    allow_shift_on_weekends: bool = ALLOW_SHIFT_ON_WEEKENDS
    allow_shift_on_holidays: bool = ALLOW_SHIFT_ON_HOLIDAYS

    @classmethod
    def from_policy_settings(cls, policy_settings: PolicySettings) -> "PolicyConfig":
        """Build policy configuration from centralized project settings."""
        return cls(
            min_abs_risk_premium_to_hedge=policy_settings.min_abs_risk_premium_to_hedge,
            min_rel_risk_premium_to_hedge=policy_settings.min_rel_risk_premium_to_hedge,
            min_abs_risk_premium_to_shift=policy_settings.min_abs_risk_premium_to_shift,
            min_rel_risk_premium_to_shift=policy_settings.min_rel_risk_premium_to_shift,
            allow_shift_on_weekends=policy_settings.allow_shift_on_weekends,
            allow_shift_on_holidays=policy_settings.allow_shift_on_holidays,
        )
# Helper to build default policy config from settings

def get_default_policy_config() -> PolicyConfig:
    """Build the default heuristic-policy configuration from project settings."""
    settings = get_default_settings()
    return PolicyConfig.from_policy_settings(settings.policy)


def _validate_supported_actions() -> None:
    """Validate that the centralized action catalog matches policy expectations."""
    expected_actions = {"do_nothing", "buy_m1_future", "shift_production"}
    if SUPPORTED_ACTIONS != expected_actions:
        raise HeuristicPolicyError(
            f"SUPPORTED_ACTIONS does not match the expected policy actions: {sorted(expected_actions)}"
        )


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the basic structure of the policy input dataframe."""
    if df.empty:
        raise HeuristicPolicyError("Input dataframe is empty.")

    if DATE_COLUMN not in df.columns:
        raise HeuristicPolicyError(
            f"Input dataframe must contain a '{DATE_COLUMN}' column."
        )

    validated_df = df.copy()
    validated_df[DATE_COLUMN] = pd.to_datetime(validated_df[DATE_COLUMN], errors="coerce")

    if validated_df[DATE_COLUMN].isna().any():
        invalid_count = int(validated_df[DATE_COLUMN].isna().sum())
        raise HeuristicPolicyError(
            f"Found {invalid_count} invalid date values in policy input dataframe."
        )

    if validated_df[DATE_COLUMN].duplicated().any():
        raise HeuristicPolicyError("Policy input dataframe contains duplicated dates.")

    return validated_df.sort_values(DATE_COLUMN).reset_index(drop=True)



def _validate_required_columns(df: pd.DataFrame, config: PolicyConfig) -> None:
    """Ensure all columns required by the policy are present."""
    required_columns = [
        config.q50_column,
        config.q90_column,
        config.spot_column,
        config.future_column,
    ]

    if required_columns == POLICY_REQUIRED_COLUMNS[1:]:
        required_columns = list(POLICY_REQUIRED_COLUMNS[1:])

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

    These engineered signals are later passed to the modular rule layer in
    `action_rules.py`.
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
    config = get_default_policy_config() if config is None else config
    _validate_supported_actions()

    policy_df = _validate_input_dataframe(df)
    _validate_required_columns(policy_df, config)
    policy_df = _build_policy_signals(policy_df, config)

    # Map policy settings into the modular action-rule layer
    action_rule_config = ActionRuleConfig.from_policy_settings(
        PolicySettings(
            min_abs_risk_premium_to_hedge=config.min_abs_risk_premium_to_hedge,
            min_rel_risk_premium_to_hedge=config.min_rel_risk_premium_to_hedge,
            min_abs_risk_premium_to_shift=config.min_abs_risk_premium_to_shift,
            min_rel_risk_premium_to_shift=config.min_rel_risk_premium_to_shift,
            allow_shift_on_weekends=config.allow_shift_on_weekends,
            allow_shift_on_holidays=config.allow_shift_on_holidays,
        )
    )

    # Apply modular action rules
    policy_df = apply_action_rules(policy_df, config=action_rule_config)

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
            DATE_COLUMN: pd.date_range("2025-01-01", periods=6, freq="D"),
            SPOT_PRICE_COLUMN: [70, 75, 80, 78, 82, 76],
            PRIMARY_FUTURE_COLUMN: [72, 73, 74, 75, 76, 77],
            Q50_COLUMN: [74, 76, 79, 80, 81, 78],
            Q90_COLUMN: [85, 88, 92, 86, 95, 84],
            WEEKEND_COLUMN: [0, 0, 0, 1, 1, 0],
            HOLIDAY_COLUMN: [0, 0, 0, 0, 0, 1],
        }
    )

    config = get_default_policy_config()
    print(config)

    policy_output = apply_heuristic_policy(example_df, config=config)
    action_summary = summarize_policy_actions(policy_output)

    print(policy_output[[
        DATE_COLUMN,
        "forecast_central",
        "forecast_tail",
        "current_m1_future",
        "tail_vs_future_abs",
        "tail_vs_future_rel",
        "recommended_action",
        "decision_reason",
    ]])
    print(action_summary)
