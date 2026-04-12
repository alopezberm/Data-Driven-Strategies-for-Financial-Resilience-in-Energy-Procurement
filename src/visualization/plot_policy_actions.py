

"""
plot_policy_actions.py

Visualization utilities focused on policy decisions and action behavior.
These plots help explain when the heuristic DSS acts, how often each action is
used, and how actions relate to risk signals over time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import FIGURES_DIR


class PolicyActionPlotError(Exception):
    """Raised when policy action plots cannot be generated safely."""


REQUIRED_POLICY_COLUMNS = {
    "date",
    "recommended_action",
}


# =========================
# Validation helpers
# =========================

def _validate_policy_df(policy_df: pd.DataFrame) -> pd.DataFrame:
    """Validate a policy dataframe and standardize its date column."""
    if policy_df.empty:
        raise PolicyActionPlotError("Policy dataframe is empty.")

    missing_columns = REQUIRED_POLICY_COLUMNS - set(policy_df.columns)
    if missing_columns:
        raise PolicyActionPlotError(
            f"Policy dataframe is missing required columns: {sorted(missing_columns)}"
        )

    result_df = policy_df.copy()
    result_df["date"] = pd.to_datetime(result_df["date"], errors="coerce")

    if result_df["date"].isna().any():
        invalid_count = int(result_df["date"].isna().sum())
        raise PolicyActionPlotError(
            f"Found {invalid_count} invalid date values in policy dataframe."
        )

    if result_df["date"].duplicated().any():
        raise PolicyActionPlotError("Policy dataframe contains duplicated dates.")

    return result_df.sort_values("date").reset_index(drop=True)



def _prepare_output_path(filename: str | None) -> Path | None:
    """Prepare an output path inside the configured figures directory."""
    if filename is None:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename


# =========================
# Plot functions
# =========================

def plot_action_frequency_bar_chart(
    policy_df: pd.DataFrame,
    title: str = "Policy Action Frequency",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the number of days assigned to each policy action."""
    df = _validate_policy_df(policy_df)
    counts = df["recommended_action"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title(title)
    plt.xlabel("Recommended Action")
    plt.ylabel("Number of Days")
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_action_share_pie_chart(
    policy_df: pd.DataFrame,
    title: str = "Policy Action Share",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the share of each recommended action as a pie chart."""
    df = _validate_policy_df(policy_df)
    counts = df["recommended_action"].value_counts().sort_index()

    plt.figure(figsize=(7, 7))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_action_timeline(
    policy_df: pd.DataFrame,
    title: str = "Policy Actions Over Time",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the sequence of recommended actions over time as discrete markers."""
    df = _validate_policy_df(policy_df)

    action_to_y = {
        "do_nothing": 0,
        "buy_m1_future": 1,
        "shift_production": 2,
    }

    action_values = df["recommended_action"].map(action_to_y)
    if action_values.isna().any():
        unknown_actions = df.loc[action_values.isna(), "recommended_action"].unique().tolist()
        raise PolicyActionPlotError(
            f"Unknown actions found in policy dataframe: {unknown_actions}"
        )

    plt.figure(figsize=(12, 4))
    plt.scatter(df["date"], action_values)
    plt.yticks(list(action_to_y.values()), list(action_to_y.keys()))
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Recommended Action")
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_actions_vs_tail_risk(
    policy_df: pd.DataFrame,
    tail_risk_column: str = "tail_vs_future_abs",
    title: str = "Policy Actions vs Tail-Risk Signal",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot a tail-risk signal over time together with discrete policy actions.

    This helps explain whether stronger risk signals tend to trigger stronger
    interventions.
    """
    df = _validate_policy_df(policy_df)

    if tail_risk_column not in df.columns:
        raise PolicyActionPlotError(
            f"Tail-risk column '{tail_risk_column}' not found in policy dataframe."
        )

    tail_risk = pd.to_numeric(df[tail_risk_column], errors="coerce")
    if tail_risk.isna().all():
        raise PolicyActionPlotError(
            f"Tail-risk column '{tail_risk_column}' could not be interpreted as numeric."
        )

    action_to_y = {
        "do_nothing": 0,
        "buy_m1_future": 1,
        "shift_production": 2,
    }
    action_values = df["recommended_action"].map(action_to_y)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["date"], tail_risk, label=tail_risk_column)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Tail-risk signal")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.scatter(df["date"], action_values, label="recommended_action")
    ax2.set_ylabel("Action level")
    ax2.set_yticks(list(action_to_y.values()))
    ax2.set_yticklabels(list(action_to_y.keys()))

    fig.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=8, freq="D"),
            "recommended_action": [
                "do_nothing",
                "buy_m1_future",
                "buy_m1_future",
                "shift_production",
                "do_nothing",
                "buy_m1_future",
                "shift_production",
                "do_nothing",
            ],
            "tail_vs_future_abs": [2.0, 10.5, 12.2, 15.8, 3.1, 9.7, 18.0, 1.5],
        }
    )

    plot_action_frequency_bar_chart(example_df, show=False)
    plot_action_share_pie_chart(example_df, show=False)
    plot_action_timeline(example_df, show=False)
    plot_actions_vs_tail_risk(example_df, show=False)