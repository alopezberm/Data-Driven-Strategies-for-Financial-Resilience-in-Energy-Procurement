"""
plots.py

Unified reporting visualization module for Sections 5, 6, and 7 of the
technical report.  Every function uses a consistent Seaborn whitegrid
aesthetic and is fully self-contained — no inline matplotlib boilerplate
needed in the notebook.

Public API
----------
Section 5 — Predictive Modeling:
  plot_quantile_spread_diagnostics(prediction_frame, ...)

Section 6 — Decision Engine:
  plot_heuristic_action_distribution(policy_df, ...)
  plot_rl_learning_curve(rewards_df, ...)
  plot_rl_action_breakdown(decisions_df, ...)

Section 7 — Backtesting & Results:
  plot_naive_baseline_2025(df, ...)          ← MDP-grounded financial baseline
  plot_savings_bar_chart(all_sims, ...)
  plot_strategy_cost_comparison(all_sims, ...)
  plot_resilience_metrics(all_sims, ...)
  plot_sensitivity_sweep(sweep_df, spot_total, ...)
  plot_production_invariance(prod_df, ...)
"""

from __future__ import annotations

from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from src.config.constants import (
    DATE_COLUMN,
    MDP_D,
    MDP_E_START,
    MDP_E_UNIT,
    SPOT_PRICE_COLUMN,
    STRATEGY_HEURISTIC_POLICY,
    STRATEGY_RL_POLICY,
    STRATEGY_SPOT_ONLY,
    STRATEGY_STATIC_HEDGE,
)

# ---------------------------------------------------------------------------
# Module-level aesthetic configuration
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.05)

_STRATEGY_STYLES: dict[str, tuple[str, str, float]] = {
    # strategy_name: (linestyle, hex_color, linewidth)
    STRATEGY_SPOT_ONLY:        ("--", "#888888", 1.4),
    STRATEGY_STATIC_HEDGE:     (":",  "#4c72b0", 1.6),
    STRATEGY_HEURISTIC_POLICY: ("-",  "#dd8452", 2.2),
    STRATEGY_RL_POLICY:        ("-",  "#55a868", 2.2),
}

_PALETTE_ACTIONS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

_EUR_FMT = mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}")
_DATE_FMT = mdates.DateFormatter("%b %Y")


class PlotsError(Exception):
    """Raised when a reporting plot cannot be generated safely."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_DTU_RED = "#990000"
_DTU_GREY = "#787878"
_DTU_DARK = "#1a1a1a"
_DTU_LIGHT_GREY = "#f5f5f5"

# RC overrides for exec-summary plots (no gridlines, clean white canvas)
_EXEC_RC = {
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
}


def _resolve_save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)


def _resolve_save_hires(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Save at 300 DPI for print-quality executive exports."""
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")


def _strategy_style(name: str) -> tuple[str, str, float]:
    return _STRATEGY_STYLES.get(name, ("-", "#aaaaaa", 1.2))


def _sim_to_series(all_sims: Sequence[pd.DataFrame]) -> dict[str, pd.Series]:
    """Build {strategy_name: cost_series} mapping from simulation DataFrames."""
    result: dict[str, pd.Series] = {}
    for df in all_sims:
        name = str(df["strategy_name"].iloc[0])
        result[name] = pd.to_numeric(df["total_cost"], errors="coerce").reset_index(drop=True)
    return result


def _sim_dates(all_sims: Sequence[pd.DataFrame]) -> pd.Series:
    """Extract the common date series from the first simulation DataFrame."""
    return pd.to_datetime(all_sims[0][DATE_COLUMN], errors="coerce").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 5 — Predictive Modeling
# ---------------------------------------------------------------------------


def plot_quantile_spread_diagnostics(
    prediction_frame: pd.DataFrame,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Two-panel diagnostic for the t+2 quantile suite.

    Left : Distribution of the tail-risk spread (q_0.9 − q_0.5).
    Right: Scatter of q_0.5 vs q_0.9 — shows the model widens intervals
           on uncertain days rather than applying a flat premium.

    Parameters
    ----------
    prediction_frame : pd.DataFrame
        Must contain columns ``q_0.5`` and ``q_0.9``.
    """
    for col in ("q_0.5", "q_0.9"):
        if col not in prediction_frame.columns:
            raise PlotsError(f"prediction_frame is missing required column '{col}'.")

    spread = prediction_frame["q_0.9"] - prediction_frame["q_0.5"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: spread distribution
    axes[0].hist(spread, bins=30, color="darkorange", edgecolor="white", alpha=0.85)
    axes[0].axvline(
        spread.mean(),
        color="black",
        ls="--",
        lw=1.2,
        label=f"Mean = {spread.mean():.1f} EUR/MWh",
    )
    axes[0].set_title("Distribution of Tail-Risk Spread  (q_0.9 − q_0.5)", fontweight="bold")
    axes[0].set_xlabel("EUR/MWh")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Right: q50 vs q90 scatter
    axes[1].scatter(
        prediction_frame["q_0.5"],
        prediction_frame["q_0.9"],
        alpha=0.4,
        s=14,
        color="steelblue",
    )
    lo = min(prediction_frame["q_0.5"].min(), prediction_frame["q_0.9"].min())
    hi = max(prediction_frame["q_0.5"].max(), prediction_frame["q_0.9"].max())
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=0.8, label="q50 = q90  (zero spread)")
    axes[1].set_title("Central vs. Tail Forecast  (q_0.5 vs q_0.9)", fontweight="bold")
    axes[1].set_xlabel("q_0.5 — Central Forecast (EUR/MWh)")
    axes[1].set_ylabel("q_0.9 — Tail Forecast (EUR/MWh)")
    axes[1].legend()

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Section 6 — Decision Engine
# ---------------------------------------------------------------------------


def plot_heuristic_action_distribution(
    policy_df: pd.DataFrame,
    title: str = "Heuristic Policy: Action Distribution",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Bar chart of how many days each heuristic action was triggered.

    Parameters
    ----------
    policy_df : pd.DataFrame
        Must contain a ``recommended_action`` column.
    """
    if "recommended_action" not in policy_df.columns:
        raise PlotsError("policy_df is missing 'recommended_action' column.")

    counts = policy_df["recommended_action"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 3.8))
    colors = _PALETTE_ACTIONS[: len(counts)]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylabel("Number of Days")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_rl_learning_curve(
    rewards_df: pd.DataFrame,
    title: str = "RL Agent Learning Curve: Cumulative Episode Reward",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Line plot of per-episode total reward during Q-learning training,
    with a smoothed trend overlay (window ≈ 5% of total episodes).

    Parameters
    ----------
    rewards_df : pd.DataFrame
        Output of ``train_q_learning_agent().rewards_history_df``.
        Must contain columns ``episode`` and ``total_reward``.
    """
    for col in ("episode", "total_reward"):
        if col not in rewards_df.columns:
            raise PlotsError(f"rewards_df is missing required column '{col}'.")

    window = max(1, len(rewards_df) // 20)
    smoothed = rewards_df["total_reward"].rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        rewards_df["episode"],
        rewards_df["total_reward"],
        alpha=0.2,
        color="steelblue",
        lw=0.8,
        label="Episode reward (raw)",
    )
    ax.plot(
        rewards_df["episode"],
        smoothed,
        color="steelblue",
        lw=2.2,
        label=f"Smoothed (window = {window})",
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Total Reward  (= M·P − h·I − energy costs)")
    ax.legend()

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_rl_action_breakdown(
    decisions_df: pd.DataFrame,
    title: str = "RL Policy: Compound Action Breakdown (2025)",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Three-panel breakdown of the RL compound action decisions.

    Left   : Distribution of chosen production volumes (units/day).
    Centre : Distribution of M1 block sizes (MWh).
    Right  : Distribution of M2 + M3 block sizes (MWh, stacked).

    Reads the NEW decisions_df columns produced by ``apply_rl_policy()``:
    ``production_units``, ``m1_block_mwh``, ``m2_block_mwh``, ``m3_block_mwh``.

    Parameters
    ----------
    decisions_df : pd.DataFrame
        Output of ``apply_rl_policy().decisions_df``.
    """
    required = ["production_units", "m1_block_mwh", "m2_block_mwh", "m3_block_mwh"]
    missing = [c for c in required if c not in decisions_df.columns]
    if missing:
        raise PlotsError(f"decisions_df is missing required columns: {missing}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Production volume histogram
    prod_counts = decisions_df["production_units"].value_counts().sort_index()
    axes[0].bar(
        prod_counts.index.astype(str),
        prod_counts.values,
        color="#5c85d6",
        edgecolor="white",
    )
    axes[0].set_title("Production Volume  (units/day)", fontweight="bold")
    axes[0].set_xlabel("P_{t+1} (units)")
    axes[0].set_ylabel("Days")
    axes[0].tick_params(axis="x", rotation=45)

    # M1 block distribution
    m1_counts = decisions_df["m1_block_mwh"].value_counts().sort_index()
    axes[1].bar(
        m1_counts.index.astype(str),
        m1_counts.values,
        color="#ff9f40",
        edgecolor="white",
        width=0.4,
    )
    axes[1].set_title("M1 Futures Block  (MWh/day)", fontweight="bold")
    axes[1].set_xlabel("M1 Block (MWh)")
    axes[1].set_ylabel("Days")

    # M2 + M3 stacked bar
    m2_counts = decisions_df["m2_block_mwh"].value_counts().sort_index().reindex(
        [0, 500, 1000], fill_value=0
    )
    m3_counts = decisions_df["m3_block_mwh"].value_counts().sort_index().reindex(
        [0, 500, 1000], fill_value=0
    )
    x_labels = [str(v) for v in [0, 500, 1000]]
    axes[2].bar(x_labels, m2_counts.values, color="#55a868", edgecolor="white", label="M2", width=0.4)
    axes[2].bar(
        x_labels,
        m3_counts.values,
        bottom=m2_counts.values,
        color="#a8d5b5",
        edgecolor="white",
        label="M3",
        width=0.4,
    )
    axes[2].set_title("M2 / M3 Futures Blocks  (MWh/day)", fontweight="bold")
    axes[2].set_xlabel("Block Size (MWh)")
    axes[2].set_ylabel("Days")
    axes[2].legend()

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Section 7 — Backtesting & Results
# ---------------------------------------------------------------------------


def plot_naive_baseline_2025(
    df: pd.DataFrame,
    show: bool = True,
    save_path: str | Path | None = None,
) -> dict[str, float]:
    """
    Plot the MDP-grounded naive baseline for 2025: the cumulative energy cost
    of the factory doing nothing — buying all required energy on the daily
    spot market without any hedging or production adjustment.

    Physical consistency enforced via MDP constants:
        E_req = e_start + e_unit × D  =  20 + 1 × 1,000  =  1,020 MWh/day

    This is the absolute financial benchmark every strategy must beat.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and ``Spot_Price_SPEL`` columns.
        Will be filtered to 2025-01-01 → 2025-12-31 (exactly 365 days).

    Returns
    -------
    dict[str, float]
        ``{"total_cost": ..., "avg_daily_cost": ..., "e_req_mwh": ...}``
    """
    if DATE_COLUMN not in df.columns or SPOT_PRICE_COLUMN not in df.columns:
        raise PlotsError(
            f"df must contain '{DATE_COLUMN}' and '{SPOT_PRICE_COLUMN}' columns."
        )

    work = df.copy()
    work[DATE_COLUMN] = pd.to_datetime(work[DATE_COLUMN], errors="coerce")
    work[SPOT_PRICE_COLUMN] = pd.to_numeric(work[SPOT_PRICE_COLUMN], errors="coerce")

    # Strict 2025 filter
    mask = (work[DATE_COLUMN] >= "2025-01-01") & (work[DATE_COLUMN] <= "2025-12-31")
    year_df = work.loc[mask].sort_values(DATE_COLUMN).reset_index(drop=True)

    if year_df.empty:
        raise PlotsError("No 2025 data found in df after filtering.")
    if len(year_df) != 365:
        raise PlotsError(
            f"Expected 365 days for 2025, got {len(year_df)}. "
            "Ensure the input covers the full calendar year."
        )

    # MDP-consistent energy requirement per day
    e_req: float = MDP_E_START + MDP_E_UNIT * MDP_D  # = 20 + 1 * 1000 = 1020 MWh/day

    daily_cost = year_df[SPOT_PRICE_COLUMN] * e_req
    cumulative = daily_cost.cumsum()
    total_cost = float(daily_cost.sum())
    avg_daily = float(daily_cost.mean())

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # Top: daily cost
    axes[0].fill_between(
        year_df[DATE_COLUMN], daily_cost, alpha=0.25, color="crimson"
    )
    axes[0].plot(year_df[DATE_COLUMN], daily_cost, lw=1.1, color="crimson")
    axes[0].axhline(
        avg_daily,
        color="black",
        ls="--",
        lw=1.2,
        label=f"Average Daily Cost: €{avg_daily:,.0f}",
    )
    axes[0].set_title(
        f"Daily Energy Cost — Naive Spot Baseline 2025  "
        f"(E_req = {e_req:,.0f} MWh/day = e_start + e_unit×D)",
        fontweight="bold",
    )
    axes[0].set_ylabel("Daily Cost (€)")
    axes[0].yaxis.set_major_formatter(_EUR_FMT)
    axes[0].legend(loc="upper right", fontsize=9)

    # Bottom: cumulative cost
    axes[1].plot(
        year_df[DATE_COLUMN],
        cumulative,
        lw=2.4,
        color="crimson",
        label=f"Cumulative Spot Cost (Total: €{total_cost:,.0f})",
    )
    axes[1].plot(
        year_df[DATE_COLUMN],
        (year_df.index + 1) * avg_daily,
        "k--",
        lw=1.2,
        label="Average Cost Accumulation",
    )
    axes[1].fill_between(
        year_df[DATE_COLUMN], cumulative, alpha=0.08, color="crimson"
    )
    axes[1].set_title(
        "Cumulative Spot Cost — The Benchmark Our DSS Must Beat",
        fontweight="bold",
    )
    axes[1].set_ylabel("Cumulative Cost (€)")
    axes[1].yaxis.set_major_formatter(_EUR_FMT)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].xaxis.set_major_formatter(_DATE_FMT)

    fig.supxlabel("Date (2025)", fontsize=10)
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"total_cost": total_cost, "avg_daily_cost": avg_daily, "e_req_mwh": e_req}


def plot_savings_bar_chart1(
    all_sims: Sequence[pd.DataFrame],
    reference_strategy: str = STRATEGY_SPOT_ONLY,
    title: str = "Absolute Savings vs. Spot-Only Baseline",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Bar chart of total annual savings (or losses) for each non-reference
    strategy relative to the spot-only baseline.

    Green bars = cheaper than spot; red bars = more expensive.

    Parameters
    ----------
    all_sims : sequence of pd.DataFrame
        Each DataFrame must contain ``strategy_name`` and ``total_cost``.
    reference_strategy : str
        Strategy used as the cost baseline (default: ``spot_only``).
    """
    series = _sim_to_series(all_sims)
    if reference_strategy not in series:
        raise PlotsError(f"Reference strategy '{reference_strategy}' not found in all_sims.")

    ref_total = float(series[reference_strategy].sum())
    savings = {
        name: ref_total - float(s.sum())
        for name, s in series.items()
        if name != reference_strategy
    }

    if not savings:
        raise PlotsError("No non-reference strategies found to plot.")

    names = list(savings.keys())
    values = list(savings.values())
    colors = ["#5cb85c" if v >= 0 else "#d9534f" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.bar_label(bars, fmt="€%.0f", padding=4, fontsize=9)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylabel("Annual Savings (€)")
    ax.yaxis.set_major_formatter(_EUR_FMT)
    ax.tick_params(axis="x", rotation=12)

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_savings_bar_chart2(
    all_sims: Sequence[pd.DataFrame],
    reference_strategy: str = STRATEGY_SPOT_ONLY,
    title: str = "Total Annual Cost & Savings vs. Spot-Only Baseline",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Bar chart of total annual cost for all strategies, including the baseline.
    Savings relative to the baseline are annotated inside the bars.

    Parameters
    ----------
    all_sims : sequence of pd.DataFrame
        Each DataFrame must contain ``strategy_name`` and ``total_cost``.
    reference_strategy : str
        Strategy used as the cost baseline (default: ``spot_only``).
    """
    series = _sim_to_series(all_sims)
    if reference_strategy not in series:
        raise PlotsError(f"Reference strategy '{reference_strategy}' not found in all_sims.")

    # Calculate total costs for ALL strategies
    costs = {name: float(s.sum()) for name, s in series.items()}
    ref_total = costs[reference_strategy]

    # Sort by total cost (ascending: cheapest first)
    sorted_costs = dict(sorted(costs.items(), key=lambda item: item[1]))

    names = list(sorted_costs.keys())
    values = list(sorted_costs.values())

    fig, ax = plt.subplots(figsize=(9, 5))

    # Match the aesthetic of plot_resilience_metrics
    palette = sns.color_palette("Blues_d", n_colors=len(names))
    bars = ax.bar(names, values, color=palette, edgecolor="white", width=0.6)
    
    # Draw a baseline reference line to make savings visually obvious
    ax.axhline(ref_total, color="black", linestyle="--", lw=1.2, label=f"Baseline ({reference_strategy})")

    # Annotate bars with Total Cost (top) and Savings (inside)
    for bar, name, val in zip(bars, names, values):
        savings = ref_total - val
        
        # Label on top of the bar: Total Cost
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() + (max(values) * 0.02), # slightly above
            f"€{val:,.0f}", 
            ha='center', va='bottom', fontsize=10, fontweight='bold' if name == reference_strategy else 'normal'
        )
            
        # Label inside the bar: Savings
        if name != reference_strategy:
            if savings > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() / 2, 
                    f"Saved:\n€{savings:,.0f}", 
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold'
                )
            elif savings < 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() / 2, 
                    f"Loss:\n€{abs(savings):,.0f}", 
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold'
                )

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylabel("Total Annual Cost (€)")
    
    # Expand Y-axis slightly to fit the top labels
    ax.set_ylim(0, max(values) * 1.15)
    
    ax.yaxis.set_major_formatter(_EUR_FMT)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper left")

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_savings_bar_chart3(
    all_sims: Sequence[pd.DataFrame],
    reference_strategy: str = "spot_only",
    title: str = "Net Margin Recovery vs. Spot Market",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Executive-style horizontal bar chart of total annual savings (or losses).
    High resolution and anti-overlap margins included.
    """
    series = _sim_to_series(all_sims)
    if reference_strategy not in series:
        raise PlotsError(f"Reference strategy '{reference_strategy}' not found in all_sims.")

    ref_total = float(series[reference_strategy].sum())
    savings = {
        name: ref_total - float(s.sum())
        for name, s in series.items()
        if name != reference_strategy
    }

    if not savings:
        raise PlotsError("No non-reference strategies found to plot.")

    # Sort by savings descending (best first)
    savings = dict(sorted(savings.items(), key=lambda item: item[1], reverse=True))

    # Executive labels mapping
    label_map = {
        "heuristic_policy": "TailRisk DSS (Recommended)",
        "heuristic_DSS": "TailRisk DSS (Recommended)",
        "static_hedge": "Static Hedge (70% M1)",
        "rl_policy": "RL Agent (Shadow Mode)"
    }

    names = [label_map.get(n, n) for n in savings.keys()]
    values = list(savings.values())

    # Sleek consulting colors: Deep teal for savings, muted coral for losses
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in values]

    # FIX 1: HIGH RESOLUTION (dpi=300) AND SLIGHTLY WIDER FIGURE
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.barh(names, values, color=colors, height=0.55, edgecolor="white")

    # Remove all spines (borders) to clean up the chart
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw a clean vertical baseline at 0
    ax.axvline(0, color='#333333', linewidth=1.5)

    # Remove x-axis completely (we will label the bars directly)
    ax.set_xticks([])
    
    # Clean y-axis styling
    ax.tick_params(axis='y', length=0, labelsize=11, labelcolor='#333333')

    # FIX 2: PREVENT OVERLAPPING BY EXPANDING X-AXIS LIMITS
    max_val = max(abs(v) for v in values)
    # Give 35% empty space on the left for the negative text, and 20% on the right
    ax.set_xlim(-max_val * 0.35, max_val * 1.2)

    # Add text labels directly outside the bars
    for bar, val, color in zip(bars, values, colors):
        # Format numbers elegantly (M for Millions, K for Thousands)
        if abs(val) >= 1_000_000:
            text = f"{'+' if val>0 else '-'}€{abs(val)/1_000_000:.2f}M"
        else:
            text = f"{'+' if val>0 else '-'}€{abs(val)/1_000:.0f}K"

        # Position text just outside the end of the bar
        if val >= 0:
            ax.text(val + (max_val * 0.02), bar.get_y() + bar.get_height()/2, text,
                    va='center', ha='left', color=color, fontweight='bold', fontsize=12)
        else:
            ax.text(val - (max_val * 0.02), bar.get_y() + bar.get_height()/2, text,
                    va='center', ha='right', color=color, fontweight='bold', fontsize=12)

    # Add Title and Subtitle with the baseline info
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98, x=0.52)
    ax.set_title(f"Annual financial impact relative to the €{ref_total/1_000_000:.1f}M Spot-Only baseline", 
                 fontsize=11, color='#666666', pad=15)

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_strategy_cost_comparison(
    all_sims: Sequence[pd.DataFrame],
    title_prefix: str = "2025 Out-of-Sample",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Two-panel comparison: daily procurement cost (top) and cumulative cost
    (bottom) for all four strategies across the evaluation period.

    Parameters
    ----------
    all_sims : sequence of pd.DataFrame
        Each DataFrame must contain ``date``, ``strategy_name``, ``total_cost``.
    """
    if not all_sims:
        raise PlotsError("all_sims is empty.")

    dates = _sim_dates(all_sims)
    series = _sim_to_series(all_sims)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for name, costs in series.items():
        ls, color, lw = _strategy_style(name)
        axes[0].plot(dates, costs, ls=ls, color=color, lw=lw, label=name)
        axes[1].plot(dates, costs.cumsum(), ls=ls, color=color, lw=lw, label=name)

    axes[0].set_title(
        f"Daily Procurement Cost — All Strategies ({title_prefix}, 365 Days)",
        fontweight="bold",
    )
    axes[0].set_ylabel("Daily Cost (€)")
    axes[0].yaxis.set_major_formatter(_EUR_FMT)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].set_title(
        f"Cumulative Procurement Cost ({title_prefix}) — Annual Savings Compound Over Time",
        fontweight="bold",
    )
    axes[1].set_ylabel("Cumulative Cost (€)")
    axes[1].yaxis.set_major_formatter(_EUR_FMT)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].xaxis.set_major_formatter(_DATE_FMT)

    fig.supxlabel("Date", fontsize=10)
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_resilience_metrics2(
    all_sims: Sequence[pd.DataFrame],
    reference_strategy: str = STRATEGY_SPOT_ONLY,
    title: str = "Resilience Metrics — 2025 Out-of-Sample",
    show: bool = True,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Two-panel resilience comparison:
      Left  : Daily cost volatility (std dev) per strategy.
      Right : 90th-percentile daily cost (tail risk exposure) per strategy.
      
    High resolution (dpi=300) and executive naming included.

    Parameters
    ----------
    all_sims : sequence of pd.DataFrame
        Each DataFrame must contain ``strategy_name`` and ``total_cost``.
    reference_strategy : str
        Used to compute the "extreme day" threshold (P90 of reference costs).

    Returns
    -------
    pd.DataFrame
        Resilience metrics table (one row per strategy).
    """
    series = _sim_to_series(all_sims)
    if reference_strategy not in series:
        raise PlotsError(f"Reference strategy '{reference_strategy}' not found.")

    spot_p90 = float(series[reference_strategy].quantile(0.90))

    rows = []
    for name, costs in series.items():
        rows.append(
            {
                "strategy": name,
                "volatility": round(float(costs.std()), 2),
                "p90_daily_cost": round(float(costs.quantile(0.90)), 2),
                "p95_daily_cost": round(float(costs.quantile(0.95)), 2),
                "max_daily_cost": round(float(costs.max()), 2),
                "n_extreme_days": int((costs > spot_p90).sum()),
            }
        )

    risk_df = pd.DataFrame(rows).sort_values("volatility").reset_index(drop=True)

    # Executive labels mapping to maintain consistency across all charts
    label_map = {
        "heuristic_policy": "TailRisk DSS (Recommended)",
        "heuristic_DSS": "TailRisk DSS (Recommended)",
        "static_hedge": "Static Hedge (70% M1)",
        "rl_policy": "RL Agent (Shadow Mode)",
        "spot_only": "Spot-Only (Baseline)"
    }
    
    # Apply the clean names to the dataframe before plotting
    risk_df["strategy"] = risk_df["strategy"].apply(lambda x: label_map.get(x, x))

    # High resolution figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=300)

    palette = sns.color_palette("Blues_d", n_colors=len(risk_df))
    axes[0].bar(
        risk_df["strategy"],
        risk_df["volatility"],
        color=palette,
        edgecolor="white",
    )
    axes[0].set_title("Daily Cost Volatility  (Std Dev)", fontweight="bold")
    axes[0].set_ylabel("Std Dev of Daily Cost (€)")
    axes[0].yaxis.set_major_formatter(_EUR_FMT)
    # Rotate slightly more to accommodate longer executive names
    axes[0].tick_params(axis="x", rotation=25)

    palette2 = sns.color_palette("Oranges_d", n_colors=len(risk_df))
    axes[1].bar(
        risk_df["strategy"],
        risk_df["p90_daily_cost"],
        color=palette2,
        edgecolor="white",
    )
    axes[1].set_title("P90 Daily Cost  (Tail Risk Exposure)", fontweight="bold")
    axes[1].set_ylabel("P90 Daily Cost (€)")
    axes[1].yaxis.set_major_formatter(_EUR_FMT)
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return risk_df

def plot_resilience_metrics1(
    all_sims: Sequence[pd.DataFrame],
    reference_strategy: str = STRATEGY_SPOT_ONLY,
    title: str = "Resilience Metrics — 2025 Out-of-Sample",
    show: bool = True,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Two-panel resilience comparison:
      Left  : Daily cost volatility (std dev) per strategy.
      Right : 90th-percentile daily cost (tail risk exposure) per strategy.

    Parameters
    ----------
    all_sims : sequence of pd.DataFrame
        Each DataFrame must contain ``strategy_name`` and ``total_cost``.
    reference_strategy : str
        Used to compute the "extreme day" threshold (P90 of reference costs).

    Returns
    -------
    pd.DataFrame
        Resilience metrics table (one row per strategy).
    """
    series = _sim_to_series(all_sims)
    if reference_strategy not in series:
        raise PlotsError(f"Reference strategy '{reference_strategy}' not found.")

    spot_p90 = float(series[reference_strategy].quantile(0.90))

    rows = []
    for name, costs in series.items():
        rows.append(
            {
                "strategy": name,
                "volatility": round(float(costs.std()), 2),
                "p90_daily_cost": round(float(costs.quantile(0.90)), 2),
                "p95_daily_cost": round(float(costs.quantile(0.95)), 2),
                "max_daily_cost": round(float(costs.max()), 2),
                "n_extreme_days": int((costs > spot_p90).sum()),
            }
        )

    risk_df = pd.DataFrame(rows).sort_values("volatility").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=300)

    palette = sns.color_palette("Blues_d", n_colors=len(risk_df))
    axes[0].bar(
        risk_df["strategy"],
        risk_df["volatility"],
        color=palette,
        edgecolor="white",
    )
    axes[0].set_title("Daily Cost Volatility  (Std Dev)", fontweight="bold")
    axes[0].set_ylabel("Std Dev of Daily Cost (€)")
    axes[0].yaxis.set_major_formatter(_EUR_FMT)
    axes[0].tick_params(axis="x", rotation=15)

    palette2 = sns.color_palette("Oranges_d", n_colors=len(risk_df))
    axes[1].bar(
        risk_df["strategy"],
        risk_df["p90_daily_cost"],
        color=palette2,
        edgecolor="white",
    )
    axes[1].set_title("P90 Daily Cost  (Tail Risk Exposure)", fontweight="bold")
    axes[1].set_ylabel("P90 Daily Cost (€)")
    axes[1].yaxis.set_major_formatter(_EUR_FMT)
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return risk_df


def plot_sensitivity_sweep(
    sweep_df: pd.DataFrame,
    spot_total: float,
    default_threshold: float = 8.0,
    title: str = "Hedge Threshold Sensitivity — 2025 Out-of-Sample",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Two-panel sensitivity analysis for the hedge-threshold sweep.

    Left  : Total cost vs. threshold (with spot-only reference line).
    Right : Savings (%) vs. threshold (with default-threshold marker).

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Must contain columns ``threshold``, ``total_cost``, ``savings_pct``.
    spot_total : float
        Total cost of the spot-only baseline (for the reference line).
    default_threshold : float
        Current default threshold — marked with a vertical line on the right panel.
    """
    required = ["threshold", "total_cost", "savings_pct"]
    missing = [c for c in required if c not in sweep_df.columns]
    if missing:
        raise PlotsError(f"sweep_df is missing required columns: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        sweep_df["threshold"],
        sweep_df["total_cost"],
        marker="o",
        color="steelblue",
        lw=2.0,
    )
    axes[0].axhline(
        spot_total,
        color="#888888",
        ls="--",
        lw=1.2,
        label=f"spot_only  (€{spot_total:,.0f})",
    )
    axes[0].set_title("Total Cost vs. Hedge Threshold", fontweight="bold")
    axes[0].set_xlabel("Threshold (EUR/MWh)")
    axes[0].set_ylabel("Total Annual Cost (€)")
    axes[0].yaxis.set_major_formatter(_EUR_FMT)
    axes[0].legend(fontsize=9)

    axes[1].plot(
        sweep_df["threshold"],
        sweep_df["savings_pct"],
        marker="s",
        color="darkorange",
        lw=2.0,
    )
    axes[1].axvline(
        default_threshold,
        color="black",
        ls=":",
        lw=1.2,
        label=f"Default ({default_threshold:.0f} EUR/MWh)",
    )
    axes[1].set_title("Savings (%) vs. Hedge Threshold", fontweight="bold")
    axes[1].set_xlabel("Threshold (EUR/MWh)")
    axes[1].set_ylabel("Savings vs. Spot-Only (%)")
    axes[1].legend(fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_production_invariance(
    prod_df: pd.DataFrame,
    title: str = "Relative Savings Across Production Levels — DSS is Scale-Invariant",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Line plot confirming that relative savings (%) are constant across all
    production levels — the DSS decision logic is scale-invariant.

    Parameters
    ----------
    prod_df : pd.DataFrame
        Must contain ``production_level`` and ``savings_pct`` columns.
    """
    for col in ("production_level", "savings_pct"):
        if col not in prod_df.columns:
            raise PlotsError(f"prod_df is missing required column '{col}'.")

    mean_savings = float(prod_df["savings_pct"].mean())

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(
        prod_df["production_level"],
        prod_df["savings_pct"],
        marker="o",
        color="teal",
        lw=2.0,
        ms=7,
    )
    ax.axhline(
        mean_savings,
        color="#888888",
        ls="--",
        lw=1.0,
        label=f"Mean savings = {mean_savings:.2f}%",
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Production Level (fraction of P_max)")
    ax.set_ylabel("Savings vs. Spot-Only (%)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    _resolve_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Executive Summary — C-Level export functions
# ---------------------------------------------------------------------------


def plot_exec_summary_business_case(
    df_results: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = False,
    per_unit_savings: float | None = None,
) -> None:
    """
    "The Bottom Line" — C-Level procurement cost bar chart.

    Two bars only: current spot-only approach vs. TailRisk DSS (best strategy).
    A large annotation highlights total EUR savings and, optionally, savings per unit.
    No gridlines, no legend, no technical jargon. Designed for executive slide decks.

    Parameters
    ----------
    df_results : pd.DataFrame
        Strategy summary table. Required columns: ``strategy_name``, ``total_cost``.
    save_path : path-like, optional
        Output PNG path. Parent directory is created if absent. Saved at 300 DPI.
    show : bool
        Display interactively (default False — export mode).
    per_unit_savings : float, optional
        EUR saved per unit produced (total savings / (D * n_days)). Displayed as
        a KPI box below the savings annotation when provided.
    """
    for col in ("strategy_name", "total_cost"):
        if col not in df_results.columns:
            raise PlotsError(f"df_results is missing required column '{col}'.")

    cost_map = df_results.set_index("strategy_name")["total_cost"].to_dict()
    if STRATEGY_SPOT_ONLY not in cost_map:
        raise PlotsError(f"df_results must contain a '{STRATEGY_SPOT_ONLY}' row.")

    spot_cost = float(cost_map[STRATEGY_SPOT_ONLY])
    alternatives = {k: float(v) for k, v in cost_map.items() if k != STRATEGY_SPOT_ONLY}
    if not alternatives:
        raise PlotsError("df_results must contain at least one non-spot strategy.")

    best_cost = min(alternatives.values())
    savings = spot_cost - best_cost
    savings_pct = savings / spot_cost * 100.0

    n_days_rows = df_results.loc[df_results["strategy_name"] == STRATEGY_SPOT_ONLY, "n_days"]
    period_label = f"{int(n_days_rows.iloc[0])}-Day Backtest" if (
        "n_days" in df_results.columns and not n_days_rows.empty
    ) else "Backtest Period"

    with plt.rc_context(_EXEC_RC):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor("white")

        bar_labels = ["Current Approach\n(Spot-Only)", "TailRisk DSS\nRecommendation"]
        bar_values = [spot_cost, best_cost]
        bar_colors = [_DTU_GREY, _DTU_RED]

        bars = ax.bar([0, 1], bar_values, color=bar_colors, width=0.5, edgecolor="none", zorder=3)

        # Value labels centered inside each bar
        for bar, val in zip(bars, bar_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val * 0.50,
                f"€{val:,.0f}",
                ha="center", va="center",
                fontsize=22, fontweight="bold", color="white",
                zorder=4,
            )

        # Double-headed savings arrow between bar tops
        gap_x = 0.5
        ax.annotate(
            "",
            xy=(gap_x, best_cost),
            xytext=(gap_x, spot_cost),
            arrowprops=dict(arrowstyle="<->", color=_DTU_RED, lw=2.5, shrinkA=0, shrinkB=0),
            zorder=5,
        )
        savings_label = f"€{savings:,.0f}\nSAVED\n({savings_pct:.1f}%)"
        ax.text(
            gap_x + 0.07,
            (spot_cost + best_cost) / 2.0,
            savings_label,
            ha="left", va="center",
            fontsize=18, fontweight="bold", color=_DTU_RED,
            linespacing=1.3, zorder=6,
        )

        # KPI box: € saved per unit produced
        if per_unit_savings is not None:
            kpi_text = f"€{per_unit_savings:.2f} saved / unit produced"
            ax.text(
                0.5, -0.11,
                kpi_text,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=12, fontweight="bold", color=_DTU_RED,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff0f0", edgecolor=_DTU_RED, linewidth=1.2),
            )

        ax.set_xlim(-0.55, 1.75)
        ax.set_ylim(0, spot_cost * 1.18)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(bar_labels, fontsize=14, fontweight="bold", color=_DTU_DARK)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.spines["bottom"].set_linewidth(0.8)

        ax.set_title(
            f"THE BOTTOM LINE\n2025 Energy Procurement Cost — {period_label}",
            fontsize=15, fontweight="bold", color=_DTU_DARK, pad=18,
        )
        fig.text(
            0.98, 0.01,
            "TailRisk Solutions · Group 17 · DTU 42578",
            ha="right", va="bottom", fontsize=7, color="#aaaaaa",
        )

        fig.tight_layout()
        _resolve_save_hires(fig, save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_exec_summary_resilience_overlay(
    df_simulation: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = False,
    hedge_action_column: str = "action_heuristic_policy",
    hedge_action_value: str | Sequence[str] = "buy_m1_future",
) -> None:
    """
    "The Protection Map" — C-Level resilience time series.

    Shows the 2025 daily spot price with green-shaded zones wherever the
    TailRisk DSS had an active forward contract in place. Tells the story:
    "When the market spiked, we were protected."

    Parameters
    ----------
    df_simulation : pd.DataFrame
        Required columns:
        - ``date`` (datetime or date-parseable str)
        - ``Spot_Price_SPEL`` (float, EUR/MWh)
        - *hedge_action_column* (str column with action labels)
    save_path : path-like, optional
        Output PNG path. Parent directory is created if absent. Saved at 300 DPI.
    show : bool
        Display interactively (default False — export mode).
    hedge_action_column : str
        Column indicating which action the DSS took each day.
    hedge_action_value : str or sequence of str
        Value or values in *hedge_action_column* that indicate a forward
        contract being active. If multiple values are provided (e.g.
        ``["buy_m1_future", "do_nothing"]``), each will be plotted with
        a distinct color (first = green, second = red) and the legend will
        show the variable names.
    """
    required = ["date", SPOT_PRICE_COLUMN, hedge_action_column]
    missing = [c for c in required if c not in df_simulation.columns]
    if missing:
        raise PlotsError(f"df_simulation is missing columns: {missing}")

    df = df_simulation.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df[SPOT_PRICE_COLUMN] = pd.to_numeric(df[SPOT_PRICE_COLUMN], errors="coerce")

    # Normalize hedge_action_value to a list of strings
    if isinstance(hedge_action_value, (str,)):
        actions = [hedge_action_value]
    else:
        actions = list(hedge_action_value)

    dates = df["date"].values
    prices = df[SPOT_PRICE_COLUMN].values

    # Colors: first -> green, second -> red, others use palette
    action_colors = ["#2e7d32", "#f3ecec"] + _PALETTE_ACTIONS

    # Build spans and stats per action
    action_spans: dict[str, list[tuple]] = {}
    action_stats: dict[str, tuple[int, float]] = {}
    for i, action in enumerate(actions):
        mask = (df[hedge_action_column] == action).values
        spans: list[tuple] = []
        for hedged, group in groupby(enumerate(mask), key=itemgetter(1)):
            indices = [g[0] for g in group]
            if hedged:
                spans.append((dates[indices[0]], dates[indices[-1]]))
        action_spans[action] = spans
        n_act = int(mask.sum())
        n_total = len(df)
        pct = n_act / n_total * 100.0 if n_total > 0 else 0.0
        action_stats[action] = (n_act, pct)

    p90_price = float(pd.Series(prices).quantile(0.90))

    with plt.rc_context(_EXEC_RC):
        fig, ax = plt.subplots(figsize=(13, 5.5))
        fig.patch.set_facecolor("white")

        # Draw protection spans for each requested action (behind the line)
        for i, action in enumerate(actions):
            color = action_colors[i]
            for span_start, span_end in action_spans.get(action, []):
                ax.axvspan(span_start, span_end, alpha=0.18, color=color, linewidth=0, zorder=1)

        # P90 danger reference line (use distinct color)
        ax.axhline(
            p90_price, color="#880000", ls="--", lw=1.2, alpha=0.6, zorder=2,
            label=f"P90 spike threshold  (€{p90_price:.0f}/MWh)",
        )

        # Spot price line
        ax.plot(df["date"], prices, color="#1a237e", lw=1.6, zorder=3, alpha=0.9)

        # Legend: one patch per action requested
        patches = []
        for i, action in enumerate(actions):
            n_act, pct = action_stats[action]
            patch = mpatches.Patch(
                color=action_colors[i], alpha=0.4,
                label=f"{action}  ({n_act}/{n_total} days · {pct:.0f}% coverage)",
            )
            patches.append(patch)
        # Add the p90 line handle too
        patches.insert(0, mpatches.Patch(color="#880000", alpha=0.4, label=f"P90 spike threshold  (€{p90_price:.0f}/MWh)"))
        ax.legend(handles=patches, loc="upper right", fontsize=10, frameon=False)

        # "PROTECTED" annotation: use the longest hedged span across all actions
        all_spans = [s for spans in action_spans.values() for s in spans]
        if all_spans:
            longest_idx = max(range(len(all_spans)), key=lambda i: all_spans[i][1] - all_spans[i][0])
            ann_start, ann_end = all_spans[longest_idx]
            ann_date = ann_start + (ann_end - ann_start) / 2
            ann_price = float(pd.Series(prices).quantile(0.60))
            ax.text(
                ann_date, ann_price, "PROTECTED",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="#1b5e20", alpha=0.75,
                zorder=5,
            )

        ax.xaxis.set_major_formatter(_DATE_FMT)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.set_ylabel("Spot Price (€/MWh)", fontsize=11, color=_DTU_DARK)
        ax.tick_params(axis="x", rotation=30, labelsize=10, colors=_DTU_DARK)
        ax.tick_params(axis="y", labelsize=10, colors=_DTU_DARK)
        ax.spines["bottom"].set_color("#cccccc")
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color("#cccccc")

        ax.set_title(
            "THE PROTECTION MAP\n"
            "2025 Daily Spot Price — When the Market Spiked, We Were Protected",
            fontsize=14, fontweight="bold", color=_DTU_DARK, pad=16,
        )
        fig.text(
            0.98, 0.01,
            "TailRisk Solutions · Group 17 · DTU 42578",
            ha="right", va="bottom", fontsize=7, color="#aaaaaa",
        )

        fig.tight_layout()
        _resolve_save_hires(fig, save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
