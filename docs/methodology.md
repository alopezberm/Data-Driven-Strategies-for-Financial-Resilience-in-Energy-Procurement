# Methodology

## Pipeline Overview

Raw → Clean → Features → Models → Decision Engine → Backtest

The system is a fully reproducible, modular pipeline implemented in `/src/`.

---

## MDP Formulation (Single Source of Truth)

The Reinforcement Learning layer is governed by a Markov Decision Process with the following fixed parameters. These are the authoritative values. Any change requires updating `src/config/constants.py` and this document simultaneously.

### System Parameters

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
| Maximum Production Capacity | P_max | 2000 | units/day |
| Daily Demand | D | 1000 | units/day |
| Maximum Warehouse Capacity | I_max | 3000 | units |
| Startup Energy | e_start | 20 | MWh |
| Variable Energy per Unit | e_unit | 1 | MWh/unit |
| Inventory Holding Cost | h | 5 | EUR/unit/day |
| Gross Profit Margin | M | 200 | EUR/unit |

### State Space

The observation at each step t contains:

- `forecast_central` — q_0.5 median price forecast (t+2 horizon)
- `forecast_tail` — q_0.9 tail forecast (t+2 horizon)
- `forecast_central_h3` — q_0.5 median price forecast (t+3 horizon, **required**)
- `forecast_tail_h3` — q_0.9 tail forecast (t+3 horizon, **required**)
- `m1_price` — Front-month futures price (Future_M1_Price)
- `spot_price` — Day-ahead spot price (Spot_Price_SPEL)
- `Spot_M1_Spread` = spot_price − m1_price (**required**; auto-computed if absent)
- `inventory` — Current warehouse stock (units)
- `inventory_bin` — Discretized inventory: 0 = low (<33%), 1 = mid (33–67%), 2 = high (>67%)

For tabular Q-learning, the state key is formed from four binned signals:

| Signal | Computation | Bin width |
|---|---|---|
| `q90_vs_m1` | forecast_tail − m1_price | 5 EUR/MWh |
| `spot_m1_spread` | spot_price − m1_price | 5 EUR/MWh |
| `q90_h3_vs_m1` | forecast_tail_h3 − m1_price | 5 EUR/MWh |
| `inventory_bin` | {0, 1, 2} | — |

### Action Space (Joint Compound, Discrete)

Each action encodes a simultaneous decision on production volume and futures block purchases:

| Dimension | Symbol | Options | Count |
|---|---|---|---|
| Production volume | P_{t+1} | {0, 100, 200, ..., 2000} units | 21 |
| M1 futures block | b_m1 | {0, 500, 1000} MWh | 3 |
| M2 futures block | b_m2 | {0, 500, 1000} MWh | 3 |
| M3 futures block | b_m3 | {0, 500, 1000} MWh | 3 |

**Total actions: 21 × 3³ = 567**

Encoding: `action_id = prod_idx * 27 + m1_idx * 9 + m2_idx * 3 + m3_idx`

Production is further constrained so that I_{t+1} ≤ I_max (no warehouse overflow).

### Reward Function (Take-or-Pay)

```
E_t       = e_start + e_unit × P_{t+1}   if P_{t+1} > 0, else 0
hedge_pay = b_m1 × p_M1 + b_m2 × p_M2 + b_m3 × p_M3     (Fixed Hedge Payment)
deficit   = max(0, E_t − b_m1 − b_m2 − b_m3)
spot_pay  = deficit × spot_price                           (Spot Payment for deficit)
I_{t+1}   = clip(I_t + P_{t+1} − D, 0, I_max)

R_t = M × P_{t+1} − h × I_{t+1} − hedge_pay − spot_pay
```

The reward is the daily gross profit net of inventory holding cost and energy procurement cost.

### Inventory Dynamics

- `I_{t+1} = clip(I_t + P_{t+1} − D, 0, I_max)`
- If `I_t + P_{t+1} < D`: a stockout occurs; inventory floors at 0.
- If `I_t + P_{t+1} > I_max`: production is capped internally (`_feasible_production`).

---

## Modeling Strategy

- Quantile regression at horizons t+2 and t+3 (q_0.5, q_0.9, q_0.95)
- Tail-risk-aware forecasting rather than point prediction
- Chronological train/validation/test split to prevent leakage

---

## Decision Framework

### Heuristic Policy (Primary Benchmark)

- Rule-based, fully interpretable
- Operates on q90 spread signals and configurable thresholds
- Actions: `do_nothing`, `buy_m1_future`, `shift_production`

### RL Policy (Experimental Extension)

- Tabular Q-learning on the 567-action compound MDP action space
- Learns from historical cost outcomes derived from the MDP reward function
- Same input signals as the heuristic policy (q90, Spot_M1_Spread, h3 forecasts)
- Training via episodic rollouts over the 2020–2024 state space

---

## Evaluation

Counterfactual backtesting on the 2025 out-of-sample holdout (365 days). No data leakage. Four strategies compared on energy procurement cost:

1. **Spot only** — buy all energy at daily spot price
2. **Static hedge** — fixed 70% hedged via M1 futures
3. **Heuristic policy** — rule-based procurement decisions
4. **RL policy** — compound-action Q-learning decisions

Metrics: total cost, tail-risk exposure (q90 exceedance days), cost volatility, resilience on extreme days.

---

## Reproducibility

- Modular `src/` implementation
- End-to-end pipeline via `src/pipeline/run_full_pipeline.py`
- Automated tests in `tests/`
- Structured outputs in `data/outputs/`
