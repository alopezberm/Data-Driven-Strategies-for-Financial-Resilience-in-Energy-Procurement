# Methodology

## Pipeline Overview
Raw → Clean → Features → Models → Decision → Backtest

The system is designed as a fully reproducible, modular pipeline implemented in `/src/`.

---

## Modeling Strategy

- Quantile regression (q50, q90)
- Tail-risk-aware forecasting instead of point prediction
- Chronological validation to avoid leakage

The modeling layer provides probabilistic signals that explicitly capture uncertainty and extreme scenarios.

---

## Decision Framework

Two types of decision policies are implemented:

### Heuristic Policy (Benchmark)
- Rule-based
- Interpretable
- Uses quantile spreads and thresholds
- Serves as the main reference strategy

### RL Policy (Experimental)
- Tabular Q-learning
- Learns actions from historical cost outcomes
- Operates on the same input signals as the heuristic policy
- Sensitive to reward design and state representation

Both policies produce daily decisions among:
- `do_nothing`
- `buy_m1_future`
- `shift_production`

---

## Evaluation

- Counterfactual simulation
- No data leakage
- Temporal validation on unseen periods

Comparison across four strategies:
- Spot only
- Static hedge
- Heuristic policy
- RL policy

Evaluation covers:

- Cost performance
- Tail risk exposure
- Stability and volatility
- Resilience under extreme conditions
- Policy behavior (actions taken over time)

---

## Reproducibility

- Modular `src/` implementation
- End-to-end pipeline execution via CLI
- Automated tests
- Structured outputs (data, tables, figures)

---

## Key Insight

The system is not only a forecasting model but a **decision support framework** that integrates:

- predictive modeling (quantiles)
- decision logic (policies)
- evaluation (backtesting)

RL is included as an **experimental extension**, while the heuristic policy remains the most robust and interpretable baseline.