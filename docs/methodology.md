# Methodology

## Pipeline Overview
Raw → Clean → Features → Models → Decision → Backtest

## Modeling Strategy
- Quantile regression (q50, q90)
- Tail-risk focus
- Chronological validation

## Decision Framework
- Heuristic policy based on thresholds and rule-based actions
- RL-based policy as an experimental extension
- Policies evaluated under the same backtesting framework

## Evaluation
- Counterfactual simulation
- No data leakage
- Temporal validation
- Comparison across four strategies:
  - Spot only
  - Static hedge
  - Heuristic policy
  - RL policy

## Reproducibility
- Modular `src/` implementation
- End-to-end pipeline execution
- Automated tests