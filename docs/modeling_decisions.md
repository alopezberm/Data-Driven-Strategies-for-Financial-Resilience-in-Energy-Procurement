# Modeling Decisions

## Why Quantile Models?
- Capture uncertainty instead of only expected values
- Focus on tail risk
- Support downstream decision-making under extreme scenarios

## Why not only point forecasts?
- Point forecasts are not sufficient for procurement decisions
- Risk-aware decisions require information about upper-tail scenarios

## Model Types
- Baseline models
- Quantile regression models

## Decision Layer
- Heuristic policy as the main interpretable decision benchmark
- RL-based policy as an experimental extension

## Validation Strategy
- Chronological split
- No leakage
- Backtesting on unseen periods

## Trade-offs
- Simplicity vs performance
- Interpretability vs complexity
- Robust benchmark vs experimental adaptive strategy