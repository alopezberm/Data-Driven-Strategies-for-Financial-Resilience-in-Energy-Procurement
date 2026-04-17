# Backtesting Design

## Objective
Evaluate DSS decisions under realistic conditions on unseen periods, focusing on cost efficiency and robustness under extreme price scenarios.

## Approach
- Walk-forward style chronological evaluation
- Daily decision-making based on model outputs and policy logic
- Counterfactual comparison of alternative procurement strategies under identical conditions

## Strategies Compared
- Spot only
- Static hedge
- Heuristic policy
- RL policy (experimental)

## Metrics

### Cost & Efficiency
- Total cost
- Savings vs spot-only (reference baseline)
- Daily cost evolution

### Risk & Tail Exposure
- Upper-tail exceedances (q90)
- Extreme cost days
- Quantile coverage and calibration

### Stability & Resilience
- Cost volatility
- Resilience indicators
- Performance under extreme scenarios

### Policy Behavior
- Action distribution (frequency and share)
- Policy timelines
- Relationship between actions and tail risk

## Key Principle
Counterfactual evaluation without future information and under consistent simulation assumptions across all strategies.

## Outputs Generated
The backtesting pipeline produces:

- Strategy-level summaries:
  - `strategy_summary_table.csv`
  - `strategy_summary_vs_spot_only.csv`
- Daily comparisons:
  - `strategy_daily_comparison.csv`
- Risk diagnostics:
  - quantile coverage and tail exceedances
  - resilience metrics
- Policy-level outputs:
  - action timelines
  - action distributions