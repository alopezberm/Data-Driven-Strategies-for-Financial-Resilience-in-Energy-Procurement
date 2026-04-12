# Backtesting Design

## Objective
Evaluate DSS decisions under realistic conditions on unseen periods.

## Approach
- Walk-forward style chronological evaluation
- Daily decision-making based on model outputs and policy logic
- Counterfactual comparison of alternative procurement strategies

## Strategies Compared
- Spot only
- Static hedge
- Heuristic policy
- RL policy (experimental)

## Metrics
- Total cost
- Savings vs reference strategy
- Tail risk exposure
- Cost stability / volatility
- Resilience indicators

## Key Principle
Counterfactual evaluation without future information and under consistent simulation assumptions across strategies.