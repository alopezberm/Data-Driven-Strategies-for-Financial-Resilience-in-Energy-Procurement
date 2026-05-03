# Data-Driven Strategies for Financial Resilience in Energy Procurement

A Data-Driven Decision Support System (DSS) to optimize industrial energy procurement and mitigate tail risks in the Spanish electricity market (Spot/Futures hedging). Developed for the Advanced Business Analytics course (42578) at DTU.

---

## 📋 Executive Summary

Industrial manufacturers in the Spanish electricity market (MIBEL) face extreme price volatility. Sudden Spot price spikes can significantly impact profitability. Traditional procurement strategies force a rigid choice between fixed contracts and Spot exposure, failing to dynamically manage **tail risks**.

This project presents an **end-to-end Decision Support System (DSS)** that acts as a virtual consultant, providing daily recommendations for:

- Financial hedging (futures contracts)
- Operational decisions (e.g., production shifting)

The goal is to **minimize expected costs under uncertainty while reducing exposure to extreme price events**.

The system is **not a forecast tool** — it is an automated decision engine. Every recommendation is auditable, threshold-driven, and validated over a full out-of-sample 2025 calendar year (365 days, zero data leakage).

---

## ⚙️ System Architecture

The solution follows a modular pipeline:

```
Raw Data → Cleaning → Feature Engineering → Modeling → Decision → Backtesting
```

### 1. Risk Prediction Engine

- Quantile forecasting (e.g., q50, q90)
- Tail-risk awareness instead of point prediction
- Captures extreme price scenarios

### 2. Decision Engine

Transforms risk signals into actions:

- Heuristic policy (baseline)
- RL-based policy (experimental extension)

### 3. Backtesting Engine

- Counterfactual simulation
- Strategy comparison:
  - Spot-only
  - Static hedge
  - Heuristic policy
	- RL policy

---

---

## 💰 Key Financial Results

Validated on 365 out-of-sample trading days (full year 2025) for a reference manufacturer with fixed daily demand of **D = 1,000 units/day** and a gross margin of **€200/unit**:

| KPI | Spot-Only Baseline | TailRisk DSS (Heuristic) | Delta |
|:----|-------------------:|-------------------------:|------:|
| **Annual Energy Cost** | €24,304,703 | €23,084,528 | **−€1,220,175** |
| **Avg. Cost per MWh** | €65.28/MWh | €62.01/MWh | −€3.27/MWh |
| **Daily Cost Volatility (σ)** | €35,373/day | €19,102/day | **−46%** |
| **P95 Daily Cost** | €127,537 | €90,627 | −€36,910 |
| **Max Daily Cost** | €147,818 | €103,275 | −31.7% |
| **Net Profit Improvement** | — | **+2.50%** | +€3.34/unit |
| **Annual Margin Recovery** | — | **€1,220,175** | +5.0% |

> 💡 **The bottom line:** The TailRisk DSS recovers **€1.22M in annual net profit** — not through revenue growth, but by eliminating avoidable energy expenditure that flows directly to operating margin.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| 🐍 **Python 3.11** | Core runtime | End-to-end pipeline orchestration |
| 📊 **Gradient Boosting** (Scikit-Learn) | Quantile Regression | q50/q90 forecasts at t+2 / t+3 horizons |
| 🧮 **Mutual Information** (Scikit-Learn) | Feature Selection | Dimensionality reduction from raw feature matrix |
| 🤖 **Tabular Q-Learning** | Reinforcement Learning | Compound 168-action policy learning |
| 🌤️ **Open-Meteo Archive API** | Weather Ingestion | 52 Spanish provinces, 2020–2025, geospatial batching |
| 🕸️ **OMIP Web Scraper** | Market Data | SPEL spot + M1–M6 futures + open interest (2,192 days) |
| 📦 **pandas / numpy** | Data Engineering | Feature matrix construction, rolling statistics |
| 📈 **matplotlib / seaborn / plotly** | Visualization | Executive dashboards, policy timelines, resilience maps |
| 🧪 **pytest** | Testing | 39 unit/integration tests; zero data leakage assertion |

---

## 🚀 How to Run

### Environment Setup

```bash
# Option 1 — Virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option 2 — Conda
conda env create -f environment.yml
conda activate tailrisk-env
```

### Test Suite (Sanity Check)

```bash
python -m pytest tests/
# Expected: 39 passed
```

### Full Pipeline (End-to-End)

```bash
python -m src.pipeline.run_full_pipeline
```

This executes: data ingestion → cleaning → feature engineering → quantile model training → heuristic decision policy → RL agent training → backtesting → figure generation.

### Regenerate Raw Holiday Data

```bash
python -m src.data.generate_holidays_raw
# Generates: data/raw/holidays/holidays_raw.csv (2020–2025 Spanish national holidays)
```

---

## 📂 Project Structure

```
.
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/              (OMIP futures, weather, holidays)
│   ├── interim/          (cleaned, merged intermediates)
│   ├── processed/        (modeling_dataset, train/val/test splits, features)
│   └── outputs/
│       ├── backtests/    (strategy simulation CSVs, resilience metrics)
│       ├── policies/     (daily decision logs)
│       └── figures/      (executive dashboards, policy timelines)
│
├── notebooks/
│   └── 08_reporting/
│       └── 01_technical_report.ipynb   ← primary reference document
│
├── src/
│   ├── config/           (paths, settings, constants)
│   ├── data/             (ingestion, cleaning, merging)
│   ├── features/         (time, lag, rolling, futures features)
│   ├── models/           (quantile regression, evaluation)
│   ├── decision/         (heuristic policy, RL agent, RL policy)
│   ├── rl/               (environment, training, evaluation)
│   ├── backtesting/      (strategy simulation, resilience metrics)
│   ├── visualization/    (forecasts, backtest results, policy actions)
│   └── pipeline/         (run_full_pipeline, run_backtest)
│
└── tests/                (39 tests: data, features, models, backtest, RL)
```

---

## ✅ System Status

- 🟢 Full pipeline implemented and reproducible
- 🟢 39/39 unit and integration tests passing
- 🟢 365-day out-of-sample backtest validated (zero data leakage)
- 🟢 Heuristic Policy production-ready (Layer 1)
- 🟡 RL Agent integrated and functional — shadow mode only (Layer 2)
- 🟢 Executive dashboards and resilience figures generated

---

## 👥 Authors

**DTU — MSc Business Analytics, Group 17** (Course 42578: Advanced Business Analytics)

| Student | ID |
|:--------|:---|
| Ignacio Ripoll González | s242875 |
| Pablo Baurier Gasch | s253159 |
| Alejandro López Bermejo | s253272 |

---

## 📄 License

Academic project — for educational purposes only.
