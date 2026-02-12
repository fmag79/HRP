# HRP Cookbook: A Practical Guide

This cookbook provides hands-on recipes for using the Hedgefund Research Platform. Each recipe includes real, runnable examples.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Data Operations](#2-data-operations)
3. [Hypothesis Management](#3-hypothesis-management)
4. [Running Backtests](#4-running-backtests)
5. [ML & Walk-Forward Validation](#5-ml--walk-forward-validation)
6. [Data Quality Monitoring](#6-data-quality-monitoring)
7. [Scheduled Jobs & Automation](#7-scheduled-jobs--automation)
8. [Using the Dashboard](#8-using-the-dashboard)
9. [Common Workflows](#9-common-workflows)
10. [Troubleshooting](#10-troubleshooting)
11. [Claude Integration (MCP Server)](#11-claude-integration-mcp-server)
12. [Ops Server & Monitoring](#12-ops-server--monitoring)
13. [Trading Execution (Tier 4)](#13-trading-execution-tier-4)
14. [Advanced Analytics (Tier 5)](#14-advanced-analytics-tier-5)

---

## 1. Getting Started

### 1.1 Initial Setup

```bash
# Activate your virtual environment
source .venv/bin/activate

# Verify installation
python -c "from hrp.api.platform import PlatformAPI; print('HRP ready!')"
```

### 1.2 Your First API Connection

```python
from hrp.api.platform import PlatformAPI
from datetime import date

# Create the API instance (this is your main entry point)
api = PlatformAPI()

# Check system health
health = api.health_check()
print(f"Status: {health['status']}")
print(f"Database: {health['database']}")
print(f"Tables: {health['tables']}")
```

**Expected output:**
```
Status: healthy
Database: connected
Tables: {'prices': 52340, 'features': 418720, 'hypotheses': 5, ...}
```

**Note on Architecture:**
- External access uses `PlatformAPI` (single entry point)
- Internal modules (`hrp/research/hypothesis.py`, `hrp/research/lineage.py`) use function-based APIs
- There are no `HypothesisRegistry`, `LineageTracker`, or `ValidationFramework` classes
- Functions like `create_hypothesis()`, `log_event()`, and `validate_strategy()` are the primary interface

### 1.3 Environment Variables

Copy `.env.example` to `.env` and edit with your values:

```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

The `.env.example` file contains all ~50 configurable variables with defaults and descriptions. Key categories:

```bash
# Core: HRP_ENVIRONMENT, HRP_DATA_DIR, HRP_DB_PATH
# Data Sources: POLYGON_API_KEY, ALPACA_API_KEY, SIMFIN_API_KEY, ...
# Notifications: RESEND_API_KEY, NOTIFICATION_EMAIL
# Auth: HRP_AUTH_ENABLED, HRP_AUTH_COOKIE_KEY
# Trading: HRP_BROKER_TYPE, IBKR_*, ROBINHOOD_*
# Portfolio: HRP_PORTFOLIO_VALUE, HRP_MAX_POSITIONS, HRP_TRADING_DRY_RUN
# Risk: HRP_USE_VAR_SIZING, HRP_MAX_PORTFOLIO_VAR_PCT
# Real-Time: HRP_REALTIME_ENABLED, HRP_REALTIME_SYMBOLS
# Thresholds: HRP_THRESHOLD_* (11 overrides)
```

### 1.4 Start the System

Once your environment is configured, you can start all HRP services:

```bash
# Start all core services (dashboard, MLflow UI, scheduler)
./scripts/startup.sh start

# Check service status
./scripts/startup.sh status

# Stop all services
./scripts/startup.sh stop
```

**Access Points:**
- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5010

**Note:** The startup script does **not** initialize the database. The database is created automatically on first access. For fresh installations, ensure you've completed section 1.1 (Initial Setup) before starting services.

For more advanced startup options (full research agent pipeline, individual services, custom ports), see [Section 7.7: System Startup Script](#77-system-startup-script).

---

## 2. Data Operations

### 2.1 Get Price Data

```python
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()

# Get prices for specific symbols
prices = api.get_prices(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date(2023, 1, 1),
    end=date(2023, 12, 31)
)

print(prices.head())
```

**Output:**
```
  symbol        date    open    high     low   close  adj_close     volume
0   AAPL  2023-01-03  130.28  130.90  124.17  125.07     124.42  112117500
1   AAPL  2023-01-04  126.89  128.66  125.08  126.36     125.70   89113600
2   AAPL  2023-01-05  127.13  127.77  124.76  125.02     124.37   80962700
...
```

### 2.2 Get Feature Data

```python
# Get pre-computed technical features
features = api.get_features(
    symbols=['AAPL', 'MSFT'],
    features=['momentum_20d', 'volatility_20d', 'rsi_14d'],
    as_of_date=date(2023, 12, 29)
)

print(features)
```

**Output:**
```
        momentum_20d  volatility_20d   rsi_14d
symbol
AAPL        0.0523         0.1842      58.34
MSFT        0.0712         0.1654      62.15
```

### 2.3 Get Universe

```python
# Get current tradeable universe
universe = api.get_universe(as_of_date=date.today())
print(f"Universe size: {len(universe)} symbols")
print(f"Sample: {universe[:10]}")
```

**Output:**
```
Universe size: 450 symbols
Sample: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'V', 'UNH', 'JNJ']
```

### 2.4 Update Universe Membership

```python
from hrp.data.universe import UniverseManager
from datetime import date

manager = UniverseManager()

# Update universe with current S&P 500 constituents
stats = manager.update_universe(as_of_date=date.today())

print(f"Total constituents: {stats['total_constituents']}")
print(f"Included in universe: {stats['included']}")
print(f"Excluded: {stats['excluded']}")
print(f"Added: {stats['added']}")
print(f"Removed: {stats['removed']}")
print(f"Exclusion reasons: {stats['exclusion_reasons']}")
```

**Output:**
```
Total constituents: 503
Included in universe: 380
Excluded: 123
Added: 2
Removed: 1
Exclusion reasons: {'excluded_sector': 80, 'penny_stock': 43}
```

**Get Historical Universe:**

```python
# Get universe as of a specific date (for backtest accuracy)
universe_2023 = manager.get_universe_at_date(date(2023, 1, 1))
print(f"Universe on 2023-01-01: {len(universe_2023)} symbols")

# Track universe changes over time
changes = manager.get_universe_changes(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31)
)
print(changes)
```

**Sector Breakdown:**

```python
sectors = manager.get_sector_breakdown(date.today())
for sector, count in sectors.items():
    print(f"{sector}: {count} symbols")
```

### 2.5 Ingest New Price Data

```python
from hrp.data.ingestion.prices import ingest_prices
from datetime import date, timedelta

# Ingest last 30 days for specific symbols
result = ingest_prices(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date.today() - timedelta(days=30),
    end=date.today(),
    source='yfinance'  # or 'polygon' if you have API key
)

print(f"Symbols succeeded: {result['symbols_success']}")
print(f"Rows inserted: {result['rows_inserted']}")
```

### 2.6 Compute Features

```python
from hrp.data.ingestion.features import compute_features
from datetime import date, timedelta

# Compute features for recent data (symbol-by-symbol)
result = compute_features(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date.today() - timedelta(days=60),
    end=date.today()
)

print(f"Features computed: {result['features_computed']}")
print(f"Rows inserted: {result['rows_inserted']}")
```

### 2.6.1 Batch Feature Computation (~10x Faster)

```python
from hrp.data.ingestion.features import compute_features_batch
from datetime import date, timedelta

# Compute features for all symbols in single vectorized pass
# This is ~10x faster than symbol-by-symbol processing
result = compute_features_batch(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],  # or None for all
    start=date.today() - timedelta(days=60),
    end=date.today()
)

print(f"Symbols processed: {result['symbols_success']}")
print(f"Features computed: {result['features_computed']}")
print(f"Rows inserted: {result['rows_inserted']}")
```

**Features computed (8 total):**
- Returns: `returns_1d`, `returns_5d`, `returns_20d`
- Momentum: `momentum_20d`, `momentum_60d`
- Volatility: `volatility_20d`, `volatility_60d`
- Volume: `volume_20d`

**Note:** The scheduled `FeatureComputationJob` automatically uses batch processing.

### 2.7 Get Point-in-Time Fundamentals

```python
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()

# Get fundamentals as they would have been known on a specific date
# This prevents look-ahead bias by only returning data where report_date <= as_of_date
fundamentals = api.get_fundamentals_as_of(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    metrics=['revenue', 'eps', 'book_value'],
    as_of_date=date(2023, 6, 30)
)

print(fundamentals)
```

**Output:**
```
  symbol      metric         value  report_date  period_end
0   AAPL     revenue  3.948000e+11   2023-05-04  2023-03-31
1   AAPL         eps  1.520000e+00   2023-05-04  2023-03-31
2   AAPL  book_value  6.240000e+01   2023-05-04  2023-03-31
3   MSFT     revenue  2.119000e+11   2023-04-25  2023-03-31
...
```

**Using Fundamentals in Backtests:**

```python
from hrp.research.backtest import get_fundamentals_for_backtest
import pandas as pd

# Get fundamentals for each date in a backtest range
dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
fundamentals = get_fundamentals_for_backtest(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    metrics=['eps', 'book_value'],
    dates=dates
)

# Returns DataFrame with MultiIndex (date, symbol) and metrics as columns
print(fundamentals.head())
```

**Output:**
```
                            eps  book_value
date       symbol
2023-01-31 AAPL     1.29        60.10
           GOOGL    1.06        21.58
           MSFT     2.32        26.44
2023-02-28 AAPL     1.29        60.10
           GOOGL    1.06        21.58
...
```

---

## 3. Hypothesis Management

### 3.1 Create a New Hypothesis

```python
from hrp.api.platform import PlatformAPI

api = PlatformAPI()

# Create a formal research hypothesis
hypothesis_id = api.create_hypothesis(
    title="20-day momentum predicts 5-day forward returns",
    thesis="""
    Stocks with strong 20-day momentum tend to continue outperforming
    over the next 5 trading days. This is based on the momentum anomaly
    documented in academic literature.
    """,
    prediction="""
    A portfolio long top-decile 20-day momentum stocks, rebalanced weekly,
    will outperform SPY by >2% annually with Sharpe > 0.5.
    """,
    falsification="""
    - Out-of-sample Sharpe ratio < 0.3
    - p-value for excess returns > 0.10
    - Performance concentrated in < 2 years
    """,
    actor='user'
)

print(f"Created hypothesis: {hypothesis_id}")
# Output: Created hypothesis: HYP-2025-001
```

### 3.2 List Hypotheses

```python
# List all hypotheses
all_hypotheses = api.list_hypotheses()
print(f"Total hypotheses: {len(all_hypotheses)}")

# List by status
draft_hypotheses = api.list_hypotheses(status='draft')
validated_hypotheses = api.list_hypotheses(status='validated')

for h in draft_hypotheses[:3]:
    print(f"- {h['hypothesis_id']}: {h['title']}")
```

### 3.3 Update Hypothesis Status

```python
# Move hypothesis to testing (after running initial backtest)
api.update_hypothesis(
    hypothesis_id='HYP-2025-001',
    status='testing',
    actor='user'
)

# After validation, mark as validated or rejected
api.update_hypothesis(
    hypothesis_id='HYP-2025-001',
    status='validated',
    outcome='Passed all validation criteria. Sharpe=0.72, p-value=0.023',
    actor='user'
)
```

### 3.4 View Hypothesis Details

```python
# Get full hypothesis details
hypothesis = api.get_hypothesis('HYP-2025-001')

print(f"Title: {hypothesis['title']}")
print(f"Status: {hypothesis['status']}")
print(f"Created: {hypothesis['created_at']}")
print(f"Thesis: {hypothesis['thesis'][:100]}...")
```

### 3.5 View Audit Trail (Lineage)

```python
# Get full history of actions for a hypothesis
lineage = api.get_lineage(hypothesis_id='HYP-2025-001')

for event in lineage:
    print(f"{event['timestamp']} | {event['event_type']} | {event['actor']}")
```

**Output:**
```
2025-01-15 10:30:00 | hypothesis_created | user
2025-01-15 11:00:00 | experiment_run | user
2025-01-15 14:30:00 | status_updated | user
2025-01-16 09:00:00 | validation_passed | system
```

---

## 4. Running Backtests

### 4.1 Simple Momentum Backtest

```python
from hrp.api.platform import PlatformAPI
from hrp.research.config import BacktestConfig, CostModel
from datetime import date

api = PlatformAPI()

# Configure the backtest
config = BacktestConfig(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'V', 'UNH', 'JNJ'],
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    sizing_method='equal',
    max_positions=5,
    max_position_pct=0.20,
    costs=CostModel(commission_pct=0.001, slippage_pct=0.001),
    name='momentum_test',
    description='Testing 20-day momentum on tech stocks'
)

# Run backtest (uses momentum signals by default)
experiment_id = api.run_backtest(
    config=config,
    hypothesis_id='HYP-2025-001',  # Link to hypothesis
    actor='user'
)

print(f"Experiment logged: {experiment_id}")
```

### 4.2 Total Return Backtest (with Dividend Reinvestment)

```python
from hrp.api.platform import PlatformAPI
from hrp.research.config import BacktestConfig
from hrp.research.backtest import get_price_data
from datetime import date

api = PlatformAPI()

# Configure backtest with total return (includes dividend reinvestment)
config = BacktestConfig(
    symbols=['AAPL', 'MSFT', 'JNJ'],  # Dividend-paying stocks
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    total_return=True,  # Enable dividend reinvestment
    name='total_return_test'
)

# Get price data with dividend adjustment
prices = get_price_data(
    symbols=config.symbols,
    start=config.start_date,
    end=config.end_date,
    adjust_splits=True,
    adjust_dividends=True  # Apply dividend adjustment
)

# The 'dividend_adjusted_close' column reflects total return
print(prices.head())
```

**Note:** The dividend adjustment uses the standard formula: for each dividend, prior prices are adjusted by factor `1 - (dividend / price_on_ex_date)`. This compounds correctly for multiple dividends and provides accurate total return calculations.

### 4.4 Backtest with Custom Signals

```python
import pandas as pd
import numpy as np
from hrp.api.platform import PlatformAPI
from hrp.research.config import BacktestConfig
from datetime import date

api = PlatformAPI()

# Get price data
prices = api.get_prices(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date(2020, 1, 1),
    end=date(2023, 12, 31)
)

# Pivot to wide format
close_prices = prices.pivot(index='date', columns='symbol', values='adj_close')

# Create custom signals (example: RSI-based mean reversion)
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

rsi = close_prices.apply(lambda x: compute_rsi(x, period=14))

# Signal: Buy when RSI < 30 (oversold), sell when RSI > 70
signals = pd.DataFrame(index=rsi.index, columns=rsi.columns)
signals[rsi < 30] = 1.0   # Buy signal
signals[rsi > 70] = -1.0  # Sell signal
signals = signals.fillna(0)

# Run backtest with custom signals
config = BacktestConfig(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    name='rsi_mean_reversion'
)

experiment_id = api.run_backtest(
    config=config,
    signals=signals,
    hypothesis_id='HYP-2025-002',
    actor='user'
)
```

### 4.5 View Experiment Results

```python
# Get experiment details
experiment = api.get_experiment(experiment_id)

print("=== Backtest Results ===")
print(f"Status: {experiment['status']}")
print(f"\nMetrics:")
for metric, value in experiment['metrics'].items():
    print(f"  {metric}: {value:.4f}")
```

**Output:**
```
=== Backtest Results ===
Status: FINISHED

Metrics:
  sharpe_ratio: 0.7234
  sortino_ratio: 1.0521
  total_return: 0.4523
  cagr: 0.1342
  max_drawdown: 0.1823
  volatility: 0.1854
  calmar_ratio: 0.7361
  win_rate: 0.5423
```

### 4.6 Compare Multiple Experiments

```python
# Compare experiments side-by-side
comparison = api.compare_experiments(
    experiment_ids=[experiment_id_1, experiment_id_2, experiment_id_3],
    metrics=['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
)

print(comparison)
```

**Output:**
```
                     sharpe_ratio  total_return  max_drawdown  win_rate
experiment_id
abc123               0.7234        0.4523        0.1823        0.5423
def456               0.5123        0.3245        0.2134        0.4823
ghi789               0.8534        0.5234        0.1523        0.5723
```

### 4.7 Multi-Factor Strategy Backtest

```python
from hrp.research.strategies import generate_multifactor_signals
from hrp.research.backtest import get_price_data, run_backtest
from hrp.research.config import BacktestConfig
from datetime import date

# Load prices
prices = get_price_data(
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META'],
    start=date(2020, 1, 1),
    end=date(2023, 12, 31)
)

# Generate multi-factor signals
# Positive weights favor higher values, negative favor lower
signals = generate_multifactor_signals(
    prices,
    feature_weights={
        "momentum_20d": 1.0,     # Favor high momentum stocks
        "volatility_60d": -0.5,  # Penalize high volatility
    },
    top_n=10,  # Hold top 10 stocks by composite score
)

# Run backtest
config = BacktestConfig(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META'],
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    name='multifactor_momentum_lowvol'
)

result = run_backtest(signals, config, prices)
print(f"Sharpe: {result.sharpe:.2f}")
print(f"Total Return: {result.total_return:.1%}")
```

**How Multi-Factor Works:**
1. Fetches feature values for all symbols on each date
2. Z-score normalizes each factor cross-sectionally (mean=0, std=1)
3. Computes weighted composite: `sum(weight * normalized_factor)`
4. Ranks stocks by composite score, selects top N

### 4.8 ML-Predicted Strategy Backtest

```python
from hrp.research.strategies import generate_ml_predicted_signals
from hrp.research.backtest import get_price_data, run_backtest
from hrp.research.config import BacktestConfig
from datetime import date

# Load prices
prices = get_price_data(
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META'],
    start=date(2020, 1, 1),
    end=date(2023, 12, 31)
)

# Generate ML-predicted signals
signals = generate_ml_predicted_signals(
    prices,
    model_type="ridge",  # Options: ridge, lasso, random_forest, lightgbm, xgboost
    features=["momentum_20d", "volatility_60d"],
    signal_method="rank",      # Options: rank, threshold, zscore
    top_pct=0.1,               # For rank: select top 10%
    train_lookback=252,        # 1 year training window
    retrain_frequency=21,      # Monthly retraining
)

# Run backtest
config = BacktestConfig(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META'],
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    name='ml_predicted_ridge'
)

result = run_backtest(signals, config, prices)
print(f"Sharpe: {result.sharpe:.2f}")
print(f"Total Return: {result.total_return:.1%}")
```

**How ML-Predicted Works:**
1. For each rebalance date (every `retrain_frequency` days):
   - Train model on past `train_lookback` days of feature data
   - Generate predictions for forward returns on all symbols
2. Convert predictions to signals using `signal_method`:
   - `rank`: Select top X% by predicted return
   - `threshold`: Select if prediction >= threshold value
   - `zscore`: Continuous signals (z-score normalized)
3. Hold positions until next rebalance

**Available Models:**
- `ridge`: Ridge regression (L2 regularization) - fast, stable
- `lasso`: Lasso regression (L1 regularization) - feature selection
- `random_forest`: Random Forest regressor - captures non-linear patterns
- `lightgbm`: LightGBM (requires lightgbm package)
- `xgboost`: XGBoost (requires xgboost package)

---

## 5. ML & Walk-Forward Validation

### 5.1 Train a Simple ML Model

```python
from hrp.ml.training import train_model
from hrp.ml.config import MLConfig
from datetime import date

# Configure ML training
config = MLConfig(
    model_type='ridge',
    target='returns_20d',
    features=['momentum_20d', 'momentum_60d', 'volatility_20d', 'rsi_14d'],
    train_start=date(2015, 1, 1),
    train_end=date(2020, 12, 31),
    validation_start=date(2021, 1, 1),
    validation_end=date(2021, 12, 31),
    test_start=date(2022, 1, 1),
    test_end=date(2023, 12, 31),
    feature_selection=True,
    max_features=10
)

# Train the model
result = train_model(
    config=config,
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    hypothesis_id='HYP-2025-001'
)

print(f"Train R²: {result['train_metrics']['r2']:.4f}")
print(f"Val R²: {result['val_metrics']['r2']:.4f}")
print(f"Test R²: {result['test_metrics']['r2']:.4f}")
print(f"Test IC: {result['test_metrics']['ic']:.4f}")
```

### 5.2 Walk-Forward Validation

```python
from hrp.ml import WalkForwardConfig, walk_forward_validate
from datetime import date

# Configure walk-forward validation
config = WalkForwardConfig(
    model_type='ridge',
    target='returns_20d',
    features=['momentum_20d', 'volatility_20d', 'rsi_14d', 'volume_20d'],
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    n_folds=5,
    window_type='expanding',  # 'expanding' or 'rolling'
    feature_selection=True,
    max_features=15,
    n_jobs=1,  # Sequential (default) - use -1 for all cores
)

# Run walk-forward validation
result = walk_forward_validate(
    config=config,
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    log_to_mlflow=True
)

# Check results
print(f"Stability Score: {result.stability_score:.4f}")  # Lower is better
print(f"Mean IC: {result.mean_ic:.4f}")
print(f"Model is stable: {result.is_stable}")  # stability_score <= 1.0

# Per-fold results
print("\nPer-Fold Results:")
for fold in result.fold_results:
    print(f"  Fold {fold.fold_index}: IC={fold.metrics['ic']:.4f}, "
          f"MSE={fold.metrics['mse']:.6f}, "
          f"Train: {fold.train_start} to {fold.train_end}")
```

### 5.2.1 Parallel Walk-Forward Validation (3-4x Speedup)

```python
from hrp.ml import WalkForwardConfig, walk_forward_validate
from datetime import date

# Configure with parallel processing
config = WalkForwardConfig(
    model_type='ridge',
    target='returns_20d',
    features=['momentum_20d', 'volatility_20d', 'rsi_14d', 'volume_20d'],
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    n_folds=5,
    window_type='expanding',
    feature_selection=True,
    max_features=15,
    n_jobs=-1,  # Use all CPU cores for parallel fold processing
)

# Run validation - folds processed in parallel
result = walk_forward_validate(
    config=config,
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    log_to_mlflow=True
)

# Timing is automatically logged:
# Timing [walk_forward_validate]: 12.345s
#   - data_fetch: 0.234s (1.9%)
#   - fold_processing: 11.890s (96.3%)

print(f"Stability Score: {result.stability_score:.4f}")
print(f"Mean IC: {result.mean_ic:.4f}")
```

**n_jobs options:**
- `n_jobs=1`: Sequential processing (default, enables feature selection caching)
- `n_jobs=-1`: Use all available CPU cores
- `n_jobs=N`: Use exactly N parallel workers

**Note:** Feature selection caching is only available in sequential mode (`n_jobs=1`).
Parallel mode trades caching for faster execution via multiple processes.

**Output:**
```
Stability Score: 0.4523
Mean IC: 0.0823
Model is stable: True

Per-Fold Results:
  Fold 0: IC=0.0912, MSE=0.000234, Train: 2015-01-01 to 2017-12-31
  Fold 1: IC=0.0845, MSE=0.000256, Train: 2015-01-01 to 2018-12-31
  Fold 2: IC=0.0789, MSE=0.000278, Train: 2015-01-01 to 2019-12-31
  Fold 3: IC=0.0756, MSE=0.000289, Train: 2015-01-01 to 2020-12-31
  Fold 4: IC=0.0812, MSE=0.000245, Train: 2015-01-01 to 2021-12-31
```

### 5.3 Generate Trading Signals from ML Predictions

```python
from hrp.ml.signals import predictions_to_signals

# Assuming you have predictions from your model
predictions = model.predict(features_df)

# Method 1: Rank-based (go long top 10%)
signals = predictions_to_signals(
    predictions=predictions,
    method='rank',
    top_pct=0.10
)

# Method 2: Threshold-based
signals = predictions_to_signals(
    predictions=predictions,
    method='threshold',
    threshold=0.02  # Buy if predicted return > 2%
)

# Method 3: Z-score normalized
signals = predictions_to_signals(
    predictions=predictions,
    method='zscore'
)
```

### 5.4 Statistical Validation

```python
from hrp.risk.validation import validate_strategy, significance_test, ValidationCriteria
import pandas as pd

# Define validation criteria
criteria = ValidationCriteria(
    min_sharpe=0.5,
    min_trades=100,
    max_drawdown=0.25,
    min_win_rate=0.40,
    min_profit_factor=1.2,
    min_oos_period_days=730  # 2 years
)

# Get strategy and benchmark returns
strategy_returns = pd.Series(...)  # Your strategy daily returns
benchmark_returns = pd.Series(...)  # SPY daily returns

# Run significance test
sig_result = significance_test(strategy_returns, benchmark_returns, alpha=0.05)
print(f"Excess Return (annualized): {sig_result['excess_return_annualized']:.2%}")
print(f"t-statistic: {sig_result['t_statistic']:.3f}")
print(f"p-value: {sig_result['p_value']:.4f}")
print(f"Statistically Significant: {sig_result['significant']}")

# Full validation
validation_result = validate_strategy(
    returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    criteria=criteria,
    trades_df=trades_df  # DataFrame of trades
)

print(f"\nValidation Passed: {validation_result.passed}")
for criterion, passed in validation_result.criteria_results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {criterion}: {status}")
```

### 5.5 Robustness Testing

```python
from hrp.risk.robustness import (
    check_parameter_sensitivity,
    check_time_stability,
    check_regime_robustness
)

# Parameter sensitivity (varies parameters +/- 20%)
sensitivity_result = check_parameter_sensitivity(
    experiments=experiment_results,  # Dict of param_set -> metrics
    baseline_key='default',
    threshold=0.5  # Max allowed degradation
)
print(f"Parameter Sensitivity: {'PASS' if sensitivity_result.passed else 'FAIL'}")

# Time stability (checks across sub-periods)
time_result = check_time_stability(
    fold_results=walk_forward_result.fold_results,
    threshold=0.5
)
print(f"Time Stability: {'PASS' if time_result.passed else 'FAIL'}")

# Regime robustness (bull/bear/sideways)
regime_result = check_regime_robustness(
    results_by_regime={
        'bull': bull_market_metrics,
        'bear': bear_market_metrics,
        'sideways': sideways_metrics
    },
    threshold=0.5
)
print(f"Regime Robustness: {'PASS' if regime_result.passed else 'FAIL'}")
```

### 5.6 Overfitting Guards & Test Set Discipline

```python
from hrp.risk import TestSetGuard, validate_strategy

# Test Set Guard - prevents excessive test set evaluations
# Automatically enforced when training with hypothesis_id
from hrp.ml import train_model, MLConfig

config = MLConfig(
    model_type='ridge',
    target='returns_20d',
    features=['momentum_20d', 'volatility_20d'],
    train_start=date(2020, 1, 1),
    train_end=date(2021, 12, 31),
    validation_start=date(2022, 1, 1),
    validation_end=date(2022, 6, 30),
    test_start=date(2022, 7, 1),
    test_end=date(2022, 12, 31),
)

# Training with hypothesis_id enables TestSetGuard automatically
result = train_model(
    config=config,
    symbols=['AAPL', 'MSFT'],
    hypothesis_id='HYP-2025-001'  # Guard tracks evaluations per hypothesis
)

# Manual guard usage for custom evaluation
guard = TestSetGuard(hypothesis_id='HYP-2025-001')

print(f"Evaluations used: {guard.evaluation_count}/3")
print(f"Remaining: {guard.remaining_evaluations}")

with guard.evaluate(metadata={"experiment": "final_validation"}):
    # Your test set evaluation code here
    test_metrics = model.evaluate(test_data)

# Override limit (explicit justification required)
# Use sparingly - typically only for final validation
with guard.evaluate(override=True, reason="Final model validation after peer review"):
    final_metrics = model.evaluate(test_data)

# Sharpe Decay Monitor - detect train/test overfitting
from hrp.risk.overfitting import SharpeDecayMonitor

monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
result = monitor.check(train_sharpe=1.5, test_sharpe=1.0)
if not result.passed:
    print(f"⚠️ Sharpe decay warning: {result.message}")

# Feature Count Validator - prevent too many features
from hrp.risk.overfitting import FeatureCountValidator

validator = FeatureCountValidator(warn_threshold=30, max_threshold=50)
result = validator.check(feature_count=25, sample_count=1000)
if not result.passed:
    raise OverfittingError(result.message)
if result.warning:
    print(f"⚠️ Feature count warning: {result.message}")

# Hyperparameter Trial Counter - limit HP search
from hrp.risk.overfitting import HyperparameterTrialCounter

counter = HyperparameterTrialCounter(hypothesis_id='HYP-2025-001', max_trials=50)
counter.log_trial(
    model_type='ridge',
    hyperparameters={'alpha': 1.0},
    metric_name='val_r2',
    metric_value=0.85,
)
print(f"HP trials remaining: {counter.remaining_trials}")
best = counter.get_best_trial()

# Target Leakage Validator - detect data leakage
from hrp.risk.overfitting import TargetLeakageValidator

leakage_validator = TargetLeakageValidator(correlation_threshold=0.95)
result = leakage_validator.check(features_df, target_series)
if not result.passed:
    raise OverfittingError(f"Leakage detected: {result.suspicious_features}")

# Strategy validation gates
metrics = {
    "sharpe": 0.80,
    "num_trades": 200,
    "max_drawdown": 0.18,
    "win_rate": 0.52,
    "profit_factor": 1.5,
    "oos_period_days": 800,
}

validation = validate_strategy(metrics)

if validation.passed:
    print(f"✅ Strategy validated! Confidence: {validation.confidence_score:.2f}")
    print(f"   Passed: {validation.passed_criteria}")
else:
    print(f"❌ Validation failed")
    print(f"   Failed: {validation.failed_criteria}")
    for criterion in validation.failed_criteria:
        print(f"   - {criterion}")
```

**Key Points:**
- TestSetGuard enforces 3-evaluation limit per hypothesis (prevents data snooping)
- All evaluations logged to database with timestamp and metadata
- Override mechanism requires explicit justification (tracked for audit)
- Integrated with PlatformAPI validation gates
- Validation criteria: Sharpe ≥0.5, 100+ trades, drawdown ≤25%, win rate ≥48%

---

## 6. Data Quality Monitoring

### 6.1 Run Quality Checks

```python
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()

# Run all quality checks
result = api.run_quality_checks(
    as_of_date=date.today(),
    send_alerts=True  # Send email if issues found
)

print(f"Health Score: {result['health_score']}/100")
print(f"Critical Issues: {result['critical_issues']}")
print(f"Warning Issues: {result['warning_issues']}")
print(f"Passed: {result['passed']}")
```

**Output:**
```
Health Score: 95/100
Critical Issues: 0
Warning Issues: 2
Passed: True
```

**Available Checks:**
- **Price Anomaly Check**: Detects >50% price moves without corporate actions
- **Completeness Check**: Identifies missing prices for active symbols
- **Gap Detection Check**: Finds missing trading days in price history
- **Stale Data Check**: Flags symbols not updated in 3+ days
- **Volume Anomaly Check**: Detects zero volume or 10x+ average volume

### 6.2 Check Specific Quality Aspects

```python
from hrp.data.quality.checks import (
    PriceAnomalyCheck,
    CompletenessCheck,
    GapDetectionCheck,
    StaleDataCheck,
    VolumeAnomalyCheck
)
from hrp.data.db import DatabaseManager
from datetime import date

db = DatabaseManager()

# Check for price anomalies (>50% moves)
price_check = PriceAnomalyCheck(db)
result = price_check.run(as_of_date=date.today())
print(f"Price anomalies found: {len(result.issues)}")
for issue in result.issues[:3]:
    print(f"  - {issue.symbol}: {issue.message}")

# Check for missing data
completeness_check = CompletenessCheck(db)
result = completeness_check.run(as_of_date=date.today())
print(f"\nMissing data issues: {len(result.issues)}")

# Check for stale data (not updated in 3+ days)
stale_check = StaleDataCheck(db, max_stale_days=3)
result = stale_check.run(as_of_date=date.today())
print(f"\nStale data issues: {len(result.issues)}")
```

### 6.3 Generate Quality Report

```python
from hrp.data.quality.report import QualityReportGenerator
from datetime import date

# Generate comprehensive report
generator = QualityReportGenerator()
report = generator.generate_report(as_of_date=date.today())

print(f"Report generated: {report.generated_at}")
print(f"Health Score: {report.health_score}")
print(f"Checks Run: {report.checks_run}")
print(f"Checks Passed: {report.checks_passed}")

print("\nIssues by Severity:")
print(f"  Critical: {report.critical_issues}")
print(f"  Warning: {report.warning_issues}")
print(f"  Info: {report.info_issues}")

# Store report in database for historical tracking
report_id = generator.store_report(report)
print(f"\nStored as report_id: {report_id}")

# View health trend over time
trend = generator.get_health_trend(days=90)
print(f"\n90-day health trend: {len(trend)} reports")
```

### 6.4 Backup & Restore Operations

```bash
# Create a backup (database + MLflow)
python -m hrp.data.backup --backup

# Verify backup integrity
python -m hrp.data.backup --verify ~/hrp-data/backups/backup_20260124_120000

# List all backups
python -m hrp.data.backup --list

# Restore from backup
python -m hrp.data.backup --restore ~/hrp-data/backups/backup_20260124_120000 \
    --target-dir ~/hrp-data-restored

# Rotate old backups (keep 30 days)
python -m hrp.data.backup --rotate --keep-days 30
```

**Programmatic Backup:**

```python
from hrp.data.backup import create_backup, verify_backup, restore_backup

# Create backup
result = create_backup(include_mlflow=True)
print(f"Backup created: {result['path']}")
print(f"Size: {result['size_mb']} MB")
print(f"Files: {len(result['files'])}")

# Verify it
if verify_backup(result['path']):
    print("Backup verified successfully")
    
# Restore if needed
restore_backup(backup_path=result['path'], target_dir="~/hrp-data-restored")
```

**Automated Backups:**

Backups are automatically scheduled when using the scheduler:

```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()

# Setup weekly backup at 2 AM Saturday, keep 30 days
scheduler.setup_weekly_backup(
    backup_time='02:00',
    day_of_week='sat',
    keep_days=30,
    include_mlflow=True
)

scheduler.start()
```

### 6.5 Historical Data Backfill

```bash
# Backfill prices for specific symbols
python -m hrp.data.backfill --symbols AAPL MSFT GOOGL \
    --start 2020-01-01 --end 2023-12-31 --prices

# Backfill entire S&P 500 universe
python -m hrp.data.backfill --universe \
    --start 2015-01-01 --all

# Resume from previous progress
python -m hrp.data.backfill --resume backfill_progress_20260124.json \
    --prices --features

# Validate backfill completeness
python -m hrp.data.backfill --symbols AAPL MSFT \
    --start 2020-01-01 --end 2023-12-31 --validate
```

**Programmatic Backfill:**

```python
from hrp.data.backfill import backfill_prices, backfill_features, validate_backfill
from datetime import date
from pathlib import Path

# Backfill prices with progress tracking
result = backfill_prices(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date(2020, 1, 1),
    end=date(2023, 12, 31),
    source='yfinance',
    batch_size=10,
    progress_file=Path('backfill_progress.json')
)

print(f"Success: {result['symbols_success']}/{result['symbols_requested']}")
print(f"Rows inserted: {result['rows_inserted']}")

# Backfill features
result = backfill_features(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date(2020, 1, 1),
    end=date(2023, 12, 31),
    batch_size=10
)

# Validate completeness
validation = validate_backfill(
    symbols=['AAPL', 'MSFT'],
    start=date(2020, 1, 1),
    end=date(2023, 12, 31),
    check_features=True
)

print(f"Valid: {validation['is_valid']}")
print(f"Complete symbols: {validation['symbols_complete']}")
if validation['gaps']:
    print(f"Symbols with gaps: {list(validation['gaps'].keys())}")
```

---

## 7. Scheduled Jobs & Automation

### 7.1 Run Jobs Manually

```bash
# Run price ingestion job
python -m hrp.agents.cli run-now --job prices

# Run universe update job
python -m hrp.agents.cli run-now --job universe

# Run feature computation job
python -m hrp.agents.cli run-now --job features

# Run for specific symbols (prices only)
python -m hrp.agents.cli run-now --job prices --symbols AAPL MSFT GOOGL
```

**What each job does:**
- **prices**: Fetches daily OHLCV data from Yahoo Finance or Polygon.io
- **universe**: Updates S&P 500 constituents from Wikipedia, applies exclusion rules
- **features**: Computes technical indicators from price data

### 7.2 Set Up Daily Ingestion

#### Option A: Background Service (Production - macOS)

**Step 1: Create launchd service file**

Create `~/Library/LaunchAgents/com.hrp.scheduler.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hrp.scheduler</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/your/.venv/bin/python</string>
        <string>/path/to/your/HRP/run_scheduler.py</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>/path/to/your/HRP</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <true/>
    
    <key>StandardOutPath</key>
    <string>/Users/your-username/hrp-data/logs/scheduler.log</string>
    
    <key>StandardErrorPath</key>
    <string>/Users/your-username/hrp-data/logs/scheduler.error.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>HRP_DB_PATH</key>
        <string>/Users/your-username/hrp-data/hrp.duckdb</string>
    </dict>
</dict>
</plist>
```

**Step 2: Load and start the service**

```bash
# Load the service (starts automatically)
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Check status
launchctl list | grep hrp

# View logs
tail -f ~/hrp-data/logs/scheduler.error.log
```

**Management commands:**

```bash
# Stop service
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Start service
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# View logs
tail -f ~/hrp-data/logs/scheduler.error.log

# Check job history
python -m hrp.agents.cli job-status
```

**Benefits:**
- ✅ Auto-starts on login
- ✅ Auto-restarts if crashed
- ✅ Runs in background
- ✅ Survives terminal closure
- ✅ Survives reboots

#### Option B: Foreground (Testing/Development)

```bash
# Run with default settings
# - Prices: 6:00 PM ET
# - Universe: 6:05 PM ET
# - Features: 6:10 PM ET
# - Backup: 2:00 AM ET
python run_scheduler.py

# Custom times
python run_scheduler.py \
    --price-time 18:00 \
    --universe-time 18:05 \
    --feature-time 18:10 \
    --backup-time 02:00

# Disable backup
python run_scheduler.py --no-backup

# Custom symbols (prices only, universe/features use all DB symbols)
python run_scheduler.py --symbols AAPL MSFT GOOGL
```

**Daily Pipeline Flow:**
```
18:00 ET → Price Ingestion
             ↓
18:05 ET → Universe Update (S&P 500 changes)
             ↓
18:10 ET → Feature Computation
             ↓
02:00 ET → Backup (next day)
```

**Event-Driven Research Pipeline:**

```bash
# Enable full autonomous research pipeline
python -m hrp.agents.run_scheduler \
    --with-research-triggers \
    --with-signal-scan \
    --with-quality-sentinel \
    --with-daily-report \
    --with-weekly-report

# This enables:
# - Lineage event polling (every 60s by default)
# - Weekly signal scan (Monday 7 PM ET)
# - Daily ML Quality Sentinel audit (6 AM ET)
# - Daily research report (7 AM ET)
# - Weekly research report (Sunday 8 PM ET)
# - Automatic agent chaining:
#   Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel
```

**Research Agent CLI Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--with-research-triggers` | off | Enable event-driven agent pipeline |
| `--trigger-poll-interval` | 60 | Lineage event poll interval (seconds) |
| `--with-signal-scan` | off | Enable weekly signal scan |
| `--signal-scan-time` | 19:00 | Time for signal scan (HH:MM) |
| `--signal-scan-day` | mon | Day for signal scan |
| `--ic-threshold` | 0.03 | Minimum IC to create hypothesis |
| `--with-quality-sentinel` | off | Enable daily ML Quality Sentinel |
| `--sentinel-time` | 06:00 | Time for quality sentinel |
| `--with-daily-report` | off | Enable daily research report (7 AM ET) |
| `--daily-report-time` | 07:00 | Time for daily report (HH:MM) |
| `--with-weekly-report` | off | Enable weekly research report (Sunday 8 PM ET) |
| `--weekly-report-time` | 20:00 | Time for weekly report (HH:MM) |

**Advanced Options:**

```bash
# Keep 60 days of backups
python run_scheduler.py --backup-keep-days 60

# Custom signal scan schedule (Tuesday 8 PM, IC > 0.05)
python -m hrp.agents.run_scheduler \
    --with-signal-scan \
    --signal-scan-day tue \
    --signal-scan-time 20:00 \
    --ic-threshold 0.05
```

#### Option C: Terminal Session (Alternative Background)

**Using nohup:**

```bash
cd /path/to/HRP
nohup python run_scheduler.py > ~/hrp-data/logs/scheduler.log 2>&1 &

# Get process ID
echo $!

# Check if running
ps aux | grep run_scheduler

# Kill it
kill <PID>
```

**Using screen:**

```bash
# Start screen session
screen -S hrp-scheduler

# Run scheduler
python run_scheduler.py

# Detach: Ctrl+A then D

# Reattach later
screen -r hrp-scheduler

# Kill session
screen -X -S hrp-scheduler quit
```

**Using tmux:**

```bash
# Start tmux session
tmux new -s hrp-scheduler

# Run scheduler
python run_scheduler.py

# Detach: Ctrl+B then D

# Reattach later
tmux attach -t hrp-scheduler

# Kill session
tmux kill-session -t hrp-scheduler
```

#### Option D: Programmatic (Custom Script)

```python
from hrp.agents.scheduler import IngestionScheduler
import signal

scheduler = IngestionScheduler()

# Set up daily jobs (US/Eastern timezone)
scheduler.setup_daily_ingestion(
    symbols=None,  # None = all universe symbols
    price_job_time='18:00',      # 6:00 PM ET (after market close)
    universe_job_time='18:05',   # 6:05 PM ET (after prices loaded)
    feature_job_time='18:10'     # 6:10 PM ET (after universe updated)
)

# Set up weekly backup
scheduler.setup_weekly_backup(
    backup_time='02:00',  # 2 AM ET
    day_of_week='sat',    # Saturday
    keep_days=30,
    include_mlflow=True,
)

# Start the scheduler
scheduler.start()

# The scheduler runs in the background
# Keep running until Ctrl+C
signal.pause()
```

### 7.3 View Job Status

```bash
# List all scheduled jobs
python -m hrp.agents.cli list-jobs

# View job execution history
python -m hrp.agents.cli job-status

# View status for specific job
python -m hrp.agents.cli job-status --job-id price_ingestion --limit 10
```

**Output:**
```
Scheduled Jobs:
  price_ingestion: Daily at 18:00 ET (next: 2025-01-23 18:00:00)
  universe_update: Daily at 18:05 ET (next: 2025-01-23 18:05:00)
  feature_computation: Daily at 18:10 ET (next: 2025-01-23 18:10:00)

Recent Executions:
  2025-01-22 18:00:05 | price_ingestion | SUCCESS | 450 symbols, 450 rows
  2025-01-22 18:05:08 | universe_update | SUCCESS | 503 constituents, 380 included
  2025-01-22 18:10:12 | feature_computation | SUCCESS | 3600 features
  2025-01-21 18:00:03 | price_ingestion | SUCCESS | 450 symbols, 450 rows
```

### 7.4 Run Jobs Programmatically

```python
from hrp.agents.jobs import PriceIngestionJob, FeatureComputationJob, UniverseUpdateJob
from datetime import date, timedelta

# Run price ingestion
price_job = PriceIngestionJob(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date.today() - timedelta(days=7),
    end=date.today()
)
result = price_job.run()
print(f"Status: {result['status']}")
print(f"Records inserted: {result['records_inserted']}")

# Run universe update
universe_job = UniverseUpdateJob(
    as_of_date=date.today(),
    actor="user:manual"
)
result = universe_job.run()
print(f"Total constituents: {result['records_fetched']}")
print(f"Included in universe: {result['records_inserted']}")
print(f"Added: {result['symbols_added']}, Removed: {result['symbols_removed']}")

# Run feature computation (depends on prices being loaded)
feature_job = FeatureComputationJob(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date.today() - timedelta(days=30),
    end=date.today()
)
result = feature_job.run()
print(f"Features computed: {result['features_computed']}")
```

### 7.5 Clear Job History

```bash
# Clear all history older than 30 days
python -m hrp.agents.cli clear-history --before 2025-01-01 --confirm

# Clear history for specific job
python -m hrp.agents.cli clear-history --job-id price_ingestion --confirm

# Clear only failed jobs
python -m hrp.agents.cli clear-history --status FAILED --confirm
```

### 7.6 Generate Research Reports

The Report Generator agent synthesizes research findings into human-readable daily and weekly summaries.

**Generate a Daily Report:**

```python
from hrp.agents.report_generator import ReportGenerator

# Generate a daily research report
daily_generator = ReportGenerator(report_type="daily")
result = daily_generator.execute()

print(f"Report: {result['report_path']}")
print(f"Tokens: {result['token_usage']['total']}")
print(f"Cost: ${result['token_usage']['estimated_cost_usd']:.4f}")

# Report includes:
# - Executive summary (hypotheses created, experiments completed, best model)
# - Hypothesis pipeline (draft, testing, validated, deployed)
# - Experiment results (top 3 with Sharpe ratios)
# - Signal analysis (best validated signals with IC)
# - Actionable insights (Claude-powered or fallback)
# - Agent activity summary (all 5 research agents)
```

**Generate a Weekly Report:**

```python
from hrp.agents.report_generator import ReportGenerator, ReportGeneratorConfig

# Generate a weekly research report with custom config
weekly_config = ReportGeneratorConfig(
    report_type="weekly",
    report_dir="docs/reports",
    lookback_days=7,
)
weekly_generator = ReportGenerator(
    report_type="weekly",
    config=weekly_config,
)
result = weekly_generator.execute()

# Weekly report includes additional sections:
# - Week at a Glance overview
# - Pipeline Velocity visualization
# - Top Hypotheses This Week
# - Model Performance comparison
# - Signal Discoveries summary
```

**Report Output Location:**

```
docs/reports/
└── 2026-01-26/
    ├── 2026-01-26-07-00-daily.md
    └── 2026-01-26-20-00-weekly.md
```

**Schedule Automated Reports:**

```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()

# Daily report at 7 AM ET (before market open)
scheduler.setup_daily_report(
    report_time='07:00',
    report_type='daily',
)

# Weekly report on Friday 6 PM ET (end of trading week)
scheduler.setup_weekly_report(
    report_time='18:00',
    day_of_week='fri',
)

scheduler.start()
```

**Report Generator Features:**
- **Data Aggregation**: Gathers data from hypotheses, MLflow experiments, lineage table, and signal discoveries
- **Claude-Powered Insights**: AI-generated action items with fallback to rule-based logic
- **Agent Tracking**: Monitors status of all research agents (Signal Scientist, Alpha Researcher, ML Scientist, ML Quality Sentinel, Validation Analyst)
- **Token Tracking**: Cost estimation for Claude API usage
- **Flexible Rendering**: Markdown output with timestamped files in dated folders

**Run Report Generator Manually:**

```bash
# Generate daily report
python -m hrp.agents.cli run-now --job report_generator --report-type daily

# Generate weekly report
python -m hrp.agents.cli run-now --job report_generator --report-type weekly
```

### 7.7 System Startup Script

The `startup.sh` script provides a convenient way to manage all HRP services from a single interface.

**Available Services:**
- **Dashboard** (Streamlit) - http://localhost:8501
- **MLflow UI** - http://localhost:5010
- **Scheduler** - Background data ingestion and research agents

**Start All Services:**

```bash
# Start all core services (dashboard, MLflow, scheduler)
./scripts/startup.sh start

# Start with all research agents enabled
./scripts/startup.sh start --full

# Start in minimal mode (ingestion only, no backup/fundamentals)
./scripts/startup.sh start --minimal
```

**Start Individual Services:**

```bash
# Dashboard only
./scripts/startup.sh start --dashboard-only

# MLflow UI only
./scripts/startup.sh start --mlflow-only

# Scheduler only
./scripts/startup.sh start --scheduler-only
```

**Service Management:**

```bash
# Check service status
./scripts/startup.sh status

# Stop all services
./scripts/startup.sh stop

# Restart all services
./scripts/startup.sh restart

# Show help
./scripts/startup.sh --help
```

**Environment Variables:**

```bash
# Custom ports
HRP_DASHBOARD_PORT=8080 ./scripts/startup.sh start
HRP_MLFLOW_PORT=5001 ./scripts/startup.sh start

# Custom scheduler times
HRP_PRICE_TIME=17:30 ./scripts/startup.sh start
HRP_UNIVERSE_TIME=17:35 ./scripts/startup.sh start
HRP_FEATURE_TIME=17:40 ./scripts/startup.sh start

# Disable backup or fundamentals
HRP_NO_BACKUP=true ./scripts/startup.sh start
HRP_NO_FUNDAMENTALS=true ./scripts/startup.sh start
```

**Full Mode (--full):**

Enables the complete research agent pipeline:
- Event-driven triggers (Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel)
- Weekly signal scan (Monday 7 PM ET)
- Daily ML Quality Sentinel (6 AM ET)
- Daily research reports (7 AM ET)
- Weekly research reports (Sunday 8 PM ET)

**Status Output Example:**

```
========================================
  HRP System Manager
========================================

  Dashboard:      RUNNING (PID: 12345, Port: 8501)
  MLflow UI:      RUNNING (PID: 12346, Port: 5010)
  Scheduler:      RUNNING (PID: 12347)

  Total: 3 running, 0 stopped

  Logs directory: /Users/your-username/hrp-data/logs
  PID directory:  /path/to/HRP/.hrp_pids
```

**Troubleshooting Port Conflicts:**

The startup script automatically detects port conflicts and provides helpful error messages:

```
[ERROR] Port 5010 is already in use (PID: 419)
[INFO] Try: HRP_MLFLOW_PORT=5011 ./scripts/startup.sh start --mlflow-only
```

**Common Port Conflicts:**

| Port | Typical Conflict | Solution |
|------|------------------|----------|
| 5000 | macOS ControlCenter (AirPlay Receiver) | Default changed to 5010, or use `HRP_MLFLOW_PORT=5011` |
| 8501 | Previous dashboard instance | Run `./scripts/startup.sh stop` first, or kill old process |
| 5010 | Another MLflow instance | Use `HRP_MLFLOW_PORT=5011 ./scripts/startup.sh start` |

**Check what's using a port:**

```bash
# See which process is using a port
lsof -i :5010

# Kill a process by PID
kill <PID>
```

**Note:** The startup script does **not** initialize the database. The database is created automatically on first access by the services. For fresh installations, run the initial setup first (see Section 1.1).

---

## 8. Using the Dashboard

### 8.1 Start the Dashboard

**Option A: Using the startup script (recommended)**

```bash
# Start dashboard along with MLflow and scheduler
./scripts/startup.sh start

# Or start dashboard only
./scripts/startup.sh start --dashboard-only
```

**Option B: Manual start**

```bash
# Start Streamlit dashboard manually
streamlit run hrp/dashboard/app.py

# Access at http://localhost:8501
```

### 8.2 Dashboard Pages

| Page | Purpose | Key Features |
|------|---------|--------------|
| **Home** | System overview | Health status, recent activity, quick stats |
| **Data Health** | Data quality | Quality scores, issue summary, trends |
| **Ingestion Status** | Job monitoring | Job history, next runs, failure alerts |
| **Hypotheses** | Research management | Create, view, update hypotheses |
| **Experiments** | Backtest results | View metrics, compare experiments, MLflow link |
| **Pipeline Progress** | Agent pipeline | Kanban view of hypothesis stages, agent launcher |
| **Agents Monitor** | Agent status | Real-time agent status, historical timeline |
| **Job Health** | Job execution | Job health metrics, error tracking |
| **Ops** | System monitoring | CPU/memory/disk, alert thresholds |
| **Trading** | Live execution | Portfolio overview, positions, trades, drift status |
| **Risk Metrics** | VaR/CVaR analysis | Portfolio VaR, per-symbol breakdown, method comparison |
| **Performance Attribution** | Strategy analysis | Brinson-Fachler, factor contributions, feature importance |
| **Backtest Performance** | Backtest visualization | Equity curves, drawdowns, rolling metrics, exports |

### 8.3 Start MLflow UI

**Option A: Using the startup script (recommended)**

```bash
# Start MLflow along with dashboard and scheduler
./scripts/startup.sh start

# Or start MLflow only
./scripts/startup.sh start --mlflow-only
```

**Option B: Manual start**

```bash
# Start MLflow UI for detailed experiment tracking
mlflow ui --backend-store-uri sqlite:///$HOME/hrp-data/mlflow/mlflow.db

# Access at http://localhost:5000
```

### 8.4 Dashboard Authentication

The dashboard uses bcrypt password hashing with session cookies. Manage users via the auth CLI.

**List users:**

```bash
python -m hrp.dashboard.auth_cli list-users
```

**Add a user:**

```bash
python -m hrp.dashboard.auth_cli add-user \
    --username admin \
    --email admin@example.com \
    --name "Admin User"
# You'll be prompted for the password
```

**Reset a password:**

```bash
python -m hrp.dashboard.auth_cli reset-password --username admin
# You'll be prompted for the new password

# Or non-interactively:
python -m hrp.dashboard.auth_cli reset-password --username admin --password "newpass"
```

**Remove a user:**

```bash
python -m hrp.dashboard.auth_cli remove-user --username olduser
```

**Environment variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `HRP_AUTH_ENABLED` | Enable/disable auth | `true` |
| `HRP_AUTH_COOKIE_KEY` | Secret for session cookies (32+ chars) | Required |
| `HRP_AUTH_USERS_FILE` | Path to users YAML | `~/hrp-data/auth/users.yaml` |

---

## 9. Common Workflows

### 9.1 Complete Research Workflow

```python
from hrp.api.platform import PlatformAPI
from hrp.research.config import BacktestConfig
from hrp.ml import WalkForwardConfig, walk_forward_validate
from hrp.risk.validation import validate_strategy, ValidationCriteria
from datetime import date

api = PlatformAPI()

# Step 1: Create hypothesis
hypothesis_id = api.create_hypothesis(
    title="Momentum + Low Volatility Factor",
    thesis="Combining momentum with low volatility improves risk-adjusted returns",
    prediction="Sharpe > 0.8, max drawdown < 20%",
    falsification="Sharpe < 0.5 or drawdown > 25%",
    actor='user'
)

# Step 2: Run initial backtest
config = BacktestConfig(
    symbols=api.get_universe(date.today())[:50],  # Top 50 stocks
    start_date=date(2018, 1, 1),
    end_date=date(2023, 12, 31),
    name='momentum_low_vol_v1'
)

experiment_id = api.run_backtest(config, hypothesis_id=hypothesis_id)
experiment = api.get_experiment(experiment_id)

# Step 3: Move to testing if initial results promising
if experiment['metrics']['sharpe_ratio'] > 0.5:
    api.update_hypothesis(hypothesis_id, status='testing')

    # Step 4: Run walk-forward validation
    wf_config = WalkForwardConfig(
        model_type='ridge',
        target='returns_20d',
        features=['momentum_20d', 'volatility_20d'],
        start_date=date(2015, 1, 1),
        end_date=date(2023, 12, 31),
        n_folds=5,
        window_type='expanding'
    )

    wf_result = walk_forward_validate(wf_config, symbols=config.symbols)

    # Step 5: Statistical validation
    criteria = ValidationCriteria(min_sharpe=0.5, max_drawdown=0.25)
    validation = validate_strategy(returns, benchmark_returns, criteria)

    if validation.passed and wf_result.is_stable:
        api.update_hypothesis(
            hypothesis_id,
            status='validated',
            outcome=f"Passed validation. Stability={wf_result.stability_score:.2f}"
        )
        print("Hypothesis VALIDATED!")
    else:
        api.update_hypothesis(
            hypothesis_id,
            status='rejected',
            outcome="Failed validation criteria"
        )
        print("Hypothesis REJECTED")
```

### 9.2 Daily Monitoring Workflow

```python
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()

# 1. Check system health
health = api.health_check()
if health['status'] != 'healthy':
    print("WARNING: System health issue!")
    print(health)

# 2. Run data quality checks
quality = api.run_quality_checks(as_of_date=date.today())
print(f"Data Health Score: {quality['health_score']}/100")

if quality['critical_issues'] > 0:
    print(f"CRITICAL: {quality['critical_issues']} issues found!")

# 3. Check deployed strategies
deployed = api.get_deployed_strategies()
print(f"\nDeployed strategies: {len(deployed)}")
for strategy in deployed:
    print(f"  - {strategy['hypothesis_id']}: {strategy['title']}")

# 4. Review recent agent activity
lineage = api.get_lineage(limit=20)
agent_actions = [e for e in lineage if e['actor'].startswith('agent:')]
print(f"\nRecent agent actions: {len(agent_actions)}")
```

### 9.3 Backfill Historical Data

```python
from hrp.data.ingestion.prices import ingest_prices
from hrp.data.ingestion.features import compute_features
from datetime import date
import time

# Get universe symbols
api = PlatformAPI()
symbols = api.get_universe(date.today())

# Backfill in batches to respect rate limits
batch_size = 10
for i in range(0, len(symbols), batch_size):
    batch = symbols[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}: {batch}")

    # Ingest prices
    result = ingest_prices(
        symbols=batch,
        start=date(2015, 1, 1),
        end=date.today(),
        source='yfinance'
    )
    print(f"  Prices: {result['rows_inserted']} rows")

    # Compute features
    result = compute_features(
        symbols=batch,
        start=date(2015, 1, 1),
        end=date.today()
    )
    print(f"  Features: {result['rows_inserted']} rows")

    # Rate limiting
    time.sleep(2)

print("Backfill complete!")
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Database Connection Errors

```python
# Check database path
import os
db_path = os.environ.get('HRP_DB_PATH', '~/hrp-data/hrp.duckdb')
print(f"Database path: {os.path.expanduser(db_path)}")

# Verify database exists
if os.path.exists(os.path.expanduser(db_path)):
    print("Database file exists")
else:
    print("Database file NOT FOUND - run initial setup")
```

#### Missing Features

```python
# Check what features are available
from hrp.data.db import DatabaseManager

db = DatabaseManager()
with db.get_connection() as conn:
    features = conn.execute("""
        SELECT DISTINCT feature_name
        FROM features
        ORDER BY feature_name
    """).fetchall()

print("Available features:")
for f in features:
    print(f"  - {f[0]}")
```

#### Symbol Not Found

```python
# Check if symbol exists in universe
from hrp.data.db import DatabaseManager
from datetime import date

db = DatabaseManager()
with db.get_connection() as conn:
    result = conn.execute("""
        SELECT symbol, date, in_universe
        FROM universe
        WHERE symbol = 'AAPL'
        ORDER BY date DESC
        LIMIT 5
    """).fetchall()

print("Symbol history:")
for r in result:
    print(f"  {r[0]} | {r[1]} | in_universe={r[2]}")
```

### 10.2 Reset Database

```bash
# WARNING: This deletes all data!
rm ~/hrp-data/hrp.duckdb

# Recreate schema
python -c "from hrp.data.schema import create_schema; create_schema()"

# Re-ingest data
python -m hrp.data.ingestion.prices --symbols AAPL MSFT GOOGL --start 2020-01-01
```

### 10.3 View Logs

```bash
# Check ingestion logs
tail -f ~/hrp-data/logs/ingestion.log

# Check agent logs
tail -f ~/hrp-data/logs/agents.log
```

### 10.4 Verify MLflow Setup

```python
import mlflow
from hrp.research.mlflow_utils import setup_mlflow

# Setup MLflow
setup_mlflow()

# List experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"  {exp.experiment_id}: {exp.name}")

# Count runs
runs = mlflow.search_runs(experiment_names=['backtests'])
print(f"\nTotal backtest runs: {len(runs)}")
```

### 10.5 Test Suite Known Issues

**FK Constraint Violations in Test Fixtures**

The test suite currently has ~97.6% pass rate (1,204/1,234 tests). The remaining 29 failures are tests that expect FK constraints which were intentionally removed from the schema due to DuckDB 1.4.3 limitations. This is a test expectation issue, not a production code bug.

**Symptoms:**
```
Constraint Error: Violates foreign key constraint because key "hypothesis_id: HYP-2026-001" 
is still referenced by a foreign key
```

**Root Cause:** Test fixtures attempt to delete parent records (hypotheses) that still have dependent records (lineage events, experiments).

**Workaround:** Tests still validate functionality correctly; the errors occur only during cleanup.

**Fix (pending):** 
- Option 1: Add `ON DELETE CASCADE` to FK relationships in schema
- Option 2: Update test fixtures to delete dependent records before parents

**Impact:** Production code is unaffected. All core functionality is operational.

---

## 11. Claude Integration (MCP Server)

HRP includes an MCP (Model Context Protocol) server that enables Claude to interact with the platform programmatically.

### 11.1 Configure Claude Desktop

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hrp-research": {
      "command": "python",
      "args": ["-m", "hrp.mcp"],
      "env": {
        "HRP_DB_PATH": "/Users/your-username/hrp-data/hrp.duckdb"
      }
    }
  }
}
```

Restart Claude Desktop after adding this configuration.

### 11.2 Available MCP Tools (22 total)

| Category | Tools | Description |
|----------|-------|-------------|
| **Hypothesis** | `list_hypotheses`, `get_hypothesis`, `create_hypothesis`, `update_hypothesis`, `get_experiments_for_hypothesis` | Manage research hypotheses |
| **Data Access** | `get_universe`, `get_features`, `get_prices`, `get_available_features`, `is_trading_day` | Access market data |
| **Backtesting** | `run_backtest`, `get_experiment`, `compare_experiments`, `analyze_results` | Run and analyze backtests |
| **ML Training** | `run_walk_forward_validation`, `get_supported_models`, `train_ml_model` | Train ML models |
| **Quality** | `run_quality_checks`, `get_health_status`, `get_data_coverage` | Monitor data quality |
| **Lineage** | `get_lineage`, `get_deployed_strategies` | Audit trail and deployments |

### 11.3 Example Claude Interactions

**List hypotheses:**
> "Show me all hypotheses in testing status"

Claude will call `list_hypotheses(status='testing')` and format the results.

**Create a hypothesis:**
> "Create a hypothesis testing whether RSI < 30 predicts positive 5-day returns"

Claude will call `create_hypothesis()` with appropriate parameters.

**Run a backtest:**
> "Run a backtest for AAPL, MSFT, GOOGL from 2020 to 2023 linked to hypothesis HYP-2026-001"

Claude will call `run_backtest()` and return the results.

**Analyze results:**
> "Analyze the results of experiment abc123"

Claude will call `analyze_results()` and provide a human-readable interpretation.

### 11.4 Security Model

- **All MCP calls are logged** with actor `agent:claude-interactive`
- **Agents cannot deploy strategies** — `approve_deployment` is not exposed
- **Full audit trail** in the lineage table

### 11.5 Start MCP Server Manually (for testing)

```bash
# Start the server (connects via stdio)
python -m hrp.mcp

# Or import and run programmatically
python -c "from hrp.mcp import mcp; mcp.run()"
```

---

## 12. Ops Server & Monitoring

The HRP Ops Server provides health endpoints, Prometheus metrics, and system monitoring for production deployments.

**Quick Start:**

```bash
# Start ops server (default: 0.0.0.0:8080)
python -m hrp.ops

# Verify it's running
curl http://localhost:8080/health
```

**Key Endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `/health` | Liveness probe - confirms server is running |
| `/ready` | Readiness probe - verifies database and API connectivity |
| `/metrics` | Prometheus metrics for scraping |

**Full Documentation:**

- **[Ops Server Guide](ops-server.md)** - Complete ops server documentation including:
  - Health endpoint details and response formats
  - Prometheus metrics reference and example queries
  - Programmatic usage with `MetricsCollector`
  - Running as a launchd service
  - Troubleshooting guide

- **[Alert Thresholds Guide](alert-thresholds.md)** - Configurable monitoring thresholds including:
  - Health score thresholds (warning/critical levels)
  - Data freshness thresholds
  - Anomaly detection thresholds
  - Model drift thresholds (PSI, KL divergence, IC decay)
  - Configuration via YAML or environment variables

---

## 13. Trading Execution (Tier 4)

### 13.1 Prerequisites

Before enabling trading:
1. Complete Tier 1-3 setup
2. Have at least one deployed strategy
3. Choose a broker and configure credentials

**Broker selection:**

```bash
# IBKR (default) — requires TWS/IB Gateway running
export HRP_BROKER_TYPE=ibkr
# See docs/operations/ibkr-setup-guide.md

# Robinhood — requires OAuth + MFA
export HRP_BROKER_TYPE=robinhood
export ROBINHOOD_USERNAME=your_email
export ROBINHOOD_PASSWORD=your_password
export ROBINHOOD_MFA_SECRET=your_totp_secret
```

### 13.2 Generate Predictions

```bash
# Test prediction job
python -m hrp.agents.run_job --job predictions --dry-run

# Run predictions for deployed models
python -m hrp.agents.run_job --job predictions
```

### 13.3 Execute Trades (Dry Run)

```bash
# Dry run - see what trades would execute
python -m hrp.agents.run_job --job live-trader --trading-dry-run

# Execute trades (DANGEROUS - only after testing)
python -m hrp.agents.run_job --job live-trader --execute-trades
```

### 13.4 Monitor Drift

```bash
# Check for model drift
python -m hrp.agents.run_job --job drift-monitor --dry-run

# With auto-rollback on drift detection
python -m hrp.agents.run_job --job drift-monitor --auto-rollback
```

### 13.5 View Positions and Trades

```python
from hrp.api.platform import PlatformAPI

api = PlatformAPI()

# Get current positions
positions = api.get_live_positions()
print(f"Positions: {len(positions)}")
print(positions)

# Get recent trades
trades = api.get_executed_trades(limit=10)
print(f"Recent trades: {len(trades)}")

# Portfolio value
value = api.get_portfolio_value()
print(f"Portfolio value: ${value:,.2f}")
```

### 13.6 Trading Dashboard

Access the Trading dashboard page at http://localhost:8501 (after starting the dashboard):
- Portfolio overview (value, P&L, positions)
- Current positions with market values
- Recent trades history
- Model drift status

### 13.7 Safety Features

Trading has multiple safety layers:
- **Dry-run mode**: Default, simulates trades without execution
- **Position limits**: Max 20 positions, 10% max per position
- **Minimum order**: $100 minimum order value
- **Drift monitoring**: Automatic monitoring before execution
- **Paper trading**: Default broker configuration (IBKR only)
- **VaR-aware sizing**: Constrain positions by VaR budget (see Section 14)
- **Auto stop-loss**: Generate stop-loss orders for all new positions
- **Rate limiting**: Robinhood broker enforces 5 req/15s with 2s order cooldown

---

## 14. Advanced Analytics (Tier 5)

### 14.1 VaR/CVaR Risk Metrics

```python
from hrp.risk.var_calculator import VaRCalculator, VaRConfig
from hrp.risk.risk_config import VaRMethod

# Calculate VaR using historical simulation
calculator = VaRCalculator()
config = VaRConfig(
    confidence_level=0.95,
    time_horizon=1,       # 1-day VaR
    method=VaRMethod.HISTORICAL,
)

result = calculator.calculate(returns=daily_returns, config=config)
print(f"VaR (95%, 1d):  {result.var:.4f}")
print(f"CVaR (95%, 1d): {result.cvar:.4f}")
print(f"Dollar VaR:     ${result.dollar_var:,.0f}")
```

**Available methods:** `PARAMETRIC` (normal/t-dist), `HISTORICAL`, `MONTE_CARLO`

See `docs/operations/var-risk-metrics.md` for the full guide.

### 14.2 VaR-Aware Position Sizing

```bash
# Enable VaR constraints for trading
export HRP_USE_VAR_SIZING=true
export HRP_MAX_PORTFOLIO_VAR_PCT=0.02    # 2% portfolio VaR limit
export HRP_MAX_POSITION_VAR_PCT=0.005    # 0.5% per-position limit
export HRP_AUTO_STOP_LOSS_PCT=0.05       # 5% auto stop-loss
```

### 14.3 Performance Attribution

```python
from hrp.research.attribution.factor_attribution import BrinsonAttribution, FactorAttribution

# Brinson-Fachler attribution (sector allocation vs selection)
brinson = BrinsonAttribution()
result = brinson.analyze(
    portfolio_returns=portfolio_df,
    benchmark_returns=benchmark_df,
    sector_weights=sector_weights,
)
print(f"Allocation effect:  {result.allocation_effect:.4f}")
print(f"Selection effect:   {result.selection_effect:.4f}")
print(f"Interaction effect: {result.interaction_effect:.4f}")

# Factor regression attribution (Fama-French)
factor_attr = FactorAttribution()
result = factor_attr.analyze(
    portfolio_returns=returns,
    method='fama_french_3',
)
```

### 14.4 Feature Importance

```python
from hrp.research.attribution.feature_importance import FeatureImportance

importance = FeatureImportance()

# Permutation importance
result = importance.analyze(model=trained_model, X=X_test, y=y_test, method='permutation')

# SHAP importance (requires shap package)
result = importance.analyze(model=trained_model, X=X_test, y=y_test, method='shap')
```

### 14.5 Intraday Data (Real-Time)

```python
from hrp.api.platform import PlatformAPI
from datetime import datetime

api = PlatformAPI()

# Get intraday price bars (minute-level)
bars = api.get_intraday_prices(
    symbols=['AAPL', 'MSFT'],
    start_ts=datetime(2026, 2, 11, 9, 30),
    end_ts=datetime(2026, 2, 11, 16, 0),
)

# Get intraday features (7 available)
features = api.get_intraday_features(
    symbols=['AAPL'],
    start_ts=datetime(2026, 2, 11, 9, 30),
    end_ts=datetime(2026, 2, 11, 16, 0),
    feature_names=['intraday_vwap', 'intraday_rsi_14', 'intraday_momentum_20'],
)
```

**Available intraday features:** `intraday_vwap`, `intraday_rsi_14`, `intraday_momentum_20`, `intraday_volatility_20`, `intraday_volume_ratio`, `intraday_price_to_open`, `intraday_range_position`

**Requires:** `POLYGON_API_KEY` environment variable for Polygon.io WebSocket data.

### 14.6 NLP Sentiment Features

```python
from hrp.data.features.sentiment_features import get_sentiment_features

# Get sentiment features for a symbol
features = get_sentiment_features(symbol='AAPL')
# Returns: sentiment_score_10k, sentiment_score_10q, sentiment_score_8k,
#          sentiment_score_avg, sentiment_momentum, sentiment_category
```

**Requires:** `ANTHROPIC_API_KEY` for Claude API-based sentiment analysis.

---

## Quick Reference

### API Methods

| Method | Description |
|--------|-------------|
| `api.get_prices(symbols, start, end)` | Get OHLCV price data |
| `api.get_features(symbols, features, as_of_date)` | Get computed features |
| `api.get_universe(as_of_date)` | Get tradeable symbol list |
| `api.create_hypothesis(...)` | Create research hypothesis |
| `api.update_hypothesis(id, status, ...)` | Update hypothesis status |
| `api.list_hypotheses(status)` | List hypotheses |
| `api.run_backtest(config, ...)` | Execute backtest |
| `api.get_experiment(id)` | Get experiment results |
| `api.compare_experiments(ids)` | Compare multiple experiments |
| `api.get_lineage(hypothesis_id)` | Get audit trail |
| `api.run_quality_checks(date)` | Run data quality checks |
| `api.health_check()` | System health status |
| `api.get_fundamentals_as_of(symbols, metrics, as_of_date)` | Get point-in-time fundamentals |
| `api.get_intraday_prices(symbols, start_ts, end_ts)` | Get minute-level OHLCV bars |
| `api.get_intraday_features(symbols, start_ts, end_ts, features)` | Get intraday computed features |
| `api.get_live_positions()` | Current broker positions |
| `api.get_executed_trades(limit)` | Trade execution history |
| `api.get_portfolio_value()` | Portfolio metrics |


### Strategy Signal Generators

| Function | Description |
|----------|-------------|
| `generate_momentum_signals(prices, lookback, top_n)` | Simple momentum strategy |
| `generate_multifactor_signals(prices, feature_weights, top_n)` | Multi-factor with configurable weights |
| `generate_ml_predicted_signals(prices, model_type, features, ...)` | ML model predictions as signals |
| `predictions_to_signals(predictions, method, ...)` | Convert predictions to trading signals |

### CLI Commands

```bash
# Jobs
python -m hrp.agents.cli run-now --job prices
python -m hrp.agents.cli run-now --job features
python -m hrp.agents.cli list-jobs
python -m hrp.agents.cli job-status

# Services
streamlit run hrp/dashboard/app.py          # Dashboard (port 8501)
mlflow ui --backend-store-uri sqlite:///$HOME/hrp-data/mlflow/mlflow.db  # MLflow (port 5000)

# Testing
pytest tests/ -v
pytest tests/test_api/ -v  # Specific module
```

### File Locations

| Item | Location |
|------|----------|
| Database | `~/hrp-data/hrp.duckdb` |
| MLflow | `~/hrp-data/mlflow/` |
| Logs | `~/hrp-data/logs/` |
| Backups | `~/hrp-data/backups/` |

---

## Next Steps

After mastering these recipes, consider:

1. **Explore the Dashboard** - Visual interface for monitoring
2. **Read the Spec** - `docs/plans/2026-01-19-hrp-spec.md` for architecture details
3. **Review the Roadmap** - `docs/plans/Roadmap.md` for implementation status
4. **Run Tests** - `pytest tests/ -v` to understand test coverage
5. **Check MLflow** - Deep dive into experiment tracking
