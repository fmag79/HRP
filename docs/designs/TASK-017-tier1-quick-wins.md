# Design: TASK-017 Tier 1 Quick Wins

**Task:** TASK-017
**Author:** Athena
**Date:** 2026-02-13
**Status:** READY FOR IMPLEMENTATION
**Project:** ~/Projects/HRP

---

## Overview

Design and implement 5 high-impact quick wins identified in TASK-016 research:

1. **Factor Library Expansion** - Quality, Value, and Statistical factors
2. **NLP Sentiment Features** - SEC EDGAR integration with production pipeline
3. **Regime Switching Strategy** - HMM-based adaptive strategy switching
4. **Risk Limit UI** - Expose Risk Manager controls in dashboard
5. **Advanced Backtesting UI** - Expose optimization functionality via dashboard

These features leverage existing infrastructure patterns for maximum ROI with minimal risk.

---

## Current State

### Existing Infrastructure (What We Build On)

| Component | Location | Status |
|-----------|----------|--------|
| Feature Registry | `hrp/data/features/registry.py` | Production |
| Feature Computer | `hrp/data/features/computation.py` | Production (45 features) |
| Feature Ingestion | `hrp/data/ingestion/features.py` | Production |
| Fundamental Source | `hrp/data/sources/fundamental_source.py` | Production (YFinance) |
| SimFin Source | `hrp/data/sources/simfin_source.py` | Production |
| EDGAR Source | `hrp/data/sources/edgar_source.py` | Implemented, not in pipeline |
| Sentiment Analyzer | `hrp/data/sentiment_analyzer.py` | Implemented, not in pipeline |
| SEC Ingestion Job | `hrp/data/ingestion/sec_ingestion.py` | Implemented, not scheduled |
| Sentiment Features | `hrp/data/features/sentiment_features.py` | Defined, not integrated |
| HMM Regime Detector | `hrp/ml/regime.py` | Production (used by Validation Analyst) |
| Strategy System | `hrp/research/strategies.py` | Production (4 preset strategies) |
| Risk Manager Agent | `hrp/agents/risk_manager.py` | Production (4 risk checks) |
| Risk Limits | `hrp/risk/limits.py` | Production (PreTradeValidator) |
| Backtest Performance | `hrp/dashboard/pages/backtest_performance.py` | Production (TASK-009) |
| Optuna Optimization | `hrp/ml/optimization.py` | Production |
| Parameter Sweep | `hrp/research/parameter_sweep.py` | Skeleton only (NotImplementedError) |
| Platform API | `hrp/api/platform.py` | Production |
| Dashboard App | `hrp/dashboard/app.py` | Production (v1.6.0, 12+ pages) |
| Database Schema | `hrp/data/schema.py` | Production (13+ tables) |

---

## Feature 1: Factor Library Expansion

### Goal
Add 12-14 new factors across Quality, Value, and Statistical categories. Extends the current 45-feature library to ~58 features.

### New Factors

#### Quality Factors (4)
| Factor | Formula | Data Source | Notes |
|--------|---------|-------------|-------|
| `roe` | Net Income / Shareholders' Equity | SimFin `net_income` + `book_value` | Higher = better |
| `roa` | Net Income / Total Assets | SimFin `net_income` + `total_assets` | Higher = better |
| `fcf_yield` | Free Cash Flow / Market Cap | SimFin FCF + YFinance `market_cap` | Requires new SimFin metric |
| `earnings_quality` | Operating CF / Net Income | SimFin OCF + `net_income` | >1 = high quality |

#### Value Factors (4)
| Factor | Formula | Data Source | Notes |
|--------|---------|-------------|-------|
| `peg_ratio` | P/E / EPS Growth Rate | YFinance `pe_ratio` + SimFin EPS | Growth = YoY EPS change |
| `ev_revenue` | Enterprise Value / Revenue | YFinance EV + SimFin `revenue` | Lower = cheaper |
| `price_to_fcf` | Price / FCF per Share | YFinance price + SimFin FCF | Lower = cheaper |
| `earnings_yield` | 1 / P/E (or E/P) | YFinance `pe_ratio` (invert) | Higher = cheaper |

#### Statistical Factors (4)
| Factor | Formula | Data Source | Notes |
|--------|---------|-------------|-------|
| `autocorrelation_5d` | 5-day return autocorrelation | Prices (returns) | Negative = mean-reverting |
| `skewness_60d` | 60-day return skewness | Prices (returns) | Negative skew = crash risk |
| `kurtosis_60d` | 60-day return kurtosis | Prices (returns) | High = fat tails |
| `downside_vol_60d` | 60-day downside volatility | Prices (returns < 0) | Lower = better |

### Components Affected

- `hrp/data/features/computation.py` - Add 12 new `compute_*()` functions + register in `FEATURE_FUNCTIONS`
- `hrp/data/sources/simfin_source.py` - Extend `get_fundamentals()` to support new metrics: `operating_cash_flow`, `free_cash_flow`, `eps`
- `hrp/data/ingestion/fundamentals_timeseries.py` - Add new metric ingestion for FCF, OCF, EPS
- `hrp/data/schema.py` - No changes needed (fundamentals table supports arbitrary metrics)
- `CLAUDE.md` - Update feature list documentation

### New Factor Computation Pattern

All new factors follow the established pattern in `computation.py`:

```python
# Quality factors use fundamentals data (point-in-time correct)
def compute_roe(prices: pd.DataFrame) -> pd.DataFrame:
    """Return on Equity = Net Income / Book Value.

    Uses point-in-time fundamentals via PlatformAPI to prevent look-ahead bias.
    Forward-fills quarterly values to daily granularity.

    Args:
        prices: DataFrame with MultiIndex (date, symbol) and OHLCV columns.
    Returns:
        DataFrame with 'roe' column, same MultiIndex.
    """
    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()
    start, end = dates.min(), dates.max()

    # Point-in-time fundamentals (no look-ahead bias)
    net_income = api.get_fundamentals_as_of(symbols, ["net_income"], end)
    book_value = api.get_fundamentals_as_of(symbols, ["book_value"], end)

    # Merge and compute
    roe = net_income["net_income"] / book_value["book_value"]

    # Forward-fill to daily, align with price index
    result = roe.reindex(prices.index).ffill()
    api.close()
    return result.to_frame(name="roe")

# Statistical factors use price data only
def compute_autocorrelation_5d(prices: pd.DataFrame) -> pd.DataFrame:
    """5-day return autocorrelation (rolling 60-day window).

    Negative autocorrelation suggests mean-reversion; positive suggests momentum.
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change(5)
    autocorr = returns.rolling(60).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    result = autocorr.stack(level="symbol", future_stack=True)
    return result.to_frame(name="autocorrelation_5d")
```

### Data Flow

```
SimFin API → fundamentals table (new metrics: OCF, FCF, EPS)
                    ↓
FeatureComputer.compute_features() → compute_roe(), compute_roa(), etc.
                    ↓
features table (new rows: roe, roa, fcf_yield, etc.)
                    ↓
PlatformAPI.get_features() → ML pipeline → Signal generation
```

### Testing Strategy

- **Unit tests:** Each `compute_*()` function tested with synthetic price/fundamental data
- **Integration tests:** End-to-end: ingestion → computation → storage → retrieval via API
- **Validation:** Cross-check computed values against known Bloomberg/YFinance values for 5 tickers
- **Edge cases:** Missing fundamentals, zero denominators, newly listed stocks

---

## Feature 2: NLP Sentiment Features

### Goal
Complete the NLP sentiment pipeline that is partially implemented. Wire together existing EDGAR source, sentiment analyzer, and sentiment features into the production pipeline.

### Current State Assessment

**Already implemented (just needs integration):**
- `hrp/data/sources/edgar_source.py` - EDGARSource with `get_filings()`, `get_filing_text()`
- `hrp/data/sentiment_analyzer.py` - SentimentAnalyzer using Claude API (Haiku model)
- `hrp/data/ingestion/sec_ingestion.py` - SECIngestionJob with `run()` method
- `hrp/data/features/sentiment_features.py` - 6 sentiment features defined

**What needs to be done:**
1. Add `sec_filings` and `sec_filing_sentiment` tables to `schema.py` TABLES dict
2. Register sentiment features in the feature registry
3. Create a launchd job for weekly SEC filing ingestion
4. Add sentiment features to the feature ingestion pipeline
5. Expose via PlatformAPI

### Components Affected

- `hrp/data/schema.py` - Add `sec_filings` and `sec_filing_sentiment` table definitions to TABLES dict
- `hrp/data/features/computation.py` - Import and register sentiment features from `sentiment_features.py`
- `hrp/data/ingestion/features.py` - Add sentiment feature computation to feature ingestion job
- `hrp/agents/jobs.py` - Register `SECIngestionJob` in the jobs registry
- `hrp/data/features/sentiment_features.py` - Refactor to follow FeatureComputer pattern
- Config / launchd plist - Add weekly SEC ingestion schedule

### Architecture Decision: FinBERT vs Claude API

**Decision: Use Claude API (already implemented in SentimentAnalyzer).**

Rationale:
1. SentimentAnalyzer already uses Claude Haiku (`claude-3-5-haiku-20241022`) - works today
2. Claude produces richer analysis (JSON with score, reasoning, key topics)
3. FinBERT would require: installing transformers + torch (~2GB), GPU/CPU inference setup, custom integration
4. API cost is manageable: ~$0.001/filing with Haiku, ~$0.50/day for full universe
5. Fallback: If Claude API is unavailable, return NaN (don't block pipeline)

**Future enhancement:** Add FinBERT as local fallback for cost reduction at scale.

### Database Tables

```sql
-- Already defined in sec_ingestion.py, needs to be added to schema.py TABLES dict
CREATE TABLE IF NOT EXISTS sec_filings (
    symbol VARCHAR(10) NOT NULL,
    cik VARCHAR(20),
    filing_type VARCHAR(10) NOT NULL,
    filing_date DATE NOT NULL,
    accession_number VARCHAR(30) NOT NULL,
    document_url TEXT,
    filing_url TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, accession_number)
);

CREATE TABLE IF NOT EXISTS sec_filing_sentiment (
    symbol VARCHAR(10) NOT NULL,
    cik VARCHAR(20),
    filing_type VARCHAR(10) NOT NULL,
    filing_date DATE NOT NULL,
    accession_number VARCHAR(30) NOT NULL,
    sentiment_score DOUBLE,
    sentiment_category VARCHAR(20),
    key_topics TEXT,
    analysis TEXT,
    model_used VARCHAR(50),
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, accession_number)
);
```

### Sentiment Feature Computation Pattern

```python
# In computation.py, register sentiment features differently
# Sentiment features don't compute from prices - they read from sec_filing_sentiment table

def compute_sentiment_score_10k(prices: pd.DataFrame) -> pd.DataFrame:
    """Most recent 10-K filing sentiment score.

    Reads from sec_filing_sentiment table. Forward-fills annually.
    Returns NaN if no filing data available (non-blocking).
    """
    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()

    # Query latest 10-K sentiment per symbol as of each date
    sentiment = api.query_readonly("""
        SELECT s.symbol, s.filing_date, s.sentiment_score
        FROM sec_filing_sentiment s
        WHERE s.filing_type = '10-K'
        AND s.symbol IN (SELECT UNNEST(?::VARCHAR[]))
        ORDER BY s.symbol, s.filing_date
    """, [symbols])

    # Create daily series by forward-filling annual 10-K scores
    result = _forward_fill_to_daily(sentiment, dates, symbols)
    api.close()
    return result.to_frame(name="sentiment_score_10k")
```

### Ingestion Pipeline

```
Weekly (Saturday):
  SECIngestionJob.run()
    → EDGARSource.get_filings() for universe symbols
    → EDGARSource.get_filing_text() for new filings
    → SentimentAnalyzer.analyze_filing() for each text
    → Store in sec_filings + sec_filing_sentiment tables
    → Log to ingestion_log table

Daily (after price ingestion):
  FeatureComputationJob
    → compute_sentiment_score_10k() - reads sec_filing_sentiment
    → compute_sentiment_score_10q() - reads sec_filing_sentiment
    → compute_sentiment_score_avg() - weighted average
    → compute_sentiment_momentum() - delta between filings
    → Store in features table
```

### Data Flow

```
SEC EDGAR API → sec_filings table → SentimentAnalyzer (Claude Haiku)
                                          ↓
                                  sec_filing_sentiment table
                                          ↓
FeatureComputer → compute_sentiment_*() → features table
                                          ↓
PlatformAPI.get_features() → ML pipeline → Signal generation
```

### Testing Strategy

- **Unit tests:** Mock EDGAR API responses, test sentiment feature computation
- **Integration tests:** End-to-end pipeline with test filings
- **Edge cases:** API failures (return NaN), empty filings, rate limiting (SEC 10 req/sec)
- **Cost test:** Verify Claude API cost per filing is within budget

---

## Feature 3: Regime Switching Strategy

### Goal
Create a new strategy that automatically switches between momentum and mean-reversion approaches based on HMM-detected market regimes. Leverages existing `RegimeDetector` infrastructure.

### Architecture Decision: Strategy Design

**Decision: Regime-Conditional Signal Blending**

Instead of a hard switch (which creates whipsaw), blend signals weighted by regime probability:

```
signal = P(bull) * momentum_signal + P(bear) * defensive_signal + P(sideways) * mean_reversion_signal
```

This is smoother, more robust, and naturally handles regime transitions.

### Components Affected

- `hrp/research/strategies.py` - Add `generate_regime_switching_signals()` function
- `hrp/research/strategies.py` - Add `regime_switching` entry to `STRATEGY_REGISTRY` and `PRESET_STRATEGIES`
- `hrp/ml/regime.py` - Add `predict_proba()` method to RegimeDetector (returns state probabilities)

### New Strategy Function

```python
def generate_regime_switching_signals(
    prices: pd.DataFrame,
    regime_config: dict[str, Any] | None = None,
    bull_weights: dict[str, float] | None = None,
    bear_weights: dict[str, float] | None = None,
    sideways_weights: dict[str, float] | None = None,
    top_n: int = 10,
    lookback: int = 252,
    retrain_frequency: int = 63,  # Quarterly regime model retrain
) -> pd.DataFrame:
    """
    Regime-switching strategy that adapts factor weights based on HMM regime detection.

    1. Fit HMM to detect current regime (bull/bear/sideways)
    2. Get regime probabilities for smooth blending
    3. Apply regime-specific factor weights to generate composite signal
    4. Select top-N stocks by blended score

    Default weights:
    - Bull: momentum_60d=1.0, volume_ratio=0.5, trend=0.5
    - Bear: volatility_60d=-1.0, dividend_yield=1.0, pb_ratio=-0.5 (defensive)
    - Sideways: rsi_14d=-1.0, price_to_sma_20d=-1.0, bb_width_20d=-0.5 (mean-reversion)
    """
```

### RegimeDetector Enhancement

```python
# Add to hrp/ml/regime.py RegimeDetector class
def predict_proba(self, prices: pd.DataFrame) -> pd.DataFrame:
    """Get regime probabilities for each time step.

    Returns DataFrame with columns: P(bull), P(bear), P(sideways)
    Rows are dates. Probabilities sum to 1.0 per row.

    Uses HMM forward algorithm (not just Viterbi decoding).
    """
    features = self._prepare_features(prices)
    features_normalized = self._normalize(features)
    proba = self.model.predict_proba(features_normalized)

    # Map state indices to regime labels
    labels = self._label_regimes(...)
    columns = [labels[i].value for i in range(self.config.n_regimes)]

    return pd.DataFrame(proba, index=features.index, columns=columns)
```

### Strategy Registration

```python
# Add to STRATEGY_REGISTRY
STRATEGY_REGISTRY["regime_switching"] = {
    "generator": "generate_regime_switching_signals",
    "params": ["regime_config", "bull_weights", "bear_weights", "sideways_weights",
               "top_n", "lookback", "retrain_frequency"],
}

# Add to PRESET_STRATEGIES
PRESET_STRATEGIES["regime_adaptive"] = {
    "bull_weights": {"momentum_60d": 1.0, "volume_ratio": 0.5, "trend": 0.5},
    "bear_weights": {"volatility_60d": -1.0, "dividend_yield": 1.0, "pb_ratio": -0.5},
    "sideways_weights": {"rsi_14d": -1.0, "price_to_sma_20d": -1.0, "bb_width_20d": -0.5},
    "top_n": 10,
    "lookback": 252,
    "retrain_frequency": 63,
}
```

### Data Flow

```
Historical Prices → RegimeDetector.fit() → HMM Model
                          ↓
Current Prices → RegimeDetector.predict_proba() → Regime Probabilities
                          ↓
Feature Store → Factor Scores (per regime sub-strategy)
                          ↓
Blended Signal = sum(P(regime) * factor_scores[regime])
                          ↓
Top-N Selection → Trading Signals
```

### Testing Strategy

- **Unit tests:** Regime detection, probability output format, signal blending math
- **Backtest validation:** Compare regime-switching vs static momentum over 5+ years
- **Regime correctness:** Verify bull/bear/sideways labels match actual market conditions (2020 COVID, 2022 bear)
- **Edge cases:** Regime transitions, insufficient history, single-regime markets

---

## Feature 4: Risk Limit UI

### Goal
Create a new Streamlit dashboard page that exposes Risk Manager controls and risk metrics visualization.

### Components Affected

- `hrp/dashboard/pages/12_Risk_Limits.py` - **NEW** dashboard page
- `hrp/dashboard/components/risk_controls.py` - **NEW** reusable risk control components

### Page Design

```
┌─────────────────────────────────────────────────┐
│ Risk Management Dashboard                        │
├─────────────────────────────────────────────────┤
│                                                   │
│ [Tab 1: Current Risk Status]                     │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │
│ │Max DD│ │VaR   │ │CVaR  │ │Conc. │ ← KPI cards│
│ │12.3% │ │2.1%  │ │3.4%  │ │28%   │            │
│ └──────┘ └──────┘ └──────┘ └──────┘            │
│                                                   │
│ [Position Sizing Limits]                         │
│ Max Position:  ████████░░ 5%    [slider]        │
│ Min Position:  █░░░░░░░░░ 1%    [slider]        │
│ Max Sector:    █████░░░░░ 25%   [slider]        │
│ Max Gross:     ██████████ 100%  [slider]        │
│                                                   │
│ [Tab 2: Risk History]                            │
│ ┌───────────────────────────────────────┐       │
│ │ Rolling VaR/CVaR chart (21/63/252d)   │       │
│ │ with limit lines overlaid             │       │
│ └───────────────────────────────────────┘       │
│                                                   │
│ [Tab 3: Risk Manager Decisions]                  │
│ ┌───────────────────────────────────────┐       │
│ │ Recent vetoes table (hypothesis, type,│       │
│ │ severity, reason, date)               │       │
│ └───────────────────────────────────────┘       │
│                                                   │
│ [Save Limits] [Reset to Defaults]    buttons     │
└─────────────────────────────────────────────────┘
```

### Tab 1: Current Risk Status

**KPI Cards (4 metrics):**
- Max Drawdown (current portfolio)
- Value-at-Risk (95%, 1-day)
- Conditional VaR (Expected Shortfall)
- Max Sector Concentration

**Position Sizing Controls (sliders that write to config):**
```python
# Sliders map directly to RiskLimits dataclass fields
limits = RiskLimits(
    max_position_pct=st.slider("Max Position %", 0.01, 0.20, 0.05, 0.01),
    min_position_pct=st.slider("Min Position %", 0.005, 0.05, 0.01, 0.005),
    max_sector_pct=st.slider("Max Sector %", 0.10, 0.50, 0.25, 0.05),
    max_gross_exposure=st.slider("Max Gross Exposure", 0.50, 1.50, 1.00, 0.10),
    max_turnover_pct=st.slider("Max Turnover %", 0.05, 0.50, 0.20, 0.05),
    max_top_n_concentration=st.slider("Top-5 Concentration", 0.20, 0.80, 0.40, 0.05),
)
```

**Save/Load Pattern:**
- Save limits to JSON file: `data/config/risk_limits.json`
- Load on startup with defaults fallback
- Risk Manager reads from same config file

### Tab 2: Risk History

- Rolling VaR chart (21d, 63d, 252d windows) with limit lines
- Rolling CVaR chart overlaid
- Drawdown underwater chart

### Tab 3: Risk Manager Decisions

- Query `lineage` table for Risk Manager vetoes
- Table: hypothesis_id, veto_type, severity, reason, timestamp
- Filter by date range and severity

### Risk Limit Persistence

```python
# New file: hrp/risk/config.py
import json
from pathlib import Path
from hrp.risk.limits import RiskLimits

RISK_CONFIG_PATH = Path("data/config/risk_limits.json")

def save_risk_limits(limits: RiskLimits) -> None:
    """Save risk limits to config file."""
    RISK_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RISK_CONFIG_PATH, "w") as f:
        json.dump(asdict(limits), f, indent=2)

def load_risk_limits() -> RiskLimits:
    """Load risk limits from config file, with defaults fallback."""
    if RISK_CONFIG_PATH.exists():
        with open(RISK_CONFIG_PATH) as f:
            data = json.load(f)
        return RiskLimits(**data)
    return RiskLimits()  # Defaults
```

### Testing Strategy

- **UI tests:** Manual verification of slider interactions and chart rendering
- **Integration tests:** Save limits → restart dashboard → verify limits loaded
- **Edge cases:** Invalid slider values, missing config file, concurrent access

---

## Feature 5: Advanced Backtesting UI

### Goal
Create a dashboard page that exposes the Optuna optimization infrastructure and allows interactive parameter exploration.

### Architecture Decision: Parameter Sweep vs Optuna

**Decision: Focus on Optuna (Bayesian optimization) since parameter_sweep.py's `_evaluate_single_combination()` raises NotImplementedError.**

The parameter sweep grid search is not implemented. Rather than implementing both, focus on Optuna which is fully production-ready with 4 samplers, pruning, and MLflow integration.

### Components Affected

- `hrp/dashboard/pages/13_Optimization.py` - **NEW** dashboard page
- `hrp/dashboard/components/optimization_controls.py` - **NEW** reusable components

### Page Design

```
┌─────────────────────────────────────────────────┐
│ Strategy Optimization                            │
├─────────────────────────────────────────────────┤
│                                                   │
│ [Configuration Panel - Sidebar]                  │
│ Strategy:    [multifactor ▼]                     │
│ Model:       [ridge ▼]                           │
│ Sampler:     [TPE ▼] (grid/random/tpe/cmaes)    │
│ Trials:      [50] (slider 10-200)                │
│ CV Folds:    [5]  (slider 3-10)                  │
│ Scoring:     [IC ▼] (ic/r2/mse/mae/sharpe)      │
│ Window:      [expanding ▼] (expanding/rolling)   │
│ Pruning:     [✓] enabled                         │
│ Date Range:  [2020-01-01] to [2025-12-31]        │
│                                                   │
│ [Run Optimization]  ← button                     │
│                                                   │
│ [Tab 1: Results]                                 │
│ ┌───────────────────────────────────────┐       │
│ │ Best Parameters Table                  │       │
│ │ Trial History Chart (score vs trial #) │       │
│ │ Parameter Importance (from Optuna)     │       │
│ └───────────────────────────────────────┘       │
│                                                   │
│ [Tab 2: Fold Analysis]                           │
│ ┌───────────────────────────────────────┐       │
│ │ Train vs Test metrics per fold         │       │
│ │ Stability score                        │       │
│ │ Overfitting detection (train >> test)  │       │
│ └───────────────────────────────────────┘       │
│                                                   │
│ [Tab 3: Study History]                           │
│ ┌───────────────────────────────────────┐       │
│ │ Previous optimization runs             │       │
│ │ Compare best params across runs        │       │
│ └───────────────────────────────────────┘       │
│                                                   │
│ [Export Results (CSV)]                            │
└─────────────────────────────────────────────────┘
```

### Configuration Panel

```python
# Sidebar configuration
with st.sidebar:
    st.header("Optimization Config")

    strategy = st.selectbox("Strategy", list(STRATEGY_REGISTRY.keys()))
    model_type = st.selectbox("Model", ["ridge", "lasso", "elasticnet", "random_forest", "lightgbm"])
    sampler = st.selectbox("Sampler", ["tpe", "random", "grid", "cmaes"])
    n_trials = st.slider("Trials", 10, 200, 50)
    n_folds = st.slider("CV Folds", 3, 10, 5)
    scoring = st.selectbox("Scoring Metric", ["ic", "r2", "mse", "mae"])
    window_type = st.selectbox("Window Type", ["expanding", "rolling"])
    enable_pruning = st.checkbox("Enable Pruning", value=True)

    start_date = st.date_input("Start Date", date(2020, 1, 1))
    end_date = st.date_input("End Date", date.today())

    # Feature selection
    available_features = FeatureRegistry().list_features(active_only=True)
    selected_features = st.multiselect("Features", available_features,
                                        default=["momentum_20d", "volatility_20d", "rsi_14d"])
```

### Running Optimization

```python
if st.button("Run Optimization"):
    with st.spinner("Running optimization..."):
        config = OptimizationConfig(
            model_type=model_type,
            target="returns_20d",
            features=selected_features,
            param_space=get_default_param_space(model_type),
            start_date=start_date,
            end_date=end_date,
            n_folds=n_folds,
            scoring_metric=scoring,
            n_trials=n_trials,
            sampler=sampler,
            enable_pruning=enable_pruning,
            window_type=window_type,
        )

        result = cross_validated_optimize(config, symbols=get_universe_symbols())

    st.session_state["optimization_result"] = result
```

### Tab 1: Results Visualization

```python
# Best parameters
st.subheader("Best Parameters")
st.json(result.best_params)
st.metric("Best Score", f"{result.best_score:.4f}")

# Trial history
fig = px.line(
    result.cv_results,
    x="trial_number",
    y="score",
    title="Optimization Progress",
)
fig.add_hline(y=result.best_score, line_dash="dash", annotation_text="Best")
st.plotly_chart(fig)

# Parameter importance (from Optuna)
import optuna
study = optuna.load_study(study_name=..., storage=...)
importances = optuna.importance.get_param_importances(study)
fig_importance = px.bar(
    x=list(importances.keys()),
    y=list(importances.values()),
    title="Parameter Importance",
)
st.plotly_chart(fig_importance)
```

### Tab 2: Fold Analysis

- Train vs test metric comparison per fold (grouped bar chart)
- Stability score (CV of test metrics)
- Overfitting warning if train >> test (>2x gap)

### Tab 3: Study History

- List previous Optuna studies from SQLite storage
- Compare best parameters and scores across runs
- Link to MLflow experiments for full details

### Default Parameter Spaces

```python
def get_default_param_space(model_type: str) -> dict:
    """Get default Optuna parameter space for a model type."""
    if model_type == "ridge":
        return {"alpha": FloatDistribution(0.01, 100.0, log=True)}
    elif model_type == "lasso":
        return {"alpha": FloatDistribution(0.001, 10.0, log=True)}
    elif model_type == "random_forest":
        return {
            "n_estimators": IntDistribution(50, 500, step=50),
            "max_depth": IntDistribution(3, 15),
            "min_samples_leaf": IntDistribution(5, 50),
        }
    elif model_type == "lightgbm":
        return {
            "n_estimators": IntDistribution(50, 500, step=50),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
            "max_depth": IntDistribution(3, 12),
            "num_leaves": IntDistribution(15, 127),
        }
```

### Testing Strategy

- **UI tests:** Manual verification of all controls and visualizations
- **Integration tests:** Run small optimization (5 trials, 3 folds) and verify results display
- **Edge cases:** No results yet, optimization failure, missing MLflow data

---

## Implementation Plan

### Phase 1: Factor Library Expansion (Priority 1, ~2 weeks)

**Sequential dependencies: SimFin metrics first, then factor computation.**

| Step | Task | Files | Verify |
|------|------|-------|--------|
| 1 | Extend SimFin source with new metrics (OCF, FCF, EPS) | `simfin_source.py` | Unit test: fetch FCF for AAPL |
| 2 | Update fundamentals ingestion for new metrics | `fundamentals_timeseries.py` | Run ingestion for 5 tickers |
| 3 | Implement 4 statistical factors (price-only) | `computation.py` | Unit tests for each |
| 4 | Implement 4 quality factors | `computation.py` | Unit tests + check vs YFinance |
| 5 | Implement 4 value factors | `computation.py` | Unit tests + check vs known values |
| 6 | Register all 12 factors in FEATURE_FUNCTIONS | `computation.py` | `registry.list_features()` returns 57+ |
| 7 | Update CLAUDE.md feature list | `CLAUDE.md` | Manual review |

**--- CHECKPOINT: Run all feature tests, verify 12 new features compute correctly ---**

### Phase 2: NLP Sentiment Features (Priority 2, ~1.5 weeks)

**Sequential dependencies: Schema first, then integration.**

| Step | Task | Files | Verify |
|------|------|-------|--------|
| 8 | Add sec_filings + sec_filing_sentiment tables to schema.py | `schema.py` | `python -m hrp.data.schema --init` |
| 9 | Integrate SECIngestionJob into jobs registry | `jobs.py` | Job appears in registry |
| 10 | Refactor sentiment_features.py to follow compute_* pattern | `sentiment_features.py` | Unit tests with mock data |
| 11 | Register 6 sentiment features in FEATURE_FUNCTIONS | `computation.py` | Registry lists sentiment_* |
| 12 | Add sentiment features to feature ingestion pipeline | `features.py` ingestion | End-to-end test |
| 13 | Create launchd plist for weekly SEC ingestion | `com.hrp.sec-ingestion.plist` | `launchctl load` succeeds |

**--- CHECKPOINT: Run sentiment pipeline end-to-end for 5 tickers ---**

### Phase 3: Regime Switching Strategy (Priority 3, ~1 week)

| Step | Task | Files | Verify |
|------|------|-------|--------|
| 14 | Add `predict_proba()` to RegimeDetector | `regime.py` | Unit test: probabilities sum to 1.0 |
| 15 | Implement `generate_regime_switching_signals()` | `strategies.py` | Unit test: signals are valid |
| 16 | Add regime_switching to STRATEGY_REGISTRY + PRESET_STRATEGIES | `strategies.py` | Strategy appears in registry |
| 17 | Backtest regime strategy vs momentum (5-year) | Manual test | Compare Sharpe ratios |

**--- CHECKPOINT: Backtest shows regime strategy is competitive ---**

### Phase 4: Risk Limit UI (Priority 4, ~1 week)

| Step | Task | Files | Verify |
|------|------|-------|--------|
| 18 | Create risk config persistence (`risk/config.py`) | `risk/config.py` **NEW** | Save/load round-trip |
| 19 | Create Risk Limit dashboard page | `pages/12_Risk_Limits.py` **NEW** | Page renders without errors |
| 20 | Add risk control components | `components/risk_controls.py` **NEW** | Sliders update state |
| 21 | Wire Risk Manager to read from config file | `risk_manager.py` | Config changes affect vetoes |

**--- CHECKPOINT: Risk page loads, sliders save, Risk Manager reads config ---**

### Phase 5: Advanced Backtesting UI (Priority 5, ~1 week)

| Step | Task | Files | Verify |
|------|------|-------|--------|
| 22 | Create optimization dashboard page | `pages/13_Optimization.py` **NEW** | Page renders |
| 23 | Create optimization control components | `components/optimization_controls.py` **NEW** | Controls work |
| 24 | Wire to Optuna `cross_validated_optimize()` | Same page | Small optimization runs successfully |
| 25 | Add results visualization (trial history, importance, fold analysis) | Same page | Charts render |
| 26 | Add study history and comparison | Same page | Previous studies listed |

**--- CHECKPOINT: Full optimization workflow works end-to-end ---**

---

## Parallel Workstreams

Phases can be partially parallelized:

```
Phase 1 (Factor Library) ─────────────────────────────┐
Phase 2 (NLP Sentiment) ──────────────────────┐       │
Phase 3 (Regime Switching) ─────────┐          │       │
                                     ↓          ↓       ↓
                              ← Integration Testing →
Phase 4 (Risk UI) ─────────────────────┐
Phase 5 (Backtest UI) ──────────────────┐
                                         ↓
                              ← Final Testing →
```

**Workstream A (Backend - can run in parallel):**
- Phase 1: Factor Library (Steps 1-7)
- Phase 2: NLP Sentiment (Steps 8-13)
- Phase 3: Regime Switching (Steps 14-17)

**Workstream B (Frontend - can run in parallel with A):**
- Phase 4: Risk Limit UI (Steps 18-21)
- Phase 5: Advanced Backtesting UI (Steps 22-26)

**Key constraint:** Phases 1-3 are independent of each other. Phases 4-5 are independent of each other. All 5 are independent — no cross-dependencies.

---

## Edge Cases & Error Handling

### Factor Library
- **Missing fundamentals:** Return NaN, don't block pipeline. ML models handle NaN via imputation.
- **Zero denominators:** Guard against division by zero in ROE, ROA (book_value=0, total_assets=0).
- **Newly listed stocks:** Less than 60 days of history → return NaN for statistical factors.
- **SimFin API failures:** Log warning, use cached values or return NaN.

### NLP Sentiment
- **Claude API failure:** Return NaN sentiment, log warning. Never block pipeline.
- **SEC EDGAR rate limiting:** Respect 10 req/sec limit. Exponential backoff on 429 errors.
- **Empty filing text:** Score as NaN, log warning.
- **API cost control:** Cap at 500 filings per ingestion run (configurable).

### Regime Switching
- **Insufficient history:** Need 252+ days for HMM fit. Return simple momentum signals if insufficient.
- **Single-regime detection:** If HMM detects only 1 regime, fall back to equal-weight blending.
- **Regime transition whipsaw:** Probability-based blending naturally smooths transitions.

### Risk Limit UI
- **Concurrent config edits:** File-based locking (already used by other HRP jobs).
- **Invalid slider values:** Streamlit widget constraints prevent invalid inputs.
- **Config file corruption:** Fall back to RiskLimits() defaults.

### Backtesting UI
- **Long-running optimization:** Show progress bar, allow cancellation via Streamlit stop button.
- **Out-of-memory:** Limit symbols list in UI, warn if > 100 symbols selected.
- **MLflow unavailable:** Graceful degradation — show results without experiment linking.

---

## Open Questions

None — all architectural decisions have been made. This design is ready for implementation.

---

## Summary

| Feature | New Files | Modified Files | Estimated Complexity | Priority |
|---------|-----------|----------------|---------------------|----------|
| Factor Library | 0 | 3-4 | Medium | 1 |
| NLP Sentiment | 0-1 | 5-6 | Medium | 2 |
| Regime Switching | 0 | 2 | Medium | 3 |
| Risk Limit UI | 3 | 1 | Low-Medium | 4 |
| Backtesting UI | 2-3 | 0 | Medium | 5 |

**Total estimated effort:** 6-7 weeks for all 5 features.
**All features are independent** — can be implemented in any order or in parallel.
