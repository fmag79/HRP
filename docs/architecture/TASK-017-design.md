# TASK-017: Tier 1 Quick Wins - Architectural Design

**Designer:** Athena (Architect Agent)
**Date:** 2026-02-13
**Status:** Design Phase
**Priority:** High
**Target Implementation:** 12-16 weeks

---

## Executive Summary

This document provides the architectural design for implementing the top 5 high-impact, medium-complexity features identified in TASK-016 research. The design leverages existing HRP infrastructure and follows established patterns to minimize technical debt while maximizing alpha generation.

**Features to Implement:**
1. Factor Library Expansion (Quality, Value, Statistical factors)
2. NLP Sentiment Features (SEC EDGAR + FinBERT/Claude)
3. Regime Switching Strategy (HMM-based)
4. Risk Limit UI (Dashboard integration)
5. Advanced Backtesting UI (Parameter sweep exposure)

**Design Principles:**
- ✅ Leverage existing patterns (feature pipeline, MLflow, dashboard)
- ✅ Maintain institutional rigor (tests, validation, overfitting guards)
- ✅ Minimize external dependencies (use existing APIs where possible)
- ✅ Ensure backward compatibility (no breaking changes)
- ✅ Follow "The Rule" - Platform API as single entry point

---

## 1. Factor Library Expansion

### 1.1 Overview

Expand the existing feature library from 45 features to 60-65 features by adding high-signal factor categories:
- **Quality Factors:** ROE, ROA, Free Cash Flow, earnings quality
- **Value Factors:** P/E, P/B, EV/EBITDA, PEG ratio
- **Statistical Factors:** Autocorrelation, skewness, kurtosis

### 1.2 Architecture

```
hrp/data/features/
├── definitions.py              # NEW FACTORS (expand existing)
├── quality_factors.py          # NEW: Quality factor definitions
├── value_factors.py            # NEW: Value factor definitions
├── statistical_factors.py      # NEW: Statistical factor definitions
└── ...
```

**Design Approach:**
- Follow existing pattern in `definitions.py` (FeatureDefinition dataclass)
- Reuse existing computation pipeline (FeatureFactory, FeatureCache)
- Leverage SimFin + YFinance for fundamental data (already integrated)
- Add fundamental-specific caching (quarterly data has lower frequency)

### 1.3 Factor Specifications

#### Quality Factors (NEW)

| Factor | Formula | Data Source | Frequency | Implementation Notes |
|--------|---------|-------------|-----------|---------------------|
| ROE | Net Income / Shareholders' Equity | SimFin | Quarterly | Rank across universe, forward-fill 90 days |
| ROA | Net Income / Total Assets | SimFin | Quarterly | Rank across universe, forward-fill 90 days |
| FCF Margin | Free Cash Flow / Revenue | SimFin | Quarterly | Rank across universe, forward-fill 90 days |
| Earnings Quality | Cash Flow from Operations / Net Income | SimFin | Quarterly | Ratio < 0.8 = poor quality, > 1.2 = high quality |
| Asset Turnover | Revenue / Total Assets | SimFin | Quarterly | Rank across universe, forward-fill 90 days |
| Gross Margin | Gross Profit / Revenue | SimFin | Quarterly | Rank across universe, forward-fill 90 days |

#### Value Factors (NEW)

| Factor | Formula | Data Source | Frequency | Implementation Notes |
|--------|---------|-------------|-----------|---------------------|
| P/E Ratio | Price / EPS (TTM) | SimFin | Quarterly | Rank across universe, forward-fill 90 days |
| P/B Ratio | Price / Book Value per Share | SimFin | Quarterly | Rank across universe, forward-fill 90 days |
| EV/EBITDA | Enterprise Value / EBITDA | SimFin | Quarterly | Rank across universe, forward-fill 90 days |
| PEG Ratio | P/E / EPS Growth Rate | SimFin | Quarterly | Compute from multi-year data, handle negative growth |
| Price/Sales | Price / Revenue per Share | SimFin | Quarterly | Rank across universe, forward-fill 90 days |

#### Statistical Factors (NEW)

| Factor | Formula | Data Source | Frequency | Implementation Notes |
|--------|---------|-------------|-----------|---------------------|
| Autocorrelation | Correlation(return_t, return_{t-1}) | Computed | Daily | 60-day rolling window |
| Skewness | Skewness of returns (60-day) | Computed | Daily | scipy.stats.skew |
| Kurtosis | Kurtosis of returns (60-day) | Computed | Daily | scipy.stats.kurtosis |
| Downside Risk | StdDev of negative returns only | Computed | Daily | 60-day window |
| Upside Potential | StdDev of positive returns only | Computed | Daily | 60-day window |

### 1.4 Integration Points

**Data Ingestion (hrp/data/ingestion/):**
- Leverage existing `FundamentalsIngestor` (already fetches SimFin data)
- Add new columns to existing tables (fundamental_metrics table)
- Update schema migrations for new columns

**Feature Computation (hrp/data/features/):**
- Add new FeatureDefinition entries to `definitions.py`
- Implement computation functions in quality_factors.py, value_factors.py, statistical_factors.py
- Add to FeatureFactory registry (automatic via decorators)

**Testing (tests/test_data/test_features.py):**
- Add unit tests for each new factor
- Validate against known values (golden set)
- Test forward-fill behavior for quarterly data
- Test rank normalization

**ML Pipeline:**
- New factors automatically available for ML models
- No changes needed to ML Scientist or MLflow integration
- Feature selection will rank new factors by mutual information

### 1.5 Dependencies

| Dependency | Version | Purpose | Status |
|------------|---------|---------|--------|
| SimFin API | Existing | Fundamental data | ✅ Already integrated |
| scipy | Existing | Statistical functions | ✅ Already integrated |
| numpy | Existing | Numerical computations | ✅ Already integrated |
| pandas | Existing | Data manipulation | ✅ Already integrated |

**New Dependencies Required:** None

### 1.6 Implementation Timeline

**Phase 1: Quality Factors** (2 weeks)
- Week 1: Data model updates, ROE/ROA/FCF implementation
- Week 2: Earnings quality, asset turnover, gross margin + testing

**Phase 2: Value Factors** (2 weeks)
- Week 1: P/E, P/B, EV/EBITDA implementation
- Week 2: PEG ratio, price/sales + testing

**Phase 3: Statistical Factors** (1 week)
- Week 1: Autocorrelation, skewness, kurtosis + testing

**Total:** ~5 weeks (including testing and validation)

### 1.7 Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SimFin data gaps | Low | Medium | Fallback to YFinance, forward-fill with validation |
| Negative P/E ratios | High | Low | Handle negative values in ranking, use absolute value |
| Seasonality in quarterly data | Medium | Low | Forward-fill with 90-day expiration, add warning on staleness |
| Feature collinearity | Medium | Low | Feature selection will handle, monitor correlation matrix |

---

## 2. NLP Sentiment Features

### 2.1 Overview

Implement sentiment analysis pipeline for SEC filings and news sources using FinBERT or Claude API. This provides an alternative data edge for alpha generation.

### 2.2 Architecture

```
hrp/data/sources/
├── sec_edgar.py               # NEW: SEC EDGAR client
└── news_sources.py            # NEW: News API client (optional future)

hrp/data/ingestion/
├── sec_ingestor.py            # NEW: SEC filing ingestion
└── sentiment_ingestor.py      # NEW: Sentiment scoring job

hrp/data/features/
├── sentiment_features.py      # NEW: Sentiment-based features
└── news_sentiment.py          # NEW: News sentiment aggregation

hrp/ml/nlp/
├── sentiment_analyzer.py      # NEW: Sentiment analysis interface
├── finbert_client.py          # NEW: FinBERT wrapper
└── claude_sentiment.py        # NEW: Claude API wrapper
```

**Design Approach:**
- Abstract sentiment interface (switch between FinBERT/Claude)
- Cache sentiment scores (SEC filings don't change)
- Rolling aggregation for time-series signals
- Integrate with existing feature pipeline

### 2.3 Data Flow

```
SEC EDGAR API
     ↓
sec_ingestor.py (fetch 10-Q/10-K filings)
     ↓
sentiment_analyzer.py (score sentiment)
     ↓
sentiment_features.py (compute rolling features)
     ↓
FeatureFactory (register with existing pipeline)
```

### 2.4 Sentiment Scoring Options

#### Option A: FinBERT (Recommended for production)

**Pros:**
- ✅ Self-hosted (no API costs)
- ✅ Fine-tuned for financial text
- ✅ Consistent performance
- ✅ No external dependency risk

**Cons:**
- ❌ GPU recommended (can run on CPU but slower)
- ❌ Model maintenance (updates, fine-tuning)

**Implementation:**
```python
from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone",
    tokenizer="yiyanghkust/finbert-tone"
)
```

#### Option B: Claude API (Recommended for experimentation)

**Pros:**
- ✅ Higher accuracy (Claude 3.5 Sonnet)
- ✅ No model maintenance
- ✅ Easy to iterate

**Cons:**
- ❌ API costs
- ❌ Rate limits
- ❌ External dependency risk

**Implementation:**
```python
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{
        "role": "user",
        "content": f"Analyze sentiment of this SEC filing text: {text}\n\n"
                   f"Return sentiment score from -1 (very negative) to +1 (very positive)"
    }]
)
```

**Recommendation:** Implement both options with feature flag, default to FinBERT for production.

### 2.5 Feature Specifications

#### Sentiment Features (NEW)

| Feature | Formula | Data Source | Frequency | Implementation Notes |
|--------|---------|-------------|-----------|---------------------|
| SEC Sentiment Score | Average sentiment from 10-Q/10-K | SEC EDGAR | Quarterly | -1 to +1 range, forward-fill 90 days |
| Sentiment Trend | Slope of sentiment over last 4 quarters | SEC EDGAR | Quarterly | Linear regression slope |
| Sentiment Momentum | Change from last quarter | SEC EDGAR | Quarterly | Delta from previous filing |
| News Sentiment (Future) | Aggregated news sentiment | News API | Daily | Rolling 7-day average |

### 2.6 Integration Points

**Data Ingestion:**
- New `SecIngester` class in `hrp/data/ingestion/sec_ingestor.py`
- Schedule via launchd (quarterly sync after earnings season)
- Store in `sec_sentiment` table (symbol, filing_date, form_type, sentiment_score)

**Sentiment Analysis:**
- Abstract `SentimentAnalyzer` interface in `hrp/ml/nlp/sentiment_analyzer.py`
- Implement `FinBERTSentimentAnalyzer` and `ClaudeSentimentAnalyzer`
- Use environment variable `SENTIMENT_ENGINE` to switch

**Feature Pipeline:**
- Add `SentimentFeature` definition to `definitions.py`
- Fetch sentiment scores from `sec_sentiment` table
- Compute rolling aggregation (momentum, trend)

**Testing:**
- Unit tests for SEC fetching (mock EDGAR API)
- Unit tests for sentiment scoring (golden set of filings)
- Integration tests for feature computation

**ML Pipeline:**
- Sentiment features automatically available for ML models
- Monitor feature importance in MLflow

### 2.7 Dependencies

| Dependency | Version | Purpose | Status |
|------------|---------|---------|--------|
| transformers | >=4.30.0 | FinBERT model | 🔜 New |
| torch | >=2.0.0 | Backend for transformers | 🔜 New |
| anthropic | >=0.18.0 | Claude API client | 🔜 New |
| requests | Existing | SEC EDGAR API | ✅ Already integrated |

**New Dependencies Required:**
- transformers (FinBERT)
- torch (ML backend)
- anthropic (optional, for Claude)

### 2.8 Implementation Timeline

**Phase 1: Infrastructure** (1 week)
- Week 1: SEC EDGAR client, sentiment analyzer interface, FinBERT integration

**Phase 2: Feature Integration** (1 week)
- Week 1: Sentiment feature definitions, rolling aggregations, testing

**Phase 3: Claude Integration (Optional)** (1 week)
- Week 1: Claude API wrapper, A/B testing vs FinBERT

**Total:** ~3-4 weeks

### 2.9 Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SEC EDGAR rate limits | Medium | Medium | Respect rate limits, schedule during off-peak hours |
| FinBERT performance | Low | Low | Cache scores, run batch processing overnight |
| Claude API costs | Medium | High | Use FinBERT for production, Claude for experimentation |
| Negative sentiment bias | Medium | Low | Calibrate scores using golden set, validate against returns |

---

## 3. Regime Switching Strategy

### 3.1 Overview

Leverage existing HMM (Hidden Markov Model) infrastructure to implement regime detection and adaptive strategy switching between momentum and mean-reversion strategies based on market regime.

### 3.2 Architecture

```
hrp/research/strategies/
├── regime_switching.py        # NEW: Regime switching strategy
└── regime_signals.py          # NEW: Regime signal generation

hrp/ml/regime/
├── regime_detector.py         # EXISTING: HMM regime detection
└── regime_classifier.py       # NEW: Classify regime type
```

**Design Approach:**
- Leverage existing `RegimeDetector` (hrp/ml/regime/regime_detector.py)
- Create new `RegimeSwitchingStrategy` that adapts based on detected regime
- Integrate with existing backtesting framework (VectorBT)
- Add regime analysis to Validation Analyst

### 3.3 Regime Detection Logic

**Existing HMM Implementation:**
```python
# hrp/ml/regime/regime_detector.py (already exists)
class RegimeDetector:
    def detect_regime(self, returns: pd.Series) -> np.ndarray:
        """Returns regime labels (0=bull, 1=bear, 2=sideways)"""
        # HMM model trained on market returns
```

**Regime Classification:**
- **Bull Market:** Positive returns, low volatility → Momentum strategy
- **Bear Market:** Negative returns, high volatility → Defensive/mean-reversion
- **Sideways:** Flat returns, moderate volatility → Mean-reversion strategy

### 3.4 Strategy Specification

#### RegimeSwitchingStrategy (NEW)

**Regime Rules:**
```
IF regime == BULL:
    Signal = MomentumSignal(RSI, MACD, Price Momentum)
ELSE IF regime == BEAR:
    Signal = MeanReversionSignal(RSI, Bollinger Bands, Volatility)
ELSE:  # SIDEWAYS
    Signal = MeanReversionSignal(RSI, Bollinger Bands)
```

**Position Sizing:**
- Bull regime: Full position size (1.0x)
- Bear regime: Reduced position size (0.5x - risk management)
- Sideways regime: Standard position size (1.0x)

**Stop Losses:**
- Bull regime: ATR trailing stop (trend-following)
- Bear regime: Fixed % stop loss (defensive)
- Sideways regime: Fixed % stop loss

### 3.5 Integration Points

**ML Pipeline:**
- Existing `RegimeDetector` already trained on market returns
- Add `RegimeClassifier` to label regimes (bull/bear/sideways)
- Train HMM on SPY returns (market benchmark)

**Backtesting:**
- Create `RegimeSwitchingStrategy` class extending `BaseStrategy`
- Implement `generate_signals()` method with regime-aware logic
- Add VectorBT backtesting with regime-specific parameter sets
- Compare vs pure momentum and pure mean-reversion strategies

**Risk Management:**
- Integrate with Risk Manager Agent (regime as additional risk factor)
- Add regime-aware position sizing rules
- Monitor regime transition risk (sudden shifts)

**Validation:**
- Validation Analyst: Add regime stress tests
- Performance analysis by regime (bull vs bear vs sideways)
- Regime transition analysis (how fast does strategy adapt?)

**Dashboard:**
- Add regime visualization to Trading page
- Show current regime with confidence score
- Display regime transition timeline

### 3.6 Testing Strategy

**Unit Tests:**
- Test regime classification logic
- Test signal generation for each regime
- Test position sizing rules

**Backtesting Tests:**
- Walk-forward validation with regime switching
- Compare performance: regime-switching vs pure momentum vs pure mean-reversion
- Analyze regime-specific performance (Sharpe, drawdown in bull/bear/sideways)

**Integration Tests:**
- End-to-end pipeline: regime detection → signal generation → backtesting
- Risk Manager integration (regime-aware veto rules)

### 3.7 Dependencies

| Dependency | Version | Purpose | Status |
|------------|---------|---------|--------|
| hmmlearn | Existing | HMM models | ✅ Already integrated |
| vectorbt | Existing | Backtesting | ✅ Already integrated |
| numpy | Existing | Numerical computations | ✅ Already integrated |
| pandas | Existing | Data manipulation | ✅ Already integrated |

**New Dependencies Required:** None

### 3.8 Implementation Timeline

**Phase 1: Strategy Implementation** (1 week)
- Week 1: RegimeSwitchingStrategy class, signal generation, backtesting

**Phase 2: Validation & Testing** (1 week)
- Week 1: Walk-forward validation, regime performance analysis, testing

**Total:** ~2-3 weeks

### 3.9 Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Regime misclassification | Medium | Medium | Use regime confidence scores, transition smoothing |
| Frequent regime switching | Medium | Medium | Add minimum regime duration (e.g., 10 days) |
| Overfitting to past regimes | Medium | Low | Walk-forward validation, cross-market testing |
| HMM parameter drift | Low | Medium | Periodic retraining, regime transition monitoring |

---

## 4. Risk Limit UI

### 4.1 Overview

Expose Risk Manager controls in the dashboard to allow real-time risk limit adjustments and monitoring without code changes.

### 4.2 Architecture

```
hrp/dashboard/pages/
├── risk_limits.py              # NEW: Risk limits configuration page

hrp/api/
├── risk_limits.py              # NEW: Risk limits API endpoints
└── risk_manager_api.py         # NEW: Risk Manager queries
```

**Design Approach:**
- Follow existing dashboard pattern (Streamlit pages)
- Add API endpoints for risk limit CRUD operations
- Expose existing Risk Manager logic (no changes to core logic)
- Persist risk limits in database (new table or config)

### 4.3 Features to Expose

**Risk Limits Configuration:**
1. Maximum drawdown threshold
2. Maximum position VaR
3. Maximum portfolio VaR
4. Maximum concentration (single ticker %)
5. Maximum correlation limit
6. Stop loss thresholds (fixed %, ATR, volatility-scaled)

**Real-Time Monitoring:**
1. Current risk metrics vs limits
2. Risk limit usage (e.g., 80% of max drawdown)
3. Alert banners when approaching limits
4. Risk Manager veto history

### 4.4 UI Design

```
Risk Limits Page
├── Overview
│   ├── Current Risk Metrics (VaR, drawdown, concentration)
│   └── Risk Limit Usage (progress bars)
├── Risk Limits Configuration
│   ├── Drawdown Threshold (slider + input)
│   ├── Position VaR Limit (slider + input)
│   ├── Portfolio VaR Limit (slider + input)
│   ├── Concentration Limit (slider + input)
│   └── Stop Loss Threshold (select: fixed%/ATR/volatility)
└── Risk Manager History
    ├── Veto history table
    └── Risk alerts timeline
```

### 4.5 Integration Points

**Dashboard (hrp/dashboard/pages/risk_limits.py):**
- New Streamlit page following existing pattern
- Use st.sidebar for navigation (add to existing menu)
- API calls to fetch/update risk limits

**API (hrp/api/risk_limits.py):**
- GET /risk/limits - Fetch current risk limits
- POST /risk/limits - Update risk limits
- GET /risk/current - Fetch current risk metrics
- GET /risk/history - Fetch risk Manager veto history

**Database:**
- New table: `risk_limits` (limit_name, limit_value, updated_at)
- New table: `risk_alerts` (alert_type, threshold, current_value, timestamp)
- Update Risk Manager to read limits from database instead of hardcoded

**Risk Manager:**
- Add `load_limits_from_db()` method
- Update veto logic to use database limits
- Add logging for limit changes

### 4.6 Dependencies

| Dependency | Version | Purpose | Status |
|------------|---------|---------|--------|
| streamlit | Existing | Dashboard UI | ✅ Already integrated |
| duckdb | Existing | Database | ✅ Already integrated |
| fastapi | Existing | API server | ✅ Already integrated |

**New Dependencies Required:** None

### 4.7 Implementation Timeline

**Phase 1: Backend API** (1 week)
- Week 1: Risk limits database schema, API endpoints, Risk Manager integration

**Phase 2: Dashboard UI** (1 week)
- Week 1: Streamlit page, risk metrics display, limit configuration

**Total:** ~2 weeks

### 4.8 Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Risk limit persistence issues | Low | Medium | Add validation, backup to config file |
| Concurrent limit updates | Low | Low | Database row locking, last-write-wins with audit |
| UI performance (slow queries) | Low | Low | Cache risk metrics, refresh every 30s |

---

## 5. Advanced Backtesting UI

### 5.1 Overview

Expose existing parameter sweep functionality via dashboard UI to enable faster hypothesis iteration and visual parameter optimization.

### 5.2 Architecture

```
hrp/dashboard/pages/
├── backtest_advanced.py        # NEW: Advanced backtesting page

hrp/api/
├── backtest_sweep.py            # NEW: Parameter sweep API endpoints
└── backtest_results.py          # NEW: Backtest results API
```

**Design Approach:**
- Leverage existing Optuna parameter sweep (already implemented)
- Expose sweep configuration via UI
- Visualize parameter space and optimization results
- Integrate with MLflow for experiment tracking

### 5.2 Features to Expose

**Parameter Sweep Configuration:**
1. Select hypothesis to optimize
2. Define parameter search space (min/max/log scale)
3. Select optimization objective (Sharpe, CAGR, max drawdown)
4. Configure Optuna sampler (TPE, Random, Grid)
5. Set trial count and timeout

**Visualization:**
1. Parameter importance plot
2. Optuna study history (best value over time)
3. Parallel coordinate plot (parameter interactions)
4. Contour plots (2D parameter relationships)
5. Top trials table

**Export & Analysis:**
1. Download best parameters as JSON
2. Export all trials to CSV
3. Compare multiple studies
4. Hyperparameter sensitivity analysis

### 5.3 UI Design

```
Advanced Backtesting Page
├── Sweep Configuration
│   ├── Hypothesis selector (dropdown)
│   ├── Parameter search space (table with min/max/log toggle)
│   ├── Objective function (Sharpe/CAGR/Max Drawdown)
│   ├── Sampler selection (TPE/Random/Grid)
│   └── Trial count & timeout (inputs)
├── Sweep Execution
│   ├── Start sweep button
│   ├── Progress bar (trials completed / total)
│   ├── Real-time best value display
│   └── Stop button
├── Results Visualization
│   ├── Parameter importance (bar chart)
│   ├── Study history (line chart)
│   ├── Parallel coordinates (multi-dimensional)
│   └── Contour plots (select 2 params)
└── Top Trials
    ├── Trials table (rank, Sharpe, params)
    ├── Download best params (button)
    ├── Export all trials (CSV button)
    └── Compare studies (select 2 studies)
```

### 5.4 Integration Points

**Dashboard (hrp/dashboard/pages/backtest_advanced.py):**
- New Streamlit page following existing pattern
- Add to navigation menu
- Use Optuna integration for visualization

**API (hrp/api/backtest_sweep.py):**
- POST /backtest/sweep - Start parameter sweep
- GET /backtest/sweep/{study_id} - Get sweep status
- GET /backtest/sweep/{study_id}/trials - Get all trials
- DELETE /backtest/sweep/{study_id} - Delete study

**MLflow Integration:**
- Log sweep to MLflow experiment (existing integration)
- Associate trials with MLflow runs
- Reuse MLflow UI for experiment comparison

**Optuna Integration:**
- Leverage existing Optuna setup (TASK completed 2026-02-04)
- Use Optuna's visualization functions
- Persist studies to DuckDB (or file-based)

### 5.5 Dependencies

| Dependency | Version | Purpose | Status |
|------------|---------|---------|--------|
| streamlit | Existing | Dashboard UI | ✅ Already integrated |
| optuna | Existing | Parameter optimization | ✅ Already integrated |
| mlflow | Existing | Experiment tracking | ✅ Already integrated |
| plotly | Existing | Interactive plots | ✅ Already integrated |

**New Dependencies Required:** None

### 5.6 Implementation Timeline

**Phase 1: Backend API** (1 week)
- Week 1: Parameter sweep API endpoints, Optuna integration, study management

**Phase 2: Dashboard UI** (1 week)
- Week 1: Streamlit page, sweep configuration, visualization, export

**Total:** ~2 weeks

### 5.7 Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Long sweep times (blocking UI) | Medium | Low | Async execution, progress polling, email notification on completion |
| Large study memory footprint | Low | Medium | Pagination, lazy loading, export to file |
| Concurrent sweep conflicts | Low | Low | Study ID isolation, queue system |

---

## 6. Integration & Dependencies

### 6.1 Cross-Feature Dependencies

```
Factor Library ──────┐
                     ├──> ML Pipeline (Feature Selection)
NLP Sentiment ───────┤
                     │
Statistical Factors ─┘
                     │
Regime Switching ────┼──> Backtesting (VectorBT)
                     │
Risk Limit UI ────────┼──> Risk Manager
                     │
Advanced Backtest UI ─┴──> Optuna + MLflow
```

### 6.2 Integration Testing Strategy

**End-to-End Tests:**
1. Factor Library → Feature Selection → ML Model Training
2. SEC Ingestion → Sentiment Analysis → Feature Computation
3. Regime Detection → Regime Switching Strategy → Backtesting
4. Risk Limit UI → Database → Risk Manager Veto Logic
5. Parameter Sweep API → Optuna → MLflow → Dashboard Visualization

**Integration Test Suites:**
- `tests/integration/test_factor_pipeline.py`
- `tests/integration/test_nlp_pipeline.py`
- `tests/integration/test_regime_strategy.py`
- `tests/integration/test_risk_limits.py`
- `tests/integration/test_backtest_sweep.py`

### 6.3 Database Schema Changes

**New Tables:**
```sql
-- SEC sentiment scores
CREATE TABLE sec_sentiment (
    symbol VARCHAR,
    filing_date DATE,
    form_type VARCHAR,
    sentiment_score FLOAT,
    created_at TIMESTAMP,
    PRIMARY KEY (symbol, filing_date)
);

-- Risk limits configuration
CREATE TABLE risk_limits (
    limit_name VARCHAR PRIMARY KEY,
    limit_value FLOAT,
    updated_at TIMESTAMP
);

-- Risk alerts history
CREATE TABLE risk_alerts (
    alert_id INTEGER PRIMARY KEY,
    alert_type VARCHAR,
    threshold FLOAT,
    current_value FLOAT,
    triggered_at TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Parameter sweep studies
CREATE TABLE backtest_studies (
    study_id VARCHAR PRIMARY KEY,
    hypothesis_id VARCHAR,
    objective VARCHAR,
    sampler VARCHAR,
    n_trials INTEGER,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

**Existing Tables (Add Columns):**
- `fundamental_metrics`: Add quality/value factor columns
- `features`: Add sentiment factor columns
- `risk_metrics`: Add regime column

### 6.4 Configuration Changes

**New Environment Variables:**
```bash
# NLP Sentiment Engine
SENTIMENT_ENGINE=finbert  # or "claude"
ANTHROPIC_API_KEY=sk-ant-...

# SEC EDGAR
SEC_EDGAR_USER_AGENT=your_email@domain.com

# Risk Limits
RISK_LIMITS_DB_ENABLED=true
```

**New Launchd Jobs:**
- `com.hrp.sec-ingestion.plist` - Quarterly SEC filing sync
- `com.hrp.sentiment-scoring.plist` - Daily sentiment scoring

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Factor Library:**
- `test_quality_factors.py` - Test each quality factor computation
- `test_value_factors.py` - Test each value factor computation
- `test_statistical_factors.py` - Test each statistical factor
- Validate against golden set (known values for specific tickers)

**NLP Sentiment:**
- `test_sec_ingestor.py` - Mock EDGAR API, test parsing
- `test_sentiment_analyzer.py` - Test scoring with sample texts
- `test_sentiment_features.py` - Test rolling aggregation

**Regime Switching:**
- `test_regime_switching.py` - Test signal generation per regime
- `test_regime_classifier.py` - Test regime classification logic

**Risk Limits UI:**
- `test_risk_limits_api.py` - Test API endpoints
- `test_risk_manager_db.py` - Test database integration

**Backtest UI:**
- `test_backtest_sweep_api.py` - Test sweep API
- `test_optuna_integration.py` - Test Optuna study management

### 7.2 Integration Tests

**Feature Pipeline Integration:**
- End-to-end: Data ingestion → Feature computation → ML model
- Validate new factors appear in MLflow feature importance
- Test feature selection with new factors

**NLP Pipeline Integration:**
- End-to-end: SEC fetch → Sentiment scoring → Feature computation
- Validate sentiment scores persist to database
- Test forward-fill behavior (quarterly data)

**Regime Strategy Integration:**
- End-to-end: Regime detection → Strategy signals → Backtesting
- Compare regime-switching vs pure strategies
- Validate regime-specific performance

### 7.3 Performance Tests

**Factor Computation Performance:**
- Benchmark new factor computation time (target: <10s for 500 tickers)
- Test with 5-year historical data

**NLP Inference Performance:**
- Benchmark FinBERT scoring time (target: <100ms per filing)
- Test batch processing (100 filings)

**Backtest Sweep Performance:**
- Benchmark parameter sweep time (target: <5 min for 100 trials)
- Test with large parameter spaces

### 7.4 Regression Tests

**Guardrails:**
- All existing tests must pass (99%+ pass rate target)
- No breaking changes to existing features
- Backward compatibility with existing data

**Performance Regression:**
- Factor computation time must not increase >20%
- Dashboard page load time <3 seconds
- API response time <500ms (p95)

---

## 8. Deployment Strategy

### 8.1 Deployment Order (Risk-Based)

**Phase 1: Risk-Low Features (Week 1-2)**
1. Factor Library (Quality factors)
2. Statistical Factors
3. Risk Limit UI (UI only, no core logic changes)

**Phase 2: Risk-Medium Features (Week 3-6)**
4. Value Factors
5. Regime Switching Strategy
6. Advanced Backtesting UI

**Phase 3: Risk-High Features (Week 7-10)**
7. NLP Sentiment (new external dependency)
8. Full integration testing
9. Documentation

**Phase 4: Validation (Week 11-12)**
10. Paper trading validation
11. Performance monitoring
12. User acceptance testing

### 8.2 Rollout Plan

**Staging Environment:**
- Deploy all features to staging first
- Run full test suite
- Manual QA by squad
- Monitor for 1 week

**Production Rollout:**
- Feature flag rollout (can disable individual features)
- Gradual rollout (10% → 50% → 100% traffic)
- Monitor error rates, performance metrics
- Rollback plan: Revert to previous commit if critical issues

**Post-Deployment:**
- Monitor for 2 weeks in production
- Collect user feedback
- Fix bugs and iterate

### 8.3 Monitoring & Alerting

**Metrics to Monitor:**
- Feature computation time (p95, p99)
- SEC ingestion success rate
- Sentiment scoring error rate
- Regime detection frequency (sudden changes)
- Risk limit changes (audit log)
- Parameter sweep success rate

**Alerts:**
- Feature computation failures >5%
- SEC ingestion failures >10%
- Sentiment API errors >10%
- Risk limit usage >90% (warning), >95% (critical)
- Backtest sweep failures

---

## 9. Documentation Requirements

### 9.1 Code Documentation

**Docstrings Required:**
- All new classes and public methods
- Parameter types and return values
- Usage examples in docstrings
- Type hints enforced (mypy)

**Comments:**
- Complex logic explanations
- Algorithm references (e.g., "HMM as in Murphy 2002")
- External API references (SEC EDGAR documentation)

### 9.2 User Documentation

**New Documentation Sections:**
- Feature Engineering Guide (how to add new factors)
- NLP Sentiment Features (usage, configuration)
- Regime Switching Strategy (usage, tuning)
- Risk Limits Configuration (dashboard guide)
- Advanced Backtesting (parameter sweep guide)

**API Documentation:**
- Auto-generated from docstrings (Sphinx)
- API reference for new endpoints

### 9.3 Architecture Documentation

**Updates Required:**
- CLAUDE.md - Add new features to feature list (update 45 → 60-65)
- Architecture diagram - Add NLP pipeline, regime switching
- Data flow diagram - Update with new components

---

## 10. Success Metrics

### 10.1 Technical Metrics

**Feature Library:**
- ✅ 20 new factors implemented (6 quality + 5 value + 4 statistical + 5 sentiment)
- ✅ All tests passing (99%+ pass rate)
- ✅ Feature computation time <10s for 500 tickers
- ✅ Feature availability in MLflow

**NLP Sentiment:**
- ✅ SEC ingestion success rate >95%
- ✅ Sentiment scoring time <100ms per filing
- ✅ Sentiment features available in ML pipeline

**Regime Switching:**
- ✅ Regime detection accuracy >70% (vs historical labels)
- ✅ Regime-switching Sharpe > max(momentum, mean-reversion) Sharpe
- ✅ Regime adaptation latency <3 days

**Risk Limit UI:**
- ✅ API response time <500ms (p95)
- ✅ Dashboard page load time <3s
- ✅ Risk limit changes persisted correctly

**Backtest UI:**
- ✅ Parameter sweep execution <5 min for 100 trials
- ✅ Visualization rendering time <2s
- ✅ Study export functional (JSON, CSV)

### 10.2 Business Metrics

**Alpha Generation:**
- Feature library: 3-5% Sharpe improvement (backtested)
- NLP sentiment: 2-4% Sharpe improvement (backtested)
- Regime switching: 3-5% Sharpe improvement vs pure strategies

**User Adoption:**
- Risk Limit UI: Daily active usage
- Advanced Backtest UI: >10 parameter sweeps/week
- NLP Sentiment: >5 hypotheses using sentiment features

**Operational Efficiency:**
- Faster hypothesis iteration (parameter sweep UI reduces time by 50%)
- Real-time risk limit adjustments (no code changes required)

---

## 11. Handoff to Forge

### 11.1 Implementation Tasks

Forge will implement features in this order (based on deployment strategy):

**Sprint 1 (Week 1-2):**
1. Factor Library - Quality Factors (ROE, ROA, FCF)
2. Risk Limit UI - API endpoints + Dashboard page
3. Testing for Sprint 1 features

**Sprint 2 (Week 3-4):**
4. Factor Library - Value Factors (P/E, P/B, EV/EBITDA)
5. Factor Library - Statistical Factors
6. Regime Switching Strategy
7. Testing for Sprint 2 features

**Sprint 3 (Week 5-6):**
8. NLP Sentiment - Infrastructure (SEC client, FinBERT)
9. NLP Sentiment - Features + Integration
10. Advanced Backtesting UI
11. Testing for Sprint 3 features

**Sprint 4 (Week 7-8):**
12. Integration testing (end-to-end)
13. Documentation updates
14. Staging deployment
15. Bug fixes and iteration

### 11.2 Key Design Decisions (Forge Must Follow)

1. **Feature Library:** Follow existing `FeatureDefinition` pattern, no breaking changes
2. **NLP Sentiment:** Implement abstract interface for FinBERT/Claude switching
3. **Regime Switching:** Leverage existing `RegimeDetector`, add `RegimeSwitchingStrategy`
4. **Risk Limit UI:** Expose existing Risk Manager logic, don't modify core
5. **Backtest UI:** Reuse Optuna integration, don't rewrite optimization logic
6. **Testing:** All new code must have unit tests + integration tests
7. **Documentation:** Docstrings required for all public APIs
8. **Type Safety:** Type hints required, mypy must pass

### 11.3 Files for Forge to Create/Modify

**New Files:**
```
hrp/data/features/quality_factors.py
hrp/data/features/value_factors.py
hrp/data/features/statistical_factors.py
hrp/data/features/sentiment_features.py
hrp/data/sources/sec_edgar.py
hrp/data/ingestion/sec_ingestor.py
hrp/ml/nlp/sentiment_analyzer.py
hrp/ml/nlp/finbert_client.py
hrp/ml/nlp/claude_sentiment.py
hrp/ml/regime/regime_classifier.py
hrp/research/strategies/regime_switching.py
hrp/dashboard/pages/risk_limits.py
hrp/dashboard/pages/backtest_advanced.py
hrp/api/risk_limits.py
hrp/api/backtest_sweep.py
tests/test_data/test_quality_factors.py
tests/test_data/test_value_factors.py
tests/test_data/test_statistical_factors.py
tests/test_data/test_nlp_pipeline.py
tests/test_research/test_regime_switching.py
tests/test_api/test_risk_limits.py
tests/test_api/test_backtest_sweep.py
tests/integration/test_factor_pipeline.py
tests/integration/test_nlp_pipeline.py
tests/integration/test_regime_strategy.py
docs/feature-engineering-guide.md
docs/nlp-sentiment-guide.md
docs/regime-switching-guide.md
docs/risk-limits-guide.md
docs/advanced-backtesting-guide.md
```

**Modified Files:**
```
hrp/data/features/definitions.py  # Add new feature definitions
hrp/data/features/__init__.py     # Import new feature modules
hrp/ml/regime/regime_detector.py  # Add regime classifier method
hrp/agents/risk_manager.py        # Load limits from database
hrp/dashboard/app.py              # Add new pages to navigation
CLAUDE.md                         # Update feature count (45 → 60-65)
```

---

## 12. Appendix

### 12.1 Technical Specifications

**SEC EDGAR API:**
- Base URL: `https://www.sec.gov/Archives/`
- Required Headers: `User-Agent` (email)
- Rate Limit: 10 requests/second
- Filing Types: 10-K (annual), 10-Q (quarterly)

**FinBERT Model:**
- Model: `yiyanghkust/finbert-tone`
- Labels: `Positive`, `Negative`, `Neutral`
- Output: Sentiment score (-1 to +1)
- Inference Time: ~50-100ms per filing (CPU), ~10-20ms (GPU)

**HMM Regime Detection:**
- Algorithm: Gaussian HMM with 3 states (bull/bear/sideways)
- Training Data: SPY daily returns (20+ years)
- Retention: Retrain quarterly to adapt to market structure changes

### 12.2 Reference Documents

**Internal:**
- TASK-016 Research: `~/Projects/HRP/docs/research/TASK-016-feature-roadmap-assessment.md`
- CLAUDE.md: Existing feature list (45 features)
- Project-Status.md: Current architecture and status

**External:**
- SEC EDGAR API: https://www.sec.gov/edgar/sec-api-documentation
- FinBERT Paper: https://arxiv.org/abs/1908.10063
- Optuna Documentation: https://optuna.readthedocs.io/
- VectorBT Documentation: https://vectorbt.dev/

### 12.3 Glossary

**Alpha:** Risk-adjusted returns above market benchmark
**Bull Market:** Rising market regime (positive returns, low volatility)
**Bear Market:** Falling market regime (negative returns, high volatility)
**Drawdown:** Peak-to-trough decline in portfolio value
**HMM:** Hidden Markov Model (statistical method for regime detection)
**P/E Ratio:** Price-to-Earnings ratio (valuation metric)
**Regime:** Market state (bull/bear/sideways) with distinct statistical properties
**ROE:** Return on Equity (profitability metric)
**VaR:** Value at Risk (quantitative risk measure)
**Walk-Forward Validation:** Time-series cross-validation method

---

## Conclusion

This architectural design provides a comprehensive plan for implementing 5 high-impact, medium-complexity features in 12-16 weeks. The design leverages existing HRP infrastructure, maintains institutional rigor, and follows established patterns to minimize technical debt.

**Key Design Principles Applied:**
- ✅ Follow existing patterns (feature pipeline, MLflow, dashboard)
- ✅ Maintain institutional rigor (tests, validation, overfitting guards)
- ✅ Minimize external dependencies (use existing APIs where possible)
- ✅ Ensure backward compatibility (no breaking changes)
- ✅ Follow "The Rule" - Platform API as single entry point

**Next Steps:**
1. Review design with squad (Jarvis, Sentinel, Forge)
2. Address feedback and refine design
3. Forge begins Sprint 1 implementation
4. Sentinel reviews code before merge
5. Deploy to staging, monitor, then production

---

**Design Complete:** 2026-02-13 15:10 CST
**Designer:** Athena (Architect Agent)
**Task:** TASK-017
**Status:** Ready for Implementation
