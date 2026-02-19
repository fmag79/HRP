# Implementation Plan: Autonomous Trading Recommendation Service

**Date:** 2026-02-19
**Depends on:** Strategic analysis in `2026-02-19-strategic-analysis-autonomous-recommendations.md`
**Goal:** Transform HRP from a quant research workbench into a product that delivers autonomous, validated trading recommendations with plain-English explanations.

---

## Architecture Overview

```
                          ┌──────────────────────────────────────┐
                          │         Consumer Interface           │
                          │  (email digests, simple web view,    │
                          │   push notifications)                │
                          └──────────────┬───────────────────────┘
                                         │ reads
                          ┌──────────────▼───────────────────────┐
                          │       Advisory Service (NEW)         │
                          │                                      │
                          │  ┌─────────────┐  ┌───────────────┐  │
                          │  │   User       │  │ Recommendation│  │
                          │  │   Profiles   │  │ Engine        │  │
                          │  └─────────────┘  └───────┬───────┘  │
                          │  ┌─────────────┐  ┌───────▼───────┐  │
                          │  │  Portfolio   │  │ Track Record  │  │
                          │  │  Constructor │  │ Tracker       │  │
                          │  └─────────────┘  └───────────────┘  │
                          │  ┌─────────────┐  ┌───────────────┐  │
                          │  │  Explainer   │  │ Paper Trading │  │
                          │  │  (NLP)       │  │ Validator     │  │
                          │  └─────────────┘  └───────────────┘  │
                          └──────────────┬───────────────────────┘
                                         │ uses
                          ┌──────────────▼───────────────────────┐
                          │          HRP Engine (EXISTS)          │
                          │  PlatformAPI → agents, ML, risk,     │
                          │  features, execution                  │
                          └──────────────────────────────────────┘
```

**The Rule still applies:** The Advisory Service accesses everything through `PlatformAPI`. New API methods are added to `platform.py` as needed.

---

## Phase 1: Foundation — Recommendation Engine + Track Record

**Duration:** ~2-3 weeks
**Goal:** Generate structured recommendations from validated hypotheses and track their outcomes.

### 1.1 Database Schema Extensions

**File:** `hrp/data/schema.py`

New tables:

```sql
-- Stores generated recommendations
recommendations (
    recommendation_id   VARCHAR PRIMARY KEY,   -- REC-2026-001
    created_at          TIMESTAMP DEFAULT current_timestamp,
    hypothesis_id       VARCHAR REFERENCES hypotheses(hypothesis_id),
    model_name          VARCHAR,
    symbol              VARCHAR NOT NULL,
    action              VARCHAR CHECK (action IN ('BUY', 'HOLD', 'SELL')),
    confidence          VARCHAR CHECK (confidence IN ('HIGH', 'MEDIUM', 'LOW')),
    signal_strength     FLOAT,                 -- raw model output
    entry_price         FLOAT,                 -- price at recommendation time
    target_price        FLOAT,                 -- model-estimated target
    stop_price          FLOAT,                 -- risk-based stop loss
    position_pct        FLOAT,                 -- recommended portfolio %
    thesis_plain        VARCHAR,               -- plain-English explanation (2-3 sentences)
    risk_plain          VARCHAR,               -- plain-English risk scenario
    time_horizon_days   INTEGER,               -- expected holding period
    status              VARCHAR CHECK (status IN ('active', 'closed_profit', 'closed_loss', 'closed_stopped', 'expired', 'cancelled')),
    closed_at           TIMESTAMP,
    close_price         FLOAT,
    realized_return     FLOAT,
    batch_id            VARCHAR                -- groups recommendations from same weekly run
)

-- Stores user/portfolio profiles for personalization
user_profiles (
    profile_id          VARCHAR PRIMARY KEY,    -- PROF-001
    created_at          TIMESTAMP DEFAULT current_timestamp,
    name                VARCHAR NOT NULL,
    risk_tolerance      INTEGER CHECK (risk_tolerance BETWEEN 1 AND 5),  -- 1=conservative, 5=aggressive
    portfolio_value     FLOAT NOT NULL,
    max_positions       INTEGER DEFAULT 20,
    max_position_pct    FLOAT DEFAULT 0.10,
    excluded_sectors    VARCHAR,               -- JSON list: ["Financials", "Energy"]
    excluded_symbols    VARCHAR,               -- JSON list: ["GOOGL"] (e.g., insider trading rules)
    preferred_horizon   VARCHAR DEFAULT 'medium',  -- short (<1mo), medium (1-3mo), long (>3mo)
    active              BOOLEAN DEFAULT TRUE
)

-- Aggregated track record for transparency
track_record (
    period_start        DATE NOT NULL,
    period_end          DATE NOT NULL,
    total_recommendations INTEGER,
    profitable          INTEGER,
    unprofitable        INTEGER,
    avg_return          FLOAT,
    avg_win             FLOAT,
    avg_loss            FLOAT,
    best_pick           VARCHAR,               -- symbol
    worst_pick          VARCHAR,
    benchmark_return    FLOAT,                 -- SPY return over same period
    excess_return       FLOAT,                 -- our return - benchmark
    PRIMARY KEY (period_start, period_end)
)
```

### 1.2 Recommendation Engine

**New file:** `hrp/advisory/recommendation_engine.py`

Core class that sits between the agent pipeline output and the consumer interface.

```python
@dataclass
class Recommendation:
    recommendation_id: str
    symbol: str
    action: Literal["BUY", "HOLD", "SELL"]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    signal_strength: float
    entry_price: float
    target_price: float
    stop_price: float
    position_pct: float
    thesis_plain: str          # "Apple shows strengthening momentum with institutional buying..."
    risk_plain: str            # "A broad market pullback could cause 5-10% drawdown..."
    time_horizon_days: int
    hypothesis_id: str
    model_name: str

class RecommendationEngine:
    """Transforms validated model predictions into actionable recommendations."""

    def __init__(self, api: PlatformAPI, profile: UserProfile):
        self.api = api
        self.profile = profile

    def generate_weekly_recommendations(self, as_of_date: date) -> list[Recommendation]:
        """
        Main entry point. Runs weekly.

        Pipeline:
        1. Get all deployed models (api.query_readonly)
        2. For each model: get latest predictions (api.predict_model)
        3. Filter by user profile constraints (sectors, symbols, risk)
        4. Rank by signal strength × confidence
        5. Apply portfolio construction (position sizing, correlation limits)
        6. Generate plain-English explanations (Explainer)
        7. Persist to recommendations table
        8. Return top 3-5 recommendations
        """

    def close_recommendation(self, recommendation_id: str, close_price: float, reason: str):
        """Close a recommendation when stop hit, target reached, or signal reverses."""

    def review_open_recommendations(self, as_of_date: date) -> list[RecommendationUpdate]:
        """Check all active recommendations: still valid? stop hit? target reached?"""
```

**Integration points:**
- Reads deployed models via `api.predict_model(model_name, symbols, as_of_date)`
- Reads CIO scores via `api.query_readonly()` on `cio_decisions` table
- Reads risk metrics via `VaRCalculator`
- Uses existing `SignalConverter` for position sizing (or new portfolio constructor, Phase 2)
- Writes to new `recommendations` table via `api.execute_write()`

### 1.3 Explainer Module

**New file:** `hrp/advisory/explainer.py`

Generates plain-English explanations from model outputs and feature data.

```python
class RecommendationExplainer:
    """Translates quantitative signals into plain-English explanations."""

    def generate_thesis(self, symbol: str, features: dict, prediction: float, model_name: str) -> str:
        """
        Input: AAPL, {momentum_20d: 0.85, rsi_14d: 62, sentiment_score_10k: 0.4}, pred=0.03
        Output: "Apple shows strong 20-day momentum with moderate buying pressure (RSI 62).
                 Recent 10-K filing sentiment is positive. The model expects a 3% return
                 over the next 20 trading days."

        Implementation:
        - Template-based for speed and consistency (no LLM call needed)
        - Maps feature values to human-readable descriptions
        - Includes key driver features (top 3 by importance)
        """

    def generate_risk_scenario(self, symbol: str, var_result: VaRResult, max_drawdown: float) -> str:
        """
        Input: AAPL, VaR_95=2.3%, max_dd=12%
        Output: "In a bad week (1-in-20 occurrence), this position could lose up to 2.3%.
                 Historical worst drawdown for this strategy was 12%. A broad market
                 correction would likely amplify this loss."
        """

    def generate_confidence_explanation(self, cio_score: CIOScore, stability_score: float) -> str:
        """
        Input: CIOScore(stat=0.8, risk=0.7, econ=0.9, cost=0.6), stability=0.4
        Output: "HIGH confidence — strong statistical backing (0.8/1.0), good risk profile,
                 excellent economic rationale. Cost efficiency is moderate. Model has been
                 stable across different market conditions."
        """
```

**Design choice — template-based, not LLM-based:** Explanations use templates with variable substitution. This is deterministic (same inputs always produce same output), fast (no API call), and auditable. Reserve LLM for special cases (weekly summary narrative).

### 1.4 Track Record Tracker

**New file:** `hrp/advisory/track_record.py`

```python
class TrackRecordTracker:
    """Tracks and reports recommendation outcomes with full transparency."""

    def compute_track_record(self, start_date: date, end_date: date) -> TrackRecordSummary:
        """
        Queries closed recommendations and computes:
        - Win rate (profitable / total closed)
        - Average return (all closed)
        - Average win, average loss
        - Best/worst picks
        - Comparison to SPY over same period
        - Excess return (our return - SPY)
        - Sharpe ratio of recommendations
        """

    def generate_weekly_report(self, as_of_date: date) -> WeeklyReport:
        """
        Produces a structured report:
        - This week's new recommendations
        - Status of open recommendations (P&L update)
        - Closed recommendations this week (outcomes)
        - Cumulative track record
        - Honest assessment: are we adding value vs. SPY?
        """

    def compute_rolling_metrics(self, window_days: int = 90) -> RollingMetrics:
        """Rolling win rate, rolling excess return, rolling Sharpe for trend analysis."""
```

### 1.5 PlatformAPI Extensions

**File:** `hrp/api/platform.py` — add methods:

```python
# Recommendation CRUD
api.create_recommendation(recommendation: Recommendation) -> str
api.get_recommendations(status: str | None, symbol: str | None, batch_id: str | None) -> pd.DataFrame
api.close_recommendation(recommendation_id: str, close_price: float, reason: str)
api.get_recommendation_by_id(recommendation_id: str) -> dict

# User profiles
api.create_user_profile(profile: UserProfile) -> str
api.get_user_profile(profile_id: str) -> dict
api.update_user_profile(profile_id: str, **updates)

# Track record
api.get_track_record(start_date: date, end_date: date) -> pd.DataFrame
api.get_recommendation_history(limit: int = 100) -> pd.DataFrame
```

### 1.6 Recommendation Agent

**New file:** `hrp/agents/recommendation_agent.py`

Scheduled agent that runs the recommendation pipeline weekly.

```python
class RecommendationAgent(ResearchAgent):
    """
    Weekly recommendation generation agent.

    Schedule: Sunday evening (prepares Monday recommendations)

    Pipeline:
    1. Load all user profiles
    2. For each profile:
        a. Get deployed models appropriate for risk tolerance
        b. Generate predictions
        c. Apply portfolio construction
        d. Generate explanations
        e. Persist recommendations
    3. Review open recommendations (close stopped/expired)
    4. Update track record
    5. Trigger notification (email digest)
    6. Log lineage event
    """
```

**Schedule:** Runs via launchd on Sunday evening, after data ingestion completes. Produces recommendations ready for Monday market open.

---

## Phase 2: Portfolio Construction + Risk Intelligence

**Duration:** ~2-3 weeks
**Goal:** Replace equal-weight sizing with proper portfolio optimization. Add pre-trade safeguards.

### 2.1 Portfolio Constructor

**New file:** `hrp/advisory/portfolio_constructor.py`

```python
@dataclass
class PortfolioConstraints:
    max_positions: int = 20
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30          # No more than 30% in one sector
    max_correlation: float = 0.70          # Don't hold highly correlated pairs
    max_weekly_turnover_pct: float = 0.20  # Limit rebalancing costs
    min_holding_days: int = 5              # Avoid churning

class PortfolioConstructor:
    """Optimized portfolio construction with constraints."""

    def construct(
        self,
        signals: pd.DataFrame,
        current_positions: dict[str, float],
        constraints: PortfolioConstraints,
        risk_model: CovarianceEstimator,
    ) -> PortfolioAllocation:
        """
        Steps:
        1. Estimate covariance matrix (Ledoit-Wolf shrinkage)
        2. Compute expected returns from signal strengths
        3. Optimize weights (minimize risk for target return, subject to constraints)
        4. Apply turnover constraint (limit distance from current portfolio)
        5. Round to whole shares
        6. Generate rebalancing orders
        """

class CovarianceEstimator:
    """Robust covariance estimation for portfolio optimization."""

    def estimate(self, returns: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
        """
        Methods:
        - ledoit_wolf: Shrinkage estimator (default, most robust)
        - sample: Raw sample covariance (baseline)
        - exponential: Exponentially weighted (more recent = more weight)
        """
```

**Integration:** Replaces the equal-weight path in `SignalConverter.signals_to_orders()`. The VaR-aware path in `PositionSizer` remains as an alternative.

### 2.2 Operational Safeguards

**New file:** `hrp/advisory/safeguards.py`

```python
class PreTradeChecks:
    """Sanity checks before any recommendation is generated or executed."""

    def check_data_freshness(self, as_of_date: date) -> CheckResult:
        """Verify price data is current (not stale by >1 trading day)."""

    def check_market_regime(self, as_of_date: date) -> CheckResult:
        """Detect extreme market conditions (VIX > 35, circuit breaker days)."""

    def check_portfolio_concentration(self, allocation: PortfolioAllocation) -> CheckResult:
        """Verify no single sector/factor dominance."""

    def check_order_reasonableness(self, orders: list[Order]) -> CheckResult:
        """Verify no single order > 5% ADV, total turnover within limits."""

class CircuitBreaker:
    """Halt recommendations under extreme conditions."""

    def should_halt(self, portfolio_daily_return: float) -> bool:
        """Halt if daily loss exceeds -3%."""

    def should_reduce(self, rolling_5d_return: float) -> bool:
        """Reduce position sizes by 50% if 5-day loss exceeds -5%."""
```

### 2.3 Historical Universe (Survivorship Bias Fix)

**New file:** `hrp/data/ingestion/historical_universe.py`

```python
class HistoricalUniverseIngestion:
    """Track S&P 500 membership changes over time."""

    def ingest_historical_membership(self):
        """
        Source: S&P 500 changes from Wikipedia/Quandl/custom CSV.
        Stores: (date, symbol, action) tuples for additions/removals.
        Enables: Point-in-time universe queries for bias-free backtesting.
        """

    def get_universe_as_of(self, as_of_date: date) -> list[str]:
        """Return the S&P 500 membership that existed on a specific date."""
```

**Schema addition:**
```sql
universe_history (
    effective_date  DATE NOT NULL,
    symbol          VARCHAR NOT NULL,
    action          VARCHAR CHECK (action IN ('ADDED', 'REMOVED')),
    reason          VARCHAR,
    PRIMARY KEY (effective_date, symbol, action)
)
```

---

## Phase 3: Consumer Interface + Notifications

**Duration:** ~2-3 weeks
**Goal:** Deliver recommendations to users via email, simple web view, and optional brokerage integration.

### 3.1 Email Digest Service

**New file:** `hrp/advisory/digest.py`

```python
class WeeklyDigest:
    """Generates and sends weekly recommendation email."""

    def generate_email(self, profile: UserProfile, recommendations: list[Recommendation],
                       open_positions: list[RecommendationUpdate],
                       track_record: TrackRecordSummary) -> EmailContent:
        """
        Structure:
        - Header: "Your Weekly Market Brief — Feb 24, 2026"
        - Section 1: "This Week's Recommendations" (3-5 picks with plain explanations)
        - Section 2: "Your Open Positions" (status, unrealized P&L)
        - Section 3: "Closed This Week" (realized outcomes)
        - Section 4: "Track Record" (win rate, cumulative return, vs SPY)
        - Footer: Risk disclaimer, unsubscribe link
        """

    def send(self, profile: UserProfile, content: EmailContent):
        """Send via existing Resend integration (hrp/notifications/)."""
```

Uses existing `hrp/notifications/` email infrastructure (Resend API).

### 3.2 Simple Web View

**New file:** `hrp/dashboard/pages/13_Recommendations.py`

A single Streamlit page optimized for non-technical users. Separate from the existing 14-page research dashboard.

```
┌──────────────────────────────────────────────────────────┐
│  📊 Your Recommendations            Track Record: +4.2% │
│                                       vs SPY: +1.8%     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  This Week (3 new)                                       │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 🟢 BUY Apple (AAPL) — HIGH confidence             │  │
│  │ "Strong momentum with institutional buying.        │  │
│  │  Positive 10-K sentiment. Expected +3-5% / 20d."  │  │
│  │ Risk: "Market pullback could cause 5-10% loss"     │  │
│  │ Size: 8% of portfolio | Stop: $182 | Target: $205  │  │
│  │ [Approve] [Skip] [Why this stock?]                 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Open Positions (8)                                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │ MSFT  +2.4%  │ NVDA -1.1%  │ UNH  +4.7%  │ ...  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Recent Outcomes                                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │ ✅ AMZN +5.2% (closed at target)                  │  │
│  │ ❌ META -3.1% (stopped out)                        │  │
│  │ ✅ LLY  +7.8% (closed at target)                  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Cumulative Track Record                                 │
│  [Equity curve chart: recommendations vs SPY]            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Brokerage Integration (One-Tap Approval)

Extends existing `LiveTradingAgent` with an approval workflow:

```python
class ApprovalWorkflow:
    """Bridges recommendations to execution with user approval."""

    def submit_for_approval(self, recommendations: list[Recommendation]):
        """Mark recommendations as pending_approval in DB."""

    def approve(self, recommendation_id: str):
        """User approves → convert to Order → submit to broker."""

    def approve_all(self):
        """Approve all pending recommendations at once."""

    def reject(self, recommendation_id: str, reason: str):
        """User rejects → mark as cancelled, log reason."""
```

---

## Phase 4: Feedback Loop + Continuous Improvement

**Duration:** ~2 weeks
**Goal:** Close the loop — recommendations inform model retraining.

### 4.1 Post-Trade Attribution

**New file:** `hrp/advisory/post_trade_attribution.py`

```python
class PostTradeAttributor:
    """Decompose recommendation outcomes into signal, timing, and sizing components."""

    def attribute(self, recommendation: ClosedRecommendation) -> Attribution:
        """
        Returns:
        - signal_contribution: Was the direction right? (buy signal for stock that went up)
        - timing_contribution: Did we enter/exit at good prices?
        - sizing_contribution: Did position size help or hurt?
        - cost_contribution: Transaction costs and slippage impact
        """

    def aggregate_attribution(self, closed_recs: list[ClosedRecommendation]) -> AggregateAttribution:
        """Over many closed recommendations, what's driving performance?"""
```

Integrates with existing `hrp/research/attribution/` framework (factor, feature, decision attribution).

### 4.2 Model Performance Monitoring

**New file:** `hrp/advisory/performance_monitor.py`

```python
class ModelPerformanceMonitor:
    """Tracks whether model predictions are accurate over time."""

    def check_prediction_accuracy(self, model_name: str, lookback_days: int = 60) -> AccuracyReport:
        """
        Compare recent predictions to actual outcomes.
        If directional accuracy drops below 50%, flag for retraining.
        If IC drops below 0.02, flag for retirement.
        """

    def check_feature_stability(self, model_name: str) -> StabilityReport:
        """
        Compare current feature importances to historical.
        Large shifts suggest model may be fitting to regime-specific noise.
        """

    def trigger_retraining(self, model_name: str, reason: str):
        """
        Log RETRAIN_TRIGGERED event to lineage.
        ML Scientist agent will pick up via event watcher and retrain.
        """
```

Integrates with existing `DriftMonitorJob` and `LineageEventWatcher`.

### 4.3 Kill Gate Calibration

Extend `KillGateEnforcer` to track its own false positive/negative rates:

```python
def calibrate_thresholds(self):
    """
    Look at hypotheses that passed kill gates and were later rejected,
    and hypotheses that were killed but similar ones later succeeded.
    Adjust thresholds to minimize false positives (killing good strategies)
    and false negatives (passing bad ones).
    """
```

---

## New Module Structure

```
hrp/
├── advisory/                          # NEW — Advisory Service layer
│   ├── __init__.py
│   ├── recommendation_engine.py       # Core: predictions → recommendations
│   ├── explainer.py                   # Plain-English explanation generator
│   ├── track_record.py               # Outcome tracking and reporting
│   ├── portfolio_constructor.py       # Optimized portfolio construction
│   ├── safeguards.py                 # Pre-trade checks, circuit breakers
│   ├── digest.py                     # Email digest generation
│   ├── approval_workflow.py          # User approval bridge to execution
│   ├── post_trade_attribution.py     # Outcome decomposition
│   └── performance_monitor.py        # Model accuracy tracking + retrain triggers
├── agents/
│   └── recommendation_agent.py        # NEW — Weekly recommendation scheduler
├── dashboard/
│   └── pages/
│       └── 13_Recommendations.py      # NEW — Consumer-facing recommendation view
└── data/
    └── ingestion/
        └── historical_universe.py     # NEW — Survivorship-bias-free universe
```

---

## Integration Map

How new modules connect to existing infrastructure:

| New Module | Reads From (existing) | Writes To (new) |
|---|---|---|
| `RecommendationEngine` | `api.predict_model()`, `api.query_readonly()` (CIO scores, hypotheses) | `recommendations` table |
| `Explainer` | Feature store (`api.get_features()`), `VaRCalculator`, `CIOScore` | `recommendations.thesis_plain`, `risk_plain` |
| `TrackRecordTracker` | `recommendations` table, `api.get_prices()` (SPY benchmark) | `track_record` table |
| `PortfolioConstructor` | `api.get_prices()` (for covariance), `recommendations` | Modifies recommendation `position_pct` |
| `Safeguards` | `api.get_prices()`, `api.run_quality_checks()` | Blocks/modifies recommendations |
| `RecommendationAgent` | All of the above | `recommendations`, `lineage` |
| `Digest` | `recommendations`, `track_record`, `user_profiles` | Sends email via `hrp/notifications/` |
| `ApprovalWorkflow` | `recommendations` | `api.record_trade()` via `LiveTradingAgent` |
| `PerformanceMonitor` | `recommendations`, `api.predict_model()` | `lineage` (RETRAIN_TRIGGERED event) |
| `HistoricalUniverse` | External sources (Wikipedia, CSV) | `universe_history` table |

---

## New Lineage Event Types

```python
# Add to hrp/research/lineage.py EventType
RECOMMENDATION_GENERATED = "RECOMMENDATION_GENERATED"    # Weekly batch created
RECOMMENDATION_APPROVED = "RECOMMENDATION_APPROVED"       # User approved
RECOMMENDATION_REJECTED = "RECOMMENDATION_REJECTED"       # User rejected
RECOMMENDATION_CLOSED = "RECOMMENDATION_CLOSED"           # Position closed
TRACK_RECORD_UPDATED = "TRACK_RECORD_UPDATED"             # Weekly track record refresh
RETRAIN_TRIGGERED = "RETRAIN_TRIGGERED"                   # Performance degradation detected
CIRCUIT_BREAKER_ACTIVATED = "CIRCUIT_BREAKER_ACTIVATED"   # Emergency halt
```

---

## New Environment Variables

```bash
# Advisory Service
HRP_ADVISORY_ENABLED=true                    # Master switch
HRP_ADVISORY_MAX_RECOMMENDATIONS=5           # Per-profile per-week cap
HRP_ADVISORY_MIN_CONFIDENCE=MEDIUM           # Minimum confidence to recommend
HRP_ADVISORY_PAPER_ONLY=true                 # Paper trading only (safety default)

# Portfolio Construction
HRP_PORTFOLIO_MAX_SECTOR_PCT=0.30            # Max 30% in one sector
HRP_PORTFOLIO_MAX_CORRELATION=0.70           # Block highly correlated pairs
HRP_PORTFOLIO_MAX_WEEKLY_TURNOVER=0.20       # 20% max weekly rebalancing

# Circuit Breakers
HRP_CIRCUIT_BREAKER_DAILY_LOSS=-0.03         # Halt at -3% daily
HRP_CIRCUIT_BREAKER_WEEKLY_LOSS=-0.05        # Reduce sizing at -5% weekly

# Notifications
HRP_DIGEST_DAY=sunday                        # Day to send weekly digest
HRP_DIGEST_HOUR=18                           # Hour to send (local time)
```

---

## Implementation Sequence

```
Phase 1 (Weeks 1-3): Foundation
├── 1a: Schema extensions (recommendations, user_profiles, track_record, universe_history)
├── 1b: RecommendationEngine + Explainer
├── 1c: TrackRecordTracker
├── 1d: PlatformAPI extensions
├── 1e: RecommendationAgent (scheduled job)
└── 1f: Tests for all Phase 1 modules

Phase 2 (Weeks 4-6): Risk + Optimization
├── 2a: PortfolioConstructor (Ledoit-Wolf, constraints)
├── 2b: Safeguards (pre-trade checks, circuit breakers)
├── 2c: HistoricalUniverseIngestion (survivorship fix)
└── 2d: Tests for all Phase 2 modules

Phase 3 (Weeks 7-9): Consumer Interface
├── 3a: WeeklyDigest email generation + sending
├── 3b: Recommendations dashboard page
├── 3c: ApprovalWorkflow (approve/reject/auto)
└── 3d: Tests for all Phase 3 modules

Phase 4 (Weeks 10-11): Feedback Loop
├── 4a: PostTradeAttributor
├── 4b: ModelPerformanceMonitor (accuracy tracking, retrain triggers)
├── 4c: Kill gate calibration
└── 4d: Tests for all Phase 4 modules
```

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Weekly recommendation batch generates 3-5 picks per profile
- [ ] Each recommendation has plain-English thesis and risk explanation
- [ ] Track record computes win rate, avg return, excess vs SPY
- [ ] All data persisted and queryable via PlatformAPI
- [ ] Lineage events logged for audit trail

### Phase 2 Complete When:
- [ ] Portfolio weights use Ledoit-Wolf shrinkage + constraints (not equal-weight)
- [ ] Pre-trade checks block stale data, extreme concentration, unreasonable orders
- [ ] Circuit breaker halts recommendations during extreme losses
- [ ] Historical universe enables survivorship-bias-free backtest comparison

### Phase 3 Complete When:
- [ ] Weekly email digest sent automatically with recommendations + track record
- [ ] Dashboard page shows recommendations, open positions, outcomes, cumulative performance
- [ ] Users can approve/reject individual recommendations
- [ ] Approved recommendations flow to execution (paper or live)

### Phase 4 Complete When:
- [ ] Closed recommendations decomposed into signal/timing/sizing/cost attribution
- [ ] Model accuracy monitored; RETRAIN_TRIGGERED event fires when accuracy drops
- [ ] Kill gate thresholds adjusted based on historical false positive/negative rates

### Overall:
- [ ] 90-day paper trading period produces meaningful track record
- [ ] Track record transparently shows wins, losses, and comparison to SPY
- [ ] A non-technical user can receive, understand, and act on recommendations with zero knowledge of ML, statistics, or trading
