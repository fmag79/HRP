# Head Start: Building Your Initial Portfolio

A systematic approach to deploying your first $10k using HRP.

## Prerequisites

Before deploying capital, ensure:

- [ ] Database populated with 5+ years of price history
- [ ] Universe updated with current S&P 500 constituents
- [ ] Features computed for all symbols
- [ ] Scheduler running daily ingestion jobs
- [ ] At least one hypothesis validated through walk-forward

## Phase 1: Validate Your Edge (Weeks 1-4)

You need statistical evidence before risking capital.

### Step 1.1: Discover Signals

```python
from hrp.agents import SignalScientist

scientist = SignalScientist(
    symbols=None,           # All universe symbols
    features=None,          # All 44 features
    forward_horizons=[5, 10, 20],
    ic_threshold=0.03,      # Minimum IC to create hypothesis
    create_hypotheses=True,
)
result = scientist.run()

print(f"Signals found: {result['signals_found']}")
print(f"Hypotheses created: {result['hypotheses_created']}")
```

### Step 1.2: Review and Promote

```python
from hrp.agents import AlphaResearcher

researcher = AlphaResearcher()
result = researcher.run()

print(f"Analyzed: {result['hypotheses_analyzed']}")
print(f"Promoted to testing: {result['promoted_to_testing']}")
```

### Step 1.3: Walk-Forward Validate

This is the critical step. Walk-forward validation prevents overfitting.

```python
from hrp.agents import MLScientist

scientist = MLScientist(
    n_folds=5,
    window_type='expanding',
    stability_threshold=1.0,  # Lower is better
)
result = scientist.run()

print(f"Hypotheses validated: {result['hypotheses_validated']}")
print(f"Hypotheses rejected: {result['hypotheses_rejected']}")
```

### Step 1.4: Quality Audit

```python
from hrp.agents import MLQualitySentinel

sentinel = MLQualitySentinel(audit_window_days=30, send_alerts=True)
result = sentinel.run()

print(f"Critical issues: {result['critical_issues']}")
print(f"Warnings: {result['warnings']}")
```

### Minimum Bar for Deployment

| Metric | Requirement |
|--------|-------------|
| Walk-forward stability | ≤ 1.0 |
| Positive IC | Majority of folds |
| Test Sharpe | > 0.5 (after costs) |
| Sharpe decay | < 50% train-to-test |
| Test set evaluations | ≤ 3 per hypothesis |

## Phase 2: Portfolio Construction

### Constraints for $10k

| Constraint | Value | Implication |
|------------|-------|-------------|
| Transaction cost | ~$1/trade | 0.01% per position on $10k |
| Minimum position | $500-1000 | Must be large enough to matter |
| Max positions | 10-20 | Diversification vs. concentration |
| Rebalance frequency | Monthly | Limit transaction costs |

### Recommended Allocation

```
Total Capital: $10,000
├── Core Portfolio (70% = $7,000)
│   ├── 10-15 positions @ $450-700 each
│   ├── Strategy: Top validated signal
│   └── Weighting: Equal-weight or signal-strength
│
└── Reserve (30% = $3,000)
    ├── Cash buffer for rebalancing
    ├── Opportunistic additions
    └── Drawdown cushion
```

### Position Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| Max single position | 10% ($1k) | Concentration risk |
| Max sector | 30% | Sector diversification |
| Stop-loss per position | 15-20% | Capital preservation |

## Phase 3: Strategy Selection

For small accounts, simplicity wins. Start with single-factor strategies.

### Option A: Momentum (Recommended)

Most robust factor historically, works across markets.

```python
from hrp.research.strategies import generate_multifactor_signals
from hrp.research.backtest import get_price_data, run_backtest
from hrp.research.config import BacktestConfig
from datetime import date

# Load prices
prices = get_price_data(universe_symbols, date(2015, 1, 1), date.today())

# Single-factor momentum
signals = generate_multifactor_signals(
    prices,
    feature_weights={"momentum_252d": 1.0},  # 12-month momentum
    top_n=15,
)

# Backtest with realistic costs
config = BacktestConfig(
    symbols=universe_symbols,
    start_date=date(2015, 1, 1),
    end_date=date.today(),
    commission=0.001,   # 10bps round-trip
    slippage=0.001,     # 10bps slippage
)

result = run_backtest(signals, config, prices)
```

### Option B: Quality + Momentum

Combine factors for potentially smoother returns.

```python
signals = generate_multifactor_signals(
    prices,
    feature_weights={
        "momentum_252d": 0.6,     # 12-month momentum
        "volatility_60d": -0.4,   # Penalize high volatility
    },
    top_n=15,
)
```

### Option C: ML-Predicted Signals

Higher complexity, requires more validation.

```python
from hrp.research.strategies import generate_ml_predicted_signals

signals = generate_ml_predicted_signals(
    prices,
    model_type="ridge",
    features=["momentum_20d", "momentum_252d", "volatility_60d", "rsi_14d"],
    signal_method="rank",
    top_pct=0.1,
    train_lookback=252,
    retrain_frequency=21,
)
```

## Phase 4: Risk Management

### Configure Trailing Stops

```python
from hrp.research.config import StopLossConfig

stop_config = StopLossConfig(
    enabled=True,
    type="atr_trailing",
    atr_multiplier=3.0,   # Wider for daily timeframe
    atr_period=14,
)
```

### Monitor Regime

```python
from hrp.ml import HMMConfig, RegimeDetector

config = HMMConfig(
    n_regimes=3,
    features=['returns_20d', 'volatility_20d'],
)

detector = RegimeDetector(config)
detector.fit(prices)

result = detector.get_regime_statistics(prices)
print(f"Current regime: {result.current_regime}")
```

Consider reducing position sizes in high-volatility regimes.

## Phase 5: Deployment Timeline

| Week | Action | Capital at Risk |
|------|--------|-----------------|
| 1-2 | Run SignalScientist, identify candidates | $0 |
| 3-4 | Walk-forward validate, select best | $0 |
| 5-8 | Paper trade using dashboard | $0 |
| 9 | Deploy first $5k (half capital) | $5k |
| 10-12 | Monitor vs. backtest expectations | $5k |
| 13+ | Add remaining $5k if results align | $10k |

## Phase 6: Ongoing Operations

### Daily

```bash
# Ensure scheduler is running
launchctl list | grep hrp

# Check for alerts
tail -20 ~/hrp-data/logs/scheduler.error.log
```

### Weekly

```python
from hrp.api.platform import PlatformAPI

api = PlatformAPI()
result = api.run_quality_checks(as_of_date=date.today(), send_alerts=True)
print(f"Health score: {result['health_score']}")
```

### Monthly

1. Review strategy performance vs. backtest
2. Check for hypothesis degradation
3. Run MLQualitySentinel audit
4. Rebalance portfolio

## Realistic Expectations

For $10k with a validated strategy:

| Metric | Target Range |
|--------|--------------|
| Annual return | 8-15% above benchmark |
| Sharpe ratio | 0.5-0.8 (after costs) |
| Max drawdown | 15-20% |
| Win rate | 50-55% |

### What Success Looks Like

- Strategy performs within 1 standard deviation of backtest
- No unexpected large losses
- Consistent adherence to rules
- Learning and iteration

### Red Flags

- Performance significantly below backtest → investigate data issues
- Multiple stop-losses triggered → review regime/volatility
- Strategy stopped working → hypothesis may be invalidated

## Checklist Before Going Live

- [ ] Hypothesis passed walk-forward validation (stability ≤ 1.0)
- [ ] Test Sharpe > 0.5 after realistic costs
- [ ] Sharpe decay < 50%
- [ ] Paper traded for 4+ weeks
- [ ] Brokerage account funded
- [ ] Stop-loss rules defined
- [ ] Maximum position sizes set
- [ ] Rebalancing schedule determined
- [ ] Alerting configured
- [ ] Emergency exit criteria defined

## Next Steps

After successful deployment:

1. **Month 3**: Evaluate adding second factor/strategy
2. **Month 6**: Consider increasing capital if results align
3. **Year 1**: Full portfolio review, hypothesis refresh
