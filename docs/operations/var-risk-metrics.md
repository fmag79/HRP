# VaR and CVaR Risk Metrics

## Overview

The HRP platform includes comprehensive Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) calculations for portfolio risk monitoring and management.

## Features

### 1. VaR Calculator

The `VaRCalculator` class provides three methods for calculating VaR:

- **Parametric Method**: Assumes returns follow a specified distribution (normal or Student's t)
- **Historical Simulation**: Uses empirical distribution of historical returns
- **Monte Carlo**: Simulates returns based on fitted distribution parameters

### 2. Risk Metrics Dashboard

Located at `hrp/dashboard/pages/11_Risk_Metrics.py`, the dashboard provides:

- **Portfolio VaR Overview**: Portfolio-level VaR and CVaR with dollar values
- **Per-Symbol VaR Breakdown**: Risk contribution by individual positions
- **Method Comparison**: Side-by-side comparison of all three VaR methods
- **Historical VaR Tracking**: Rolling VaR over time with breach analysis

## Usage

### Basic VaR Calculation

```python
from hrp.data.risk.var_calculator import VaRCalculator
from hrp.data.risk.risk_config import VaRConfig, VaRMethod
import numpy as np

# Create calculator
calculator = VaRCalculator()

# Generate or load returns data
returns = np.array([0.01, -0.02, 0.015, -0.005, ...])  # Daily returns

# Calculate VaR
result = calculator.calculate(
    returns=returns,
    portfolio_value=100000.0,  # Optional: for dollar VaR
)

print(f"VaR: {result.var * 100:.2f}%")
print(f"CVaR: {result.cvar * 100:.2f}%")
print(f"VaR (Dollar): ${result.var_dollar:,.2f}")
```

### Custom Configuration

```python
from hrp.data.risk.risk_config import VaRConfig, VaRMethod, Distribution

# 99% VaR with 10-day horizon using t-distribution
config = VaRConfig(
    confidence_level=0.99,
    time_horizon=10,
    method=VaRMethod.PARAMETRIC,
    distribution=Distribution.T,
    df=5.0,  # Degrees of freedom for t-distribution
)

calculator = VaRCalculator(config)
result = calculator.calculate(returns)
```

### Comparing Methods

```python
# Calculate VaR using all three methods
results = calculator.calculate_all_methods(
    returns=returns,
    portfolio_value=100000.0,
    confidence_level=0.95,
    time_horizon=1,
)

for method_name, result in results.items():
    print(f"{method_name}: VaR = {result.var*100:.2f}%, CVaR = {result.cvar*100:.2f}%")
```

## Configuration Options

### VaRConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_level` | float | 0.95 | Confidence level (e.g., 0.95 for 95% VaR) |
| `time_horizon` | int | 1 | Time horizon in days |
| `method` | VaRMethod | PARAMETRIC | Calculation method |
| `distribution` | Distribution | NORMAL | Distribution assumption |
| `n_simulations` | int | 10000 | Number of Monte Carlo simulations |
| `window_size` | int | 252 | Historical window size in days |
| `df` | float | 5.0 | Degrees of freedom for t-distribution |

### Pre-configured Options

```python
from hrp.data.risk.risk_config import (
    VAR_95_1D,   # 95% VaR, 1-day horizon
    VAR_99_1D,   # 99% VaR, 1-day horizon
    VAR_95_10D,  # 95% VaR, 10-day horizon
    VAR_99_10D,  # 99% VaR, 10-day horizon
    MC_VAR_95_1D, # 95% VaR using Monte Carlo with t-distribution
)

calculator = VaRCalculator(VAR_99_1D)
```

## VaR Methods Explained

### Parametric VaR

**Formula**: `VaR = -μT + σ√T * z`

Where:
- μ = mean return
- σ = standard deviation of returns
- T = time horizon
- z = quantile of distribution at confidence level

**Pros**:
- Fast to calculate
- Smooth estimates
- Works well for normally distributed returns

**Cons**:
- Assumes specific distribution (may not capture fat tails)
- Poor performance with non-normal returns

### Historical Simulation

**Method**: Uses empirical distribution of historical returns

For multi-day VaR, calculates overlapping N-day returns to capture path dependency.

**Pros**:
- No distribution assumptions
- Captures actual historical behavior
- Accounts for fat tails and skewness

**Cons**:
- Limited by historical data
- May not reflect future risk
- Sensitive to lookback period

### Monte Carlo

**Method**: Simulates future returns based on fitted distribution

1. Fit distribution to historical returns
2. Simulate many paths
3. Calculate VaR from simulated distribution

**Pros**:
- Flexible (can use any distribution)
- Can model complex scenarios
- Smooth estimates with large simulations

**Cons**:
- Computationally intensive
- Quality depends on distribution fit
- May overfit to recent history

## Dashboard Usage

### Accessing the Dashboard

1. Start the HRP dashboard: `streamlit run hrp/dashboard/app.py`
2. Navigate to "Risk Metrics" in the sidebar

### Dashboard Features

#### Configuration Controls

- **Confidence Level**: 90%, 95%, or 99%
- **Time Horizon**: 1, 5, 10, or 21 days
- **Method**: Parametric, Historical, or Monte Carlo
- **Distribution**: Normal or Student's t

#### Portfolio VaR Overview

Displays four key metrics:
- **Portfolio VaR**: Maximum expected loss (as percentage)
- **Portfolio CVaR**: Expected loss in worst-case scenarios
- **VaR (Dollar)**: VaR in dollar terms based on current portfolio value
- **CVaR (Dollar)**: CVaR in dollar terms

Plus a histogram showing the return distribution with VaR/CVaR thresholds.

#### Per-Symbol VaR Breakdown

Shows VaR contribution for each position:
- Symbol
- Position Value
- VaR % (risk as percentage of position)
- CVaR %
- VaR $ (dollar risk)
- CVaR $
- VaR/Value ratio

Includes a bar chart showing VaR contribution by position.

#### Method Comparison

Compares VaR calculated using all three methods:
- Side-by-side VaR and CVaR values
- CVaR/VaR ratio for each method
- Grouped bar chart for visual comparison

#### Historical VaR Tracking

Shows VaR evolution over time:
- Rolling VaR and CVaR (configurable window size)
- Actual daily returns overlaid
- **Breach Analysis**:
  - Number of VaR breaches (days when loss exceeded VaR)
  - Breach rate (percentage of days with breaches)
  - Model calibration status

## Environment Configuration

> **All VaR and risk variables are documented in `.env.example`.** Set them in your `.env` file:

```bash
HRP_USE_VAR_SIZING=true              # Enable VaR-aware position sizing
HRP_MAX_PORTFOLIO_VAR_PCT=0.02       # 2% portfolio VaR limit (daily, 95%)
HRP_MAX_POSITION_VAR_PCT=0.005       # 0.5% per-position VaR limit
HRP_AUTO_STOP_LOSS_PCT=0.05          # 5% auto stop-loss (optional)
```

## Integration with Trading

### VaR-Aware Position Sizing

The VaR calculator integrates with the position sizer (TASK-010) to ensure trades respect portfolio VaR limits:

```python
from hrp.execution.position_sizer import PositionSizer
from hrp.data.risk.risk_config import VAR_95_1D

sizer = PositionSizer(
    max_position_size=0.10,  # 10% max per position
    var_budget=0.05,          # 5% total VaR budget
    var_config=VAR_95_1D,
)

# Position sizing respects both position limit and VaR budget
size = sizer.calculate_size(
    signal_strength=0.8,
    symbol="AAPL",
    portfolio_value=100000,
)
```

See `docs/operations/position-sizing.md` for details.

## Risk Limits

### Recommended VaR Budgets

| Portfolio Type | VaR Budget (95%) | CVaR Budget |
|----------------|------------------|-------------|
| Conservative | 2-3% | 3-4% |
| Moderate | 3-5% | 5-7% |
| Aggressive | 5-10% | 8-15% |

### VaR Monitoring

The platform supports automated VaR monitoring:

1. **Daily VaR Calculation**: Runs at market close
2. **Breach Alerts**: Notifications when VaR is exceeded
3. **Model Calibration**: Tracks breach rate vs. expected rate
4. **Escalation**: Alerts when breach rate deviates significantly

## Mathematical Foundations

### CVaR Definition

CVaR (also called Expected Shortfall) is the expected loss given that loss exceeds VaR:

```
CVaR_α = E[Loss | Loss > VaR_α]
```

Where α is the confidence level.

### Important Properties

1. **CVaR ≥ VaR**: CVaR is always greater than or equal to VaR
2. **Subadditivity**: CVaR is subadditive (VaR is not), making it a coherent risk measure
3. **Tail Risk**: CVaR captures tail risk better than VaR

### Multi-Day VaR Scaling

For parametric VaR, the square root rule applies:

```
VaR_T = VaR_1 * √T
```

For historical and Monte Carlo, actual multi-day paths are used to avoid this approximation.

## Testing

Comprehensive tests are available in `tests/test_risk/`:

- **Unit Tests**: `test_var_calculator.py` (19 test cases)
- **Dashboard Tests**: `test_dashboard/test_risk_metrics_page.py` (comprehensive)
- **Integration Tests**: End-to-end VaR calculation workflows

Run tests:

```bash
# All VaR tests
pytest tests/test_risk/test_var_calculator.py -v

# Dashboard tests
pytest tests/test_dashboard/test_risk_metrics_page.py -v

# All risk tests
pytest tests/test_risk/ -v
```

## Troubleshooting

### "Insufficient data" errors

**Problem**: VaR calculation requires at least 30 valid return observations.

**Solution**: Ensure you have enough historical data. For new positions, use portfolio-level VaR until sufficient symbol-specific history is available.

### VaR seems too low/high

**Problem**: VaR estimates don't match expectations.

**Solution**:
1. Check the confidence level (95% vs 99% makes a big difference)
2. Verify the time horizon (1-day vs 10-day)
3. Try different methods (historical vs parametric)
4. Check for data quality issues (outliers, gaps)

### Breach rate significantly deviates from expected

**Problem**: VaR is breached much more or less than expected (e.g., 10% breach rate at 95% confidence).

**Solution**:
1. **Too many breaches**: Model underestimates risk
   - Use t-distribution instead of normal
   - Increase window size to capture more history
   - Use Monte Carlo with fat-tailed distribution
2. **Too few breaches**: Model overestimates risk
   - Reduce window size to focus on recent volatility
   - Use parametric method with normal distribution

### Monte Carlo results unstable

**Problem**: Monte Carlo VaR varies significantly between runs.

**Solution**:
1. Increase `n_simulations` (default: 10,000)
2. Set random seed for reproducibility
3. Check distribution fit quality

## References

- **Jorion, P.** (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*
- **McNeil, A. J., Frey, R., & Embrechts, P.** (2015). *Quantitative Risk Management*
- **Basel Committee** (2019). *Minimum capital requirements for market risk*

## Related Documentation

- [Position Sizing](position-sizing.md) - VaR-aware position sizing
- [Risk Management](risk-management.md) - Overall risk management framework
- [Trading Dashboard](../dashboard/trading.md) - Portfolio monitoring
