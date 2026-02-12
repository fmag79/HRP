# Tier 4: Trading Setup Guide

Complete setup guide for enabling live/paper trading in HRP.

## Overview

Tier 4 adds live trading execution capabilities:
- Multi-broker support: IBKR and Robinhood (configurable via `HRP_BROKER_TYPE`)
- Daily prediction generation for deployed models
- Signal-to-order conversion with VaR-aware risk limits
- 5 order types: market, limit, stop_loss, stop_limit, trailing_stop
- Position tracking and P&L monitoring
- Model drift detection with auto-rollback option
- Auto stop-loss generation

## Prerequisites

- Tier 1-3 complete (data, research, ML, ops)
- At least one deployed strategy (status = 'deployed')
- A supported broker account (IBKR or Robinhood)

## Broker Selection

HRP supports two brokers. Set `HRP_BROKER_TYPE` to choose:

| Feature | IBKR | Robinhood |
|---------|------|-----------|
| `HRP_BROKER_TYPE` | `ibkr` (default) | `robinhood` |
| Order Types | market, limit, stop, stop-limit, trailing | market, limit, stop, stop-limit, trailing |
| Paper Trading | Yes (port 7497) | No (live only) |
| Authentication | TWS/Gateway | OAuth + MFA (pyotp) |
| Commission | Per-trade | Commission-free |
| Rate Limiting | Custom | Built-in (5 req/15s, 2s order cooldown) |

### IBKR Setup

See `docs/operations/ibkr-setup-guide.md` for full IBKR configuration.

### Robinhood Setup

```bash
# Required environment variables
export HRP_BROKER_TYPE=robinhood
export ROBINHOOD_USERNAME=your_email@example.com
export ROBINHOOD_PASSWORD=your_password
export ROBINHOOD_TOTP_SECRET=your_totp_secret   # from Robinhood authenticator setup
```

**Key files:**
- `hrp/execution/robinhood_broker.py` — Broker client implementing BaseBroker protocol
- `hrp/execution/robinhood_auth.py` — MFA session management with pyotp TOTP
- `hrp/execution/rate_limiter.py` — Thread-safe token bucket rate limiter

**Test connection:**

```python
from hrp.execution.robinhood_broker import RobinhoodBroker

broker = RobinhoodBroker()
print(f"Connected: {broker.is_connected()}")
positions = broker.get_positions()
print(f"Positions: {len(positions)}")
```

## Database Migration

**Run schema updates:**

```bash
python -m hrp.data.schema --init
```

**Verify tables:**

```bash
python -c "
from hrp.data.db import get_db

conn = get_db(read_only=True)
tables = conn.execute(\"SHOW TABLES\").fetchdf()
print('Trading tables:', [t for t in tables['name'] if 'trade' in t or 'position' in t])
"
```

Expected tables: `executed_trades`, `live_positions`

## Job Configuration

### 1. Daily Predictions

**Schedule:** Daily at 6:15 PM ET (after feature computation)

```bash
# Test manually
python -m hrp.agents.run_job --job predictions --dry-run
python -m hrp.agents.run_job --job predictions
```

**Check logs:**

```bash
tail -f ~/hrp-data/logs/predictions.log
```

### 2. Drift Monitoring

**Schedule:** Daily at 7:00 PM ET (after predictions)

```bash
# Test drift check
python -m hrp.agents.run_job --job drift-monitor --dry-run
```

### 3. Live Trading (DISABLED by default)

**IMPORTANT:** Live trader is disabled by default for safety.

```bash
# Test in dry-run mode first
python -m hrp.agents.run_job --job live-trader --trading-dry-run

# When ready, execute trades (DANGEROUS)
python -m hrp.agents.run_job --job live-trader --execute-trades
```

## Portfolio & Position Sizing

All portfolio sizing variables can be set in `.env` (see `.env.example` for the full list):

```bash
# Portfolio sizing (in .env)
HRP_PORTFOLIO_VALUE=100000        # Portfolio value in dollars
HRP_MAX_POSITIONS=20              # Max concurrent positions
HRP_MAX_POSITION_PCT=0.10         # 10% max per position
HRP_MIN_ORDER_VALUE=100           # Minimum order value in dollars
HRP_TRADING_DRY_RUN=true          # Dry-run mode (safe default)
```

## VaR-Aware Position Sizing

Position sizing can be constrained by VaR budgets. Configure in `.env`:

```bash
HRP_USE_VAR_SIZING=true
HRP_MAX_PORTFOLIO_VAR_PCT=0.02    # 2% portfolio VaR (daily, 95% confidence)
HRP_MAX_POSITION_VAR_PCT=0.005    # 0.5% per-position VaR
HRP_AUTO_STOP_LOSS_PCT=0.05       # 5% auto stop-loss on new positions
```

The `LiveTradingAgent` automatically:
1. Calculates VaR for each proposed position using historical returns
2. Reduces position size if it would breach VaR limits
3. Generates stop-loss orders for all new positions
4. Polls order status until filled or timeout

See `docs/operations/var-risk-metrics.md` for VaR calculator details.

## Verification Checklist

### Pre-Trading

- [ ] `HRP_BROKER_TYPE` set (`ibkr` or `robinhood`)
- [ ] Broker connection test passes
- [ ] For IBKR: TWS/Gateway running and connected
- [ ] For Robinhood: `ROBINHOOD_*` env vars set and MFA verified
- [ ] At least one strategy deployed
- [ ] Daily predictions job runs successfully
- [ ] Drift monitoring job runs successfully
- [ ] Trading dashboard page (`10_Trading.py`) displays correctly

### Trading Enabled

- [ ] Live trader runs in dry-run mode without errors
- [ ] Orders generated match expected signals
- [ ] Position limits enforced (max 20 positions, 10% each)
- [ ] Minimum order value check works ($100)
- [ ] VaR limits enforced (if `HRP_USE_VAR_SIZING=true`)
- [ ] Auto stop-loss orders generated correctly
- [ ] Drift monitoring detects test drift
- [ ] Dashboard shows positions and trades

### Production Ready

- [ ] Tested in paper trading for 1+ week (IBKR only — Robinhood has no paper mode)
- [ ] No drift detected on deployed models
- [ ] Order execution matches backtest costs
- [ ] Position reconciliation works correctly
- [ ] Rollback mechanism tested
- [ ] Emergency stop procedure documented

## Monitoring

### Dashboard

Access trading dashboard:

```bash
streamlit run hrp/dashboard/app.py
```

Navigate to **Trading** page:
- Portfolio overview (value, P&L, positions)
- Current positions table
- Recent trades history
- Model performance and drift status

### Logs

Monitor execution logs:

```bash
# Predictions
tail -f ~/hrp-data/logs/predictions.log

# Drift monitoring
tail -f ~/hrp-data/logs/drift-monitor.log

# Live trading
tail -f ~/hrp-data/logs/live-trader.log
```

### Alerts

Email notifications configured via Resend (see `.env`):
- Job failures
- Drift detection
- Order execution errors

## Emergency Procedures

### Stop All Trading

```bash
# Stop running jobs
pkill -f "hrp.agents.run_job"
```

### Manual Rollback

```python
from hrp.api.platform import PlatformAPI

api = PlatformAPI()

# Manual rollback
api.rollback_deployment(
    model_name="momentum_v1",
    to_version=None,  # Previous version
    actor="user",
    reason="Manual emergency rollback"
)
```

### Close All Positions

Manually close positions via TWS/Gateway, then sync:

```python
from hrp.execution.broker import IBKRBroker, BrokerConfig
from hrp.execution.positions import PositionTracker
from hrp.api.platform import PlatformAPI

api = PlatformAPI()
config = BrokerConfig(...)  # From env

with IBKRBroker(config) as broker:
    tracker = PositionTracker(broker, api)
    positions = tracker.sync_positions()
    tracker.persist_positions()
```

## Troubleshooting

### No Predictions Generated

**Check:**
1. Deployed strategies exist: `api.get_deployed_strategies()`
2. Model has production version in MLflow
3. Universe has symbols: `api.get_universe()`
4. Feature data up to date

### Orders Not Submitted

**Check:**
1. Broker connection: Test connection script
2. TWS/Gateway running
3. API enabled in TWS settings
4. Portfolio value > 0
5. Signals generated (check logs)

### Drift False Positives

**Solutions:**
1. Increase drift threshold in job config
2. Collect more baseline data
3. Disable auto-rollback
4. Manual drift investigation

## Next Steps

1. Run in paper trading for 1-2 weeks
2. Compare live execution to backtest performance
3. Monitor slippage and commissions
4. Adjust position sizing if needed
5. Enable auto-rollback once drift monitoring validated
6. Document any edge cases or issues

## Support

- **IBKR Issues:** See `docs/operations/ibkr-setup-guide.md`
- **Robinhood Issues:** Check `~/hrp-data/logs/live-trader.log` for auth/rate-limit errors
- **VaR/Risk Metrics:** See `docs/operations/var-risk-metrics.md`
- **Job Failures:** Check `~/hrp-data/logs/` for error details
- **API Errors:** Enable debug logging in `.env`: `LOG_LEVEL=DEBUG`
