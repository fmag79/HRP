# Tier 4: Trading Setup Guide

Complete setup guide for enabling live/paper trading in HRP.

## Overview

Tier 4 adds live trading execution capabilities:
- Daily prediction generation for deployed models
- Signal-to-order conversion with risk limits
- IBKR broker integration for order execution
- Position tracking and P&L monitoring
- Model drift detection with auto-rollback option

## Prerequisites

- Tier 1-3 complete (data, research, ML, ops)
- At least one deployed strategy (status = 'deployed')
- Interactive Brokers paper trading account
- TWS/IB Gateway installed and configured

See `docs/operations/ibkr-setup-guide.md` for IBKR setup.

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

## Verification Checklist

### Pre-Trading

- [ ] IBKR TWS/Gateway running and connected
- [ ] Broker connection test passes
- [ ] At least one strategy deployed
- [ ] Daily predictions job runs successfully
- [ ] Drift monitoring job runs successfully
- [ ] Trading dashboard page displays correctly

### Trading Enabled

- [ ] Live trader runs in dry-run mode without errors
- [ ] Orders generated match expected signals
- [ ] Position limits enforced (max 20 positions, 10% each)
- [ ] Minimum order value check works ($100)
- [ ] Drift monitoring detects test drift
- [ ] Dashboard shows positions and trades

### Production Ready

- [ ] Tested in paper trading for 1+ week
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
- **Job Failures:** Check `~/hrp-data/logs/` for error details
- **API Errors:** Enable debug logging in `.env`: `LOG_LEVEL=DEBUG`
