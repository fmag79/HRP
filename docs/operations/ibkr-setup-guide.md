# Interactive Brokers Setup Guide

Guide for setting up Interactive Brokers (IBKR) paper trading for HRP.

## Prerequisites

- Interactive Brokers account (paper trading)
- TWS (Trader Workstation) or IB Gateway installed
- Python 3.11+ with ib-insync library

## Step 1: Create Paper Trading Account

1. Log into [IBKR Account Management](https://www.interactivebrokers.com)
2. Navigate to Settings → Paper Trading
3. Create paper trading account (username format: `DU123456`)
4. Note your paper trading credentials

## Step 2: Install TWS/IB Gateway

**Option A: TWS (Trader Workstation)** - Full GUI
- Download from [IBKR TWS Download](https://www.interactivebrokers.com/en/trading/tws.php)
- Install for your platform

**Option B: IB Gateway** - Headless (recommended for servers)
- Download from [IBKR Gateway Download](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
- Lighter weight, no GUI

## Step 3: Configure API Access

1. Launch TWS/Gateway and log in with paper trading credentials
2. Navigate to: **File → Global Configuration → API → Settings**
3. Enable API:
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Read-Only API (disable for live trading)
   - Socket port: `7497` (paper trading default)
   - Trusted IP: `127.0.0.1`
4. Click **OK** and restart TWS/Gateway

## Step 4: Configure HRP Environment

Add to `.env`:

```bash
# Interactive Brokers Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # Paper trading port (4002 for live IB Gateway)
IBKR_CLIENT_ID=1  # Unique client ID (1-32)
IBKR_ACCOUNT=DU123456  # Your paper trading account
IBKR_PAPER_TRADING=true
```

## Step 5: Test Connection

```bash
# Test broker connection
python -c "
from hrp.execution.broker import IBKRBroker, BrokerConfig

config = BrokerConfig(
    host='127.0.0.1',
    port=7497,
    client_id=1,
    account='DU123456',
    paper_trading=True,
)

with IBKRBroker(config) as broker:
    print(f'Connected: {broker.is_connected()}')
"
```

Expected output: `Connected: True`

## Step 6: Verify Positions and Orders

```python
from hrp.execution.broker import IBKRBroker, BrokerConfig
from hrp.execution.positions import PositionTracker
from hrp.api.platform import PlatformAPI

api = PlatformAPI()
config = BrokerConfig(
    host='127.0.0.1',
    port=7497,
    client_id=1,
    account='DU123456',
    paper_trading=True,
)

with IBKRBroker(config) as broker:
    tracker = PositionTracker(broker, api)
    positions = tracker.sync_positions()
    print(f"Synced {len(positions)} positions")
```

## Troubleshooting

### Connection Refused

**Problem:** `ConnectionError: IBKR connection failed`

**Solutions:**
1. Verify TWS/Gateway is running
2. Check API is enabled in settings
3. Verify port number (7497 for paper, 7496 for live TWS)
4. Check firewall allows localhost connections

### Authentication Failed

**Problem:** `Authentication failed`

**Solutions:**
1. Verify paper trading credentials
2. Log in manually to TWS/Gateway first
3. Check account number matches environment variable

### API Not Enabled

**Problem:** `API connection rejected`

**Solutions:**
1. File → Global Configuration → API → Settings
2. Enable "Enable ActiveX and Socket Clients"
3. Restart TWS/Gateway

### Trusted IP

**Problem:** `Connection rejected from untrusted IP`

**Solutions:**
1. Add `127.0.0.1` to Trusted IPs in API settings
2. Or disable IP restriction (not recommended for production)

## Production Considerations

### Live Trading (DO NOT enable without review)

To switch to live trading:

1. **Update environment:**
   ```bash
   IBKR_PORT=7496  # Live TWS
   IBKR_ACCOUNT=U123456  # Live account (not DU)
   IBKR_PAPER_TRADING=false
   ```

2. **Disable Read-Only API** in TWS settings

3. **Enable live trader job:**
   ```bash
   python -m hrp.agents.run_job --job live-trader --execute-trades
   ```

### Security Checklist

- [ ] API credentials stored in `.env` (not committed)
- [ ] TWS/Gateway protected with strong password
- [ ] Firewall restricts API access to localhost only
- [ ] Read-Only API enabled initially
- [ ] Test all operations in paper trading first
- [ ] Position limits configured correctly
- [ ] Drift monitoring enabled before live trading
- [ ] Emergency stop mechanism tested

## Resources

- [IBKR API Documentation](https://interactivebrokers.github.io/tws-api/)
- [ib_insync Documentation](https://ib-insync.readthedocs.io/)
- [TWS API Reference](https://www.interactivebrokers.com/en/software/api/api.htm)
