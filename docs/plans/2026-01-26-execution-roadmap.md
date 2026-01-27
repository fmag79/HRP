# Execution Roadmap: From Signals to Auto-Rebalance

**Date**: 2026-01-26
**Status**: Parked (complete Intelligence + Production tiers first)
**Ultimate Goal**: Auto-rebalance

## Context

With the platform buildout nearing completion, the question arose: how do we get from signals to executable trades?

### Current Gap

```
Research Layer              Execution Layer (Not Built)
──────────────              ────────────────────────────
Signals (symbol, strength)  → Trade Sheet (ticker, shares, price)
Backtest results            → Position Sizer
Walk-forward validation     → Order Manager
                            → Broker Integration (IB)
                            → Execution Monitor
```

The platform produces signals but not actionable trade instructions.

## Execution Levels

| Level | Description | User Role | Complexity |
|-------|-------------|-----------|------------|
| **1. Manual** | Trade sheet CSV → execute in brokerage | Full control | Low |
| **2. Semi-auto** | Platform sends orders, user approves each | Approval gate | Medium |
| **3. Auto-rebalance** | Platform executes on schedule per rules | Monitoring only | High |

**Target**: Level 3 (Auto-rebalance)

## Required Components

### Tier 4: Execution Layer

```
hrp/execution/
├── trade_sheet.py       # Generate actionable trade list
├── position_sizer.py    # Calculate shares from signals + capital
├── order_manager.py     # Create, track, cancel orders
├── ib_client.py         # Interactive Brokers API wrapper
├── rebalancer.py        # Scheduled portfolio rebalancing
└── monitor.py           # Execution quality, slippage tracking
```

### Trade Sheet Generator

Input:
- Signals DataFrame (symbol, signal_strength, date)
- Available capital
- Current positions (from broker or manual)
- Configuration (max positions, position sizing method)

Output:
```
| Action | Symbol | Shares | Limit Price | Est. Cost | Signal |
|--------|--------|--------|-------------|-----------|--------|
| BUY    | NVDA   | 3      | $142.50     | $427.50   | 0.85   |
| BUY    | META   | 2      | $615.00     | $1,230.00 | 0.72   |
| SELL   | XYZ    | 10     | $45.00      | $450.00   | -0.30  |
```

### Position Sizing Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Equal-weight | Same $ per position | Simple, diversified |
| Signal-weighted | $ proportional to signal strength | Concentrate on conviction |
| Risk-parity | $ inversely proportional to volatility | Volatility-balanced |
| Kelly | Optimal sizing based on edge + odds | Aggressive growth |

### Auto-Rebalance Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Scheduler (Daily)                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Signal Generator (from ML/factors)          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Position Sizer                        │
│  • Current portfolio (from IB)                          │
│  • Target portfolio (from signals)                      │
│  • Rebalance threshold (e.g., 5% drift)                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Trade Sheet Generator                  │
│  • Respect position limits                              │
│  • Minimize transaction costs                           │
│  • Tax-loss harvesting (optional)                       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Order Manager                         │
│  • Limit orders (TWAP, VWAP optional)                   │
│  • Order tracking                                        │
│  • Partial fill handling                                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│               Interactive Brokers API                    │
│  • ib_insync library                                    │
│  • Paper trading mode for testing                       │
│  • Production with safety limits                        │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Execution Monitor                       │
│  • Fill quality (slippage vs. expected)                 │
│  • Daily P&L reconciliation                             │
│  • Alerts on failures                                   │
└─────────────────────────────────────────────────────────┘
```

## Safety Controls

Per CLAUDE.md: Agents **cannot** deploy strategies. Only users approve.

| Control | Description |
|---------|-------------|
| Daily loss limit | Halt trading if daily loss exceeds threshold |
| Position limits | Max % per position, per sector |
| Order size limits | Max shares per order |
| Approval gates | User confirmation for orders above threshold |
| Kill switch | Manual override to halt all trading |
| Paper mode | Test execution without real money |

## Prerequisites

Before building Tier 4:

- [ ] **Intelligence Tier (90% → 100%)**: Complete remaining agent work
- [ ] **Production Tier (0% → 100%)**: Security, ops monitoring, error handling

## Implementation Sequence

1. **Trade Sheet Generator** — Manual execution with CSV export
2. **IB Paper Trading** — Test order flow without real money
3. **Semi-auto Mode** — Orders require user approval
4. **Auto-rebalance** — Scheduled execution with safety limits

## Initial Portfolio Approach ($10k)

Documented in `docs/operations/head-start.md`:

1. Validate edge through agent pipeline
2. Paper trade 4+ weeks
3. Deploy $5k with manual execution
4. Add remaining $5k if results align
5. Graduate to semi-auto, then auto-rebalance

## Open Questions (To Revisit)

- Rebalance frequency: Daily? Weekly? On drift threshold?
- Order types: Market, limit, or adaptive?
- Tax considerations: Wash sale rules, short-term vs long-term
- Multi-account support needed?

---

**Next Action**: Complete Intelligence + Production tiers, then revisit this roadmap.
