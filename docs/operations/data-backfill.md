# Historical Data Backfill Guide

## Overview

The backfill system enables loading historical data for new symbols or date ranges. It supports:

- Price data (OHLCV)
- Computed features (momentum, volatility, etc.)
- Corporate actions (splits, dividends)

Key features:
- **Resumable**: Progress tracking allows resuming failed backfills
- **Rate-limited**: Respects API rate limits
- **Batch processing**: Memory-efficient processing of large symbol lists
- **Validation**: Detects gaps in data coverage

---

## When to Use Backfill vs Regular Ingestion

| Scenario | Use |
|----------|-----|
| Daily updates for existing symbols | Regular ingestion (`PriceIngestionJob`) |
| Adding new symbols to universe | Backfill |
| Filling gaps in historical data | Backfill |
| Loading 5+ years of history | Backfill |
| Loading data for 50+ symbols | Backfill |

**Rule of thumb:** Use backfill for bulk historical loading, regular ingestion for incremental updates.

---

## Backfill Workflow

### 1. Plan Your Backfill

Determine:
- Which symbols need data
- Date range required
- Data types needed (prices, features, corporate actions)

### 2. Start the Backfill

```bash
# Backfill prices for specific symbols
python -m hrp.data.backfill \
    --symbols AAPL MSFT GOOGL \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --prices

# Backfill all data types for entire universe
python -m hrp.data.backfill \
    --universe \
    --start 2015-01-01 \
    --all
```

### 3. Monitor Progress

Progress is displayed during execution:
```
Backfill progress: 45/100 symbols (45.0%)
Processing batch 5/10...
```

### 4. Resume if Needed

If the backfill fails or is interrupted:
```bash
python -m hrp.data.backfill \
    --resume backfill_progress_20260124_143052.json \
    --prices
```

### 5. Validate Completeness

```bash
python -m hrp.data.backfill \
    --symbols AAPL MSFT GOOGL \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --validate
```

---

## CLI Reference

### Basic Usage

```bash
python -m hrp.data.backfill [OPTIONS]
```

### Symbol Selection

| Option | Description |
|--------|-------------|
| `--symbols AAPL MSFT` | Specific symbols to backfill |
| `--universe` | All symbols in current universe |

### Date Range

| Option | Description |
|--------|-------------|
| `--start YYYY-MM-DD` | Start date (required) |
| `--end YYYY-MM-DD` | End date (default: today) |

### Data Types

| Option | Description |
|--------|-------------|
| `--prices` | Backfill price data (OHLCV) |
| `--features` | Backfill computed features |
| `--corporate-actions` | Backfill splits and dividends |
| `--all` | Backfill all data types |

### Other Options

| Option | Description |
|--------|-------------|
| `--batch-size N` | Symbols per batch (default: 10) |
| `--resume FILE` | Resume from progress file |
| `--validate` | Validate completeness only |

---

## Rate Limiting

The backfill system respects API rate limits to avoid being blocked.

### Default Limits

| Source | Limit |
|--------|-------|
| Yahoo Finance | 2000 requests/hour |
| Polygon.io | Varies by plan |

### How It Works

1. Rate limiter tracks requests per time window
2. Before each API call, checks if limit reached
3. If at limit, waits until window resets
4. Progress continues after wait

### Adjusting Rate Limits

For custom limits, modify `hrp/data/backfill.py`:

```python
rate_limiter = RateLimiter(max_requests=1000, time_window=3600)
```

---

## Progress Tracking

### How Progress is Saved

Progress is saved to a JSON file after each batch:

```json
{
  "completed_symbols": ["AAPL", "MSFT", "GOOGL"],
  "failed_symbols": ["INVALID"],
  "total_symbols": 100,
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "timestamp": "2026-01-24T14:30:52"
}
```

### Progress File Location

Default: Current directory with timestamp
```
backfill_progress_20260124_143052.json
```

### Resuming from Progress

```bash
# Resume with same parameters (encoded in progress file)
python -m hrp.data.backfill --resume backfill_progress_20260124_143052.json --prices

# Or specify symbols/dates again (progress file filters already-done symbols)
python -m hrp.data.backfill \
    --symbols AAPL MSFT GOOGL AMZN \
    --start 2020-01-01 \
    --resume backfill_progress_20260124_143052.json \
    --prices
```

### What Happens to Failed Symbols

Failed symbols are:
1. Logged with error message
2. Recorded in progress file
3. **Retried** on resume (not skipped)

---

## Validation

### What Gets Validated

1. **Symbol coverage**: All requested symbols have data
2. **Date coverage**: No gaps in trading days
3. **Feature coverage**: Features exist for all dates with prices

### Running Validation

```bash
python -m hrp.data.backfill \
    --symbols AAPL MSFT \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --validate
```

### Interpreting Results

```
Validation Results:
  Symbols checked: 2
  Symbols with data: 2
  Missing symbols: 0
  Date range gaps: 0
  Feature coverage: 100%
  Status: VALID
```

If gaps are found:
```
Validation Results:
  Symbols checked: 2
  Symbols with data: 1
  Missing symbols: 1 (INVALID)
  Date range gaps: 3
    - AAPL: 2020-03-15 to 2020-03-17
  Feature coverage: 95%
  Status: INVALID
```

---

## Troubleshooting

### "Rate limit exceeded"

**Cause:** Too many API requests in short time

**Solution:**
1. Wait for rate limit window to reset (usually 1 hour)
2. Resume with `--resume` flag
3. Reduce batch size for slower, steadier progress

### "Symbol not found"

**Cause:** Invalid ticker symbol or delisted stock

**Solution:**
1. Verify symbol is valid
2. Check if symbol was renamed or delisted
3. Failed symbols are logged and can be reviewed

### "Connection timeout"

**Cause:** Network issues or API server unavailable

**Solution:**
1. Check internet connection
2. Resume with `--resume` flag
3. Transient errors are auto-retried (3 attempts)

### "Not enough disk space"

**Cause:** Large backfill filling disk

**Solution:**
1. Check available space: `df -h`
2. Backfill in smaller date ranges
3. Archive old backups

### "Foreign key constraint failed"

**Cause:** Symbol not in symbols table

**Solution:**
The backfill system auto-creates missing symbols. If this error occurs:
1. Check if symbol is valid
2. Try again - transient database issue

### "Progress file not found"

**Cause:** Specified resume file doesn't exist

**Solution:**
1. Check file path
2. List progress files: `ls backfill_progress_*.json`
3. Start fresh backfill if file lost

---

## Best Practices

1. **Start small**: Test with a few symbols before large backfills
2. **Use batches**: Default batch size of 10 balances speed and reliability
3. **Monitor progress**: Check logs for errors during long backfills
4. **Validate after**: Always run `--validate` after backfill completes
5. **Keep progress files**: Don't delete until backfill fully verified
6. **Off-peak hours**: Run large backfills during market closed hours

---

## Examples

### Example 1: New Universe Setup

Load 5 years of history for S&P 500:

```bash
# Step 1: Backfill prices (takes longest)
python -m hrp.data.backfill \
    --universe \
    --start 2020-01-01 \
    --prices

# Step 2: Backfill corporate actions
python -m hrp.data.backfill \
    --universe \
    --start 2020-01-01 \
    --corporate-actions

# Step 3: Compute features (requires prices)
python -m hrp.data.backfill \
    --universe \
    --start 2020-01-01 \
    --features

# Step 4: Validate
python -m hrp.data.backfill \
    --universe \
    --start 2020-01-01 \
    --validate
```

### Example 2: Add New Symbol

```bash
python -m hrp.data.backfill \
    --symbols NEWSTOCK \
    --start 2020-01-01 \
    --all
```

### Example 3: Fill Gaps

```bash
# First, identify gaps
python -m hrp.data.backfill \
    --symbols AAPL \
    --start 2020-01-01 \
    --validate

# Then fill them
python -m hrp.data.backfill \
    --symbols AAPL \
    --start 2020-03-15 \
    --end 2020-03-20 \
    --prices
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Backfill prices | `python -m hrp.data.backfill --symbols AAPL --start 2020-01-01 --prices` |
| Backfill all data | `python -m hrp.data.backfill --symbols AAPL --start 2020-01-01 --all` |
| Backfill universe | `python -m hrp.data.backfill --universe --start 2020-01-01 --prices` |
| Resume backfill | `python -m hrp.data.backfill --resume progress.json --prices` |
| Validate data | `python -m hrp.data.backfill --symbols AAPL --start 2020-01-01 --validate` |
