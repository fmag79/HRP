# Remaining Technical Indicator Features Implementation Plan

> **Status:** ✅ **COMPLETED** (as of 2026-01-25) - All planned features implemented except `efi_13d`. Platform now has 44 total features.

**Original Goal:** Add comprehensive technical indicator features including RSI, SMA, MACD, ADX, Bollinger Bands, Stochastic, and other standard indicators for systematic trading research.

**Architecture:** Features are computed in two locations: `hrp/data/ingestion/features.py` (for batch ingestion) and `hrp/data/features/computation.py` (for on-demand computation).

**Tech Stack:** pandas, numpy, DuckDB, pytest

> **Note:** This plan document is preserved for historical reference. The implementation is complete. See `CLAUDE.md` for the current feature list.

---

## Summary of Features (Implementation Status)

### Group 1: Basic Returns & Momentum
| Feature | Formula | Status |
|---------|---------|--------|
| `returns_60d` | pct_change(60) | ✅ **Done** |
| `returns_252d` | pct_change(252) | ✅ **Done** |
| `momentum_252d` | pct_change(252) | ✅ **Done** |
| `volume_ratio` | volume / volume_avg_20d | ✅ **Done** |
| `roc_10d` | (close - close_10d_ago) / close_10d_ago * 100 | ✅ **Done** |

### Group 2: Moving Averages & Trend
| Feature | Formula | Status |
|---------|---------|--------|
| `sma_20d` | close.rolling(20).mean() | ✅ **Done** |
| `sma_50d` | close.rolling(50).mean() | ✅ **Done** |
| `sma_200d` | close.rolling(200).mean() | ✅ **Done** |
| `price_to_sma_20d` | close / sma_20d | ✅ **Done** |
| `price_to_sma_50d` | close / sma_50d | ✅ **Done** |
| `price_to_sma_200d` | close / sma_200d | ✅ **Done** |
| `trend` | +1 if close > sma_200d, -1 otherwise | ✅ **Done** |

### Group 3: Oscillators
| Feature | Formula | Status |
|---------|---------|--------|
| `rsi_14d` | RSI(14) | ✅ **Done** |
| `cci_20d` | (TP - SMA(TP)) / (0.015 * Mean Deviation) | ✅ **Done** |
| `williams_r_14d` | Williams %R (14-day) | ✅ **Done** |
| `stoch_k_14d` | Stochastic %K (14-day) | ✅ **Done** |
| `stoch_d_14d` | Stochastic %D (3-day SMA of %K) | ✅ **Done** |

### Group 4: MACD
| Feature | Formula | Status |
|---------|---------|--------|
| `macd_line` | EMA(12) - EMA(26) | ✅ **Done** |
| `macd_signal` | EMA(9) of MACD line | ✅ **Done** |
| `macd_histogram` | MACD line - Signal line | ✅ **Done** |

### Group 5: Volatility & Volume
| Feature | Formula | Status |
|---------|---------|--------|
| `atr_14d` | Average True Range (14-day) | ✅ **Done** |
| `bb_upper_20d` | SMA(20) + 2*std(20) | ✅ **Done** |
| `bb_lower_20d` | SMA(20) - 2*std(20) | ✅ **Done** |
| `bb_width_20d` | (upper - lower) / middle | ✅ **Done** |
| `obv` | On-Balance Volume (cumulative) | ✅ **Done** |
| `efi_13d` | Elder's Force Index (13-day EMA smoothed) | ❌ **Not Implemented** |

### Group 6: Trend Strength
| Feature | Formula | Status |
|---------|---------|--------|
| `adx_14d` | Average Directional Index (14-day) | ✅ **Done** |

**Original Plan: 27 features** → **26 implemented, 1 remaining (`efi_13d`)**

### Additional Features Implemented (Beyond Original Plan)
| Feature | Description | Status |
|---------|-------------|--------|
| `ema_12d` | 12-day EMA | ✅ **Done** |
| `ema_26d` | 26-day EMA | ✅ **Done** |
| `ema_crossover` | EMA-12 vs EMA-26 signal | ✅ **Done** |
| `mfi_14d` | Money Flow Index | ✅ **Done** |
| `vwap_20d` | 20-day rolling VWAP approximation | ✅ **Done** |
| `market_cap` | Market capitalization (fundamental) | ✅ **Done** |
| `pe_ratio` | P/E ratio (fundamental) | ✅ **Done** |
| `pb_ratio` | P/B ratio (fundamental) | ✅ **Done** |
| `dividend_yield` | Dividend yield (fundamental) | ✅ **Done** |
| `ev_ebitda` | EV/EBITDA (fundamental) | ✅ **Done** |

**Total Features Available: 44**

---

## Task 1: Add RSI Computation Function ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py:19-70`
- Modify: `hrp/data/ingestion/features.py:191-262`

**Step 1: Write the failing test for RSI**

Add to `tests/test_data/test_features.py` after the `test_compute_volatility_60d` test (around line 585):

```python
def test_compute_rsi_14d(self):
    """Test rsi_14d computation function."""
    from hrp.data.features.computation import compute_rsi_14d

    # Create sample price data with known movement pattern
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")

    data = []
    base_price = 100.0
    for i, d in enumerate(dates):
        # Alternate between gains and losses for predictable RSI
        if i % 2 == 0:
            price = base_price * (1 + 0.01 * (i // 2))
        else:
            price = base_price * (1 + 0.005 * (i // 2))
        data.append({"date": d, "symbol": "AAPL", "close": price})

    df = pd.DataFrame(data)
    df = df.set_index(["date", "symbol"])

    # Compute RSI
    result = compute_rsi_14d(df)

    # Check result structure
    assert isinstance(result, pd.DataFrame)
    assert "rsi_14d" in result.columns
    assert len(result) > 0

    # RSI should be between 0 and 100
    valid_values = result["rsi_14d"].dropna()
    assert (valid_values >= 0).all()
    assert (valid_values <= 100).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_features.py::TestFeatureComputation::test_compute_rsi_14d -v`
Expected: FAIL with "cannot import name 'compute_rsi_14d'"

**Step 3: Implement RSI computation in computation.py**

Add after `compute_volatility_60d` function (around line 64):

```python
def compute_rsi_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over 14 periods

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with RSI values (0-100 scale)
    """
    close = prices["close"].unstack(level="symbol")

    # Calculate price changes
    delta = close.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate average gain and loss using exponential moving average
    avg_gain = gains.ewm(span=14, adjust=False).mean()
    avg_loss = losses.ewm(span=14, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases (no losses = RSI of 100, no gains = RSI of 0)
    rsi = rsi.replace([np.inf, -np.inf], np.nan)

    result = rsi.stack(level="symbol", future_stack=True)
    return result.to_frame(name="rsi_14d")
```

**Step 4: Add to FEATURE_FUNCTIONS registry**

Update the `FEATURE_FUNCTIONS` dict (around line 67):

```python
FEATURE_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "momentum_20d": compute_momentum_20d,
    "volatility_60d": compute_volatility_60d,
    "rsi_14d": compute_rsi_14d,
}
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_data/test_features.py::TestFeatureComputation::test_compute_rsi_14d -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add RSI-14 computation function

Implements the 14-day Relative Strength Index indicator using
exponential moving average for smoothing, as per industry standard.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add SMA Computation Functions ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing tests for SMA functions**

Add to `tests/test_data/test_features.py`:

```python
def test_compute_sma_20d(self):
    """Test sma_20d computation function."""
    from hrp.data.features.computation import compute_sma_20d

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = []
    for i, d in enumerate(dates):
        data.append({"date": d, "symbol": "AAPL", "close": 100.0 + i})

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_sma_20d(df)

    assert isinstance(result, pd.DataFrame)
    assert "sma_20d" in result.columns

    # First 19 values should be NaN
    valid_values = result["sma_20d"].dropna()
    assert len(valid_values) > 0

def test_compute_sma_50d(self):
    """Test sma_50d computation function."""
    from hrp.data.features.computation import compute_sma_50d

    dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 + i} for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_sma_50d(df)

    assert isinstance(result, pd.DataFrame)
    assert "sma_50d" in result.columns

def test_compute_sma_200d(self):
    """Test sma_200d computation function."""
    from hrp.data.features.computation import compute_sma_200d

    dates = pd.date_range("2022-01-01", "2023-06-30", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 + i * 0.1} for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_sma_200d(df)

    assert isinstance(result, pd.DataFrame)
    assert "sma_200d" in result.columns
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_features.py::TestFeatureComputation::test_compute_sma_20d -v`
Expected: FAIL

**Step 3: Implement SMA functions**

Add to `hrp/data/features/computation.py` after the RSI function:

```python
def compute_sma_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Simple Moving Average.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with SMA values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    result = sma.stack(level="symbol", future_stack=True)
    return result.to_frame(name="sma_20d")


def compute_sma_50d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 50-day Simple Moving Average.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with SMA values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=50).mean()
    result = sma.stack(level="symbol", future_stack=True)
    return result.to_frame(name="sma_50d")


def compute_sma_200d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 200-day Simple Moving Average.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with SMA values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=200).mean()
    result = sma.stack(level="symbol", future_stack=True)
    return result.to_frame(name="sma_200d")
```

**Step 4: Update FEATURE_FUNCTIONS registry**

```python
FEATURE_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "momentum_20d": compute_momentum_20d,
    "volatility_60d": compute_volatility_60d,
    "rsi_14d": compute_rsi_14d,
    "sma_20d": compute_sma_20d,
    "sma_50d": compute_sma_50d,
    "sma_200d": compute_sma_200d,
}
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data/test_features.py::TestFeatureComputation -v`
Expected: PASS for all SMA tests

**Step 6: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add SMA computation functions (20, 50, 200 day)

Implements simple moving average indicators at three standard
lookback periods commonly used for trend analysis.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add Price-to-SMA Ratio Functions ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing tests**

```python
def test_compute_price_to_sma_20d(self):
    """Test price_to_sma_20d computation function."""
    from hrp.data.features.computation import compute_price_to_sma_20d

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 + i} for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_price_to_sma_20d(df)

    assert isinstance(result, pd.DataFrame)
    assert "price_to_sma_20d" in result.columns

    # Values should be around 1.0 (price close to SMA)
    valid_values = result["price_to_sma_20d"].dropna()
    assert (valid_values > 0).all()

def test_compute_price_to_sma_50d(self):
    """Test price_to_sma_50d computation function."""
    from hrp.data.features.computation import compute_price_to_sma_50d

    dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 + i} for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_price_to_sma_50d(df)

    assert isinstance(result, pd.DataFrame)
    assert "price_to_sma_50d" in result.columns

def test_compute_price_to_sma_200d(self):
    """Test price_to_sma_200d computation function."""
    from hrp.data.features.computation import compute_price_to_sma_200d

    dates = pd.date_range("2022-01-01", "2023-06-30", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 + i * 0.1} for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_price_to_sma_200d(df)

    assert isinstance(result, pd.DataFrame)
    assert "price_to_sma_200d" in result.columns
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_features.py::TestFeatureComputation::test_compute_price_to_sma_20d -v`
Expected: FAIL

**Step 3: Implement price-to-SMA functions**

```python
def compute_price_to_sma_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price-to-SMA-20 ratio (close / sma_20d).

    Values > 1 indicate price above SMA (bullish), < 1 indicates below (bearish).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with price-to-SMA ratio values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    ratio = close / sma
    result = ratio.stack(level="symbol", future_stack=True)
    return result.to_frame(name="price_to_sma_20d")


def compute_price_to_sma_50d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price-to-SMA-50 ratio (close / sma_50d).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with price-to-SMA ratio values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=50).mean()
    ratio = close / sma
    result = ratio.stack(level="symbol", future_stack=True)
    return result.to_frame(name="price_to_sma_50d")


def compute_price_to_sma_200d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price-to-SMA-200 ratio (close / sma_200d).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with price-to-SMA ratio values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=200).mean()
    ratio = close / sma
    result = ratio.stack(level="symbol", future_stack=True)
    return result.to_frame(name="price_to_sma_200d")
```

**Step 4: Update FEATURE_FUNCTIONS registry**

```python
FEATURE_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "momentum_20d": compute_momentum_20d,
    "volatility_60d": compute_volatility_60d,
    "rsi_14d": compute_rsi_14d,
    "sma_20d": compute_sma_20d,
    "sma_50d": compute_sma_50d,
    "sma_200d": compute_sma_200d,
    "price_to_sma_20d": compute_price_to_sma_20d,
    "price_to_sma_50d": compute_price_to_sma_50d,
    "price_to_sma_200d": compute_price_to_sma_200d,
}
```

**Step 5: Run tests**

Run: `pytest tests/test_data/test_features.py::TestFeatureComputation -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add price-to-SMA ratio indicators

Implements price/SMA ratios for 20, 50, and 200 day periods.
Values > 1 indicate price above moving average (bullish signal).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add Volume Ratio and Additional Returns/Momentum ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing tests**

```python
def test_compute_volume_ratio(self):
    """Test volume_ratio computation function."""
    from hrp.data.features.computation import compute_volume_ratio

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0, "volume": 1000000 + i * 10000}
            for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_volume_ratio(df)

    assert isinstance(result, pd.DataFrame)
    assert "volume_ratio" in result.columns

    # Volume ratio should be positive
    valid_values = result["volume_ratio"].dropna()
    assert (valid_values > 0).all()

def test_compute_returns_60d(self):
    """Test returns_60d computation function."""
    from hrp.data.features.computation import compute_returns_60d

    dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 * (1 + 0.001 * i)}
            for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_returns_60d(df)

    assert isinstance(result, pd.DataFrame)
    assert "returns_60d" in result.columns

def test_compute_returns_252d(self):
    """Test returns_252d computation function."""
    from hrp.data.features.computation import compute_returns_252d

    dates = pd.date_range("2022-01-01", "2023-06-30", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 * (1 + 0.0005 * i)}
            for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_returns_252d(df)

    assert isinstance(result, pd.DataFrame)
    assert "returns_252d" in result.columns

def test_compute_momentum_252d(self):
    """Test momentum_252d computation function."""
    from hrp.data.features.computation import compute_momentum_252d

    dates = pd.date_range("2022-01-01", "2023-06-30", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 * (1 + 0.0005 * i)}
            for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_momentum_252d(df)

    assert isinstance(result, pd.DataFrame)
    assert "momentum_252d" in result.columns
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_features.py::TestFeatureComputation::test_compute_volume_ratio -v`
Expected: FAIL

**Step 3: Implement the functions**

```python
def compute_volume_ratio(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume ratio (current volume / 20-day average volume).

    Values > 1 indicate higher than average volume (increased interest).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'volume' column

    Returns:
        DataFrame with volume ratio values
    """
    volume = prices["volume"].unstack(level="symbol")
    avg_volume = volume.rolling(window=20).mean()
    ratio = volume / avg_volume
    result = ratio.stack(level="symbol", future_stack=True)
    return result.to_frame(name="volume_ratio")


def compute_returns_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 60-day return (quarterly).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with return values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change(60)
    result = returns.stack(level="symbol", future_stack=True)
    return result.to_frame(name="returns_60d")


def compute_returns_252d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 252-day return (annual).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with return values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change(252)
    result = returns.stack(level="symbol", future_stack=True)
    return result.to_frame(name="returns_252d")


def compute_momentum_252d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 252-day momentum (annual trailing return).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with momentum values
    """
    close = prices["close"].unstack(level="symbol")
    momentum = close.pct_change(252)
    result = momentum.stack(level="symbol", future_stack=True)
    return result.to_frame(name="momentum_252d")
```

**Step 4: Update FEATURE_FUNCTIONS registry**

```python
FEATURE_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "momentum_20d": compute_momentum_20d,
    "momentum_252d": compute_momentum_252d,
    "volatility_60d": compute_volatility_60d,
    "rsi_14d": compute_rsi_14d,
    "sma_20d": compute_sma_20d,
    "sma_50d": compute_sma_50d,
    "sma_200d": compute_sma_200d,
    "price_to_sma_20d": compute_price_to_sma_20d,
    "price_to_sma_50d": compute_price_to_sma_50d,
    "price_to_sma_200d": compute_price_to_sma_200d,
    "volume_ratio": compute_volume_ratio,
    "returns_60d": compute_returns_60d,
    "returns_252d": compute_returns_252d,
}
```

**Step 5: Run tests**

Run: `pytest tests/test_data/test_features.py::TestFeatureComputation -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add volume ratio and extended return periods

Adds volume_ratio (volume/avg_20d), returns_60d, returns_252d,
and momentum_252d indicators to complete the spec requirements.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update Ingestion Pipeline with New Features ✅ COMPLETED

**Files:**
- Modify: `hrp/data/ingestion/features.py:191-262`

**Step 1: Update `_compute_all_features` function**

Modify `hrp/data/ingestion/features.py` to add the new features in the `_compute_all_features` function:

```python
def _compute_all_features(df: pd.DataFrame, symbol: str, version: str) -> pd.DataFrame:
    """
    Compute all technical features from price data.

    Returns long-form DataFrame with columns: symbol, date, feature_name, value, version
    """
    if df.empty:
        return pd.DataFrame()

    # Use adjusted close for returns
    df = df.copy()
    df['price'] = df['adj_close'].fillna(df['close'])

    features = []

    # === RETURNS ===
    df['returns_1d'] = df['price'].pct_change(1)
    df['returns_5d'] = df['price'].pct_change(5)
    df['returns_20d'] = df['price'].pct_change(20)
    df['returns_60d'] = df['price'].pct_change(60)
    df['returns_252d'] = df['price'].pct_change(252)

    # === MOMENTUM ===
    df['momentum_20d'] = df['price'].pct_change(20)
    df['momentum_60d'] = df['price'].pct_change(60)
    df['momentum_252d'] = df['price'].pct_change(252)

    # === VOLATILITY ===
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df['volatility_20d'] = df['log_return'].rolling(window=20).std() * np.sqrt(252)
    df['volatility_60d'] = df['log_return'].rolling(window=60).std() * np.sqrt(252)

    # === VOLUME ===
    df['volume_20d'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_20d']

    # === RSI ===
    delta = df['price'].diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    avg_gain = gains.ewm(span=14, adjust=False).mean()
    avg_loss = losses.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    df['rsi_14d'] = df['rsi_14d'].replace([np.inf, -np.inf], np.nan)

    # === SMA ===
    df['sma_20d'] = df['price'].rolling(window=20).mean()
    df['sma_50d'] = df['price'].rolling(window=50).mean()
    df['sma_200d'] = df['price'].rolling(window=200).mean()

    # === PRICE TO SMA ===
    df['price_to_sma_20d'] = df['price'] / df['sma_20d']
    df['price_to_sma_50d'] = df['price'] / df['sma_50d']
    df['price_to_sma_200d'] = df['price'] / df['sma_200d']

    # Convert wide format to long format
    feature_columns = [
        'returns_1d', 'returns_5d', 'returns_20d', 'returns_60d', 'returns_252d',
        'momentum_20d', 'momentum_60d', 'momentum_252d',
        'volatility_20d', 'volatility_60d',
        'volume_20d', 'volume_ratio',
        'rsi_14d',
        'sma_20d', 'sma_50d', 'sma_200d',
        'price_to_sma_20d', 'price_to_sma_50d', 'price_to_sma_200d',
    ]

    for feature_name in feature_columns:
        feature_data = df[['date', feature_name]].copy()
        feature_data['symbol'] = symbol
        feature_data['feature_name'] = feature_name
        feature_data['value'] = feature_data[feature_name]
        feature_data['version'] = version
        feature_data = feature_data[['symbol', 'date', 'feature_name', 'value', 'version']]
        feature_data = feature_data.dropna(subset=['value'])
        features.append(feature_data)

    if not features:
        return pd.DataFrame()

    return pd.concat(features, ignore_index=True)
```

**Step 2: Update docstring to list all features**

Update the docstring at the top of `compute_features` function to list all features:

```python
def compute_features(
    symbols: Optional[list[str]] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    version: str = "v1",
) -> dict[str, Any]:
    """
    Compute technical features for given symbols.

    Computes:
    - returns_1d, returns_5d, returns_20d, returns_60d, returns_252d: N-day returns
    - momentum_20d, momentum_60d, momentum_252d: N-day momentum
    - volatility_20d, volatility_60d: N-day rolling volatility (annualized)
    - volume_20d: 20-day average volume
    - volume_ratio: current volume / 20-day average volume
    - rsi_14d: 14-day Relative Strength Index
    - sma_20d, sma_50d, sma_200d: Simple Moving Averages
    - price_to_sma_20d, price_to_sma_50d, price_to_sma_200d: Price to SMA ratios
    ...
    """
```

**Step 3: Run feature computation tests**

Run: `pytest tests/test_data/ -v -k feature`
Expected: PASS

**Step 4: Commit**

```bash
git add hrp/data/ingestion/features.py
git commit -m "$(cat <<'EOF'
feat(features): update ingestion pipeline with all spec features

Adds RSI, SMA, price-to-SMA ratios, volume_ratio, and extended
return/momentum periods to the batch feature ingestion pipeline.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Register New Features in Default Registration ✅ COMPLETED

**Files:**
- Modify: `hrp/data/features/computation.py:559-610`

**Step 1: Update `register_default_features` function**

Add all new features to the registration:

```python
def register_default_features(db_path: str | None = None) -> None:
    """
    Register default features in the registry.

    Registers all standard technical indicators for reproducibility.

    Args:
        db_path: Optional path to database (defaults to standard location)
    """
    from hrp.data.features.registry import FeatureRegistry

    registry = FeatureRegistry(db_path)

    features = [
        {
            "feature_name": "momentum_20d",
            "version": "v1",
            "computation_fn": compute_momentum_20d,
            "description": "20-day momentum (trailing return). Calculated as pct_change(20).",
        },
        {
            "feature_name": "momentum_252d",
            "version": "v1",
            "computation_fn": compute_momentum_252d,
            "description": "252-day momentum (annual trailing return).",
        },
        {
            "feature_name": "volatility_60d",
            "version": "v1",
            "computation_fn": compute_volatility_60d,
            "description": "60-day annualized volatility. Rolling std of returns * sqrt(252).",
        },
        {
            "feature_name": "rsi_14d",
            "version": "v1",
            "computation_fn": compute_rsi_14d,
            "description": "14-day Relative Strength Index (0-100 scale).",
        },
        {
            "feature_name": "sma_20d",
            "version": "v1",
            "computation_fn": compute_sma_20d,
            "description": "20-day Simple Moving Average.",
        },
        {
            "feature_name": "sma_50d",
            "version": "v1",
            "computation_fn": compute_sma_50d,
            "description": "50-day Simple Moving Average.",
        },
        {
            "feature_name": "sma_200d",
            "version": "v1",
            "computation_fn": compute_sma_200d,
            "description": "200-day Simple Moving Average.",
        },
        {
            "feature_name": "price_to_sma_20d",
            "version": "v1",
            "computation_fn": compute_price_to_sma_20d,
            "description": "Price to 20-day SMA ratio. >1 = above SMA, <1 = below.",
        },
        {
            "feature_name": "price_to_sma_50d",
            "version": "v1",
            "computation_fn": compute_price_to_sma_50d,
            "description": "Price to 50-day SMA ratio. >1 = above SMA, <1 = below.",
        },
        {
            "feature_name": "price_to_sma_200d",
            "version": "v1",
            "computation_fn": compute_price_to_sma_200d,
            "description": "Price to 200-day SMA ratio. >1 = above SMA, <1 = below.",
        },
        {
            "feature_name": "volume_ratio",
            "version": "v1",
            "computation_fn": compute_volume_ratio,
            "description": "Volume ratio (volume / 20-day avg volume). >1 = above average.",
        },
        {
            "feature_name": "returns_60d",
            "version": "v1",
            "computation_fn": compute_returns_60d,
            "description": "60-day return (quarterly).",
        },
        {
            "feature_name": "returns_252d",
            "version": "v1",
            "computation_fn": compute_returns_252d,
            "description": "252-day return (annual).",
        },
    ]

    for feature in features:
        try:
            existing = registry.get(feature["feature_name"], feature["version"])
            if existing:
                logger.debug(
                    f"Feature {feature['feature_name']} ({feature['version']}) already registered"
                )
                continue

            registry.register_feature(
                feature_name=feature["feature_name"],
                computation_fn=feature["computation_fn"],
                version=feature["version"],
                description=feature["description"],
                is_active=True,
            )
            logger.info(f"Registered feature: {feature['feature_name']} ({feature['version']})")

        except Exception as e:
            logger.debug(f"Could not register {feature['feature_name']}: {e}")
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): register all new features in default registration

Updates register_default_features() to include all new technical
indicators for automatic registration on module import.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update CLAUDE.md Documentation ✅ COMPLETED

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update feature list in CLAUDE.md**

Find the feature computation section and update it to reflect all available features:

```markdown
### Available Features

| Feature | Description | Lookback |
|---------|-------------|----------|
| `returns_1d` | 1-day return | 1 day |
| `returns_5d` | 5-day return | 5 days |
| `returns_20d` | 20-day return (monthly) | 20 days |
| `returns_60d` | 60-day return (quarterly) | 60 days |
| `returns_252d` | 252-day return (annual) | 252 days |
| `momentum_20d` | 20-day momentum | 20 days |
| `momentum_60d` | 60-day momentum | 60 days |
| `momentum_252d` | 252-day momentum (annual) | 252 days |
| `volatility_20d` | 20-day rolling volatility (annualized) | 20 days |
| `volatility_60d` | 60-day rolling volatility (annualized) | 60 days |
| `volume_20d` | 20-day average volume | 20 days |
| `volume_ratio` | volume / volume_avg_20d | 20 days |
| `rsi_14d` | Relative Strength Index | 14 days |
| `sma_20d` | Simple Moving Average | 20 days |
| `sma_50d` | Simple Moving Average | 50 days |
| `sma_200d` | Simple Moving Average | 200 days |
| `price_to_sma_20d` | close / sma_20d | 20 days |
| `price_to_sma_50d` | close / sma_50d | 50 days |
| `price_to_sma_200d` | close / sma_200d | 200 days |
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: update CLAUDE.md with complete feature list

Documents all 19 technical indicator features now available
in the HRP feature store.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Final Verification ✅ COMPLETED

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Verify feature computation works end-to-end**

```bash
python -c "
from hrp.data.ingestion.features import compute_features
from datetime import date, timedelta
result = compute_features(
    symbols=['AAPL'],
    start=date.today() - timedelta(days=30),
    end=date.today()
)
print(f'Features computed: {result}')
"
```

**Step 3: Verify feature stats show all features**

```bash
python -m hrp.data.ingestion.features --stats
```

Expected: Should show 19 unique features

---

## Summary ✅ COMPLETED

This plan added technical indicators to complete the HRP spec requirements. **All planned features are now implemented** except `efi_13d` (Elder's Force Index).

**Implemented:**
1. **RSI** (`rsi_14d`) - Momentum oscillator
2. **SMA** (`sma_20d`, `sma_50d`, `sma_200d`) - Trend indicators
3. **Price-to-SMA** (`price_to_sma_20d`, `price_to_sma_50d`, `price_to_sma_200d`) - Mean reversion signals
4. **Volume Ratio** (`volume_ratio`) - Liquidity/interest indicator
5. **Extended Returns** (`returns_60d`, `returns_252d`) - Longer-term returns
6. **Extended Momentum** (`momentum_252d`) - Annual momentum
7. **OBV** (`obv`) - On-Balance Volume
8. **ATR** (`atr_14d`) - Volatility measure
9. **ADX** (`adx_14d`) - Trend strength
10. **MACD** (`macd_line`, `macd_signal`, `macd_histogram`) - Momentum
11. **CCI** (`cci_20d`) - Overbought/oversold
12. **Williams %R** (`williams_r_14d`) - Momentum oscillator
13. **ROC** (`roc_10d`) - Rate of change
14. **Trend** (`trend`) - Binary trend indicator
15. **Bollinger Bands** (`bb_upper_20d`, `bb_lower_20d`, `bb_width_20d`) - Volatility
16. **Stochastic** (`stoch_k_14d`, `stoch_d_14d`) - Momentum oscillator

**Additional features implemented beyond original plan:**
- `ema_12d`, `ema_26d`, `ema_crossover` - EMA signals
- `mfi_14d` - Money Flow Index
- `vwap_20d` - VWAP approximation
- Fundamental features: `market_cap`, `pe_ratio`, `pb_ratio`, `dividend_yield`, `ev_ebitda`

All features are implemented in both:
- `hrp/data/features/computation.py` (on-demand computation with FEATURE_FUNCTIONS registry)
- `hrp/data/ingestion/features.py` (batch ingestion pipeline)

---

## Task 9: Add OBV (On-Balance Volume) ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing test**

```python
def test_compute_obv(self):
    """Test OBV (On-Balance Volume) computation function."""
    from hrp.data.features.computation import compute_obv

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = []
    for i, d in enumerate(dates):
        # Alternating up/down days
        close = 100.0 + (1 if i % 2 == 0 else -0.5)
        data.append({"date": d, "symbol": "AAPL", "close": close, "volume": 1000000})

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_obv(df)

    assert isinstance(result, pd.DataFrame)
    assert "obv" in result.columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_features.py::TestFeatureComputation::test_compute_obv -v`
Expected: FAIL

**Step 3: Implement OBV**

```python
def compute_obv(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV).

    OBV is a cumulative indicator that adds volume on up days
    and subtracts volume on down days.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close', 'volume' columns

    Returns:
        DataFrame with OBV values
    """
    close = prices["close"].unstack(level="symbol")
    volume = prices["volume"].unstack(level="symbol")

    # Calculate price direction: +1 for up, -1 for down, 0 for unchanged
    direction = np.sign(close.diff())

    # OBV = cumulative sum of signed volume
    obv = (direction * volume).cumsum()

    result = obv.stack(level="symbol", future_stack=True)
    return result.to_frame(name="obv")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add On-Balance Volume (OBV) indicator

OBV is a cumulative volume indicator that adds volume on up days
and subtracts on down days, useful for confirming price trends.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Add ATR (Average True Range) ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing test**

```python
def test_compute_atr_14d(self):
    """Test ATR-14 computation function."""
    from hrp.data.features.computation import compute_atr_14d

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = []
    for i, d in enumerate(dates):
        base = 100.0 + i * 0.1
        data.append({
            "date": d, "symbol": "AAPL",
            "high": base + 2, "low": base - 2, "close": base
        })

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_atr_14d(df)

    assert isinstance(result, pd.DataFrame)
    assert "atr_14d" in result.columns

    # ATR should be positive
    valid_values = result["atr_14d"].dropna()
    assert (valid_values > 0).all()
```

**Step 2: Run test to verify it fails**

**Step 3: Implement ATR**

```python
def compute_atr_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Average True Range (ATR).

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = 14-day EMA of True Range

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with ATR values
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    prev_close = close.shift(1)

    # True Range components
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # True Range = max of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, level=0)
    if isinstance(true_range, pd.Series):
        true_range = true_range.unstack(level=0)

    # ATR = 14-day EMA of True Range
    atr = true_range.ewm(span=14, adjust=False).mean()

    result = atr.stack(level="symbol", future_stack=True)
    return result.to_frame(name="atr_14d")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add Average True Range (ATR-14) indicator

ATR measures market volatility by decomposing the entire range
of an asset price for that period.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Add ADX (Average Directional Index) ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing test**

```python
def test_compute_adx_14d(self):
    """Test ADX-14 computation function."""
    from hrp.data.features.computation import compute_adx_14d

    dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    data = []
    for i, d in enumerate(dates):
        base = 100.0 + i * 0.2  # Trending up
        data.append({
            "date": d, "symbol": "AAPL",
            "high": base + 1, "low": base - 1, "close": base
        })

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_adx_14d(df)

    assert isinstance(result, pd.DataFrame)
    assert "adx_14d" in result.columns

    # ADX should be between 0 and 100
    valid_values = result["adx_14d"].dropna()
    assert (valid_values >= 0).all()
    assert (valid_values <= 100).all()
```

**Step 2: Run test to verify it fails**

**Step 3: Implement ADX**

```python
def compute_adx_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Average Directional Index (ADX).

    ADX measures trend strength (0-100) regardless of direction.
    Values > 25 indicate strong trend, < 20 indicate weak/no trend.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with ADX values (0-100 scale)
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()

    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM and -DM
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Smoothed TR, +DM, -DM (14-period EMA)
    atr = true_range.ewm(span=14, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(span=14, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(span=14, adjust=False).mean()

    # +DI and -DI
    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=14, adjust=False).mean()

    # Handle edge cases
    adx = adx.replace([np.inf, -np.inf], np.nan)

    result = adx.stack(level="symbol", future_stack=True)
    return result.to_frame(name="adx_14d")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add Average Directional Index (ADX-14) indicator

ADX measures trend strength on a 0-100 scale regardless of direction.
Values > 25 indicate a strong trend.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Add MACD (Moving Average Convergence Divergence) ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing tests**

```python
def test_compute_macd(self):
    """Test MACD computation functions."""
    from hrp.data.features.computation import (
        compute_macd_line, compute_macd_signal, compute_macd_histogram
    )

    dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 + i * 0.1}
            for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])

    macd_line = compute_macd_line(df)
    macd_signal = compute_macd_signal(df)
    macd_hist = compute_macd_histogram(df)

    assert "macd_line" in macd_line.columns
    assert "macd_signal" in macd_signal.columns
    assert "macd_histogram" in macd_hist.columns
```

**Step 2: Run test to verify it fails**

**Step 3: Implement MACD functions**

```python
def compute_macd_line(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD Line (EMA-12 minus EMA-26).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with MACD line values
    """
    close = prices["close"].unstack(level="symbol")
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26

    result = macd_line.stack(level="symbol", future_stack=True)
    return result.to_frame(name="macd_line")


def compute_macd_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD Signal Line (9-day EMA of MACD line).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with MACD signal values
    """
    close = prices["close"].unstack(level="symbol")
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal = macd_line.ewm(span=9, adjust=False).mean()

    result = signal.stack(level="symbol", future_stack=True)
    return result.to_frame(name="macd_signal")


def compute_macd_histogram(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD Histogram (MACD line minus Signal line).

    Positive histogram = bullish momentum, Negative = bearish.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with MACD histogram values
    """
    close = prices["close"].unstack(level="symbol")
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal

    result = histogram.stack(level="symbol", future_stack=True)
    return result.to_frame(name="macd_histogram")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add MACD indicator (line, signal, histogram)

Implements Moving Average Convergence Divergence with standard
12/26/9 parameters. All three components available as separate features.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Add CCI (Commodity Channel Index) ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing test**

```python
def test_compute_cci_20d(self):
    """Test CCI-20 computation function."""
    from hrp.data.features.computation import compute_cci_20d

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = []
    for i, d in enumerate(dates):
        base = 100.0 + i * 0.1
        data.append({
            "date": d, "symbol": "AAPL",
            "high": base + 1, "low": base - 1, "close": base
        })

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_cci_20d(df)

    assert isinstance(result, pd.DataFrame)
    assert "cci_20d" in result.columns
```

**Step 2: Run test to verify it fails**

**Step 3: Implement CCI**

```python
def compute_cci_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Commodity Channel Index (CCI).

    CCI = (Typical Price - SMA of TP) / (0.015 * Mean Deviation)
    where Typical Price = (High + Low + Close) / 3

    Values > +100 indicate overbought, < -100 indicate oversold.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with CCI values
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    # Typical Price
    tp = (high + low + close) / 3

    # SMA of Typical Price
    tp_sma = tp.rolling(window=20).mean()

    # Mean Deviation
    mean_dev = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())

    # CCI
    cci = (tp - tp_sma) / (0.015 * mean_dev)

    result = cci.stack(level="symbol", future_stack=True)
    return result.to_frame(name="cci_20d")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add Commodity Channel Index (CCI-20) indicator

CCI measures deviation from statistical mean. Values outside
+/-100 indicate overbought/oversold conditions.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Add Williams %R ✅ COMPLETED (as `williams_r_14d`)

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing test**

```python
def test_compute_wr_14d(self):
    """Test Williams %R computation function."""
    from hrp.data.features.computation import compute_wr_14d

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = []
    for i, d in enumerate(dates):
        base = 100.0 + i * 0.1
        data.append({
            "date": d, "symbol": "AAPL",
            "high": base + 1, "low": base - 1, "close": base
        })

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_wr_14d(df)

    assert isinstance(result, pd.DataFrame)
    assert "wr_14d" in result.columns

    # Williams %R should be between -100 and 0
    valid_values = result["wr_14d"].dropna()
    assert (valid_values >= -100).all()
    assert (valid_values <= 0).all()
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Williams %R**

```python
def compute_wr_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Williams %R.

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Range is -100 to 0. Values near 0 = overbought, near -100 = oversold.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with Williams %R values (-100 to 0)
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    highest_high = high.rolling(window=14).max()
    lowest_low = low.rolling(window=14).min()

    wr = ((highest_high - close) / (highest_high - lowest_low)) * -100

    result = wr.stack(level="symbol", future_stack=True)
    return result.to_frame(name="wr_14d")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add Williams %R (WR-14) indicator

Williams %R is a momentum oscillator measuring overbought/oversold
conditions on a -100 to 0 scale.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Add ROC (Rate of Change) ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing test**

```python
def test_compute_roc_10d(self):
    """Test ROC-10 computation function."""
    from hrp.data.features.computation import compute_roc_10d

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 * (1 + 0.01 * i)}
            for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_roc_10d(df)

    assert isinstance(result, pd.DataFrame)
    assert "roc_10d" in result.columns
```

**Step 2: Run test to verify it fails**

**Step 3: Implement ROC**

```python
def compute_roc_10d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 10-day Rate of Change (ROC).

    ROC = ((Close - Close_n_periods_ago) / Close_n_periods_ago) * 100

    Measures the percentage change over 10 periods.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with ROC values (percentage)
    """
    close = prices["close"].unstack(level="symbol")
    roc = ((close - close.shift(10)) / close.shift(10)) * 100

    result = roc.stack(level="symbol", future_stack=True)
    return result.to_frame(name="roc_10d")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add Rate of Change (ROC-10) indicator

ROC measures the percentage change in price over 10 periods,
useful for identifying momentum and potential reversals.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Add Trend Indicator ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing test**

```python
def test_compute_trend(self):
    """Test trend indicator computation function."""
    from hrp.data.features.computation import compute_trend

    dates = pd.date_range("2022-01-01", "2023-06-30", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 + i * 0.1}
            for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_trend(df)

    assert isinstance(result, pd.DataFrame)
    assert "trend" in result.columns

    # Trend should be +1 or -1
    valid_values = result["trend"].dropna()
    assert valid_values.isin([1, -1]).all()
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Trend**

```python
def compute_trend(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate binary trend indicator based on price vs 200-day SMA.

    +1 = Bullish (price above SMA-200)
    -1 = Bearish (price below SMA-200)

    This simple trend filter is robust and avoids curve fitting.
    Research shows it's effective for systematic trading.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with trend values (+1 or -1)
    """
    close = prices["close"].unstack(level="symbol")
    sma_200 = close.rolling(window=200).mean()

    # +1 if price > SMA, -1 otherwise
    trend = np.where(close > sma_200, 1, -1)
    trend_df = pd.DataFrame(trend, index=close.index, columns=close.columns)

    result = trend_df.stack(level="symbol", future_stack=True)
    return result.to_frame(name="trend")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add binary trend indicator (price vs SMA-200)

Simple and robust trend filter: +1 when price above 200-day SMA,
-1 when below. Effective for systematic trading without overfitting.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Add EFI (Elder's Force Index) ❌ NOT IMPLEMENTED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing test**

```python
def test_compute_efi_13d(self):
    """Test Elder's Force Index computation function."""
    from hrp.data.features.computation import compute_efi_13d

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = []
    for i, d in enumerate(dates):
        data.append({
            "date": d, "symbol": "AAPL",
            "close": 100.0 + i * 0.1, "volume": 1000000
        })

    df = pd.DataFrame(data).set_index(["date", "symbol"])
    result = compute_efi_13d(df)

    assert isinstance(result, pd.DataFrame)
    assert "efi_13d" in result.columns
```

**Step 2: Run test to verify it fails**

**Step 3: Implement EFI**

```python
def compute_efi_13d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 13-day Elder's Force Index (EFI).

    Force Index = (Close - Previous Close) * Volume
    EFI = 13-day EMA of Force Index

    Measures buying/selling pressure. Positive = buying pressure,
    Negative = selling pressure.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close', 'volume' columns

    Returns:
        DataFrame with EFI values
    """
    close = prices["close"].unstack(level="symbol")
    volume = prices["volume"].unstack(level="symbol")

    # Force Index = price change * volume
    force_index = close.diff() * volume

    # 13-day EMA smoothing
    efi = force_index.ewm(span=13, adjust=False).mean()

    result = efi.stack(level="symbol", future_stack=True)
    return result.to_frame(name="efi_13d")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add Elder's Force Index (EFI-13) indicator

EFI measures buying/selling pressure by combining price change
with volume, smoothed with 13-day EMA.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Add Bollinger Bands ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing tests**

```python
def test_compute_bollinger_bands(self):
    """Test Bollinger Bands computation functions."""
    from hrp.data.features.computation import (
        compute_bb_upper_20d, compute_bb_lower_20d, compute_bb_width_20d
    )

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = [{"date": d, "symbol": "AAPL", "close": 100.0 + np.sin(i/5)}
            for i, d in enumerate(dates)]

    df = pd.DataFrame(data).set_index(["date", "symbol"])

    upper = compute_bb_upper_20d(df)
    lower = compute_bb_lower_20d(df)
    width = compute_bb_width_20d(df)

    assert "bb_upper_20d" in upper.columns
    assert "bb_lower_20d" in lower.columns
    assert "bb_width_20d" in width.columns

    # Upper should be greater than lower
    valid_upper = upper["bb_upper_20d"].dropna()
    valid_lower = lower["bb_lower_20d"].dropna()
    assert (valid_upper.values > valid_lower.values).all()
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Bollinger Bands**

```python
def compute_bb_upper_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Bollinger Band Upper (SMA + 2*std).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with upper band values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    upper = sma + (2 * std)

    result = upper.stack(level="symbol", future_stack=True)
    return result.to_frame(name="bb_upper_20d")


def compute_bb_lower_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Bollinger Band Lower (SMA - 2*std).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with lower band values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    lower = sma - (2 * std)

    result = lower.stack(level="symbol", future_stack=True)
    return result.to_frame(name="bb_lower_20d")


def compute_bb_width_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Bollinger Band Width ((upper - lower) / middle).

    Measures volatility. High width = high volatility, low width = low volatility.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with band width values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    width = (upper - lower) / sma

    result = width.stack(level="symbol", future_stack=True)
    return result.to_frame(name="bb_width_20d")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add Bollinger Bands (upper, lower, width)

Implements 20-day Bollinger Bands with 2 standard deviations.
Band width measures volatility - useful for squeeze detection.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Add Stochastic Oscillator ✅ COMPLETED

**Files:**
- Test: `tests/test_data/test_features.py`
- Modify: `hrp/data/features/computation.py`

**Step 1: Write failing tests**

```python
def test_compute_stochastic(self):
    """Test Stochastic Oscillator computation functions."""
    from hrp.data.features.computation import compute_stoch_k_14d, compute_stoch_d_14d

    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    data = []
    for i, d in enumerate(dates):
        base = 100.0 + np.sin(i/10) * 5
        data.append({
            "date": d, "symbol": "AAPL",
            "high": base + 1, "low": base - 1, "close": base
        })

    df = pd.DataFrame(data).set_index(["date", "symbol"])

    stoch_k = compute_stoch_k_14d(df)
    stoch_d = compute_stoch_d_14d(df)

    assert "stoch_k_14d" in stoch_k.columns
    assert "stoch_d_14d" in stoch_d.columns

    # Stochastic should be between 0 and 100
    valid_k = stoch_k["stoch_k_14d"].dropna()
    assert (valid_k >= 0).all()
    assert (valid_k <= 100).all()
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Stochastic Oscillator**

```python
def compute_stoch_k_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Stochastic %K (Fast Stochastic).

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100

    Range is 0-100. > 80 = overbought, < 20 = oversold.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with Stochastic %K values (0-100)
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()

    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100

    result = stoch_k.stack(level="symbol", future_stack=True)
    return result.to_frame(name="stoch_k_14d")


def compute_stoch_d_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Stochastic %D (3-day SMA of %K, Slow Stochastic).

    %D is the signal line for Stochastic oscillator.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with Stochastic %D values (0-100)
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()

    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    stoch_d = stoch_k.rolling(window=3).mean()

    result = stoch_d.stack(level="symbol", future_stack=True)
    return result.to_frame(name="stoch_d_14d")
```

**Step 4: Add to FEATURE_FUNCTIONS and run tests**

**Step 5: Commit**

```bash
git add tests/test_data/test_features.py hrp/data/features/computation.py
git commit -m "$(cat <<'EOF'
feat(features): add Stochastic Oscillator (%K and %D)

Implements 14-day Stochastic with %K (fast) and %D (slow/signal).
Useful for identifying overbought/oversold conditions.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Update Ingestion Pipeline with All New Features ✅ COMPLETED

**Files:**
- Modify: `hrp/data/ingestion/features.py`

Update the `_compute_all_features` function to include all 27 new features. Add the computations for:
- OBV, ATR, ADX, MACD (3), CCI, Williams %R, ROC, Trend, EFI
- Bollinger Bands (3), Stochastic (2)

Update the `feature_columns` list to include all new features.

**Commit**

```bash
git add hrp/data/ingestion/features.py
git commit -m "$(cat <<'EOF'
feat(features): update ingestion pipeline with all 27 new indicators

Adds OBV, ATR, ADX, MACD, CCI, Williams %R, ROC, Trend, EFI,
Bollinger Bands, and Stochastic to the batch ingestion pipeline.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: Final Verification and Documentation Update ✅ COMPLETED

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run E2E feature computation verification**

```bash
python -c "
from hrp.data.ingestion.features import compute_features, get_feature_stats
from datetime import date, timedelta

# Compute features for a test symbol
result = compute_features(
    symbols=['AAPL'],
    start=date.today() - timedelta(days=30),
    end=date.today()
)
print(f'Computation result: {result}')

# Verify all features are present
stats = get_feature_stats()
print(f'Total features: {stats[\"unique_features\"]}')
print('Feature coverage:')
for fc in stats['feature_coverage']:
    print(f'  {fc[\"feature_name\"]}')
"
```

Expected: Should show 38 unique features computed successfully.

**Step 3: Update CLAUDE.md with complete feature list**

Update the feature documentation to show all 38 features now available.

**Step 4: Update Project-Status.md**

Update `docs/plans/Project-Status.md`:
- Change Feature Store status from "14+ technical indicators" to "38 technical indicators"
- Update Tier 2 completion percentage if appropriate

**Step 5: Commit all documentation**

```bash
git add CLAUDE.md docs/plans/Project-Status.md
git commit -m "$(cat <<'EOF'
docs: update documentation with complete 38-feature list

Updates CLAUDE.md and Project-Status.md to reflect all technical
indicators now available: OBV, ATR, ADX, MACD, CCI, Williams %R,
ROC, Trend, EFI, Bollinger Bands, Stochastic, and more.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Complete Feature Summary (44 Total - Current Implementation)

| Category | Features |
|----------|----------|
| **Returns** | `returns_1d`, `returns_5d`, `returns_20d`, `returns_60d`, `returns_252d` |
| **Momentum** | `momentum_20d`, `momentum_60d`, `momentum_252d`, `roc_10d` |
| **Volatility** | `volatility_20d`, `volatility_60d`, `atr_14d` |
| **Volume** | `volume_20d`, `volume_ratio`, `obv` |
| **Moving Averages** | `sma_20d`, `sma_50d`, `sma_200d`, `ema_12d`, `ema_26d` |
| **EMA Signals** | `ema_crossover` |
| **Price Ratios** | `price_to_sma_20d`, `price_to_sma_50d`, `price_to_sma_200d` |
| **Trend** | `trend`, `adx_14d` |
| **Oscillators** | `rsi_14d`, `cci_20d`, `williams_r_14d`, `stoch_k_14d`, `stoch_d_14d`, `mfi_14d` |
| **MACD** | `macd_line`, `macd_signal`, `macd_histogram` |
| **Bollinger** | `bb_upper_20d`, `bb_lower_20d`, `bb_width_20d` |
| **VWAP** | `vwap_20d` |
| **Fundamental** | `market_cap`, `pe_ratio`, `pb_ratio`, `dividend_yield`, `ev_ebitda` |

**Not Implemented:** `efi_13d` (Elder's Force Index)

**Sources:**
- [QuantInsti: Best Trend Indicators](https://blog.quantinsti.com/indicators-build-trend-following-strategy/)
- [Graham Capital: Trend Following Primer](https://www.grahamcapital.com/wp-content/uploads/2024/04/Trend-Following-Primer_January-2022.pdf)
- [Quantified Strategies: Trend Following](https://www.quantifiedstrategies.com/trend-following-trading-strategy/)
