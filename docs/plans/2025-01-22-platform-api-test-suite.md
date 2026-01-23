# Platform API Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve 70%+ code coverage on Platform API with comprehensive unit tests, integration tests, and synthetic data generators.

**Architecture:** Add input validation to platform.py (fixes 20 failing tests), create synthetic data generators for deterministic test fixtures, add unit tests for uncovered methods (backtest, split adjustment, corporate actions), and create integration test for full backtest flow with mocked MLflow.

**Tech Stack:** pytest, pytest-cov, unittest.mock, pandas, DuckDB

---

## Task 1: Add Input Validation to platform.py

**Files:**
- Modify: `hrp/api/platform.py:70-180` (data operations)
- Modify: `hrp/api/platform.py:349-520` (hypothesis operations)

**Step 1: Add validation helper at top of PlatformAPI class**

Add after line 64 (after `logger.debug("PlatformAPI initialized")`):

```python
    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def _validate_not_empty(self, value: str, field_name: str) -> None:
        """Validate that a string is not empty or whitespace-only."""
        if not value or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")

    def _validate_positive(self, value: int, field_name: str) -> None:
        """Validate that an integer is positive."""
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")

    def _validate_not_future(self, d: date, field_name: str) -> None:
        """Validate that a date is not in the future."""
        if d > date.today():
            raise ValueError(f"{field_name} cannot be in the future")

    def _validate_date_range(self, start: date, end: date) -> None:
        """Validate that start date is not after end date."""
        if start > end:
            raise ValueError("start date must be <= end date")

    def _validate_symbols_in_universe(self, symbols: List[str], as_of_date: date = None) -> None:
        """Validate that all symbols are in the universe."""
        # Get valid symbols from universe
        if as_of_date:
            query = """
                SELECT DISTINCT symbol FROM universe
                WHERE in_universe = TRUE AND date = ?
            """
            result = self._db.fetchall(query, (as_of_date,))
        else:
            query = """
                SELECT DISTINCT symbol FROM universe
                WHERE in_universe = TRUE
            """
            result = self._db.fetchall(query)

        valid_symbols = {row[0] for row in result}
        invalid = [s for s in symbols if s not in valid_symbols]

        if invalid:
            invalid_str = ", ".join(sorted(invalid))
            if as_of_date:
                raise ValueError(f"Invalid symbols not in universe as of {as_of_date}: {invalid_str}")
            else:
                raise ValueError(f"Invalid symbols not found in universe: {invalid_str}")
```

**Step 2: Add validation to get_prices()**

Add at start of `get_prices()` method (after the docstring, before `if not symbols:`):

```python
        # Validate inputs
        self._validate_not_future(start, "start date")
        self._validate_not_future(end, "end date")
        self._validate_date_range(start, end)
```

After the existing `if not symbols:` check, add:

```python
        self._validate_symbols_in_universe(symbols)
```

**Step 3: Add validation to get_features()**

Add at start of `get_features()` method (after the docstring, before `if not symbols:`):

```python
        # Validate inputs
        self._validate_not_future(as_of_date, "as_of_date")
```

After the existing empty checks, add:

```python
        self._validate_symbols_in_universe(symbols, as_of_date)
```

**Step 4: Add validation to get_universe()**

Add at start of `get_universe()` method (after docstring):

```python
        self._validate_not_future(as_of_date, "as_of_date")
```

**Step 5: Add validation to create_hypothesis()**

Add at start of `create_hypothesis()` method (after docstring):

```python
        # Validate inputs
        self._validate_not_empty(title, "title")
        self._validate_not_empty(thesis, "thesis")
        self._validate_not_empty(prediction, "prediction")
        self._validate_not_empty(falsification, "falsification")
        self._validate_not_empty(actor, "actor")
```

**Step 6: Add validation to update_hypothesis()**

Add at start of `update_hypothesis()` method (after docstring):

```python
        # Validate inputs
        self._validate_not_empty(hypothesis_id, "hypothesis_id")
        self._validate_not_empty(status, "status")
        self._validate_not_empty(actor, "actor")
```

**Step 7: Add validation to list_hypotheses()**

Add at start of `list_hypotheses()` method (after docstring):

```python
        self._validate_positive(limit, "limit")
```

**Step 8: Add validation to get_hypothesis()**

Add at start of `get_hypothesis()` method (after docstring):

```python
        self._validate_not_empty(hypothesis_id, "hypothesis_id")
```

**Step 9: Run validation tests**

Run: `source .venv/bin/activate && pytest tests/test_api/test_platform.py::TestPlatformAPIValidation -v`
Expected: All 20 validation tests PASS

**Step 10: Commit**

```bash
git add hrp/api/platform.py
git commit -m "feat(api): add input validation to PlatformAPI methods

Add validation helpers and input validation for:
- Date validation (not future, valid range)
- Symbol validation (must be in universe)
- String validation (not empty/whitespace)
- Numeric validation (positive integers)

Fixes 20 failing validation tests.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Synthetic Data Generator

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/synthetic.py`
- Test: `tests/fixtures/test_synthetic.py`

**Step 1: Create fixtures package**

Create `tests/fixtures/__init__.py`:

```python
"""Test fixtures and synthetic data generators."""

from tests.fixtures.synthetic import (
    generate_prices,
    generate_features,
    generate_corporate_actions,
    generate_universe,
)

__all__ = [
    "generate_prices",
    "generate_features",
    "generate_corporate_actions",
    "generate_universe",
]
```

**Step 2: Create synthetic data generator**

Create `tests/fixtures/synthetic.py`:

```python
"""
Synthetic data generators for deterministic test fixtures.

All generators use a seed parameter for reproducibility.
"""

from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


def generate_prices(
    symbols: List[str],
    start: date,
    end: date,
    seed: int = 42,
    base_prices: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Generate synthetic price data with realistic random walk.

    Args:
        symbols: List of ticker symbols
        start: Start date
        end: End date
        seed: Random seed for reproducibility
        base_prices: Optional dict of symbol -> base price

    Returns:
        DataFrame with columns: symbol, date, open, high, low, close, adj_close, volume
    """
    np.random.seed(seed)

    if base_prices is None:
        base_prices = {s: 100.0 + i * 50 for i, s in enumerate(symbols)}

    dates = pd.date_range(start, end, freq="B")
    data = []

    for symbol in symbols:
        price = base_prices.get(symbol, 100.0)

        for d in dates:
            # Random walk with slight upward drift
            daily_return = np.random.normal(0.0005, 0.02)
            price = price * (1 + daily_return)

            # Generate OHLC from close
            intraday_vol = abs(np.random.normal(0, 0.01))
            high = price * (1 + intraday_vol)
            low = price * (1 - intraday_vol)
            open_price = low + (high - low) * np.random.random()

            volume = int(np.random.uniform(500000, 5000000))

            data.append({
                "symbol": symbol,
                "date": d.date(),
                "open": round(open_price, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(price, 4),
                "adj_close": round(price, 4),
                "volume": volume,
            })

    return pd.DataFrame(data)


def generate_features(
    symbols: List[str],
    feature_names: List[str],
    as_of_date: date,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic feature data.

    Args:
        symbols: List of ticker symbols
        feature_names: List of feature names
        as_of_date: Date for features
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: symbol, date, feature_name, value, version
    """
    np.random.seed(seed)

    data = []
    for symbol in symbols:
        for feature in feature_names:
            # Generate realistic ranges based on feature name
            if "momentum" in feature.lower():
                value = np.random.uniform(-0.3, 0.5)
            elif "volatility" in feature.lower():
                value = np.random.uniform(0.1, 0.6)
            elif "rsi" in feature.lower():
                value = np.random.uniform(20, 80)
            else:
                value = np.random.uniform(-1, 1)

            data.append({
                "symbol": symbol,
                "date": as_of_date,
                "feature_name": feature,
                "value": round(value, 6),
                "version": "v1",
            })

    return pd.DataFrame(data)


def generate_corporate_actions(
    symbols: List[str],
    start: date,
    end: date,
    action_types: List[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic corporate actions.

    Args:
        symbols: List of ticker symbols
        start: Start date
        end: End date
        action_types: Types to generate (default: ['split', 'dividend'])
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: symbol, date, action_type, factor, source
    """
    np.random.seed(seed)

    if action_types is None:
        action_types = ["split", "dividend"]

    data = []
    days_range = (end - start).days

    for symbol in symbols:
        # Each symbol has 0-2 corporate actions
        n_actions = np.random.randint(0, 3)

        for _ in range(n_actions):
            action_date = start + timedelta(days=np.random.randint(0, days_range))
            action_type = np.random.choice(action_types)

            if action_type == "split":
                # Common split ratios
                factor = np.random.choice([2.0, 3.0, 4.0, 0.5, 0.25])
            else:
                # Dividend as percentage of price (1-5%)
                factor = round(np.random.uniform(0.01, 0.05), 4)

            data.append({
                "symbol": symbol,
                "date": action_date,
                "action_type": action_type,
                "factor": factor,
                "source": "synthetic",
            })

    return pd.DataFrame(data)


def generate_universe(
    symbols: List[str],
    dates: List[date],
    sectors: Optional[dict] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic universe membership data.

    Args:
        symbols: List of ticker symbols
        dates: List of dates for universe snapshots
        sectors: Optional dict of symbol -> sector
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: symbol, date, in_universe, sector, market_cap
    """
    np.random.seed(seed)

    if sectors is None:
        default_sectors = ["Technology", "Healthcare", "Consumer", "Industrial", "Energy"]
        sectors = {s: np.random.choice(default_sectors) for s in symbols}

    data = []
    for symbol in symbols:
        base_market_cap = np.random.uniform(10e9, 500e9)

        for d in dates:
            # Small random variation in market cap
            market_cap = base_market_cap * np.random.uniform(0.9, 1.1)

            data.append({
                "symbol": symbol,
                "date": d,
                "in_universe": True,
                "sector": sectors.get(symbol, "Unknown"),
                "market_cap": round(market_cap, 2),
            })

    return pd.DataFrame(data)
```

**Step 3: Create test for synthetic generators**

Create `tests/fixtures/test_synthetic.py`:

```python
"""Tests for synthetic data generators."""

from datetime import date

import pandas as pd
import pytest

from tests.fixtures.synthetic import (
    generate_prices,
    generate_features,
    generate_corporate_actions,
    generate_universe,
)


class TestGeneratePrices:
    """Tests for price data generator."""

    def test_generates_correct_columns(self):
        """Generated prices have all required columns."""
        df = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10))
        expected = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
        assert list(df.columns) == expected

    def test_deterministic_with_seed(self):
        """Same seed produces same data."""
        df1 = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10), seed=42)
        df2 = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10), seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds produce different data."""
        df1 = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10), seed=42)
        df2 = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10), seed=99)
        assert not df1["close"].equals(df2["close"])

    def test_high_low_relationship(self):
        """High is always >= low."""
        df = generate_prices(["AAPL", "MSFT"], date(2023, 1, 1), date(2023, 12, 31))
        assert (df["high"] >= df["low"]).all()

    def test_multiple_symbols(self):
        """Generates data for multiple symbols."""
        df = generate_prices(["AAPL", "MSFT", "GOOGL"], date(2023, 1, 1), date(2023, 1, 10))
        assert set(df["symbol"].unique()) == {"AAPL", "MSFT", "GOOGL"}


class TestGenerateFeatures:
    """Tests for feature data generator."""

    def test_generates_correct_columns(self):
        """Generated features have all required columns."""
        df = generate_features(["AAPL"], ["momentum_20d"], date(2023, 1, 5))
        expected = ["symbol", "date", "feature_name", "value", "version"]
        assert list(df.columns) == expected

    def test_deterministic_with_seed(self):
        """Same seed produces same data."""
        df1 = generate_features(["AAPL"], ["momentum_20d"], date(2023, 1, 5), seed=42)
        df2 = generate_features(["AAPL"], ["momentum_20d"], date(2023, 1, 5), seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_multiple_features(self):
        """Generates all requested features."""
        features = ["momentum_20d", "volatility_60d", "rsi_14d"]
        df = generate_features(["AAPL"], features, date(2023, 1, 5))
        assert set(df["feature_name"].unique()) == set(features)


class TestGenerateCorporateActions:
    """Tests for corporate actions generator."""

    def test_generates_correct_columns(self):
        """Generated actions have all required columns."""
        df = generate_corporate_actions(["AAPL"], date(2023, 1, 1), date(2023, 12, 31))
        if not df.empty:
            expected = ["symbol", "date", "action_type", "factor", "source"]
            assert list(df.columns) == expected

    def test_split_factors_valid(self):
        """Split factors are valid ratios."""
        df = generate_corporate_actions(
            ["AAPL", "MSFT", "GOOGL"],
            date(2020, 1, 1),
            date(2023, 12, 31),
            action_types=["split"],
            seed=42,
        )
        if not df.empty:
            assert (df["factor"] > 0).all()


class TestGenerateUniverse:
    """Tests for universe generator."""

    def test_generates_correct_columns(self):
        """Generated universe has all required columns."""
        df = generate_universe(["AAPL"], [date(2023, 1, 1)])
        expected = ["symbol", "date", "in_universe", "sector", "market_cap"]
        assert list(df.columns) == expected

    def test_all_symbols_included(self):
        """All symbols appear in universe."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        df = generate_universe(symbols, [date(2023, 1, 1)])
        assert set(df["symbol"].unique()) == set(symbols)
```

**Step 4: Run synthetic data tests**

Run: `source .venv/bin/activate && pytest tests/fixtures/test_synthetic.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/fixtures/
git commit -m "feat(tests): add synthetic data generators for test fixtures

Add deterministic generators for:
- Price data with realistic random walk
- Feature data with appropriate value ranges
- Corporate actions (splits, dividends)
- Universe membership data

All generators support seed parameter for reproducibility.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Unit Tests for Uncovered Methods

**Files:**
- Modify: `tests/test_api/test_platform.py`

**Step 1: Add TestPlatformAPICorporateActions class**

Add after `TestPlatformAPICalendar` class (around line 1413):

```python
class TestPlatformAPICorporateActions:
    """Tests for corporate actions data operations."""

    def test_get_corporate_actions_empty_symbols_raises(self, test_api):
        """get_corporate_actions should reject empty symbols list."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            test_api.get_corporate_actions([], date(2023, 1, 1), date(2023, 12, 31))

    def test_get_corporate_actions_no_data(self, populated_db):
        """get_corporate_actions returns empty DataFrame when no actions exist."""
        result = populated_db.get_corporate_actions(
            ["AAPL"], date(2023, 1, 1), date(2023, 1, 10)
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_corporate_actions_with_data(self, populated_db):
        """get_corporate_actions returns data when actions exist."""
        # Insert a corporate action
        populated_db._db.execute(
            """
            INSERT INTO corporate_actions (symbol, date, action_type, factor, source)
            VALUES ('AAPL', '2023-01-05', 'split', 2.0, 'test')
            """
        )

        result = populated_db.get_corporate_actions(
            ["AAPL"], date(2023, 1, 1), date(2023, 1, 10)
        )
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["action_type"] == "split"
        assert result.iloc[0]["factor"] == 2.0

    def test_get_corporate_actions_date_range_filter(self, populated_db):
        """get_corporate_actions filters by date range."""
        # Insert actions on different dates
        populated_db._db.execute(
            """
            INSERT INTO corporate_actions (symbol, date, action_type, factor, source)
            VALUES
                ('AAPL', '2023-01-05', 'split', 2.0, 'test'),
                ('AAPL', '2023-06-15', 'dividend', 0.23, 'test')
            """
        )

        result = populated_db.get_corporate_actions(
            ["AAPL"], date(2023, 1, 1), date(2023, 1, 31)
        )
        assert len(result) == 1
        assert result.iloc[0]["action_type"] == "split"

    def test_get_corporate_actions_returns_all_columns(self, populated_db):
        """get_corporate_actions returns all expected columns."""
        populated_db._db.execute(
            """
            INSERT INTO corporate_actions (symbol, date, action_type, factor, source)
            VALUES ('AAPL', '2023-01-05', 'split', 2.0, 'test')
            """
        )

        result = populated_db.get_corporate_actions(
            ["AAPL"], date(2023, 1, 1), date(2023, 1, 10)
        )
        expected_columns = ["symbol", "date", "action_type", "factor", "source"]
        for col in expected_columns:
            assert col in result.columns
```

**Step 2: Add TestPlatformAPISplitAdjustment class**

Add after `TestPlatformAPICorporateActions`:

```python
class TestPlatformAPISplitAdjustment:
    """Tests for price split adjustment."""

    def test_adjust_empty_dataframe(self, test_api):
        """adjust_prices_for_splits handles empty DataFrame."""
        empty_df = pd.DataFrame(columns=["symbol", "date", "close"])
        result = test_api.adjust_prices_for_splits(empty_df)
        assert "split_adjusted_close" in result.columns

    def test_adjust_no_splits(self, populated_db):
        """adjust_prices_for_splits returns unadjusted when no splits."""
        prices = populated_db.get_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10))
        result = populated_db.adjust_prices_for_splits(prices)

        assert "split_adjusted_close" in result.columns
        # With no splits, adjusted should equal close
        pd.testing.assert_series_equal(
            result["split_adjusted_close"],
            result["close"],
            check_names=False,
        )

    def test_adjust_single_split(self, populated_db):
        """adjust_prices_for_splits applies single split correctly."""
        # Insert a 2:1 split on Jan 5
        populated_db._db.execute(
            """
            INSERT INTO corporate_actions (symbol, date, action_type, factor, source)
            VALUES ('AAPL', '2023-01-05', 'split', 2.0, 'test')
            """
        )

        prices = populated_db.get_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10))
        result = populated_db.adjust_prices_for_splits(prices)

        # Prices before split should be multiplied by factor
        before_split = result[result["date"] < pd.Timestamp("2023-01-05")]
        after_split = result[result["date"] >= pd.Timestamp("2023-01-05")]

        if not before_split.empty and not after_split.empty:
            # Before split: adjusted = close * 2
            for _, row in before_split.iterrows():
                assert row["split_adjusted_close"] == pytest.approx(row["close"] * 2.0)

            # After split: adjusted = close (unchanged)
            for _, row in after_split.iterrows():
                assert row["split_adjusted_close"] == pytest.approx(row["close"])

    def test_adjust_multiple_splits_same_symbol(self, populated_db):
        """adjust_prices_for_splits handles multiple splits for same symbol."""
        # Insert two splits
        populated_db._db.execute(
            """
            INSERT INTO corporate_actions (symbol, date, action_type, factor, source)
            VALUES
                ('AAPL', '2023-01-03', 'split', 2.0, 'test'),
                ('AAPL', '2023-01-06', 'split', 3.0, 'test')
            """
        )

        prices = populated_db.get_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10))
        result = populated_db.adjust_prices_for_splits(prices)

        # Prices before first split should have both factors applied (2 * 3 = 6)
        before_first = result[result["date"] < pd.Timestamp("2023-01-03")]
        if not before_first.empty:
            for _, row in before_first.iterrows():
                assert row["split_adjusted_close"] == pytest.approx(row["close"] * 6.0)

    def test_adjust_preserves_original_columns(self, populated_db):
        """adjust_prices_for_splits preserves all original columns."""
        prices = populated_db.get_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10))
        original_columns = set(prices.columns)

        result = populated_db.adjust_prices_for_splits(prices)

        for col in original_columns:
            assert col in result.columns
```

**Step 3: Add TestPlatformAPIBacktest class with mocks**

Add after `TestPlatformAPISplitAdjustment`:

```python
class TestPlatformAPIBacktest:
    """Tests for backtest operations with mocked dependencies."""

    @patch("hrp.api.platform.log_backtest")
    @patch("hrp.api.platform.execute_backtest")
    @patch("hrp.api.platform.get_price_data")
    @patch("hrp.api.platform.generate_momentum_signals")
    def test_run_backtest_success(
        self, mock_signals, mock_prices, mock_backtest, mock_log, test_api
    ):
        """run_backtest completes successfully with mocked dependencies."""
        from hrp.research.config import BacktestConfig, BacktestResult

        # Setup mocks
        mock_prices.return_value = pd.DataFrame({
            "symbol": ["AAPL"] * 10,
            "date": pd.date_range("2023-01-01", periods=10),
            "close": [100 + i for i in range(10)],
        })
        mock_signals.return_value = pd.DataFrame({
            "AAPL": [1] * 10,
        }, index=pd.date_range("2023-01-01", periods=10))
        mock_backtest.return_value = BacktestResult(
            config=BacktestConfig(symbols=["AAPL"]),
            metrics={"sharpe_ratio": 1.5, "total_return": 0.25},
            equity_curve=pd.Series([100, 105, 110]),
            trades=pd.DataFrame(),
        )
        mock_log.return_value = "run-123"

        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )

        result = test_api.run_backtest(config)

        assert result == "run-123"
        mock_prices.assert_called_once()
        mock_backtest.assert_called_once()
        mock_log.assert_called_once()

    @patch("hrp.api.platform.log_backtest")
    @patch("hrp.api.platform.execute_backtest")
    @patch("hrp.api.platform.get_price_data")
    def test_run_backtest_with_custom_signals(
        self, mock_prices, mock_backtest, mock_log, test_api
    ):
        """run_backtest uses provided signals instead of generating."""
        from hrp.research.config import BacktestConfig, BacktestResult

        mock_prices.return_value = pd.DataFrame({
            "symbol": ["AAPL"] * 10,
            "date": pd.date_range("2023-01-01", periods=10),
            "close": [100 + i for i in range(10)],
        })
        mock_backtest.return_value = BacktestResult(
            config=BacktestConfig(symbols=["AAPL"]),
            metrics={"sharpe_ratio": 1.2},
            equity_curve=pd.Series([100, 102]),
            trades=pd.DataFrame(),
        )
        mock_log.return_value = "run-456"

        config = BacktestConfig(symbols=["AAPL"], start_date=date(2023, 1, 1), end_date=date(2023, 12, 31))
        custom_signals = pd.DataFrame({"AAPL": [0, 1, 1, 0]})

        result = test_api.run_backtest(config, signals=custom_signals)

        assert result == "run-456"

    @patch("hrp.api.platform.log_backtest")
    @patch("hrp.api.platform.execute_backtest")
    @patch("hrp.api.platform.get_price_data")
    @patch("hrp.api.platform.generate_momentum_signals")
    def test_run_backtest_links_hypothesis(
        self, mock_signals, mock_prices, mock_backtest, mock_log, test_api
    ):
        """run_backtest links experiment to hypothesis when provided."""
        from hrp.research.config import BacktestConfig, BacktestResult

        mock_prices.return_value = pd.DataFrame()
        mock_signals.return_value = pd.DataFrame()
        mock_backtest.return_value = BacktestResult(
            config=BacktestConfig(), metrics={}, equity_curve=pd.Series(), trades=pd.DataFrame()
        )
        mock_log.return_value = "run-789"

        # Create a hypothesis first
        hyp_id = test_api.create_hypothesis(
            title="Test", thesis="Test", prediction="Test", falsification="Test", actor="user"
        )

        config = BacktestConfig(symbols=["AAPL"], start_date=date(2023, 1, 1), end_date=date(2023, 12, 31))
        test_api.run_backtest(config, hypothesis_id=hyp_id)

        # Verify experiment was linked
        experiments = test_api.get_experiments_for_hypothesis(hyp_id)
        assert "run-789" in experiments

    @patch("hrp.api.platform.log_backtest")
    @patch("hrp.api.platform.execute_backtest")
    @patch("hrp.api.platform.get_price_data")
    @patch("hrp.api.platform.generate_momentum_signals")
    def test_run_backtest_logs_lineage(
        self, mock_signals, mock_prices, mock_backtest, mock_log, test_api
    ):
        """run_backtest logs event to lineage table."""
        from hrp.research.config import BacktestConfig, BacktestResult

        mock_prices.return_value = pd.DataFrame()
        mock_signals.return_value = pd.DataFrame()
        mock_backtest.return_value = BacktestResult(
            config=BacktestConfig(),
            metrics={"sharpe_ratio": 1.5, "total_return": 0.2},
            equity_curve=pd.Series(),
            trades=pd.DataFrame()
        )
        mock_log.return_value = "run-lineage-test"

        config = BacktestConfig(symbols=["AAPL"], start_date=date(2023, 1, 1), end_date=date(2023, 12, 31))
        test_api.run_backtest(config)

        lineage = test_api.get_lineage(experiment_id="run-lineage-test")
        assert len(lineage) >= 1
        assert any(e["event_type"] == "experiment_run" for e in lineage)
```

**Step 4: Run all new unit tests**

Run: `source .venv/bin/activate && pytest tests/test_api/test_platform.py::TestPlatformAPICorporateActions tests/test_api/test_platform.py::TestPlatformAPISplitAdjustment tests/test_api/test_platform.py::TestPlatformAPIBacktest -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/test_api/test_platform.py
git commit -m "test(api): add unit tests for corporate actions, splits, and backtest

Add test classes:
- TestPlatformAPICorporateActions: corporate actions data operations
- TestPlatformAPISplitAdjustment: price split adjustment logic
- TestPlatformAPIBacktest: backtest operations with mocked MLflow

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add Database Migration Tests

**Files:**
- Create: `tests/test_data/test_migrations.py`

**Step 1: Create migration tests file**

Create `tests/test_data/test_migrations.py`:

```python
"""Tests for database schema migrations and integrity."""

import os
import tempfile

import pytest

from hrp.data.db import DatabaseManager, get_db
from hrp.data.schema import TABLES, create_tables, verify_schema


@pytest.fixture
def fresh_db():
    """Create a fresh temporary database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()

    yield db_path

    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        if os.path.exists(db_path + ext):
            os.remove(db_path + ext)


class TestDatabaseMigrations:
    """Tests for database schema creation and migrations."""

    def test_create_tables_fresh_db(self, fresh_db):
        """create_tables successfully creates schema on empty database."""
        create_tables(fresh_db)

        db = get_db(fresh_db)
        result = db.fetchone("SELECT 1")
        assert result == (1,)

    def test_create_tables_idempotent(self, fresh_db):
        """Running create_tables twice doesn't error."""
        create_tables(fresh_db)
        create_tables(fresh_db)  # Should not raise

        assert verify_schema(fresh_db)

    def test_all_tables_exist(self, fresh_db):
        """All expected tables are created."""
        create_tables(fresh_db)

        db = get_db(fresh_db)
        for table_name in TABLES.keys():
            result = db.fetchone(f"SELECT COUNT(*) FROM {table_name}")
            assert result is not None, f"Table {table_name} does not exist"

    def test_verify_schema_returns_true(self, fresh_db):
        """verify_schema returns True when all tables exist."""
        create_tables(fresh_db)
        assert verify_schema(fresh_db) is True

    def test_verify_schema_returns_false_missing_table(self, fresh_db):
        """verify_schema returns False when tables are missing."""
        # Don't create tables
        assert verify_schema(fresh_db) is False

    def test_foreign_key_hypothesis_experiments(self, fresh_db):
        """FK constraint enforced on hypothesis_experiments."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # Try to insert experiment link without hypothesis - should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id)
                VALUES ('HYP-FAKE-001', 'exp-123')
                """
            )

    def test_unique_constraint_hypothesis_id(self, fresh_db):
        """Unique constraint on hypothesis_id is enforced."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # Insert a hypothesis
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
            VALUES ('HYP-2023-001', 'Test', 'Thesis', 'Prediction')
            """
        )

        # Try to insert duplicate - should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
                VALUES ('HYP-2023-001', 'Test2', 'Thesis2', 'Prediction2')
                """
            )

    def test_unique_constraint_prices(self, fresh_db):
        """Unique constraint on prices (symbol, date) is enforced."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # First need to add symbol
        db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

        # Insert a price
        db.execute(
            """
            INSERT INTO prices (symbol, date, close)
            VALUES ('AAPL', '2023-01-01', 150.0)
            """
        )

        # Try to insert duplicate - should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO prices (symbol, date, close)
                VALUES ('AAPL', '2023-01-01', 151.0)
                """
            )

    def test_check_constraint_hypothesis_status(self, fresh_db):
        """Check constraint on hypothesis status is enforced."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # Try to insert invalid status - should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
                VALUES ('HYP-2023-001', 'Test', 'Thesis', 'Prediction', 'invalid_status')
                """
            )

    def test_check_constraint_positive_price(self, fresh_db):
        """Check constraint on positive close price is enforced."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

        # Try to insert negative price - should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO prices (symbol, date, close)
                VALUES ('AAPL', '2023-01-01', -10.0)
                """
            )

    def test_table_creation_order(self, fresh_db):
        """Tables are created in correct FK dependency order."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # Verify we can insert in dependency order
        db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
            VALUES ('HYP-2023-001', 'Test', 'Thesis', 'Prediction')
            """
        )
        db.execute(
            """
            INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id)
            VALUES ('HYP-2023-001', 'exp-001')
            """
        )

        # Verify data was inserted
        result = db.fetchone("SELECT COUNT(*) FROM hypothesis_experiments")
        assert result[0] == 1
```

**Step 2: Run migration tests**

Run: `source .venv/bin/activate && pytest tests/test_data/test_migrations.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_data/test_migrations.py
git commit -m "test(data): add database migration and schema integrity tests

Test coverage for:
- Schema creation on fresh database
- Idempotent table creation
- All tables exist after migration
- Foreign key constraints (hypothesis_experiments)
- Unique constraints (hypothesis_id, prices PK)
- Check constraints (status enum, positive prices)
- Table creation order respects FK dependencies

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add Integration Test for Full Backtest Flow

**Files:**
- Modify: `tests/test_api/test_platform.py`

**Step 1: Add full flow integration test**

Add to `TestPlatformAPIIntegration` class (around line 1260):

```python
    @patch("hrp.api.platform.log_backtest")
    @patch("hrp.api.platform.execute_backtest")
    @patch("hrp.api.platform.get_price_data")
    @patch("hrp.api.platform.generate_momentum_signals")
    def test_full_backtest_flow_with_lineage(
        self, mock_signals, mock_prices, mock_backtest, mock_log, test_api
    ):
        """
        Integration test: Complete flow from hypothesis to backtest with lineage.

        Flow:
        1. Create hypothesis
        2. Run backtest linked to hypothesis
        3. Verify experiment linked
        4. Update hypothesis status
        5. Verify complete lineage trail
        """
        from hrp.research.config import BacktestConfig, BacktestResult

        # Setup mocks
        mock_prices.return_value = pd.DataFrame({
            "symbol": ["AAPL"] * 20,
            "date": pd.date_range("2023-01-01", periods=20),
            "close": [100 + i * 0.5 for i in range(20)],
        })
        mock_signals.return_value = pd.DataFrame({
            "AAPL": [1] * 20,
        }, index=pd.date_range("2023-01-01", periods=20))
        mock_backtest.return_value = BacktestResult(
            config=BacktestConfig(symbols=["AAPL"]),
            metrics={
                "sharpe_ratio": 1.8,
                "total_return": 0.35,
                "max_drawdown": -0.12,
            },
            equity_curve=pd.Series([100, 110, 120, 130, 135]),
            trades=pd.DataFrame({"symbol": ["AAPL"], "action": ["buy"]}),
        )
        mock_log.return_value = "exp-integration-001"

        # Step 1: Create hypothesis
        hyp_id = test_api.create_hypothesis(
            title="Integration Test Strategy",
            thesis="Momentum continues in trending markets",
            prediction="Sharpe > 1.5 in backtest",
            falsification="Sharpe < 1.0 or negative returns",
            actor="user",
        )
        assert hyp_id.startswith("HYP-")

        # Step 2: Run backtest linked to hypothesis
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            name="integration_test_backtest",
        )
        exp_id = test_api.run_backtest(
            config,
            hypothesis_id=hyp_id,
            actor="user",
        )
        assert exp_id == "exp-integration-001"

        # Step 3: Verify experiment linked to hypothesis
        experiments = test_api.get_experiments_for_hypothesis(hyp_id)
        assert exp_id in experiments

        # Step 4: Update hypothesis to validated (based on results)
        test_api.update_hypothesis(
            hyp_id,
            status="validated",
            outcome="Backtest Sharpe of 1.8 exceeds threshold",
            actor="user",
        )

        hyp = test_api.get_hypothesis(hyp_id)
        assert hyp["status"] == "validated"
        assert "1.8" in hyp["outcome"]

        # Step 5: Verify complete lineage trail
        lineage = test_api.get_lineage(hypothesis_id=hyp_id)
        event_types = [e["event_type"] for e in lineage]

        # All expected events present
        assert "hypothesis_created" in event_types
        assert "experiment_run" in event_types
        assert "hypothesis_updated" in event_types

        # Verify experiment_run has correct details
        exp_event = next(e for e in lineage if e["event_type"] == "experiment_run")
        assert exp_event["experiment_id"] == exp_id
        assert exp_event["actor"] == "user"
        assert exp_event["details"]["sharpe_ratio"] == 1.8

        # Events are in chronological order (most recent first in get_lineage)
        timestamps = [e["timestamp"] for e in lineage]
        assert timestamps == sorted(timestamps, reverse=True)
```

**Step 2: Run integration test**

Run: `source .venv/bin/activate && pytest tests/test_api/test_platform.py::TestPlatformAPIIntegration::test_full_backtest_flow_with_lineage -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_api/test_platform.py
git commit -m "test(api): add full backtest flow integration test

Test complete workflow:
1. Create hypothesis
2. Run backtest linked to hypothesis (mocked)
3. Verify experiment-hypothesis linking
4. Update hypothesis based on results
5. Verify complete lineage trail with all events

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Verify Coverage and Final Tests

**Step 1: Run all platform API tests**

Run: `source .venv/bin/activate && pytest tests/test_api/test_platform.py -v`
Expected: All tests PASS (should be ~115+ tests)

**Step 2: Check coverage**

Run: `source .venv/bin/activate && pytest tests/test_api/test_platform.py --cov=hrp/api/platform --cov-report=term-missing`
Expected: Coverage >= 70%

**Step 3: Run full test suite**

Run: `source .venv/bin/activate && pytest tests/ -v --ignore=tests/test_integration/`
Expected: No regressions

**Step 4: Final commit with coverage badge update (if needed)**

```bash
git add -A
git commit -m "test(api): complete Platform API test suite - 70%+ coverage

Summary of changes:
- Added input validation to platform.py (20 tests now pass)
- Created synthetic data generators for deterministic fixtures
- Added unit tests for corporate actions, split adjustment, backtest
- Added database migration/schema integrity tests
- Added full backtest flow integration test with mocked MLflow

Coverage: XX% on hrp/api/platform.py

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

- [x] All 22 validation tests pass (Task 1 - completed)
- [x] Synthetic data generators work and have tests (Task 2 - completed)
- [ ] Corporate actions tests pass
- [ ] Split adjustment tests pass
- [ ] Backtest tests pass with mocks
- [ ] Migration tests pass
- [ ] Integration test passes
- [ ] Overall coverage >= 70%
- [ ] No regressions in existing tests

## Progress Log

### 2025-01-22
- **Task 1 completed**: Added input validation helpers and validation to `get_prices`, `get_features`, `get_universe`, `create_hypothesis`, `update_hypothesis`, `list_hypotheses`, `get_hypothesis`. Fixed test fixtures (added `symbols` table inserts). All 22 validation tests pass.
- **Task 2 completed**: Created `tests/fixtures/` package with synthetic data generators for prices, features, corporate actions, and universe. All 12 generator tests pass.
