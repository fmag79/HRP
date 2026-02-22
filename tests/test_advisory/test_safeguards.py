"""Tests for the safeguards module."""

from datetime import date, timedelta

import pytest

from hrp.advisory.safeguards import CheckResult, CircuitBreaker, PreTradeChecks


@pytest.fixture
def safeguard_db(test_db):
    """Test database with prices for safeguard checks."""
    from hrp.data.db import get_db

    db = get_db(test_db)

    # Insert SPY prices
    db.execute(
        "INSERT INTO symbols (symbol, name) VALUES ('SPY', 'S&P 500 ETF') "
        "ON CONFLICT DO NOTHING"
    )
    for i in range(30):
        d = date(2024, 1, 1) + timedelta(days=i)
        price = 500 - i * 0.5  # Slight decline
        db.execute(
            "INSERT INTO prices (symbol, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            ["SPY", d, price, price * 1.01, price * 0.99, price, 5000000],
        )

    yield test_db


@pytest.fixture
def checks(safeguard_db):
    from hrp.api.platform import PlatformAPI
    return PreTradeChecks(PlatformAPI())


@pytest.fixture
def breaker(safeguard_db):
    from hrp.api.platform import PlatformAPI
    return CircuitBreaker(PlatformAPI())


class TestPreTradeChecks:
    def test_data_freshness_pass(self, checks):
        # Data goes up to 2024-01-30, checking as of 2024-01-30
        result = checks.check_data_freshness(date(2024, 1, 30))
        assert result.passed
        assert result.check_name == "data_freshness"

    def test_data_freshness_fail_stale(self, checks):
        # Data only up to 2024-01-30, checking as of 2024-02-15
        result = checks.check_data_freshness(date(2024, 2, 15))
        assert not result.passed
        assert "days old" in result.message

    def test_market_regime_normal(self, checks):
        # Slow decline, not a crash
        result = checks.check_market_regime(date(2024, 1, 15))
        assert result.passed

    def test_run_all_checks(self, checks):
        results = checks.run_all_checks(date(2024, 1, 15))
        assert len(results) >= 2
        assert all(isinstance(r, CheckResult) for r in results)

    def test_portfolio_concentration_pass(self, checks):
        weights = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25}
        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology",
            "GOOGL": "Communication", "AMZN": "Consumer Discretionary",
        }
        result = checks.check_portfolio_concentration(weights, sector_map)
        assert not result.passed  # Tech = 50% > 30%

    def test_portfolio_concentration_diversified(self, checks):
        weights = {"AAPL": 0.20, "GOOGL": 0.20, "AMZN": 0.20, "JNJ": 0.20, "JPM": 0.20}
        sector_map = {
            "AAPL": "Technology", "GOOGL": "Communication",
            "AMZN": "Consumer Discretionary", "JNJ": "Healthcare", "JPM": "Financials",
        }
        result = checks.check_portfolio_concentration(weights, sector_map)
        assert result.passed  # Each sector = 20% < 30%


class TestCircuitBreaker:
    def test_no_halt_on_empty_portfolio(self, breaker):
        should_halt, reason = breaker.should_halt(date(2024, 1, 15))
        assert not should_halt

    def test_no_reduce_on_empty_portfolio(self, breaker):
        should_reduce, multiplier = breaker.should_reduce_size(date(2024, 1, 15))
        assert not should_reduce
        assert multiplier == 1.0
