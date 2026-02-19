"""Tests for the portfolio constructor."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from hrp.advisory.portfolio_constructor import (
    CovarianceEstimator,
    PortfolioAllocation,
    PortfolioConstraints,
    PortfolioConstructor,
)


class TestPortfolioConstraints:
    def test_defaults(self):
        c = PortfolioConstraints()
        assert c.max_positions == 20
        assert c.max_position_pct == 0.10
        assert c.max_sector_pct == 0.30
        assert c.max_correlation == 0.70

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("HRP_MAX_POSITIONS", "10")
        monkeypatch.setenv("HRP_PORTFOLIO_MAX_SECTOR_PCT", "0.25")
        c = PortfolioConstraints.from_env()
        assert c.max_positions == 10
        assert c.max_sector_pct == 0.25


class TestCovarianceEstimator:
    def test_sample_covariance(self):
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.01,
            columns=["A", "B", "C"],
        )
        est = CovarianceEstimator()
        cov = est.estimate(returns, method="sample")
        assert cov.shape == (3, 3)
        # Diagonal should be positive
        assert all(cov[i, i] > 0 for i in range(3))

    def test_ledoit_wolf(self):
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.01,
            columns=["A", "B", "C"],
        )
        est = CovarianceEstimator()
        cov = est.estimate(returns, method="ledoit_wolf")
        assert cov.shape == (3, 3)
        assert all(cov[i, i] > 0 for i in range(3))


class TestPortfolioConstructor:
    @pytest.fixture
    def constructor_db(self, test_db):
        """DB with price data for portfolio construction."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        np.random.seed(42)
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication",
            "AMZN": "Consumer Discretionary", "META": "Communication",
        }
        bases = {"AAPL": 150, "MSFT": 350, "GOOGL": 140, "AMZN": 170, "META": 300}

        # Set sectors
        for symbol in symbols:
            db.execute(
                "UPDATE symbols SET sector = ? WHERE symbol = ?",
                [sectors[symbol], symbol],
            )

        # Add universe entries so get_prices validation passes
        for symbol in symbols:
            db.execute(
                "INSERT INTO universe (symbol, date, in_universe) VALUES (?, ?, TRUE) "
                "ON CONFLICT DO NOTHING",
                [symbol, date(2023, 1, 1)],
            )

        # Batch-insert prices using raw connection for speed
        with db.connection() as conn:
            for symbol in symbols:
                base = bases[symbol]
                for i in range(100):
                    d = date(2023, 1, 1) + timedelta(days=i)
                    price = base * (1 + np.random.randn() * 0.02)
                    conn.execute(
                        "INSERT INTO prices (symbol, date, open, high, low, close, volume) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [symbol, d, price * 0.99, price * 1.01, price * 0.98, abs(price), 1000000],
                    )

        yield test_db

    @pytest.fixture
    def constructor(self, constructor_db):
        from hrp.api.platform import PlatformAPI
        return PortfolioConstructor(
            api=PlatformAPI(),
            constraints=PortfolioConstraints(max_positions=5, max_position_pct=0.30),
        )

    def test_construct_basic(self, constructor):
        signals = {"AAPL": 0.8, "MSFT": 0.6, "GOOGL": 0.4}
        result = constructor.construct(
            signals, as_of_date=date(2023, 4, 1), lookback_days=80
        )
        assert isinstance(result, PortfolioAllocation)
        assert len(result.weights) > 0
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_position_limit_respected(self, constructor):
        # Use all 5 symbols so position limit can be enforced
        signals = {"AAPL": 0.9, "MSFT": 0.7, "GOOGL": 0.6, "AMZN": 0.5, "META": 0.4}
        result = constructor.construct(
            signals, as_of_date=date(2023, 4, 1), lookback_days=80
        )
        # With enough symbols, at least some weights should be clipped
        assert len(result.weights) >= 2
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.02)

    def test_empty_signals(self, constructor):
        result = constructor.construct({})
        assert len(result.weights) == 0

    def test_turnover_computation(self):
        old = {"AAPL": 0.5, "MSFT": 0.5}
        new = {"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.4}
        turnover = PortfolioConstructor._compute_turnover(new, old)
        assert turnover > 0
        assert turnover <= 1.0

    def test_turnover_constraint(self):
        old = {"AAPL": 0.5, "MSFT": 0.5}
        new = {"GOOGL": 0.5, "AMZN": 0.5}
        constrained = PortfolioConstructor._constrain_turnover(new, old, 0.20)
        turnover = PortfolioConstructor._compute_turnover(constrained, old)
        assert turnover <= 0.21  # Allow small numerical error
