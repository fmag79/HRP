"""Tests for post-trade attribution."""

from datetime import date, timedelta

import pytest

from hrp.advisory.post_trade_attribution import (
    AggregateAttribution,
    Attribution,
    PostTradeAttributor,
)


@pytest.fixture
def attribution_db(test_db):
    """Test database with closed recommendations and prices."""
    from hrp.data.db import get_db

    db = get_db(test_db)

    # Insert SPY prices
    db.execute(
        "INSERT INTO symbols (symbol, name) VALUES ('SPY', 'S&P 500 ETF') ON CONFLICT DO NOTHING"
    )
    for symbol in ["AAPL", "SPY"]:
        base = {"AAPL": 150, "SPY": 500}[symbol]
        for i in range(60):
            d = date(2024, 1, 1) + timedelta(days=i)
            price = base + i * 0.3
            db.execute(
                "INSERT INTO prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                [symbol, d, price, price * 1.01, price * 0.98, price, 1000000],
            )

    # Insert a closed recommendation
    db.execute(
        "INSERT INTO recommendations "
        "(recommendation_id, symbol, action, confidence, signal_strength, "
        "entry_price, close_price, realized_return, position_pct, "
        "status, created_at, closed_at, model_name, batch_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "REC-ATTR-001", "AAPL", "BUY", "HIGH", 0.8,
            150.0, 160.0, 0.067, 0.08,
            "closed_profit", "2024-01-05", "2024-01-20",
            "test_model", "BATCH-ATTR",
        ],
    )
    # Insert a losing recommendation
    db.execute(
        "INSERT INTO recommendations "
        "(recommendation_id, symbol, action, confidence, signal_strength, "
        "entry_price, close_price, realized_return, position_pct, "
        "status, created_at, closed_at, model_name, batch_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "REC-ATTR-002", "AAPL", "BUY", "MEDIUM", 0.5,
            160.0, 155.0, -0.031, 0.05,
            "closed_loss", "2024-01-21", "2024-02-10",
            "test_model", "BATCH-ATTR",
        ],
    )

    yield test_db


@pytest.fixture
def attributor(attribution_db):
    from hrp.api.platform import PlatformAPI
    return PostTradeAttributor(PlatformAPI())


class TestPostTradeAttributor:
    def test_attribute_single(self, attributor):
        attr = attributor.attribute("REC-ATTR-001")
        assert attr is not None
        assert isinstance(attr, Attribution)
        assert attr.symbol == "AAPL"
        assert attr.total_return == pytest.approx(0.067, abs=0.001)
        assert attr.signal_correct is True

    def test_attribute_losing_trade(self, attributor):
        attr = attributor.attribute("REC-ATTR-002")
        assert attr is not None
        assert attr.total_return < 0
        # BUY at 160, close at 155 — direction wrong
        assert attr.signal_correct is False

    def test_attribute_nonexistent(self, attributor):
        attr = attributor.attribute("REC-NONEXISTENT")
        assert attr is None

    def test_aggregate_attribution(self, attributor):
        agg = attributor.aggregate_attribution(
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        assert agg is not None
        assert isinstance(agg, AggregateAttribution)
        assert agg.total_recommendations == 2
        assert agg.signal_accuracy == 0.5  # 1 right, 1 wrong

    def test_aggregate_empty_period(self, attributor):
        agg = attributor.aggregate_attribution(
            start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
        )
        assert agg is None
