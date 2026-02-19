"""Tests for the recommendation engine."""

import os
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from hrp.advisory.recommendation_engine import (
    Recommendation,
    RecommendationEngine,
    RecommendationUpdate,
    _next_batch_id,
)


@pytest.fixture
def advisory_db(test_db):
    """Test database with advisory tables and sample data."""
    from hrp.data.db import get_db

    db = get_db(test_db)

    # Insert a deployed model
    db.execute(
        """
        INSERT INTO model_deployments
        (deployment_id, model_name, model_version, environment, status, deployed_by)
        VALUES (1, 'test_model', 'v1', 'production', 'active', 'user')
        """
    )

    # Insert some prices for SPY and test symbols
    for symbol in ["AAPL", "MSFT", "GOOGL", "SPY"]:
        db.execute(
            "INSERT INTO symbols (symbol, name) VALUES (?, ?) ON CONFLICT DO NOTHING",
            [symbol, f"{symbol} Inc."],
        )
        # Add universe entry so get_prices validation passes
        db.execute(
            "INSERT INTO universe (symbol, date, in_universe) VALUES (?, ?, TRUE) "
            "ON CONFLICT DO NOTHING",
            [symbol, date(2024, 1, 1)],
        )
        base = {"AAPL": 150, "MSFT": 350, "GOOGL": 140, "SPY": 500}[symbol]
        for i in range(30):
            d = date(2024, 1, 1) + timedelta(days=i)
            price = base + i * 0.5
            db.execute(
                "INSERT INTO prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                [symbol, d, price * 0.99, price * 1.01, price * 0.98, price, 1000000],
            )

    yield test_db


@pytest.fixture
def engine(advisory_db):
    """RecommendationEngine instance with test database."""
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI()
    return RecommendationEngine(api=api)


class TestBatchIdGeneration:
    def test_batch_id_format(self):
        batch_id = _next_batch_id()
        assert batch_id.startswith("BATCH-")
        assert len(batch_id) > 10


class TestRecommendation:
    def test_to_dict(self):
        rec = Recommendation(
            recommendation_id="REC-2026-001",
            symbol="AAPL",
            action="BUY",
            confidence="HIGH",
            signal_strength=0.75,
            entry_price=150.0,
            target_price=165.0,
            stop_price=140.0,
            position_pct=0.08,
            thesis_plain="Test thesis",
            risk_plain="Test risk",
            time_horizon_days=20,
            hypothesis_id="HYP-001",
            model_name="test_model",
            batch_id="BATCH-001",
        )
        d = rec.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["action"] == "BUY"
        assert d["confidence"] == "HIGH"


class TestRecommendationEngine:
    def test_signal_to_confidence_high(self, engine):
        assert engine._signal_to_confidence(0.8) == "HIGH"

    def test_signal_to_confidence_medium(self, engine):
        assert engine._signal_to_confidence(0.5) == "MEDIUM"

    def test_signal_to_confidence_low(self, engine):
        assert engine._signal_to_confidence(0.2) == "LOW"

    def test_stop_loss_pct_by_risk(self, engine):
        assert engine._stop_loss_pct(1) == 0.03
        assert engine._stop_loss_pct(3) == 0.07
        assert engine._stop_loss_pct(5) == 0.15

    def test_position_size_bounded(self, engine):
        size = engine._position_size(0.9, 5)
        assert size <= 0.10  # Max position

    def test_confidence_rank(self, engine):
        assert engine._confidence_rank("HIGH") > engine._confidence_rank("MEDIUM")
        assert engine._confidence_rank("MEDIUM") > engine._confidence_rank("LOW")

    def test_get_deployed_models(self, engine):
        models = engine._get_deployed_models()
        assert len(models) >= 1
        assert models[0]["model_name"] == "test_model"

    def test_get_latest_prices(self, engine):
        prices = engine._get_latest_prices(["AAPL"], date(2024, 1, 15))
        assert "AAPL" in prices
        assert prices["AAPL"] > 0


class TestRecommendationPersistence:
    def test_persist_and_retrieve(self, advisory_db):
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI()
        engine = RecommendationEngine(api=api)

        rec = Recommendation(
            recommendation_id="REC-TEST-001",
            symbol="AAPL",
            action="BUY",
            confidence="HIGH",
            signal_strength=0.75,
            entry_price=150.0,
            target_price=165.0,
            stop_price=140.0,
            position_pct=0.08,
            thesis_plain="Test thesis",
            risk_plain="Test risk",
            time_horizon_days=20,
            hypothesis_id=None,
            model_name="test_model",
            batch_id="BATCH-TEST-001",
        )
        engine._persist_recommendation(rec)

        # Retrieve
        result = api.get_recommendations(status="active")
        assert not result.empty
        assert "REC-TEST-001" in result["recommendation_id"].values

    def test_close_recommendation(self, advisory_db):
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI()
        engine = RecommendationEngine(api=api)

        # Insert a recommendation
        rec = Recommendation(
            recommendation_id="REC-CLOSE-001",
            symbol="MSFT",
            action="BUY",
            confidence="MEDIUM",
            signal_strength=0.5,
            entry_price=350.0,
            target_price=385.0,
            stop_price=330.0,
            position_pct=0.05,
            thesis_plain="Close test",
            risk_plain="Close risk",
            time_horizon_days=20,
            hypothesis_id=None,
            model_name="test_model",
            batch_id="BATCH-CLOSE",
        )
        engine._persist_recommendation(rec)

        # Close it
        engine.close_recommendation("REC-CLOSE-001", 360.0, "target_reached")

        # Verify
        result = api.fetchone_readonly(
            "SELECT status, realized_return FROM recommendations "
            "WHERE recommendation_id = 'REC-CLOSE-001'"
        )
        assert result is not None
        assert result[0] == "closed_profit"
        assert float(result[1]) > 0  # Profit


class TestReviewOpenRecommendations:
    def test_close_stopped_recommendation(self, advisory_db):
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI()
        engine = RecommendationEngine(api=api)

        # Insert recommendation where current price is below stop
        api.execute_write(
            "INSERT INTO recommendations "
            "(recommendation_id, symbol, action, confidence, signal_strength, "
            "entry_price, target_price, stop_price, position_pct, "
            "time_horizon_days, status, model_name, batch_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "REC-STOP-001", "AAPL", "BUY", "HIGH", 0.8,
                200.0, 220.0, 190.0, 0.08,  # Stop at 190, but AAPL price is ~150
                20, "active", "test", "BATCH-STOP",
                datetime(2024, 1, 1),
            ],
        )

        updates = engine.review_open_recommendations(date(2024, 1, 15))
        assert len(updates) >= 1

        stopped = [u for u in updates if u.recommendation_id == "REC-STOP-001"]
        assert len(stopped) == 1
        assert stopped[0].status == "closed_stopped"
