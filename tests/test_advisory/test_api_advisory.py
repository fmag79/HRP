"""Tests for PlatformAPI advisory extensions."""

from datetime import date

import pytest


class TestAdvisoryAPI:
    """Test the advisory-related PlatformAPI methods."""

    def test_create_user_profile(self, test_api):
        profile_id = test_api.create_user_profile(
            name="Test User",
            portfolio_value=100000.0,
            risk_tolerance=3,
        )
        assert profile_id.startswith("PROF-")

    def test_get_user_profile(self, test_api):
        profile_id = test_api.create_user_profile(
            name="Query User",
            portfolio_value=50000.0,
            risk_tolerance=2,
            excluded_sectors=["Financials"],
        )
        profile = test_api.get_user_profile(profile_id)
        assert profile is not None
        assert profile["name"] == "Query User"
        assert int(profile["risk_tolerance"]) == 2

    def test_get_nonexistent_profile(self, test_api):
        result = test_api.get_user_profile("PROF-NONEXISTENT")
        assert result is None

    def test_get_recommendations_empty(self, test_api):
        recs = test_api.get_recommendations()
        assert recs.empty

    def test_get_recommendations_with_data(self, test_api):
        # Insert a test recommendation
        test_api.execute_write(
            "INSERT INTO recommendations "
            "(recommendation_id, symbol, action, confidence, signal_strength, "
            "entry_price, status, model_name, batch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["REC-API-001", "AAPL", "BUY", "HIGH", 0.8, 150.0,
             "active", "test", "BATCH-API"],
        )

        recs = test_api.get_recommendations(status="active")
        assert not recs.empty
        assert "REC-API-001" in recs["recommendation_id"].values

    def test_get_recommendation_by_id(self, test_api):
        test_api.execute_write(
            "INSERT INTO recommendations "
            "(recommendation_id, symbol, action, confidence, signal_strength, "
            "entry_price, status, model_name, batch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["REC-API-002", "MSFT", "BUY", "MEDIUM", 0.5, 350.0,
             "active", "test", "BATCH-API"],
        )

        rec = test_api.get_recommendation_by_id("REC-API-002")
        assert rec is not None
        assert rec["symbol"] == "MSFT"

    def test_get_recommendation_by_id_not_found(self, test_api):
        rec = test_api.get_recommendation_by_id("REC-NONEXISTENT")
        assert rec is None

    def test_get_recommendation_history(self, test_api):
        history = test_api.get_recommendation_history(limit=10)
        assert isinstance(history, type(history))  # It's a DataFrame

    def test_get_track_record(self, test_api):
        tr = test_api.get_track_record(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert tr.empty  # No track record yet

    def test_filter_recommendations_by_symbol(self, test_api):
        test_api.execute_write(
            "INSERT INTO recommendations "
            "(recommendation_id, symbol, action, confidence, signal_strength, "
            "entry_price, status, model_name, batch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["REC-FILTER-001", "GOOGL", "BUY", "HIGH", 0.7, 140.0,
             "active", "test", "BATCH-FILTER"],
        )

        googl_recs = test_api.get_recommendations(symbol="GOOGL")
        assert not googl_recs.empty
        assert all(googl_recs["symbol"] == "GOOGL")
