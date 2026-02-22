"""Tests for the track record tracker."""

from datetime import date, datetime

import pytest

from hrp.advisory.track_record import TrackRecordTracker, TrackRecordSummary


@pytest.fixture
def tracker_db(test_db):
    """Test database with recommendation history."""
    from hrp.data.db import get_db

    db = get_db(test_db)

    # Insert SPY prices for benchmark
    db.execute(
        "INSERT INTO symbols (symbol, name) VALUES ('SPY', 'S&P 500 ETF') "
        "ON CONFLICT DO NOTHING"
    )
    for i in range(100):
        d = date(2024, 1, 1) + __import__("datetime").timedelta(days=i)
        price = 500 + i * 0.3
        db.execute(
            "INSERT INTO prices (symbol, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            ["SPY", d, price, price * 1.01, price * 0.99, price, 5000000],
        )

    # Insert closed recommendations
    closed_recs = [
        ("REC-2024-001", "AAPL", "BUY", "HIGH", 0.8, 150.0, 160.0, 0.067, "closed_profit", "2024-01-05", "2024-01-20"),
        ("REC-2024-002", "MSFT", "BUY", "MEDIUM", 0.5, 350.0, 340.0, -0.029, "closed_loss", "2024-01-05", "2024-01-25"),
        ("REC-2024-003", "GOOGL", "BUY", "HIGH", 0.7, 140.0, 155.0, 0.107, "closed_profit", "2024-01-10", "2024-02-01"),
        ("REC-2024-004", "AMZN", "BUY", "LOW", 0.3, 170.0, 162.0, -0.047, "closed_stopped", "2024-01-15", "2024-02-05"),
        ("REC-2024-005", "NVDA", "BUY", "HIGH", 0.9, 600.0, 650.0, 0.083, "closed_profit", "2024-01-20", "2024-02-10"),
    ]

    for rec in closed_recs:
        db.execute(
            "INSERT INTO recommendations "
            "(recommendation_id, symbol, action, confidence, signal_strength, "
            "entry_price, close_price, realized_return, status, "
            "created_at, closed_at, model_name, batch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'test', 'BATCH-1')",
            list(rec),
        )

    yield test_db


@pytest.fixture
def tracker(tracker_db):
    from hrp.api.platform import PlatformAPI
    return TrackRecordTracker(PlatformAPI())


class TestTrackRecordTracker:
    def test_compute_track_record(self, tracker):
        summary = tracker.compute_track_record(
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        assert isinstance(summary, TrackRecordSummary)
        assert summary.closed_recommendations == 5
        assert summary.profitable == 3
        assert summary.unprofitable == 2
        assert 0 < summary.win_rate < 1  # 3/5 = 0.6

    def test_win_rate_calculation(self, tracker):
        summary = tracker.compute_track_record(
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        assert summary.win_rate == pytest.approx(0.6, abs=0.01)

    def test_avg_return(self, tracker):
        summary = tracker.compute_track_record(
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        # Average of [0.067, -0.029, 0.107, -0.047, 0.083] = 0.0362
        assert summary.avg_return > 0

    def test_best_and_worst_pick(self, tracker):
        summary = tracker.compute_track_record(
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        assert summary.best_pick == "GOOGL"  # 10.7%
        assert summary.worst_pick == "AMZN"  # -4.7%

    def test_empty_track_record(self, tracker):
        summary = tracker.compute_track_record(
            start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
        )
        assert summary.closed_recommendations == 0
        assert summary.win_rate == 0.0

    def test_generate_weekly_report(self, tracker):
        report = tracker.generate_weekly_report(date(2024, 2, 15))
        assert report.report_date == date(2024, 2, 15)
        assert isinstance(report.track_record, TrackRecordSummary)

    def test_persist_track_record(self, tracker):
        summary = tracker.compute_track_record(
            start_date=date(2024, 1, 1), end_date=date(2024, 3, 31)
        )
        tracker.persist_track_record(summary)

        # Verify it was written
        from hrp.api.platform import PlatformAPI
        api = PlatformAPI()
        result = api.fetchone_readonly(
            "SELECT total_recommendations FROM track_record "
            "WHERE period_start = ? AND period_end = ?",
            [date(2024, 1, 1), date(2024, 3, 31)],
        )
        assert result is not None
