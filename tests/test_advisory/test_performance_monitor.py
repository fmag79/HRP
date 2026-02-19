"""Tests for model performance monitor."""

from datetime import date, timedelta

import pytest

from hrp.advisory.performance_monitor import AccuracyReport, ModelPerformanceMonitor


@pytest.fixture
def monitor_db(test_db):
    """Test database with closed recommendations for accuracy checking."""
    from hrp.data.db import get_db

    db = get_db(test_db)

    # Insert model recommendations with known outcomes
    today = date.today()
    recs = [
        # 6 correct direction predictions (signal > 0, return > 0)
        ("REC-MON-001", "AAPL", 0.8, 150.0, 160.0, 0.067, today - timedelta(days=10)),
        ("REC-MON-002", "MSFT", 0.6, 300.0, 315.0, 0.050, today - timedelta(days=15)),
        ("REC-MON-003", "GOOGL", 0.7, 140.0, 148.0, 0.057, today - timedelta(days=20)),
        ("REC-MON-004", "AAPL", 0.5, 155.0, 162.0, 0.045, today - timedelta(days=25)),
        ("REC-MON-005", "MSFT", 0.9, 310.0, 330.0, 0.065, today - timedelta(days=30)),
        ("REC-MON-006", "GOOGL", 0.4, 145.0, 150.0, 0.034, today - timedelta(days=35)),
        # 4 wrong direction predictions (signal > 0, return < 0)
        ("REC-MON-007", "AAPL", 0.6, 160.0, 155.0, -0.031, today - timedelta(days=12)),
        ("REC-MON-008", "MSFT", 0.5, 315.0, 308.0, -0.022, today - timedelta(days=18)),
        ("REC-MON-009", "GOOGL", 0.7, 148.0, 140.0, -0.054, today - timedelta(days=22)),
        ("REC-MON-010", "AAPL", 0.3, 162.0, 158.0, -0.025, today - timedelta(days=28)),
    ]

    for rec_id, sym, signal, entry, close, ret, closed_at in recs:
        db.execute(
            "INSERT INTO recommendations "
            "(recommendation_id, symbol, action, confidence, signal_strength, "
            "entry_price, close_price, realized_return, position_pct, "
            "status, created_at, closed_at, model_name, batch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rec_id, sym, "BUY", "HIGH", signal,
                entry, close, ret, 0.05,
                "closed_profit" if ret > 0 else "closed_loss",
                closed_at - timedelta(days=5), closed_at,
                "test_model", "BATCH-MON",
            ],
        )

    # Insert recs for a second model (bad model — all wrong)
    for i, (sym, signal, ret) in enumerate([
        ("AAPL", 0.8, -0.05),
        ("MSFT", 0.7, -0.04),
        ("GOOGL", 0.6, -0.03),
        ("AAPL", 0.5, -0.02),
        ("MSFT", 0.9, -0.06),
    ]):
        rec_id = f"REC-BAD-{i+1:03d}"
        closed_at = today - timedelta(days=10 + i * 5)
        db.execute(
            "INSERT INTO recommendations "
            "(recommendation_id, symbol, action, confidence, signal_strength, "
            "entry_price, close_price, realized_return, position_pct, "
            "status, created_at, closed_at, model_name, batch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rec_id, sym, "BUY", "HIGH", signal,
                150.0, 145.0, ret, 0.05,
                "closed_loss", closed_at - timedelta(days=5), closed_at,
                "bad_model", "BATCH-BAD",
            ],
        )

    yield test_db


@pytest.fixture
def monitor(monitor_db):
    from hrp.api.platform import PlatformAPI
    return ModelPerformanceMonitor(PlatformAPI())


class TestModelPerformanceMonitor:
    def test_check_accuracy_good_model(self, monitor):
        report = monitor.check_prediction_accuracy("test_model")
        assert isinstance(report, AccuracyReport)
        assert report.total_predictions == 10
        assert report.directional_accuracy == pytest.approx(0.6, abs=0.01)
        assert report.degraded is False  # 60% > 50% threshold

    def test_check_accuracy_bad_model(self, monitor):
        report = monitor.check_prediction_accuracy("bad_model")
        assert report.total_predictions == 5
        assert report.directional_accuracy == pytest.approx(0.0, abs=0.01)
        assert report.degraded is True

    def test_check_accuracy_nonexistent_model(self, monitor):
        report = monitor.check_prediction_accuracy("no_such_model")
        assert report.total_predictions == 0
        assert report.degraded is False
        assert "Insufficient" in report.message

    def test_check_all_models(self, monitor):
        reports = monitor.check_all_models()
        assert len(reports) == 2
        model_names = {r.model_name for r in reports}
        assert model_names == {"test_model", "bad_model"}

    def test_run_monitoring_cycle(self, monitor):
        reports = monitor.run_monitoring_cycle()
        assert len(reports) == 2
        degraded = [r for r in reports if r.degraded]
        assert len(degraded) == 1
        assert degraded[0].model_name == "bad_model"

    def test_accuracy_report_message(self, monitor):
        report = monitor.check_prediction_accuracy("bad_model")
        assert "DEGRADED" in report.message
        assert "retraining recommended" in report.message
