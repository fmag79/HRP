"""Tests for the kill gate calibrator."""

import json
from datetime import date, datetime, timedelta

import pytest

from hrp.advisory.kill_gate_calibrator import (
    CalibrationReport,
    KillGateCalibrator,
    ThresholdRecommendation,
)


@pytest.fixture
def calibration_db(test_db):
    """Test database with kill gate history for calibration analysis."""
    from hrp.data.db import get_db

    db = get_db(test_db)
    today = date.today()

    # Create hypotheses in various states
    hypotheses = [
        # Killed and stayed rejected (true positive)
        ("HYP-TP-001", "TP hypothesis 1", "rejected"),
        ("HYP-TP-002", "TP hypothesis 2", "rejected"),
        # Killed but later reopened (false positive)
        ("HYP-FP-001", "FP hypothesis 1", "testing"),
        # Passed kill gates, later rejected (false negative)
        ("HYP-FN-001", "FN hypothesis 1", "rejected"),
        ("HYP-FN-002", "FN hypothesis 2", "rejected"),
        # Passed kill gates, succeeded (true negative)
        ("HYP-TN-001", "TN hypothesis 1", "validated"),
        ("HYP-TN-002", "TN hypothesis 2", "deployed"),
    ]

    for hyp_id, title, status in hypotheses:
        db.execute(
            "INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status) "
            "VALUES (?, ?, ?, ?, ?)",
            [hyp_id, title, "Test thesis", "Test prediction", status],
        )

    # Add kill gate triggered events (killed hypotheses)
    lineage_id = 1
    killed_events = [
        ("HYP-TP-001", "baseline_sharpe_too_low", today - timedelta(days=30)),
        ("HYP-TP-002", "max_drawdown_exceeded", today - timedelta(days=45)),
        ("HYP-FP-001", "train_sharpe_too_high", today - timedelta(days=60)),
    ]
    for hyp_id, reason, ts in killed_events:
        db.execute(
            "INSERT INTO lineage (lineage_id, event_type, actor, hypothesis_id, details, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                lineage_id,
                "kill_gate_triggered",
                "agent:kill-gate-enforcer",
                hyp_id,
                json.dumps({"reason": reason}),
                ts,
            ],
        )
        lineage_id += 1

    # Add kill gate complete events (passed hypotheses)
    passed_events = [
        ("HYP-FN-001", today - timedelta(days=20)),
        ("HYP-FN-002", today - timedelta(days=35)),
        ("HYP-TN-001", today - timedelta(days=50)),
        ("HYP-TN-002", today - timedelta(days=55)),
    ]
    for hyp_id, ts in passed_events:
        db.execute(
            "INSERT INTO lineage (lineage_id, event_type, actor, hypothesis_id, details, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                lineage_id,
                "kill_gate_enforcer_complete",
                "agent:kill-gate-enforcer",
                hyp_id,
                json.dumps({"result": "passed"}),
                ts,
            ],
        )
        lineage_id += 1

    # Add a hypothesis_updated event for the false positive (reopened after kill)
    db.execute(
        "INSERT INTO lineage (lineage_id, event_type, actor, hypothesis_id, details, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            lineage_id,
            "hypothesis_updated",
            "user",
            "HYP-FP-001",
            json.dumps({"new_status": "testing"}),
            today - timedelta(days=50),
        ],
    )

    yield test_db


@pytest.fixture
def calibrator(calibration_db):
    from hrp.api.platform import PlatformAPI
    return KillGateCalibrator(PlatformAPI())


class TestKillGateCalibrator:
    def test_calibrate_basic(self, calibrator):
        report = calibrator.calibrate(lookback_days=180)
        assert isinstance(report, CalibrationReport)
        assert report.total_hypotheses == 7  # 3 killed + 4 passed
        assert report.killed == 3
        assert report.passed == 4

    def test_false_positives_detected(self, calibrator):
        report = calibrator.calibrate(lookback_days=180)
        # HYP-FP-001 was killed but later reopened to 'testing'
        assert report.false_positives >= 1
        assert "HYP-FP-001" in report.false_positive_ids

    def test_false_negatives_detected(self, calibrator):
        report = calibrator.calibrate(lookback_days=180)
        # HYP-FN-001 and HYP-FN-002 passed gates but are now rejected
        assert report.false_negatives >= 2
        assert "HYP-FN-001" in report.false_negative_ids
        assert "HYP-FN-002" in report.false_negative_ids

    def test_gate_stats(self, calibrator):
        report = calibrator.calibrate(lookback_days=180)
        assert len(report.gate_stats) > 0
        assert "baseline_sharpe_too_low" in report.gate_stats
        assert report.gate_stats["baseline_sharpe_too_low"]["trigger_count"] == 1

    def test_recommendations_generated(self, calibrator):
        report = calibrator.calibrate(lookback_days=180)
        assert len(report.recommendations) > 0

    def test_empty_calibration(self, test_db):
        """Test with no kill gate history."""
        from hrp.api.platform import PlatformAPI
        calibrator = KillGateCalibrator(PlatformAPI())
        report = calibrator.calibrate(lookback_days=180)
        assert report.total_hypotheses == 0
        assert "Insufficient data" in report.recommendations[0]

    def test_suggest_thresholds(self, calibrator):
        report = calibrator.calibrate(lookback_days=180)
        suggestions = calibrator.suggest_thresholds(report)
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert isinstance(s, ThresholdRecommendation)
            assert s.direction in ("tighten", "loosen")
            assert s.confidence in ("high", "medium", "low")

    def test_persist_calibration(self, calibrator):
        report = calibrator.calibrate(lookback_days=180)
        calibrator.persist_calibration(report)

        # Verify lineage event was logged
        df = calibrator.api.query_readonly(
            "SELECT * FROM lineage WHERE actor = 'agent:kill-gate-calibrator'"
        )
        assert not df.empty
