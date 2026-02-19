"""Tests for the weekly digest email generation."""

from datetime import date

import pytest

from hrp.advisory.digest import DigestContent, WeeklyDigest
from hrp.advisory.track_record import TrackRecordSummary, WeeklyReport


@pytest.fixture
def sample_report():
    """Sample weekly report for testing."""
    return WeeklyReport(
        report_date=date(2024, 2, 15),
        new_recommendations=[
            {
                "recommendation_id": "REC-001",
                "symbol": "AAPL",
                "action": "BUY",
                "confidence": "HIGH",
                "signal_strength": 0.8,
                "entry_price": 150.0,
                "target_price": 165.0,
                "stop_price": 140.0,
                "thesis_plain": "Strong momentum with institutional buying.",
                "risk_plain": "Market pullback could cause 5-10% loss.",
            },
        ],
        open_positions=[
            {"symbol": "MSFT", "entry_price": 350.0, "signal_strength": 0.6},
        ],
        closed_this_week=[
            {"symbol": "GOOGL", "realized_return": 0.052, "status": "closed_profit"},
        ],
        track_record=TrackRecordSummary(
            period_start=date(2024, 1, 1),
            period_end=date(2024, 2, 15),
            total_recommendations=10,
            closed_recommendations=8,
            profitable=5,
            unprofitable=3,
            win_rate=0.625,
            avg_return=0.032,
            avg_win=0.06,
            avg_loss=-0.025,
            best_pick="NVDA",
            best_return=0.12,
            worst_pick="AMZN",
            worst_return=-0.05,
            total_return=0.256,
            benchmark_return=0.02,
            excess_return=0.012,
            sharpe_ratio=1.2,
        ),
    )


class TestWeeklyDigest:
    def test_generate_content(self, sample_report):
        digest = WeeklyDigest()
        content = digest.generate(sample_report)
        assert isinstance(content, DigestContent)
        assert "Feb 15, 2024" in content.subject
        assert "Weekly Market Brief" in content.subject

    def test_html_contains_recommendations(self, sample_report):
        digest = WeeklyDigest()
        content = digest.generate(sample_report)
        assert "AAPL" in content.html_body
        assert "Strong momentum" in content.html_body
        assert "HIGH" in content.html_body

    def test_html_contains_track_record(self, sample_report):
        digest = WeeklyDigest()
        content = digest.generate(sample_report)
        assert "Win Rate" in content.html_body
        assert "62%" in content.html_body or "63%" in content.html_body  # 62.5% rounds

    def test_html_contains_closed(self, sample_report):
        digest = WeeklyDigest()
        content = digest.generate(sample_report)
        assert "GOOGL" in content.html_body
        assert "5.2%" in content.html_body

    def test_html_contains_disclaimer(self, sample_report):
        digest = WeeklyDigest()
        content = digest.generate(sample_report)
        assert "not financial advice" in content.html_body.lower()

    def test_text_version(self, sample_report):
        digest = WeeklyDigest()
        content = digest.generate(sample_report)
        assert "AAPL" in content.text_body
        assert "Win Rate" in content.text_body
        assert "not financial advice" in content.text_body.lower()

    def test_empty_report(self):
        empty_report = WeeklyReport(
            report_date=date(2024, 2, 15),
            new_recommendations=[],
            open_positions=[],
            closed_this_week=[],
            track_record=TrackRecordSummary(
                period_start=date(2024, 1, 1),
                period_end=date(2024, 2, 15),
                total_recommendations=0,
                closed_recommendations=0,
                profitable=0,
                unprofitable=0,
                win_rate=0.0,
                avg_return=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                best_pick=None,
                best_return=0.0,
                worst_pick=None,
                worst_return=0.0,
                total_return=0.0,
                benchmark_return=0.0,
                excess_return=0.0,
                sharpe_ratio=None,
            ),
        )
        digest = WeeklyDigest()
        content = digest.generate(empty_report)
        assert isinstance(content, DigestContent)
        assert "Weekly Market Brief" in content.subject
