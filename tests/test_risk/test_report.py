"""Tests for validation report generation."""

import pytest

from hrp.risk.report import ValidationReport, generate_validation_report


class TestValidationReport:
    """Tests for validation report generation."""

    @pytest.fixture
    def sample_data(self):
        """Sample validation data."""
        return {
            "hypothesis_id": "HYP-2025-001",
            "title": "Momentum predicts returns",
            "metrics": {
                "sharpe": 0.83,
                "cagr": 0.124,
                "max_drawdown": 0.182,
                "win_rate": 0.54,
                "profit_factor": 1.45,
                "num_trades": 847,
                "oos_period_days": 730,
            },
            "significance_test": {
                "t_statistic": 2.34,
                "p_value": 0.0098,
                "significant": True,
                "excess_return_annualized": 0.042,
            },
            "robustness": {
                "parameter_sensitivity": "PASS",
                "time_stability": "PASS",
                "regime_analysis": "PASS",
            },
            "validation_passed": True,
            "confidence_score": 0.72,
        }

    def test_generate_report_markdown(self, sample_data):
        """Test generating markdown report."""
        report = generate_validation_report(sample_data)
        
        assert isinstance(report, str)
        assert "HYP-2025-001" in report
        assert "VALIDATED" in report
        assert "0.83" in report  # Sharpe ratio

    def test_report_contains_sections(self, sample_data):
        """Test report contains all required sections."""
        report = generate_validation_report(sample_data)
        
        assert "## Summary" in report
        assert "## Performance Metrics" in report
        assert "## Statistical Significance" in report
        assert "## Robustness" in report
        assert "## Recommendation" in report

    def test_report_validation_passed(self, sample_data):
        """Test report shows correct status when passed."""
        report = generate_validation_report(sample_data)
        
        assert "VALIDATED" in report
        assert "Approved for paper trading" in report

    def test_report_validation_failed(self, sample_data):
        """Test report shows correct status when failed."""
        sample_data["validation_passed"] = False
        report = generate_validation_report(sample_data)
        
        assert "REJECTED" in report
        assert "did not meet validation criteria" in report

    def test_validation_report_class(self, sample_data):
        """Test ValidationReport class."""
        validator = ValidationReport(hypothesis_id="HYP-2025-001")
        
        # Remove hypothesis_id from data since class will add it
        data = {k: v for k, v in sample_data.items() if k != "hypothesis_id"}
        report = validator.generate(data)
        
        assert "HYP-2025-001" in report
        assert isinstance(report, str)

    def test_report_without_significance_test(self, sample_data):
        """Test report generation without significance test."""
        sample_data.pop("significance_test")
        report = generate_validation_report(sample_data)
        
        assert "No significance test performed" in report

    def test_report_with_empty_robustness(self, sample_data):
        """Test report generation with empty robustness checks."""
        sample_data["robustness"] = {}
        report = generate_validation_report(sample_data)
        
        assert "## Robustness" in report
