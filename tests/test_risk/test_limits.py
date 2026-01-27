"""Tests for portfolio risk limits."""

import pytest
import pandas as pd
import numpy as np

from hrp.risk.limits import RiskLimits, ValidationReport, LimitViolation


class TestRiskLimits:
    """Tests for RiskLimits configuration."""

    def test_default_limits(self):
        """Default limits are conservative institutional."""
        limits = RiskLimits()
        assert limits.max_position_pct == 0.05
        assert limits.min_position_pct == 0.01
        assert limits.max_position_adv_pct == 0.10
        assert limits.max_sector_pct == 0.25
        assert limits.max_unknown_sector_pct == 0.10
        assert limits.max_gross_exposure == 1.00
        assert limits.min_gross_exposure == 0.80
        assert limits.max_turnover_pct == 0.20
        assert limits.max_top_n_concentration == 0.40
        assert limits.top_n_for_concentration == 5
        assert limits.min_adv_dollars == 1_000_000

    def test_custom_limits(self):
        """Custom limits override defaults."""
        limits = RiskLimits(
            max_position_pct=0.10,
            max_sector_pct=0.30,
        )
        assert limits.max_position_pct == 0.10
        assert limits.max_sector_pct == 0.30
        # Other defaults unchanged
        assert limits.min_position_pct == 0.01


class TestValidationReport:
    """Tests for ValidationReport."""

    def test_empty_report_is_valid(self):
        """Report with no violations is valid."""
        report = ValidationReport(violations=[], clips=[], warnings=[])
        assert report.is_valid
        assert len(report.violations) == 0

    def test_report_with_violations_invalid(self):
        """Report with violations is invalid."""
        violation = LimitViolation(
            limit_name="max_position_pct",
            symbol="AAPL",
            limit_value=0.05,
            actual_value=0.08,
            action="rejected",
        )
        report = ValidationReport(violations=[violation], clips=[], warnings=[])
        assert not report.is_valid
        assert len(report.violations) == 1

    def test_report_with_clips_is_valid(self):
        """Report with only clips (no violations) is valid."""
        clip = LimitViolation(
            limit_name="max_position_pct",
            symbol="AAPL",
            limit_value=0.05,
            actual_value=0.08,
            action="clipped",
        )
        report = ValidationReport(violations=[], clips=[clip], warnings=[])
        assert report.is_valid
        assert len(report.clips) == 1


class TestLimitViolation:
    """Tests for LimitViolation."""

    def test_violation_fields(self):
        """LimitViolation stores all fields."""
        violation = LimitViolation(
            limit_name="max_sector_pct",
            symbol="AAPL",
            limit_value=0.25,
            actual_value=0.30,
            action="clipped",
            details="Technology sector",
        )
        assert violation.limit_name == "max_sector_pct"
        assert violation.symbol == "AAPL"
        assert violation.limit_value == 0.25
        assert violation.actual_value == 0.30
        assert violation.action == "clipped"
        assert violation.details == "Technology sector"


class TestPreTradeValidator:
    """Tests for PreTradeValidator."""

    @pytest.fixture
    def sample_signals(self):
        """Sample signals DataFrame."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "AAPL": [0.10, 0.10, 0.10, 0.10, 0.10],
                "MSFT": [0.08, 0.08, 0.08, 0.08, 0.08],
                "GOOGL": [0.06, 0.06, 0.06, 0.06, 0.06],
                "AMZN": [0.04, 0.04, 0.04, 0.04, 0.04],
                "META": [0.02, 0.02, 0.02, 0.02, 0.02],
            },
            index=dates,
        )

    @pytest.fixture
    def sample_prices(self):
        """Sample prices DataFrame."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "AAPL": [150.0, 151.0, 152.0, 153.0, 154.0],
                "MSFT": [250.0, 251.0, 252.0, 253.0, 254.0],
                "GOOGL": [100.0, 101.0, 102.0, 103.0, 104.0],
                "AMZN": [120.0, 121.0, 122.0, 123.0, 124.0],
                "META": [200.0, 201.0, 202.0, 203.0, 204.0],
            },
            index=dates,
        )

    @pytest.fixture
    def sample_adv(self):
        """Sample ADV DataFrame (shares)."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "AAPL": [50_000_000] * 5,
                "MSFT": [30_000_000] * 5,
                "GOOGL": [20_000_000] * 5,
                "AMZN": [40_000_000] * 5,
                "META": [25_000_000] * 5,
            },
            index=dates,
        )

    @pytest.fixture
    def sample_sectors(self):
        """Sample sector mapping."""
        return pd.Series({
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary",
            "META": "Technology",
        })

    def test_validator_default_mode_is_clip(self):
        """Default validation mode is clip."""
        from hrp.risk.limits import PreTradeValidator
        from hrp.risk.costs import MarketImpactModel

        validator = PreTradeValidator(
            limits=RiskLimits(),
            cost_model=MarketImpactModel(),
        )
        assert validator.mode == "clip"

    def test_clip_position_above_max(self, sample_signals, sample_prices, sample_adv, sample_sectors):
        """Positions above max_position_pct are clipped."""
        from hrp.risk.limits import PreTradeValidator
        from hrp.risk.costs import MarketImpactModel

        limits = RiskLimits(max_position_pct=0.05)
        validator = PreTradeValidator(limits=limits, cost_model=MarketImpactModel())

        validated, report = validator.validate(
            signals=sample_signals,
            prices=sample_prices,
            sectors=sample_sectors,
            adv=sample_adv,
        )

        # AAPL was 10%, should be clipped to 5%
        assert validated["AAPL"].iloc[0] == pytest.approx(0.05)
        # MSFT was 8%, should be clipped to 5%
        assert validated["MSFT"].iloc[0] == pytest.approx(0.05)
        # GOOGL was 6%, should be clipped to 5%
        assert validated["GOOGL"].iloc[0] == pytest.approx(0.05)
        # AMZN was 4%, below max, unchanged
        assert validated["AMZN"].iloc[0] == pytest.approx(0.04)
        # META was 2%, below max, unchanged
        assert validated["META"].iloc[0] == pytest.approx(0.02)

        # Should have clips for AAPL, MSFT, GOOGL
        assert len(report.clips) >= 3

    def test_clip_position_below_min(self, sample_signals, sample_prices, sample_adv, sample_sectors):
        """Positions below min_position_pct are zeroed out."""
        from hrp.risk.limits import PreTradeValidator
        from hrp.risk.costs import MarketImpactModel

        limits = RiskLimits(min_position_pct=0.03)
        validator = PreTradeValidator(limits=limits, cost_model=MarketImpactModel())

        validated, report = validator.validate(
            signals=sample_signals,
            prices=sample_prices,
            sectors=sample_sectors,
            adv=sample_adv,
        )

        # META was 2%, below min 3%, should be zeroed
        assert validated["META"].iloc[0] == 0.0
        # AMZN was 4%, above min, unchanged
        assert validated["AMZN"].iloc[0] == pytest.approx(0.04)

    def test_strict_mode_rejects_on_violation(self, sample_signals, sample_prices, sample_adv, sample_sectors):
        """Strict mode raises exception on violation."""
        from hrp.risk.limits import PreTradeValidator, RiskLimitViolationError
        from hrp.risk.costs import MarketImpactModel

        limits = RiskLimits(max_position_pct=0.05)
        validator = PreTradeValidator(
            limits=limits,
            cost_model=MarketImpactModel(),
            mode="strict",
        )

        with pytest.raises(RiskLimitViolationError):
            validator.validate(
                signals=sample_signals,
                prices=sample_prices,
                sectors=sample_sectors,
                adv=sample_adv,
            )

    def test_warn_mode_allows_violations(self, sample_signals, sample_prices, sample_adv, sample_sectors):
        """Warn mode logs warnings but allows violations."""
        from hrp.risk.limits import PreTradeValidator
        from hrp.risk.costs import MarketImpactModel

        limits = RiskLimits(max_position_pct=0.05)
        validator = PreTradeValidator(
            limits=limits,
            cost_model=MarketImpactModel(),
            mode="warn",
        )

        validated, report = validator.validate(
            signals=sample_signals,
            prices=sample_prices,
            sectors=sample_sectors,
            adv=sample_adv,
        )

        # Signals unchanged in warn mode
        assert validated["AAPL"].iloc[0] == pytest.approx(0.10)
        # But warnings recorded
        assert len(report.warnings) >= 3

    def test_sector_exposure_clipping(self, sample_signals, sample_prices, sample_adv, sample_sectors):
        """Sector exposure above max is pro-rata reduced."""
        from hrp.risk.limits import PreTradeValidator
        from hrp.risk.costs import MarketImpactModel

        # Technology sector: AAPL (10%) + MSFT (8%) + GOOGL (6%) + META (2%) = 26%
        limits = RiskLimits(max_sector_pct=0.20, max_position_pct=0.15)
        validator = PreTradeValidator(limits=limits, cost_model=MarketImpactModel())

        validated, report = validator.validate(
            signals=sample_signals,
            prices=sample_prices,
            sectors=sample_sectors,
            adv=sample_adv,
        )

        # Technology should be reduced to 20% total
        tech_symbols = ["AAPL", "MSFT", "GOOGL", "META"]
        tech_exposure = sum(validated[s].iloc[0] for s in tech_symbols)
        assert tech_exposure == pytest.approx(0.20, abs=0.01)

        # Consumer Discretionary (AMZN) unchanged
        assert validated["AMZN"].iloc[0] == pytest.approx(0.04)

    def test_concentration_limit_clipping(self, sample_prices, sample_adv, sample_sectors):
        """Top N concentration is reduced when exceeded."""
        from hrp.risk.limits import PreTradeValidator
        from hrp.risk.costs import MarketImpactModel

        # Create concentrated signals: top 3 = 70%
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        signals = pd.DataFrame(
            {
                "AAPL": [0.30] * 5,
                "MSFT": [0.25] * 5,
                "GOOGL": [0.15] * 5,
                "AMZN": [0.05] * 5,
                "META": [0.05] * 5,
            },
            index=dates,
        )

        limits = RiskLimits(
            max_top_n_concentration=0.50,
            top_n_for_concentration=3,
            max_position_pct=0.35,
            max_sector_pct=1.0,  # Disable sector limit
        )
        validator = PreTradeValidator(limits=limits, cost_model=MarketImpactModel())

        validated, report = validator.validate(
            signals=signals,
            prices=sample_prices,
            sectors=sample_sectors,
            adv=sample_adv,
        )

        # Top 3 should be <= 50%
        top_3 = validated.iloc[0].nlargest(3).sum()
        assert top_3 <= 0.51  # Allow small tolerance

    def test_gross_exposure_scaling(self, sample_prices, sample_adv, sample_sectors):
        """Gross exposure above max is scaled down."""
        from hrp.risk.limits import PreTradeValidator
        from hrp.risk.costs import MarketImpactModel

        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        # Total exposure = 120%
        signals = pd.DataFrame(
            {
                "AAPL": [0.30] * 5,
                "MSFT": [0.30] * 5,
                "GOOGL": [0.25] * 5,
                "AMZN": [0.20] * 5,
                "META": [0.15] * 5,
            },
            index=dates,
        )

        limits = RiskLimits(
            max_gross_exposure=1.00,
            max_position_pct=0.35,
            max_sector_pct=1.0,
        )
        validator = PreTradeValidator(limits=limits, cost_model=MarketImpactModel())

        validated, report = validator.validate(
            signals=signals,
            prices=sample_prices,
            sectors=sample_sectors,
            adv=sample_adv,
        )

        # Total exposure should be <= 100%
        total = validated.iloc[0].sum()
        assert total <= 1.01  # Allow small tolerance

    def test_liquidity_filter(self, sample_signals, sample_prices, sample_sectors):
        """Symbols below min ADV are filtered out."""
        from hrp.risk.limits import PreTradeValidator
        from hrp.risk.costs import MarketImpactModel

        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        # META has low ADV
        adv = pd.DataFrame(
            {
                "AAPL": [50_000_000] * 5,
                "MSFT": [30_000_000] * 5,
                "GOOGL": [20_000_000] * 5,
                "AMZN": [40_000_000] * 5,
                "META": [100] * 5,  # Very low volume: 100 shares * $200 = $20,000
            },
            index=dates,
        )

        limits = RiskLimits(min_adv_dollars=1_000_000)
        validator = PreTradeValidator(limits=limits, cost_model=MarketImpactModel())

        validated, report = validator.validate(
            signals=sample_signals,
            prices=sample_prices,
            sectors=sample_sectors,
            adv=adv,
        )
        assert validated["META"].iloc[0] == 0.0
