"""Tests for market impact cost model."""

import pytest
import numpy as np

from hrp.risk.costs import MarketImpactModel, CostBreakdown


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_cost_breakdown_total(self):
        """Total equals sum of components."""
        breakdown = CostBreakdown(
            commission=1.50,
            spread=2.00,
            market_impact=3.00,
            total=6.50,
            total_pct=0.00065,
        )
        assert breakdown.total == 6.50
        assert breakdown.total_pct == pytest.approx(0.00065)

    def test_cost_breakdown_immutable_fields(self):
        """CostBreakdown fields are accessible."""
        breakdown = CostBreakdown(
            commission=1.0,
            spread=1.0,
            market_impact=1.0,
            total=3.0,
            total_pct=0.0003,
        )
        assert breakdown.commission == 1.0
        assert breakdown.spread == 1.0
        assert breakdown.market_impact == 1.0


class TestMarketImpactModel:
    """Tests for MarketImpactModel."""

    def test_default_parameters(self):
        """Default parameters are sensible."""
        model = MarketImpactModel()
        assert model.eta == 0.1
        assert model.spread_bps == 5.0
        assert model.commission_per_share == 0.005
        assert model.commission_min == 1.00
        assert model.commission_max_pct == 0.005

    def test_small_trade_commission_minimum(self):
        """Small trades hit commission minimum."""
        model = MarketImpactModel()
        # 10 shares at $50 = $500 trade
        breakdown = model.estimate_cost(
            shares=10,
            price=50.0,
            adv=1_000_000,
            volatility=0.02,
        )
        # Commission should be minimum $1.00
        assert breakdown.commission == 1.00

    def test_large_trade_commission_per_share(self):
        """Large trades use per-share commission."""
        model = MarketImpactModel()
        # 10,000 shares at $50 = $500,000 trade
        breakdown = model.estimate_cost(
            shares=10_000,
            price=50.0,
            adv=1_000_000,
            volatility=0.02,
        )
        # Commission = 10,000 * $0.005 = $50
        # But capped at 0.5% of trade = $2,500
        assert breakdown.commission == 50.0

    def test_commission_cap(self):
        """Commission capped at max percentage."""
        model = MarketImpactModel()
        # 1,000,000 shares at $1 = $1,000,000 trade
        breakdown = model.estimate_cost(
            shares=1_000_000,
            price=1.0,
            adv=10_000_000,
            volatility=0.02,
        )
        # Per-share: 1,000,000 * $0.005 = $5,000
        # Cap: 0.5% of $1,000,000 = $5,000
        assert breakdown.commission == 5000.0

    def test_spread_cost_calculation(self):
        """Spread cost calculated correctly."""
        model = MarketImpactModel(spread_bps=10.0)
        breakdown = model.estimate_cost(
            shares=1000,
            price=100.0,
            adv=1_000_000,
            volatility=0.02,
        )
        # Spread = 10 bps * $100,000 = $100
        assert breakdown.spread == pytest.approx(100.0)

    def test_market_impact_square_root(self):
        """Market impact follows square-root law."""
        model = MarketImpactModel(eta=0.1, spread_bps=0.0)
        model.commission_min = 0.0
        model.commission_per_share = 0.0

        # Trade 10,000 shares (1% of ADV)
        breakdown = model.estimate_cost(
            shares=10_000,
            price=100.0,
            adv=1_000_000,
            volatility=0.02,
        )
        # Impact = 0.1 * 0.02 * sqrt(0.01) * $1,000,000
        # = 0.1 * 0.02 * 0.1 * $1,000,000 = $200
        expected_impact = 0.1 * 0.02 * np.sqrt(10_000 / 1_000_000) * (10_000 * 100.0)
        assert breakdown.market_impact == pytest.approx(expected_impact, rel=0.01)

    def test_market_impact_scales_with_size(self):
        """Larger trades have proportionally higher impact."""
        model = MarketImpactModel(eta=0.1, spread_bps=0.0)
        model.commission_min = 0.0
        model.commission_per_share = 0.0

        small = model.estimate_cost(shares=1000, price=100.0, adv=1_000_000, volatility=0.02)
        large = model.estimate_cost(shares=10_000, price=100.0, adv=1_000_000, volatility=0.02)

        # 10x shares but impact scales with sqrt, so ~3.16x impact per dollar
        # But total impact also scales with trade size, so 10 * 3.16 / 10 = 3.16x
        ratio = large.market_impact / small.market_impact
        assert ratio == pytest.approx(np.sqrt(10) * 10, rel=0.01)

    def test_zero_adv_handling(self):
        """Zero ADV doesn't crash, returns high cost."""
        model = MarketImpactModel()
        breakdown = model.estimate_cost(
            shares=100,
            price=50.0,
            adv=0,
            volatility=0.02,
        )
        # Should return a valid breakdown with high impact
        assert breakdown.total > 0
        assert not np.isnan(breakdown.total)
        assert not np.isinf(breakdown.total)

    def test_total_cost_percentage(self):
        """Total cost percentage calculated correctly."""
        model = MarketImpactModel()
        breakdown = model.estimate_cost(
            shares=1000,
            price=100.0,
            adv=1_000_000,
            volatility=0.02,
        )
        expected_pct = breakdown.total / (1000 * 100.0)
        assert breakdown.total_pct == pytest.approx(expected_pct)

    def test_estimate_cost_returns_breakdown(self):
        """estimate_cost returns CostBreakdown instance."""
        model = MarketImpactModel()
        result = model.estimate_cost(
            shares=100,
            price=50.0,
            adv=1_000_000,
            volatility=0.02,
        )
        assert isinstance(result, CostBreakdown)
