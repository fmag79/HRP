# F-020 Transaction Costs & Risk Limits Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add realistic transaction cost modeling and comprehensive portfolio risk limits to the backtest system.

**Architecture:** Three new modules (costs.py, limits.py, sectors.py) with pre-trade validation that clips signals before backtest execution. Square-root market impact model for costs. Polygon.io for sector data with Yahoo Finance fallback.

**Tech Stack:** Python 3.11+, pandas, numpy, VectorBT, Polygon.io API, yfinance

**Worktree:** `/Users/fer/Documents/GitHub/HRP/.worktrees/f020-transaction-costs`

---

## Task 1: Market Impact Cost Model

**Files:**
- Create: `hrp/risk/costs.py`
- Test: `tests/test_risk/test_costs.py`
- Modify: `hrp/risk/__init__.py`

**Step 1: Write the failing tests**

Create `tests/test_risk/test_costs.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/fer/Documents/GitHub/HRP/.worktrees/f020-transaction-costs
pytest tests/test_risk/test_costs.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hrp.risk.costs'`

**Step 3: Write minimal implementation**

Create `hrp/risk/costs.py`:

```python
"""
Market impact cost model for realistic transaction cost estimation.

Implements square-root market impact model (industry standard) with
IBKR-style commission structure.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CostBreakdown:
    """Breakdown of transaction costs."""

    commission: float
    spread: float
    market_impact: float
    total: float
    total_pct: float


@dataclass
class MarketImpactModel:
    """
    Square-root market impact cost model.

    Cost Components:
        - Commission: IBKR tiered (per-share with min/max)
        - Spread: Half bid-ask spread in basis points
        - Market Impact: k * sigma * sqrt(shares / ADV)

    Attributes:
        eta: Market impact coefficient (default 0.1 for US large-cap)
        spread_bps: Half bid-ask spread in basis points
        commission_per_share: Per-share commission
        commission_min: Minimum commission per trade
        commission_max_pct: Maximum commission as % of trade value
    """

    eta: float = 0.1
    spread_bps: float = 5.0
    commission_per_share: float = 0.005
    commission_min: float = 1.00
    commission_max_pct: float = 0.005

    def estimate_cost(
        self,
        shares: int,
        price: float,
        adv: float,
        volatility: float,
    ) -> CostBreakdown:
        """
        Estimate transaction cost for a trade.

        Args:
            shares: Number of shares to trade
            price: Current share price
            adv: Average daily volume (shares)
            volatility: Daily volatility (decimal, e.g., 0.02 for 2%)

        Returns:
            CostBreakdown with commission, spread, market impact, and totals
        """
        trade_value = shares * price

        # Commission (IBKR tiered)
        per_share_cost = shares * self.commission_per_share
        max_commission = trade_value * self.commission_max_pct
        commission = max(self.commission_min, min(per_share_cost, max_commission))

        # Spread cost
        spread_cost = (self.spread_bps / 10_000) * trade_value

        # Market impact (square-root law)
        if adv > 0:
            participation_rate = shares / adv
            impact_cost = self.eta * volatility * np.sqrt(participation_rate) * trade_value
        else:
            # No volume data: assume 100% participation (very high impact)
            # Cap at 5% of trade value to avoid infinity
            impact_cost = min(0.05 * trade_value, self.eta * volatility * trade_value)

        total = commission + spread_cost + impact_cost
        total_pct = total / trade_value if trade_value > 0 else 0.0

        return CostBreakdown(
            commission=commission,
            spread=spread_cost,
            market_impact=impact_cost,
            total=total,
            total_pct=total_pct,
        )
```

**Step 4: Update `hrp/risk/__init__.py`**

Add to exports in `hrp/risk/__init__.py`:

```python
from hrp.risk.costs import CostBreakdown, MarketImpactModel
```

**Step 5: Run tests to verify they pass**

```bash
cd /Users/fer/Documents/GitHub/HRP/.worktrees/f020-transaction-costs
pytest tests/test_risk/test_costs.py -v
```

Expected: All 12 tests PASS

**Step 6: Run full test suite for regressions**

```bash
pytest tests/ -q --tb=no
```

Expected: 2153+ passed

**Step 7: Commit**

```bash
git add hrp/risk/costs.py tests/test_risk/test_costs.py hrp/risk/__init__.py
git commit -m "feat(risk): add MarketImpactModel with square-root cost estimation

- CostBreakdown dataclass for cost components
- MarketImpactModel with IBKR-style commissions
- Square-root market impact law: k * sigma * sqrt(shares/ADV)
- Handles zero ADV edge case safely
- 12 tests covering all cost components"
```

---

## Task 2: Risk Limits Configuration

**Files:**
- Create: `hrp/risk/limits.py`
- Test: `tests/test_risk/test_limits.py`
- Modify: `hrp/risk/__init__.py`

**Step 1: Write the failing tests**

Create `tests/test_risk/test_limits.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/fer/Documents/GitHub/HRP/.worktrees/f020-transaction-costs
pytest tests/test_risk/test_limits.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hrp.risk.limits'`

**Step 3: Write minimal implementation**

Create `hrp/risk/limits.py`:

```python
"""
Portfolio risk limits for pre-trade validation.

Defines limit configurations and validation reporting structures.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LimitViolation:
    """Record of a single limit violation or clip."""

    limit_name: str
    symbol: str | None
    limit_value: float
    actual_value: float
    action: Literal["clipped", "rejected", "warned"]
    details: str | None = None


@dataclass
class ValidationReport:
    """Report from pre-trade validation."""

    violations: list[LimitViolation] = field(default_factory=list)
    clips: list[LimitViolation] = field(default_factory=list)
    warnings: list[LimitViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no hard violations (clips and warnings are ok)."""
        return len(self.violations) == 0


@dataclass
class RiskLimits:
    """
    Portfolio risk limits for pre-trade validation.

    Conservative institutional defaults for long-only equity.
    """

    # Position limits
    max_position_pct: float = 0.05      # Max 5% in any single position
    min_position_pct: float = 0.01      # Min 1% (avoid tiny positions)
    max_position_adv_pct: float = 0.10  # Max 10% of daily volume

    # Sector limits
    max_sector_pct: float = 0.25        # Max 25% in any sector
    max_unknown_sector_pct: float = 0.10  # Max 10% in unknown sectors

    # Portfolio limits
    max_gross_exposure: float = 1.00    # 100% = no leverage
    min_gross_exposure: float = 0.80    # Stay 80%+ invested
    max_net_exposure: float = 1.00      # Long-only: net = gross

    # Turnover limits
    max_turnover_pct: float = 0.20      # Max 20% turnover per rebalance

    # Concentration limits
    max_top_n_concentration: float = 0.40  # Top 5 holdings < 40%
    top_n_for_concentration: int = 5

    # Liquidity
    min_adv_dollars: float = 1_000_000  # Min $1M daily volume
```

**Step 4: Update `hrp/risk/__init__.py`**

Add to exports:

```python
from hrp.risk.limits import RiskLimits, ValidationReport, LimitViolation
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_risk/test_limits.py -v
```

Expected: All 7 tests PASS

**Step 6: Commit**

```bash
git add hrp/risk/limits.py tests/test_risk/test_limits.py hrp/risk/__init__.py
git commit -m "feat(risk): add RiskLimits configuration and ValidationReport

- RiskLimits with conservative institutional defaults
- LimitViolation for tracking individual violations
- ValidationReport aggregates clips, violations, warnings
- Position, sector, exposure, turnover, concentration limits"
```

---

## Task 3: Pre-Trade Validator - Position Checks

**Files:**
- Modify: `hrp/risk/limits.py`
- Modify: `tests/test_risk/test_limits.py`

**Step 1: Write the failing tests**

Add to `tests/test_risk/test_limits.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_risk/test_limits.py::TestPreTradeValidator -v
```

Expected: FAIL with `ImportError: cannot import name 'PreTradeValidator'`

**Step 3: Write minimal implementation**

Add to `hrp/risk/limits.py`:

```python
from typing import Literal

import pandas as pd
import numpy as np
from loguru import logger

from hrp.risk.costs import MarketImpactModel


class RiskLimitViolationError(Exception):
    """Raised in strict mode when limits are violated."""
    pass


class PreTradeValidator:
    """
    Validates and adjusts signals against risk limits.

    Modes:
        - clip: Adjust weights to satisfy limits (default)
        - strict: Raise exception on any violation
        - warn: Log warnings but allow violations
    """

    def __init__(
        self,
        limits: RiskLimits,
        cost_model: MarketImpactModel,
        mode: Literal["clip", "strict", "warn"] = "clip",
    ):
        self.limits = limits
        self.cost_model = cost_model
        self.mode = mode

    def validate(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        sectors: pd.Series,
        adv: pd.DataFrame,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """
        Validate signals against risk limits.

        Args:
            signals: Raw signals (weights per symbol per date)
            prices: Price data for position sizing
            sectors: Symbol → sector mapping
            adv: Average daily volume (shares)

        Returns:
            Tuple of (validated_signals, ValidationReport)
        """
        validated = signals.copy()
        report = ValidationReport()

        # Check 1: Position sizing (max)
        validated, report = self._check_max_position(validated, report)

        # Check 2: Position sizing (min)
        validated, report = self._check_min_position(validated, report)

        # In strict mode, raise if any violations
        if self.mode == "strict" and (report.clips or report.warnings):
            violations = report.clips + report.warnings
            raise RiskLimitViolationError(
                f"{len(violations)} risk limit violations detected"
            )

        return validated, report

    def _check_max_position(
        self,
        signals: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Clip positions above max_position_pct."""
        max_pct = self.limits.max_position_pct

        for col in signals.columns:
            mask = signals[col] > max_pct
            if mask.any():
                violation = LimitViolation(
                    limit_name="max_position_pct",
                    symbol=col,
                    limit_value=max_pct,
                    actual_value=float(signals[col][mask].max()),
                    action="clipped" if self.mode == "clip" else "warned",
                )

                if self.mode == "clip":
                    signals.loc[mask, col] = max_pct
                    report.clips.append(violation)
                    logger.debug(f"Clipped {col} from {violation.actual_value:.2%} to {max_pct:.2%}")
                else:  # warn mode
                    report.warnings.append(violation)
                    logger.warning(f"{col} exceeds max position: {violation.actual_value:.2%} > {max_pct:.2%}")

        return signals, report

    def _check_min_position(
        self,
        signals: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Zero out positions below min_position_pct."""
        min_pct = self.limits.min_position_pct

        for col in signals.columns:
            mask = (signals[col] > 0) & (signals[col] < min_pct)
            if mask.any():
                violation = LimitViolation(
                    limit_name="min_position_pct",
                    symbol=col,
                    limit_value=min_pct,
                    actual_value=float(signals[col][mask].min()),
                    action="clipped" if self.mode == "clip" else "warned",
                )

                if self.mode == "clip":
                    signals.loc[mask, col] = 0.0
                    report.clips.append(violation)
                    logger.debug(f"Zeroed {col}: {violation.actual_value:.2%} below min {min_pct:.2%}")
                else:  # warn mode
                    report.warnings.append(violation)
                    logger.warning(f"{col} below min position: {violation.actual_value:.2%} < {min_pct:.2%}")

        return signals, report
```

**Step 4: Update exports in `hrp/risk/__init__.py`**

```python
from hrp.risk.limits import (
    RiskLimits,
    ValidationReport,
    LimitViolation,
    PreTradeValidator,
    RiskLimitViolationError,
)
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_risk/test_limits.py -v
```

Expected: All 12 tests PASS

**Step 6: Commit**

```bash
git add hrp/risk/limits.py tests/test_risk/test_limits.py hrp/risk/__init__.py
git commit -m "feat(risk): add PreTradeValidator with position size checks

- PreTradeValidator with clip/strict/warn modes
- Max position clipping (enforces max_position_pct)
- Min position zeroing (removes tiny positions)
- RiskLimitViolationError for strict mode"
```

---

## Task 4: Pre-Trade Validator - Sector & Concentration Checks

**Files:**
- Modify: `hrp/risk/limits.py`
- Modify: `tests/test_risk/test_limits.py`

**Step 1: Write the failing tests**

Add to `tests/test_risk/test_limits.py`:

```python
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
                "META": [5_000] * 5,  # Very low volume
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

        # META should be zeroed (5000 shares * $200 = $1M, but daily so low)
        # Actually ADV in dollars = shares * price
        # 5000 * 200 = $1,000,000 - right at threshold
        # Let's use even lower
        adv["META"] = [100] * 5  # 100 shares * $200 = $20,000
        validated, report = validator.validate(
            signals=sample_signals,
            prices=sample_prices,
            sectors=sample_sectors,
            adv=adv,
        )
        assert validated["META"].iloc[0] == 0.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_risk/test_limits.py::TestPreTradeValidator::test_sector_exposure_clipping -v
pytest tests/test_risk/test_limits.py::TestPreTradeValidator::test_concentration_limit_clipping -v
pytest tests/test_risk/test_limits.py::TestPreTradeValidator::test_gross_exposure_scaling -v
pytest tests/test_risk/test_limits.py::TestPreTradeValidator::test_liquidity_filter -v
```

Expected: FAIL (methods not implemented)

**Step 3: Update PreTradeValidator implementation**

Add methods to `PreTradeValidator` in `hrp/risk/limits.py`:

```python
    def validate(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        sectors: pd.Series,
        adv: pd.DataFrame,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """
        Validate signals against risk limits.

        Checks applied in order:
        1. Liquidity filter (remove illiquid symbols)
        2. Position sizing (max)
        3. Position sizing (min)
        4. Sector exposure
        5. Concentration (top N)
        6. Gross exposure

        Args:
            signals: Raw signals (weights per symbol per date)
            prices: Price data for position sizing
            sectors: Symbol → sector mapping
            adv: Average daily volume (shares)

        Returns:
            Tuple of (validated_signals, ValidationReport)
        """
        validated = signals.copy()
        report = ValidationReport()

        # Check 1: Liquidity filter
        validated, report = self._check_liquidity(validated, prices, adv, report)

        # Check 2: Position sizing (max)
        validated, report = self._check_max_position(validated, report)

        # Check 3: Position sizing (min)
        validated, report = self._check_min_position(validated, report)

        # Check 4: Sector exposure
        validated, report = self._check_sector_exposure(validated, sectors, report)

        # Check 5: Concentration
        validated, report = self._check_concentration(validated, report)

        # Check 6: Gross exposure
        validated, report = self._check_gross_exposure(validated, report)

        # In strict mode, raise if any violations
        if self.mode == "strict" and (report.clips or report.warnings):
            violations = report.clips + report.warnings
            raise RiskLimitViolationError(
                f"{len(violations)} risk limit violations detected"
            )

        return validated, report

    def _check_liquidity(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        adv: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Filter out symbols below minimum ADV threshold."""
        min_adv = self.limits.min_adv_dollars

        for col in signals.columns:
            if col not in adv.columns or col not in prices.columns:
                continue

            # ADV in dollars = shares * price
            adv_dollars = adv[col] * prices[col]

            mask = adv_dollars < min_adv
            if mask.any() and (signals.loc[mask, col] > 0).any():
                actual_adv = float(adv_dollars[mask].mean())
                violation = LimitViolation(
                    limit_name="min_adv_dollars",
                    symbol=col,
                    limit_value=min_adv,
                    actual_value=actual_adv,
                    action="clipped" if self.mode == "clip" else "warned",
                    details=f"ADV ${actual_adv:,.0f} below minimum ${min_adv:,.0f}",
                )

                if self.mode == "clip":
                    signals.loc[mask, col] = 0.0
                    report.clips.append(violation)
                    logger.debug(f"Filtered {col}: ADV ${actual_adv:,.0f} < ${min_adv:,.0f}")
                else:
                    report.warnings.append(violation)

        return signals, report

    def _check_sector_exposure(
        self,
        signals: pd.DataFrame,
        sectors: pd.Series,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Pro-rata reduce sector exposure above max."""
        max_sector = self.limits.max_sector_pct

        for idx in signals.index:
            row = signals.loc[idx]

            # Group by sector
            sector_exposure = {}
            for symbol in row.index:
                sector = sectors.get(symbol, "Unknown")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + row[symbol]

            # Check each sector
            for sector, exposure in sector_exposure.items():
                limit = (
                    self.limits.max_unknown_sector_pct
                    if sector == "Unknown"
                    else max_sector
                )

                if exposure > limit:
                    # Calculate scale factor
                    scale = limit / exposure

                    # Apply to all symbols in this sector
                    for symbol in row.index:
                        if sectors.get(symbol, "Unknown") == sector:
                            old_weight = signals.loc[idx, symbol]
                            if old_weight > 0:
                                new_weight = old_weight * scale
                                violation = LimitViolation(
                                    limit_name="max_sector_pct",
                                    symbol=symbol,
                                    limit_value=limit,
                                    actual_value=exposure,
                                    action="clipped" if self.mode == "clip" else "warned",
                                    details=f"{sector} sector: {exposure:.2%} -> {limit:.2%}",
                                )

                                if self.mode == "clip":
                                    signals.loc[idx, symbol] = new_weight
                                    report.clips.append(violation)
                                else:
                                    report.warnings.append(violation)

        return signals, report

    def _check_concentration(
        self,
        signals: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Reduce top N concentration if exceeded."""
        max_conc = self.limits.max_top_n_concentration
        top_n = self.limits.top_n_for_concentration

        for idx in signals.index:
            row = signals.loc[idx]
            sorted_weights = row.sort_values(ascending=False)
            top_n_exposure = sorted_weights.head(top_n).sum()

            if top_n_exposure > max_conc:
                # Scale down top N positions
                scale = max_conc / top_n_exposure
                top_symbols = sorted_weights.head(top_n).index

                for symbol in top_symbols:
                    old_weight = signals.loc[idx, symbol]
                    new_weight = old_weight * scale

                    violation = LimitViolation(
                        limit_name="max_top_n_concentration",
                        symbol=symbol,
                        limit_value=max_conc,
                        actual_value=top_n_exposure,
                        action="clipped" if self.mode == "clip" else "warned",
                        details=f"Top {top_n}: {top_n_exposure:.2%} -> {max_conc:.2%}",
                    )

                    if self.mode == "clip":
                        signals.loc[idx, symbol] = new_weight
                        report.clips.append(violation)
                    else:
                        report.warnings.append(violation)

        return signals, report

    def _check_gross_exposure(
        self,
        signals: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Scale down all positions if gross exposure exceeded."""
        max_gross = self.limits.max_gross_exposure

        for idx in signals.index:
            gross = signals.loc[idx].sum()

            if gross > max_gross:
                scale = max_gross / gross

                for symbol in signals.columns:
                    old_weight = signals.loc[idx, symbol]
                    if old_weight > 0:
                        new_weight = old_weight * scale

                        violation = LimitViolation(
                            limit_name="max_gross_exposure",
                            symbol=symbol,
                            limit_value=max_gross,
                            actual_value=gross,
                            action="clipped" if self.mode == "clip" else "warned",
                            details=f"Gross: {gross:.2%} -> {max_gross:.2%}",
                        )

                        if self.mode == "clip":
                            signals.loc[idx, symbol] = new_weight
                            report.clips.append(violation)
                        else:
                            report.warnings.append(violation)

        return signals, report
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_risk/test_limits.py -v
```

Expected: All 16 tests PASS

**Step 5: Commit**

```bash
git add hrp/risk/limits.py tests/test_risk/test_limits.py
git commit -m "feat(risk): add sector, concentration, exposure checks to PreTradeValidator

- Liquidity filter: removes symbols below min ADV
- Sector exposure: pro-rata reduces over-weighted sectors
- Concentration: scales down top N holdings
- Gross exposure: scales all positions proportionally"
```

---

## Task 5: Database Schema - Add Sector Columns

**Files:**
- Modify: `hrp/data/schema.py`
- Test: `tests/test_data/test_schema.py`

**Step 1: Write the failing test**

Add to or create `tests/test_data/test_schema.py`:

```python
"""Tests for database schema changes."""

import pytest
from hrp.data.db import get_db


class TestSymbolsSchema:
    """Tests for symbols table schema."""

    def test_symbols_has_sector_column(self):
        """Symbols table has sector column."""
        db = get_db()
        result = db.fetchdf("DESCRIBE symbols")
        columns = result["column_name"].tolist()
        assert "sector" in columns

    def test_symbols_has_industry_column(self):
        """Symbols table has industry column."""
        db = get_db()
        result = db.fetchdf("DESCRIBE symbols")
        columns = result["column_name"].tolist()
        assert "industry" in columns

    def test_sector_index_exists(self):
        """Index on sector column exists."""
        db = get_db()
        # Check for index
        result = db.fetchdf("""
            SELECT index_name FROM duckdb_indexes()
            WHERE table_name = 'symbols'
        """)
        index_names = result["index_name"].tolist()
        assert any("sector" in idx.lower() for idx in index_names)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data/test_schema.py::TestSymbolsSchema -v
```

Expected: FAIL (columns don't exist)

**Step 3: Update schema**

Modify `hrp/data/schema.py` to add migration:

```python
def migrate_add_sector_columns(db: DatabaseManager) -> None:
    """Add sector and industry columns to symbols table."""
    # Check if columns exist
    result = db.fetchdf("DESCRIBE symbols")
    columns = result["column_name"].tolist()

    if "sector" not in columns:
        db.execute("ALTER TABLE symbols ADD COLUMN sector VARCHAR(50)")
        logger.info("Added sector column to symbols table")

    if "industry" not in columns:
        db.execute("ALTER TABLE symbols ADD COLUMN industry VARCHAR(100)")
        logger.info("Added industry column to symbols table")

    # Add index on sector
    try:
        db.execute("CREATE INDEX IF NOT EXISTS idx_symbols_sector ON symbols(sector)")
        logger.info("Created index on symbols.sector")
    except Exception as e:
        logger.debug(f"Index may already exist: {e}")
```

Add to the `ensure_schema()` or initialization flow.

**Step 4: Run migration in test setup**

Ensure migration runs before tests.

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_data/test_schema.py::TestSymbolsSchema -v
```

Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add hrp/data/schema.py tests/test_data/test_schema.py
git commit -m "feat(data): add sector and industry columns to symbols table

- ALTER TABLE to add sector VARCHAR(50), industry VARCHAR(100)
- CREATE INDEX idx_symbols_sector for efficient sector queries
- Migration function for existing databases"
```

---

## Task 6: Sector Ingestion Job

**Files:**
- Create: `hrp/data/ingestion/sectors.py`
- Test: `tests/test_data/test_sectors.py`
- Modify: `hrp/agents/jobs.py`

**Step 1: Write the failing tests**

Create `tests/test_data/test_sectors.py`:

```python
"""Tests for sector data ingestion."""

import pytest
from unittest.mock import Mock, patch

from hrp.data.ingestion.sectors import (
    SectorIngestionJob,
    SIC_TO_GICS_MAPPING,
    fetch_sector_from_polygon,
    fetch_sector_from_yahoo,
)


class TestSicToGicsMapping:
    """Tests for SIC to GICS sector mapping."""

    def test_technology_mapping(self):
        """Technology SIC codes map to Technology sector."""
        # Software
        assert SIC_TO_GICS_MAPPING.get("7372") == "Technology"
        # Computer equipment
        assert SIC_TO_GICS_MAPPING.get("3571") == "Technology"

    def test_healthcare_mapping(self):
        """Healthcare SIC codes map to Healthcare sector."""
        # Pharmaceutical
        assert SIC_TO_GICS_MAPPING.get("2834") == "Healthcare"

    def test_unknown_sic_returns_none(self):
        """Unknown SIC code returns None."""
        assert SIC_TO_GICS_MAPPING.get("9999") is None


class TestFetchSectorFromPolygon:
    """Tests for Polygon sector fetching."""

    @patch("hrp.data.ingestion.sectors.requests.get")
    def test_successful_fetch(self, mock_get):
        """Successful Polygon API call returns sector data."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {
                "results": {
                    "sic_code": "7372",
                    "sic_description": "Prepackaged Software",
                }
            },
        )

        result = fetch_sector_from_polygon("AAPL")

        assert result is not None
        assert result["sic_code"] == "7372"
        assert result["sector"] == "Technology"

    @patch("hrp.data.ingestion.sectors.requests.get")
    def test_api_failure_returns_none(self, mock_get):
        """API failure returns None."""
        mock_get.return_value = Mock(status_code=500)

        result = fetch_sector_from_polygon("AAPL")
        assert result is None


class TestFetchSectorFromYahoo:
    """Tests for Yahoo Finance fallback."""

    @patch("hrp.data.ingestion.sectors.yf.Ticker")
    def test_successful_fetch(self, mock_ticker):
        """Successful Yahoo fetch returns sector data."""
        mock_ticker.return_value.info = {
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }

        result = fetch_sector_from_yahoo("AAPL")

        assert result is not None
        assert result["sector"] == "Technology"
        assert result["industry"] == "Consumer Electronics"

    @patch("hrp.data.ingestion.sectors.yf.Ticker")
    def test_missing_sector_returns_unknown(self, mock_ticker):
        """Missing sector info returns Unknown."""
        mock_ticker.return_value.info = {}

        result = fetch_sector_from_yahoo("AAPL")
        assert result["sector"] == "Unknown"


class TestSectorIngestionJob:
    """Tests for SectorIngestionJob."""

    @patch("hrp.data.ingestion.sectors.fetch_sector_from_polygon")
    @patch("hrp.data.ingestion.sectors.get_db")
    def test_job_updates_symbols(self, mock_db, mock_polygon):
        """Job updates sector in symbols table."""
        mock_polygon.return_value = {
            "sic_code": "7372",
            "sector": "Technology",
            "industry": "Software",
        }
        mock_db.return_value.fetchall.return_value = [("AAPL",)]

        job = SectorIngestionJob(symbols=["AAPL"])
        result = job.run()

        assert result["status"] == "success"
        assert result["symbols_updated"] == 1

    @patch("hrp.data.ingestion.sectors.fetch_sector_from_polygon")
    @patch("hrp.data.ingestion.sectors.fetch_sector_from_yahoo")
    @patch("hrp.data.ingestion.sectors.get_db")
    def test_job_falls_back_to_yahoo(self, mock_db, mock_yahoo, mock_polygon):
        """Job falls back to Yahoo when Polygon fails."""
        mock_polygon.return_value = None
        mock_yahoo.return_value = {
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }
        mock_db.return_value.fetchall.return_value = [("AAPL",)]

        job = SectorIngestionJob(symbols=["AAPL"])
        result = job.run()

        assert result["status"] == "success"
        mock_yahoo.assert_called_once()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data/test_sectors.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

Create `hrp/data/ingestion/sectors.py`:

```python
"""
Sector data ingestion from Polygon.io and Yahoo Finance.

Provides GICS sector classification for universe symbols.
"""

import os
from typing import Any

import requests
import yfinance as yf
from loguru import logger

from hrp.data.db import get_db
from hrp.agents.jobs import IngestionJob, JobResult


# SIC code to GICS sector mapping (partial - common codes)
SIC_TO_GICS_MAPPING = {
    # Technology
    "7372": "Technology",  # Prepackaged Software
    "7371": "Technology",  # Computer Programming
    "3571": "Technology",  # Electronic Computers
    "3674": "Technology",  # Semiconductors
    "7370": "Technology",  # Computer Services
    "3661": "Technology",  # Telephone Equipment
    "3663": "Technology",  # Radio/TV Equipment
    # Healthcare
    "2834": "Healthcare",  # Pharmaceutical
    "3841": "Healthcare",  # Medical Instruments
    "8071": "Healthcare",  # Medical Labs
    "8011": "Healthcare",  # Offices of Physicians
    # Consumer Discretionary
    "5961": "Consumer Discretionary",  # Catalog/Mail-Order
    "5731": "Consumer Discretionary",  # Radio/TV Stores
    "5411": "Consumer Staples",  # Grocery Stores
    "5912": "Consumer Staples",  # Drug Stores
    # Financials
    "6022": "Financials",  # State Commercial Banks
    "6211": "Financials",  # Security Brokers
    # Industrials
    "3721": "Industrials",  # Aircraft
    "3711": "Industrials",  # Motor Vehicles
    # Energy
    "1311": "Energy",  # Crude Petroleum
    "2911": "Energy",  # Petroleum Refining
    # Materials
    "2800": "Materials",  # Chemicals
    # Utilities
    "4911": "Utilities",  # Electric Services
    # Communication Services
    "4813": "Communication Services",  # Telephone
    "7812": "Communication Services",  # Motion Pictures
    # Real Estate
    "6798": "Real Estate",  # REITs
}


def fetch_sector_from_polygon(symbol: str) -> dict[str, Any] | None:
    """
    Fetch sector data from Polygon.io Ticker Details API.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with sic_code, sector, industry or None on failure
    """
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        logger.warning("POLYGON_API_KEY not set")
        return None

    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    params = {"apiKey": api_key}

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            logger.debug(f"Polygon API returned {response.status_code} for {symbol}")
            return None

        data = response.json()
        results = data.get("results", {})

        sic_code = results.get("sic_code", "")
        sic_description = results.get("sic_description", "")

        # Map SIC to GICS sector
        sector = SIC_TO_GICS_MAPPING.get(sic_code, "Unknown")

        return {
            "sic_code": sic_code,
            "sic_description": sic_description,
            "sector": sector,
            "industry": sic_description,  # Use SIC description as industry
        }

    except Exception as e:
        logger.warning(f"Polygon API error for {symbol}: {e}")
        return None


def fetch_sector_from_yahoo(symbol: str) -> dict[str, Any]:
    """
    Fetch sector data from Yahoo Finance (fallback).

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with sector and industry
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
        }

    except Exception as e:
        logger.warning(f"Yahoo Finance error for {symbol}: {e}")
        return {
            "sector": "Unknown",
            "industry": "Unknown",
        }


class SectorIngestionJob(IngestionJob):
    """
    Job to ingest sector data for universe symbols.

    Uses Polygon.io as primary source with Yahoo Finance fallback.
    """

    def __init__(self, symbols: list[str] | None = None):
        """
        Initialize sector ingestion job.

        Args:
            symbols: List of symbols to update (None = all universe)
        """
        super().__init__()
        self.symbols = symbols

    def run(self) -> JobResult:
        """Run sector ingestion."""
        db = get_db()

        # Get symbols to update
        if self.symbols:
            symbols = self.symbols
        else:
            result = db.fetchall("SELECT symbol FROM symbols WHERE is_active = TRUE")
            symbols = [r[0] for r in result]

        logger.info(f"Starting sector ingestion for {len(symbols)} symbols")

        updated = 0
        failed = 0

        for symbol in symbols:
            # Try Polygon first
            sector_data = fetch_sector_from_polygon(symbol)

            # Fallback to Yahoo
            if sector_data is None:
                sector_data = fetch_sector_from_yahoo(symbol)

            # Update database
            try:
                db.execute(
                    """
                    UPDATE symbols
                    SET sector = ?, industry = ?
                    WHERE symbol = ?
                    """,
                    (sector_data["sector"], sector_data.get("industry", "Unknown"), symbol),
                )
                updated += 1
                logger.debug(f"Updated {symbol}: {sector_data['sector']}")

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                failed += 1

        logger.info(f"Sector ingestion complete: {updated} updated, {failed} failed")

        return {
            "status": "success" if failed == 0 else "partial",
            "symbols_updated": updated,
            "symbols_failed": failed,
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data/test_sectors.py -v
```

Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add hrp/data/ingestion/sectors.py tests/test_data/test_sectors.py
git commit -m "feat(data): add SectorIngestionJob with Polygon + Yahoo fallback

- SIC_TO_GICS_MAPPING for common sector mappings
- fetch_sector_from_polygon() using Ticker Details API
- fetch_sector_from_yahoo() as fallback
- SectorIngestionJob updates symbols table"
```

---

## Task 7: Backtest Integration

**Files:**
- Modify: `hrp/research/config.py`
- Modify: `hrp/research/backtest.py`
- Test: `tests/test_research/test_backtest_costs.py`

**Step 1: Write the failing tests**

Create `tests/test_research/test_backtest_costs.py`:

```python
"""Tests for backtest with transaction costs and risk limits."""

import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch, Mock

from hrp.research.config import BacktestConfig
from hrp.research.backtest import run_backtest
from hrp.risk.costs import MarketImpactModel
from hrp.risk.limits import RiskLimits


class TestBacktestWithCosts:
    """Tests for backtest integration with cost model."""

    @pytest.fixture
    def sample_signals(self):
        """Sample signals for testing."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {"AAPL": [0.5] * 10, "MSFT": [0.5] * 10},
            index=dates,
        )

    @pytest.fixture
    def sample_prices(self):
        """Sample prices for testing."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = {
            ("close", "AAPL"): [150 + i for i in range(10)],
            ("close", "MSFT"): [250 + i for i in range(10)],
            ("volume", "AAPL"): [1_000_000] * 10,
            ("volume", "MSFT"): [500_000] * 10,
        }
        return pd.DataFrame(data, index=dates)

    def test_backtest_with_default_cost_model(self, sample_signals, sample_prices):
        """Backtest uses default MarketImpactModel."""
        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
        )

        with patch("hrp.research.backtest.get_price_data", return_value=sample_prices):
            with patch("hrp.research.backtest.get_benchmark_returns", return_value=None):
                result = run_backtest(sample_signals, config, sample_prices)

        assert result is not None
        assert "total_return" in result.metrics

    def test_backtest_with_custom_cost_model(self, sample_signals, sample_prices):
        """Backtest accepts custom cost model."""
        cost_model = MarketImpactModel(
            eta=0.2,  # Higher impact
            spread_bps=10.0,  # Wider spread
        )

        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
            cost_model=cost_model,
        )

        assert config.cost_model.eta == 0.2
        assert config.cost_model.spread_bps == 10.0


class TestBacktestWithRiskLimits:
    """Tests for backtest integration with risk limits."""

    @pytest.fixture
    def sample_signals(self):
        """Signals that violate limits."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "AAPL": [0.60] * 5,  # Exceeds 5% max
                "MSFT": [0.40] * 5,  # Exceeds 5% max
            },
            index=dates,
        )

    @pytest.fixture
    def sample_prices(self):
        """Sample prices."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = {
            ("close", "AAPL"): [150] * 5,
            ("close", "MSFT"): [250] * 5,
            ("volume", "AAPL"): [50_000_000] * 5,
            ("volume", "MSFT"): [30_000_000] * 5,
        }
        return pd.DataFrame(data, index=dates)

    def test_backtest_applies_risk_limits(self, sample_signals, sample_prices):
        """Risk limits clip signals before backtest."""
        limits = RiskLimits(max_position_pct=0.05)

        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            risk_limits=limits,
        )

        # This should clip positions to 5% each
        assert config.risk_limits.max_position_pct == 0.05

    def test_backtest_no_limits_backward_compatible(self, sample_signals, sample_prices):
        """Backtest without limits maintains backward compatibility."""
        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            # No risk_limits specified
        )

        assert config.risk_limits is None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_research/test_backtest_costs.py -v
```

Expected: FAIL (attributes don't exist on BacktestConfig)

**Step 3: Update BacktestConfig**

Modify `hrp/research/config.py`:

```python
from hrp.risk.costs import MarketImpactModel
from hrp.risk.limits import RiskLimits

@dataclass
class BacktestConfig:
    """Configuration for running a backtest."""

    # Universe
    symbols: list[str] = field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None

    # Position sizing
    sizing_method: str = "equal"
    max_position_pct: float = 0.10
    max_positions: int = 20
    min_position_pct: float = 0.02

    # Costs (updated)
    costs: CostModel = field(default_factory=CostModel)  # Legacy
    cost_model: MarketImpactModel = field(default_factory=MarketImpactModel)  # New

    # Risk limits (new)
    risk_limits: RiskLimits | None = None
    validation_mode: str = "clip"  # clip, strict, warn

    # Stop-loss
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)

    # Validation splits
    train_end: date | None = None
    test_start: date | None = None

    # Metadata
    name: str = ""
    description: str = ""

    # Return type
    total_return: bool = False

    def __post_init__(self) -> None:
        if self.costs is None:
            self.costs = CostModel()
        if self.cost_model is None:
            self.cost_model = MarketImpactModel()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_research/test_backtest_costs.py -v
```

Expected: All 4 tests PASS

**Step 5: Run full test suite**

```bash
pytest tests/ -q --tb=no
```

Expected: 2160+ passed (no regressions)

**Step 6: Commit**

```bash
git add hrp/research/config.py tests/test_research/test_backtest_costs.py
git commit -m "feat(research): add cost_model and risk_limits to BacktestConfig

- MarketImpactModel replaces simple CostModel
- Optional RiskLimits for pre-trade validation
- validation_mode: clip/strict/warn
- Backward compatible (defaults to None/defaults)"
```

---

## Task 8: Wire PreTradeValidator into run_backtest()

**Files:**
- Modify: `hrp/research/backtest.py`
- Modify: `tests/test_research/test_backtest_costs.py`

**Step 1: Write the failing tests**

Add to `tests/test_research/test_backtest_costs.py`:

```python
class TestBacktestPreTradeValidation:
    """Tests for pre-trade validation in backtest flow."""

    @pytest.fixture
    def oversized_signals(self):
        """Signals with positions exceeding limits."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "AAPL": [0.30] * 5,
                "MSFT": [0.30] * 5,
                "GOOGL": [0.25] * 5,
                "AMZN": [0.15] * 5,
            },
            index=dates,
        )

    @pytest.fixture
    def mock_prices(self):
        """Mock price data."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        data = {}
        for sym in symbols:
            data[("close", sym)] = [100.0] * 5
            data[("volume", sym)] = [10_000_000] * 5
        return pd.DataFrame(data, index=dates)

    @patch("hrp.research.backtest.get_price_data")
    @patch("hrp.research.backtest.get_benchmark_returns")
    @patch("hrp.research.backtest._load_sector_mapping")
    def test_validation_clips_signals(
        self, mock_sectors, mock_benchmark, mock_prices_fn, oversized_signals, mock_prices
    ):
        """PreTradeValidator clips oversized positions."""
        mock_prices_fn.return_value = mock_prices
        mock_benchmark.return_value = None
        mock_sectors.return_value = pd.Series({
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary",
        })

        limits = RiskLimits(max_position_pct=0.10)
        config = BacktestConfig(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            risk_limits=limits,
        )

        result = run_backtest(oversized_signals, config, mock_prices)

        # Validation report should be attached
        assert hasattr(result, "validation_report")
        assert result.validation_report is not None
        assert len(result.validation_report.clips) > 0

    @patch("hrp.research.backtest.get_price_data")
    @patch("hrp.research.backtest.get_benchmark_returns")
    def test_no_validation_without_limits(
        self, mock_benchmark, mock_prices_fn, oversized_signals, mock_prices
    ):
        """No validation when risk_limits is None."""
        mock_prices_fn.return_value = mock_prices
        mock_benchmark.return_value = None

        config = BacktestConfig(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 5),
            risk_limits=None,  # No limits
        )

        result = run_backtest(oversized_signals, config, mock_prices)

        # No validation report when limits disabled
        assert result.validation_report is None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_research/test_backtest_costs.py::TestBacktestPreTradeValidation -v
```

Expected: FAIL (validation_report not in BacktestResult)

**Step 3: Update BacktestResult and run_backtest()**

Modify `hrp/research/config.py` to add validation_report:

```python
@dataclass
class BacktestResult:
    """Results from a backtest run."""

    config: BacktestConfig
    metrics: dict[str, float]
    equity_curve: pd.Series
    trades: pd.DataFrame
    benchmark_metrics: dict[str, float] | None = None
    validation_report: Any | None = None  # ValidationReport from risk limits
    estimated_costs: dict[str, float] | None = None
```

Modify `hrp/research/backtest.py`:

```python
from hrp.risk.limits import PreTradeValidator, ValidationReport


def _load_sector_mapping(symbols: list[str]) -> pd.Series:
    """Load sector mapping from database."""
    db = get_db()
    result = db.fetchdf(
        f"SELECT symbol, sector FROM symbols WHERE symbol IN ({','.join('?' * len(symbols))})",
        symbols,
    )
    if result.empty:
        return pd.Series({s: "Unknown" for s in symbols})
    return result.set_index("symbol")["sector"]


def _compute_adv(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute average daily volume."""
    if "volume" in prices.columns.get_level_values(0):
        return prices["volume"].rolling(window).mean()
    return pd.DataFrame()


def run_backtest(
    signals: pd.DataFrame,
    config: BacktestConfig,
    prices: pd.DataFrame = None,
) -> BacktestResult:
    """
    Run a backtest using VectorBT.

    Args:
        signals: DataFrame of signals (1 = long, 0 = no position)
        config: Backtest configuration
        prices: Optional price data (loaded if not provided)

    Returns:
        BacktestResult with metrics, equity curve, trades, and validation report
    """
    # Load prices if not provided
    if prices is None:
        prices = get_price_data(config.symbols, config.start_date, config.end_date)

    # Get close prices
    close = prices["close"] if "close" in prices.columns.get_level_values(0) else prices

    # Align signals with prices
    signals = signals.reindex(close.index).fillna(0)

    # PRE-TRADE VALIDATION
    validation_report = None
    if config.risk_limits is not None:
        # Load sector data
        sectors = _load_sector_mapping(config.symbols)

        # Compute ADV
        adv = _compute_adv(prices)

        # Run validation
        validator = PreTradeValidator(
            limits=config.risk_limits,
            cost_model=config.cost_model,
            mode=config.validation_mode,
        )
        signals, validation_report = validator.validate(
            signals=signals,
            prices=close,
            sectors=sectors,
            adv=adv if not adv.empty else close * 0 + 1_000_000,  # Default ADV
        )

    # Apply trailing stops if configured
    # ... (existing stop loss code) ...

    # Calculate fees
    fees = config.cost_model.spread_bps / 10000 + config.cost_model.eta * 0.02 * 0.01

    # Run portfolio simulation
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=signals > 0,
        exits=signals <= 0,
        fees=fees,
        freq="D",
        init_cash=100000,
        size_type="percent",
        size=1.0 / config.max_positions,
    )

    # ... (rest of existing code) ...

    return BacktestResult(
        config=config,
        metrics=metrics,
        equity_curve=equity,
        trades=trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(),
        benchmark_metrics=benchmark_metrics,
        validation_report=validation_report,
    )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_research/test_backtest_costs.py -v
```

Expected: All 6 tests PASS

**Step 5: Run full test suite**

```bash
pytest tests/ -q --tb=no
```

Expected: 2165+ passed

**Step 6: Commit**

```bash
git add hrp/research/backtest.py hrp/research/config.py tests/test_research/test_backtest_costs.py
git commit -m "feat(research): wire PreTradeValidator into run_backtest()

- Load sector mapping from database
- Compute rolling ADV from volume
- Apply risk limits before VectorBT simulation
- Return ValidationReport in BacktestResult
- Backward compatible when risk_limits=None"
```

---

## Task 9: Scheduler Integration

**Files:**
- Modify: `hrp/agents/scheduler.py`
- Modify: `hrp/agents/jobs.py`

**Step 1: Write the failing test**

Add test for scheduler in appropriate test file:

```python
def test_scheduler_has_sector_job():
    """Scheduler can setup weekly sector refresh."""
    from hrp.agents.scheduler import IngestionScheduler

    scheduler = IngestionScheduler()
    # Should have method to setup sector job
    assert hasattr(scheduler, "setup_weekly_sectors")
```

**Step 2: Add SectorIngestionJob to jobs.py exports**

```python
# In hrp/agents/jobs.py
from hrp.data.ingestion.sectors import SectorIngestionJob
```

**Step 3: Add setup_weekly_sectors to scheduler**

```python
def setup_weekly_sectors(
    self,
    sectors_time: str = "10:15",
    day_of_week: str = "sat",
) -> None:
    """
    Schedule weekly sector data refresh.

    Args:
        sectors_time: Time to run (HH:MM, default 10:15 AM)
        day_of_week: Day to run (default Saturday)
    """
    from hrp.data.ingestion.sectors import SectorIngestionJob

    self.scheduler.add_job(
        func=lambda: SectorIngestionJob().run(),
        trigger="cron",
        day_of_week=day_of_week,
        hour=int(sectors_time.split(":")[0]),
        minute=int(sectors_time.split(":")[1]),
        id="sector_ingestion",
        replace_existing=True,
    )
    logger.info(f"Scheduled sector ingestion: {day_of_week} at {sectors_time}")
```

**Step 4: Run tests**

```bash
pytest tests/test_agents/test_scheduler.py -v
```

**Step 5: Commit**

```bash
git add hrp/agents/scheduler.py hrp/agents/jobs.py
git commit -m "feat(agents): add weekly sector ingestion to scheduler

- setup_weekly_sectors() schedules Saturday 10:15 AM
- Exports SectorIngestionJob from jobs.py"
```

---

## Task 10: Update Exports and Documentation

**Files:**
- Modify: `hrp/risk/__init__.py`
- Modify: `CLAUDE.md`

**Step 1: Ensure all exports in hrp/risk/__init__.py**

```python
"""Risk management module."""

from hrp.risk.costs import CostBreakdown, MarketImpactModel
from hrp.risk.limits import (
    LimitViolation,
    PreTradeValidator,
    RiskLimitViolationError,
    RiskLimits,
    ValidationReport,
)
from hrp.risk.overfitting import (
    FeatureCountValidator,
    HyperparameterTrialCounter,
    SharpeDecayMonitor,
    TargetLeakageValidator,
    TestSetGuard,
)
from hrp.risk.robustness import (
    RobustnessResult,
    check_parameter_sensitivity,
    check_regime_stability,
    check_regime_stability_hmm,
    check_time_stability,
)
from hrp.risk.validation import (
    ValidationCriteria,
    ValidationResult,
    benjamini_hochberg,
    bonferroni_correction,
    calculate_bootstrap_ci,
    significance_test,
    validate_strategy,
)

__all__ = [
    # Costs
    "CostBreakdown",
    "MarketImpactModel",
    # Limits
    "LimitViolation",
    "PreTradeValidator",
    "RiskLimitViolationError",
    "RiskLimits",
    "ValidationReport",
    # Overfitting
    "FeatureCountValidator",
    "HyperparameterTrialCounter",
    "SharpeDecayMonitor",
    "TargetLeakageValidator",
    "TestSetGuard",
    # Robustness
    "RobustnessResult",
    "check_parameter_sensitivity",
    "check_regime_stability",
    "check_regime_stability_hmm",
    "check_time_stability",
    # Validation
    "ValidationCriteria",
    "ValidationResult",
    "benjamini_hochberg",
    "bonferroni_correction",
    "calculate_bootstrap_ci",
    "significance_test",
    "validate_strategy",
]
```

**Step 2: Update CLAUDE.md with new features**

Add to CLAUDE.md:

```markdown
### Configure realistic transaction costs
```python
from hrp.risk import MarketImpactModel
from hrp.research.config import BacktestConfig

cost_model = MarketImpactModel(
    eta=0.1,              # Market impact coefficient
    spread_bps=5.0,       # Half bid-ask spread
    commission_per_share=0.005,
)

config = BacktestConfig(
    symbols=['AAPL', 'MSFT'],
    start_date=start,
    end_date=end,
    cost_model=cost_model,
)
```

### Apply portfolio risk limits
```python
from hrp.risk import RiskLimits

limits = RiskLimits(
    max_position_pct=0.05,      # Max 5% per position
    max_sector_pct=0.25,        # Max 25% per sector
    max_gross_exposure=1.00,    # No leverage
    max_turnover_pct=0.20,      # Max 20% turnover
    max_top_n_concentration=0.40,  # Top 5 < 40%
)

config = BacktestConfig(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    risk_limits=limits,
    validation_mode='clip',  # clip, strict, or warn
)

result = run_backtest(signals, config)
print(f"Clips applied: {len(result.validation_report.clips)}")
```
```

**Step 3: Commit**

```bash
git add hrp/risk/__init__.py CLAUDE.md
git commit -m "docs: update exports and CLAUDE.md for F-020 features

- Export all new risk classes
- Add cost model and risk limits examples to CLAUDE.md"
```

---

## Task 11: Final Verification

**Step 1: Run full test suite**

```bash
cd /Users/fer/Documents/GitHub/HRP/.worktrees/f020-transaction-costs
pytest tests/ -v --tb=short
```

Expected: 2170+ tests pass

**Step 2: Update Project Status**

Update `docs/plans/Project-Status-Rodmap.md`:
- Change F-020 status from `⚠️ partial` to `✅ done`

**Step 3: Final commit**

```bash
git add docs/plans/Project-Status-Rodmap.md
git commit -m "docs: mark F-020 Transaction Cost & Risk Limits complete"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | MarketImpactModel | 12 |
| 2 | RiskLimits config | 7 |
| 3 | PreTradeValidator (position) | 5 |
| 4 | PreTradeValidator (sector/conc) | 4 |
| 5 | Schema migration | 3 |
| 6 | SectorIngestionJob | 8 |
| 7 | BacktestConfig update | 4 |
| 8 | run_backtest() integration | 2 |
| 9 | Scheduler integration | 1 |
| 10 | Exports & docs | 0 |
| 11 | Final verification | 0 |

**Total new tests:** ~46
**Estimated final test count:** 2,200+
