# CIO Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the CIO Agent that makes strategic CONTINUE/CONDITIONAL/KILL/PIVOT decisions on validated hypotheses and manages a simulated paper trading portfolio.

**Architecture:** The CIO Agent extends SDKAgent (Claude API integration) and uses PlatformAPI for database access. It implements a 4-dimension scoring framework (statistical, risk, economic, cost) and manages a $1M paper portfolio with equal-risk weighting. Advisory mode only - all decisions require user approval.

**Tech Stack:** Python 3.11+, dataclasses, anthropic (Claude API), pytest, DuckDB, PlatformAPI, APScheduler

---

## Prerequisites

Before starting, verify the current codebase structure:

```bash
# Check SDKAgent base class exists
ls hrp/agents/base.py  # Should exist

# Check PlatformAPI
ls hrp/api/platform.py  # Should exist

# Check existing research agents for reference
ls hrp/agents/research_agents.py  # Reference for structure

# Run tests to ensure clean baseline
pytest tests/ -v --tb=short
```

---

## Task 1: Create CIO Agent Dataclasses

**Files:**
- Create: `hrp/agents/cio.py`
- Test: `tests/test_agents/test_cio_dataclasses.py`

**Step 1: Write the failing test**

Create `tests/test_agents/test_cio_dataclasses.py`:

```python
"""Tests for CIO Agent dataclasses."""

import pytest
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

# Import what we're about to create
from hrp.agents.cio import CIOScore, CIODecision, CIOReport


class TestCIOScore:
    """Test CIOScore dataclass."""

    def test_create_cio_score(self):
        """Test creating a CIOScore with all dimensions."""
        score = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
        )
        assert score.hypothesis_id == "HYP-2026-001"
        assert score.statistical == 0.85
        assert score.risk == 0.78
        assert score.economic == 0.88
        assert score.cost == 0.75

    def test_total_score_calculation(self):
        """Test that total_score is the average of 4 dimensions."""
        score = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.8,
            risk=0.6,
            economic=0.9,
            cost=0.7,
        )
        # (0.8 + 0.6 + 0.9 + 0.7) / 4 = 0.75
        assert score.total == 0.75

    def test_decision_continue(self):
        """Test CONTINUE decision for score >= 0.75."""
        score = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
        )
        assert score.total >= 0.75
        assert score.decision == "CONTINUE"

    def test_decision_conditional(self):
        """Test CONDITIONAL decision for score 0.50-0.74."""
        score = CIOScore(
            hypothesis_id="HYP-2026-002",
            statistical=0.6,
            risk=0.55,
            economic=0.62,
            cost=0.58,
        )
        assert 0.50 <= score.total < 0.75
        assert score.decision == "CONDITIONAL"

    def test_decision_kill(self):
        """Test KILL decision for score < 0.50."""
        score = CIOScore(
            hypothesis_id="HYP-2026-003",
            statistical=0.3,
            risk=0.35,
            economic=0.28,
            cost=0.31,
        )
        assert score.total < 0.50
        assert score.decision == "KILL"

    def test_critical_failure_auto_pivot(self):
        """Test PIVOT decision when critical_failure is True."""
        score = CIOScore(
            hypothesis_id="HYP-2026-004",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
            critical_failure=True,  # Override score
        )
        assert score.total >= 0.75  # Would be CONTINUE
        assert score.decision == "PIVOT"  # But critical failure overrides


class TestCIODecision:
    """Test CIODecision dataclass."""

    def test_create_decision_with_continue(self):
        """Test creating a CONTINUE decision."""
        score = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
        )
        decision = CIODecision(
            hypothesis_id="HYP-2026-001",
            decision="CONTINUE",
            score=score,
            rationale="Strong candidate across all dimensions",
            evidence={"mlflow_run": "abc123"},
            paper_allocation=0.042,  # 4.2% allocation
        )
        assert decision.decision == "CONTINUE"
        assert decision.paper_allocation == 0.042
        assert decision.pivot_direction is None

    def test_create_decision_with_pivot(self):
        """Test creating a PIVOT decision with direction."""
        score = CIOScore(
            hypothesis_id="HYP-2026-005",
            statistical=0.5,
            risk=0.4,
            economic=0.6,
            cost=0.5,
            critical_failure=True,
        )
        decision = CIODecision(
            hypothesis_id="HYP-2026-005",
            decision="PIVOT",
            score=score,
            rationale="Target leakage detected - use lagged features",
            evidence={"leakage_correlation": 0.97},
            pivot_direction="Investigate lagged RSI signals (t-1, t-2)",
        )
        assert decision.decision == "PIVOT"
        assert decision.pivot_direction == "Investigate lagged RSI signals (t-1, t-2)"
        assert decision.paper_allocation is None


class TestCIOReport:
    """Test CIOReport dataclass."""

    def test_create_report(self):
        """Test creating a complete CIO report."""
        score1 = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
        )
        decision1 = CIODecision(
            hypothesis_id="HYP-2026-001",
            decision="CONTINUE",
            score=score1,
            rationale="Strong candidate",
            evidence={"run_id": "abc123"},
            paper_allocation=0.042,
        )

        score2 = CIOScore(
            hypothesis_id="HYP-2026-003",
            statistical=0.3,
            risk=0.35,
            economic=0.28,
            cost=0.31,
        )
        decision2 = CIODecision(
            hypothesis_id="HYP-2026-003",
            decision="KILL",
            score=score2,
            rationale="Insufficient statistical evidence",
            evidence={},
        )

        report = CIOReport(
            report_date=date(2026, 1, 26),
            decisions=[decision1, decision2],
            portfolio_state={
                "nav": 1023456.78,
                "cash": 50000.00,
                "positions_count": 5,
            },
            market_regime="Bull Market",
            next_actions=[
                {"priority": 1, "action": "Approve CONTINUE decisions", "deadline": "2026-01-27"},
            ],
            report_path="docs/reports/2026-01-26/09-00-cio-decision.md",
        )

        assert len(report.decisions) == 2
        assert report.portfolio_state["nav"] == 1023456.78
        assert report.market_regime == "Bull Market"
        assert len(report.next_actions) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio_dataclasses.py -v`

Expected: FAIL with "ImportError: cannot import name 'CIOScore' from 'hrp.agents.cio'"

**Step 3: Write minimal implementation**

Create `hrp/agents/cio.py`:

```python
"""
CIO Agent - Chief Investment Officer Agent.

Makes strategic decisions about research lines and manages paper portfolio.
Advisory mode: presents recommendations, awaits user approval.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional


@dataclass
class CIOScore:
    """
    Balanced score across 4 dimensions for a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis being scored
        statistical: Statistical quality score (0-1)
        risk: Risk profile score (0-1)
        economic: Economic rationale score (0-1)
        cost: Cost realism score (0-1)
        critical_failure: Whether a critical failure was detected
    """

    hypothesis_id: str
    statistical: float
    risk: float
    economic: float
    cost: float
    critical_failure: bool = False

    @property
    def total(self) -> float:
        """Calculate total score as average of 4 dimensions."""
        return (self.statistical + self.risk + self.economic + self.cost) / 4

    @property
    def decision(self) -> Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]:
        """
        Map score to decision.

        Returns:
            CONTINUE: Score >= 0.75, no critical failure
            CONDITIONAL: Score 0.50-0.74, no critical failure
            KILL: Score < 0.50, no critical failure
            PIVOT: Critical failure detected (overrides score)
        """
        if self.critical_failure:
            return "PIVOT"

        if self.total >= 0.75:
            return "CONTINUE"
        if self.total >= 0.50:
            return "CONDITIONAL"
        return "KILL"


@dataclass
class CIODecision:
    """
    Single decision for a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis this decision is for
        decision: One of CONTINUE, CONDITIONAL, KILL, PIVOT
        score: The CIOScore that led to this decision
        rationale: Human-readable explanation
        evidence: Supporting data (MLflow runs, reports, metrics)
        paper_allocation: For CONTINUE decisions, portfolio weight (0-1)
        pivot_direction: For PIVOT decisions, suggested redirect
    """

    hypothesis_id: str
    decision: Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]
    score: CIOScore
    rationale: str
    evidence: dict
    paper_allocation: Optional[float] = None
    pivot_direction: Optional[str] = None


@dataclass
class CIOReport:
    """
    Complete weekly CIO report.

    Attributes:
        report_date: When the report was generated
        decisions: All decisions made in this review cycle
        portfolio_state: Current paper portfolio state
        market_regime: Current market regime context
        next_actions: Prioritized action items
        report_path: Path to the generated markdown report
    """

    report_date: date
    decisions: list[CIODecision]
    portfolio_state: dict
    market_regime: str
    next_actions: list[dict]
    report_path: str
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_cio_dataclasses.py -v`

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add tests/test_agents/test_cio_dataclasses.py hrp/agents/cio.py
git commit -m "feat(cio): add CIO Agent dataclasses

Implements CIOScore, CIODecision, and CIOReport dataclasses.
CIOScore includes 4-dimension scoring with decision logic.
CIODecision wraps score with rationale and evidence.
CIOReport aggregates decisions for weekly reporting.

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 2: Create CIOAgent Class Skeleton

**Files:**
- Modify: `hrp/agents/cio.py`
- Test: `tests/test_agents/test_cio_agent.py`

**Step 1: Write the failing test**

Create `tests/test_agents/test_cio_agent.py`:

```python
"""Tests for CIOAgent class."""

import pytest
from unittest.mock import Mock, patch

from hrp.agents.cio import CIOAgent


class TestCIOAgentInit:
    """Test CIOAgent initialization."""

    def test_init_with_defaults(self):
        """Test CIOAgent can be initialized with defaults."""
        with patch("hrp.agents.cio.PlatformAPI") as mock_api:
            agent = CIOAgent()

            assert agent.agent_name == "cio"
            assert agent.agent_version == "1.0.0"
            assert agent.api is not None
            assert agent.thresholds["min_sharpe"] == 1.0
            assert agent.thresholds["max_drawdown"] == 0.20

    def test_init_with_custom_thresholds(self):
        """Test CIOAgent accepts custom thresholds."""
        with patch("hrp.agents.cio.PlatformAPI") as mock_api:
            custom_thresholds = {
                "min_sharpe": 1.5,
                "max_drawdown": 0.15,
            }
            agent = CIOAgent(thresholds=custom_thresholds)

            assert agent.thresholds["min_sharpe"] == 1.5
            assert agent.thresholds["max_drawdown"] == 0.15
            # Defaults still present
            assert agent.thresholds["sharpe_decay_limit"] == 0.50

    def test_init_with_passed_api(self):
        """Test CIOAgent accepts a PlatformAPI instance."""
        with patch("hrp.agents.cio.PlatformAPI") as mock_api_class:
            mock_api = Mock()
            agent = CIOAgent(api=mock_api)

            assert agent.api == mock_api
            # PlatformAPI not called again
            mock_api_class.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio_agent.py::TestCIOAgentInit -v`

Expected: FAIL with "Cannot import CIOAgent from hrp.agents.cio"

**Step 3: Write minimal implementation**

Add to `hrp/agents/cio.py` (after the dataclasses):

```python
# Add imports at top
from hrp.agents.base import SDKAgent
from hrp.api.platform import PlatformAPI


# Add after CIOReport dataclass
class CIOAgent(SDKAgent):
    """
    Chief Investment Officer Agent.

    Makes strategic decisions about research lines and manages paper portfolio.
    Advisory mode: presents recommendations, awaits user approval.
    """

    agent_name = "cio"
    agent_version = "1.0.0"

    DEFAULT_THRESHOLDS = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.20,
        "sharpe_decay_limit": 0.50,
        "min_ic": 0.03,
        "max_turnover": 0.50,
        "critical_sharpe_decay": 0.75,
        "critical_target_leakage": 0.95,
        "critical_max_drawdown": 0.35,
        "min_profitable_regimes": 2,
    }

    def __init__(
        self,
        api: PlatformAPI | None = None,
        anthropic_client=None,  # Passed to SDKAgent
        thresholds: dict | None = None,
    ):
        """
        Initialize CIO Agent.

        Args:
            api: PlatformAPI instance (created if None)
            anthropic_client: Anthropic client for Claude API
            thresholds: Custom decision thresholds
        """
        super().__init__(anthropic_client=anthropic_client)
        self.api = api or PlatformAPI()
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_cio_agent.py::TestCIOAgentInit -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/cio.py tests/test_agents/test_cio_agent.py
git commit -m "feat(cio): add CIOAgent class skeleton

Implements CIOAgent extending SDKAgent with default thresholds.
Accepts custom thresholds and PlatformAPI injection.

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 3: Implement Statistical Dimension Scoring

**Files:**
- Modify: `hrp/agents/cio.py`
- Test: `tests/test_agents/test_cio_scoring.py`

**Step 1: Write the failing test**

Create `tests/test_agents/test_cio_scoring.py`:

```python
"""Tests for CIO scoring dimensions."""

import pytest
from unittest.mock import Mock, patch
from datetime import date

from hrp.agents.cio import CIOAgent, CIOScore


class TestStatisticalScoring:
    """Test statistical dimension scoring."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent()

    def test_score_statistical_perfect(self, agent):
        """Test perfect statistical score (all metrics at target)."""
        # Mock experiment data with perfect metrics
        experiment_data = {
            "sharpe": 1.5,  # Above 1.0 target
            "stability_score": 0.5,  # At target
            "mean_ic": 0.05,  # Above 0.03 target
            "fold_cv": 1.0,  # Below 2.0 target
        }

        score = agent._score_statistical_dimension("HYP-2026-001", experiment_data)

        # All at or above target should give ~1.0
        assert score >= 0.9
        assert score <= 1.0

    def test_score_statistical_mixed(self, agent):
        """Test mixed statistical score (some good, some bad)."""
        experiment_data = {
            "sharpe": 1.2,  # Above target
            "stability_score": 1.5,  # Above (worse than) target
            "mean_ic": 0.02,  # Below target
            "fold_cv": 2.5,  # Above (worse than) target
        }

        score = agent._score_statistical_dimension("HYP-2026-001", experiment_data)

        # Mixed should give middle score
        assert 0.3 <= score <= 0.7

    def test_score_statistical_poor(self, agent):
        """Test poor statistical score (all metrics below target)."""
        experiment_data = {
            "sharpe": 0.4,  # Below 0.5 bad threshold
            "stability_score": 2.5,  # Above 2.0 bad threshold
            "mean_ic": 0.005,  # Below 0.01 bad threshold
            "fold_cv": 3.5,  # Above 3.0 bad threshold
        }

        score = agent._score_statistical_dimension("HYP-2026-001", experiment_data)

        # All bad should give ~0
        assert score >= 0.0
        assert score <= 0.2

    def test_score_statistical_linear_sharpe(self, agent):
        """Test Sharpe scoring is linear: 0.5->0, 1.0->0.5, 1.5->1.0."""
        # At bad threshold (0.5)
        assert agent._score_sharpe(0.5) == pytest.approx(0.0, abs=0.01)
        # At target (1.0)
        assert agent._score_sharpe(1.0) == pytest.approx(0.5, abs=0.01)
        # At good threshold (1.5)
        assert agent._score_sharpe(1.5) == pytest.approx(1.0, abs=0.01)
        # Clamp below bad
        assert agent._score_sharpe(0.3) == 0.0
        # Clamp above good
        assert agent._score_sharpe(2.0) == 1.0

    def test_score_statistical_linear_stability(self, agent):
        """Test stability scoring is linear: 2.0->0, 1.0->0.5, 0.5->1.0."""
        # Lower is better for stability
        # At bad threshold (2.0)
        assert agent._score_stability(2.0) == pytest.approx(0.0, abs=0.01)
        # At target (1.0)
        assert agent._score_stability(1.0) == pytest.approx(0.5, abs=0.01)
        # At good threshold (0.5)
        assert agent._score_stability(0.5) == pytest.approx(1.0, abs=0.01)

    def test_score_statistical_linear_ic(self, agent):
        """Test IC scoring is linear: 0.01->0, 0.03->0.5, 0.05->1.0."""
        # At bad threshold (0.01)
        assert agent._score_ic(0.01) == pytest.approx(0.0, abs=0.01)
        # At target (0.03)
        assert agent._score_ic(0.03) == pytest.approx(0.5, abs=0.01)
        # At good threshold (0.05)
        assert agent._score_ic(0.05) == pytest.approx(1.0, abs=0.01)

    def test_score_statistical_linear_fold_cv(self, agent):
        """Test fold CV scoring is linear: 3.0->0, 2.0->0.5, 1.0->1.0."""
        # Lower is better for CV
        assert agent._score_fold_cv(3.0) == pytest.approx(0.0, abs=0.01)
        assert agent._score_fold_cv(2.0) == pytest.approx(0.5, abs=0.01)
        assert agent._score_fold_cv(1.0) == pytest.approx(1.0, abs=0.01)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestStatisticalScoring -v`

Expected: FAIL with "CIOAgent has no attribute '_score_statistical_dimension'"

**Step 3: Write minimal implementation**

Add to `hrp/agents/cio.py` (inside CIOAgent class):

```python
    def _score_sharpe(self, sharpe: float) -> float:
        """
        Score Sharpe ratio: 0.5->0, 1.0->0.5, 1.5->1.0.

        Args:
            sharpe: Walk-forward Sharpe ratio

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.5, 1.0, 1.5
        if sharpe <= bad:
            return 0.0
        if sharpe >= good:
            return 1.0
        return (sharpe - bad) / (good - bad)

    def _score_stability(self, stability: float) -> float:
        """
        Score stability score: 2.0->0, 1.0->0.5, 0.5->1.0.

        Lower is better for stability.

        Args:
            stability: Stability score from walk-forward validation

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 2.0, 1.0, 0.5
        if stability >= bad:
            return 0.0
        if stability <= good:
            return 1.0
        return (bad - stability) / (bad - good)

    def _score_ic(self, ic: float) -> float:
        """
        Score Information Coefficient: 0.01->0, 0.03->0.5, 0.05->1.0.

        Args:
            ic: Mean Information Coefficient

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.01, 0.03, 0.05
        if ic <= bad:
            return 0.0
        if ic >= good:
            return 1.0
        return (ic - bad) / (good - bad)

    def _score_fold_cv(self, cv: float) -> float:
        """
        Score fold coefficient of variation: 3.0->0, 2.0->0.5, 1.0->1.0.

        Lower is better for CV (less variability across folds).

        Args:
            cv: Coefficient of variation across folds

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 3.0, 2.0, 1.0
        if cv >= bad:
            return 0.0
        if cv <= good:
            return 1.0
        return (bad - cv) / (bad - good)

    def _score_statistical_dimension(
        self,
        hypothesis_id: str,
        experiment_data: dict,
    ) -> float:
        """
        Score statistical quality dimension.

        Averages scores for: Sharpe, stability, IC, fold CV.

        Args:
            hypothesis_id: The hypothesis being scored
            experiment_data: Dict with sharpe, stability_score, mean_ic, fold_cv

        Returns:
            Score 0-1
        """
        sharpe_score = self._score_sharpe(experiment_data.get("sharpe", 0))
        stability_score = self._score_stability(experiment_data.get("stability_score", 2.0))
        ic_score = self._score_ic(experiment_data.get("mean_ic", 0))
        fold_cv_score = self._score_fold_cv(experiment_data.get("fold_cv", 3.0))

        return (sharpe_score + stability_score + ic_score + fold_cv_score) / 4
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestStatisticalScoring -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/cio.py tests/test_agents/test_cio_scoring.py
git commit -m "feat(cio): implement statistical dimension scoring

Adds linear scoring for Sharpe, stability, IC, and fold CV.
Implements _score_statistical_dimension averaging all metrics.

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 4: Implement Risk Dimension Scoring

**Files:**
- Modify: `hrp/agents/cio.py`
- Modify: `tests/test_agents/test_cio_scoring.py`

**Step 1: Write the failing test**

Add to `tests/test_agents/test_cio_scoring.py`:

```python
class TestRiskScoring:
    """Test risk dimension scoring."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent()

    def test_score_risk_perfect(self, agent):
        """Test perfect risk score (all metrics at target)."""
        risk_data = {
            "max_drawdown": 0.10,  # Below 20% target (good)
            "volatility": 0.10,  # Below 15% target (good)
            "regime_stable": True,  # Binary good
            "sharpe_decay_ok": True,  # Binary good
        }

        score = agent._score_risk_dimension("HYP-2026-001", risk_data)

        assert score >= 0.9

    def test_score_risk_mixed(self, agent):
        """Test mixed risk score."""
        risk_data = {
            "max_drawdown": 0.18,  # Near target
            "volatility": 0.14,  # Near target
            "regime_stable": True,
            "sharpe_decay_ok": False,  # Bad
        }

        score = agent._score_risk_dimension("HYP-2026-001", risk_data)

        # 3 good (including 2 linear near target) + 1 bad = middle score
        assert 0.4 <= score <= 0.7

    def test_score_risk_linear_max_dd(self, agent):
        """Test max drawdown scoring: 30%->0, 20%->0.5, 10%->1.0."""
        # At bad threshold (30%)
        assert agent._score_max_drawdown(0.30) == pytest.approx(0.0, abs=0.01)
        # At target (20%)
        assert agent._score_max_drawdown(0.20) == pytest.approx(0.5, abs=0.01)
        # At good threshold (10%)
        assert agent._score_max_drawdown(0.10) == pytest.approx(1.0, abs=0.01)

    def test_score_risk_linear_volatility(self, agent):
        """Test volatility scoring: 25%->0, 15%->0.5, 10%->1.0."""
        assert agent._score_volatility(0.25) == pytest.approx(0.0, abs=0.01)
        assert agent._score_volatility(0.15) == pytest.approx(0.5, abs=0.01)
        assert agent._score_volatility(0.10) == pytest.approx(1.0, abs=0.01)

    def test_score_risk_binary_regime(self, agent):
        """Test regime stability is binary."""
        assert agent._score_regime_stability(True) == 1.0
        assert agent._score_regime_stability(False) == 0.0

    def test_score_risk_binary_sharpe_decay(self, agent):
        """Test Sharpe decay is binary (<= 50% is good)."""
        assert agent._score_sharpe_decay(0.40) == 1.0  # Below limit
        assert agent._score_sharpe_decay(0.50) == 1.0  # At limit
        assert agent._score_sharpe_decay(0.60) == 0.0  # Above limit

    def test_check_critical_failure_risk(self, agent):
        """Test critical failure detection in risk dimension."""
        # No critical failure
        risk_data = {
            "max_drawdown": 0.20,
            "volatility": 0.15,
            "regime_stable": True,
            "sharpe_decay": 0.40,
        }
        assert agent._check_critical_failures_risk(risk_data) is False

        # Critical: Max DD > 35%
        risk_data["max_drawdown"] = 0.40
        assert agent._check_critical_failures_risk(risk_data) is True

        # Critical: Sharpe decay > 75%
        risk_data["max_drawdown"] = 0.20
        risk_data["sharpe_decay"] = 0.80
        assert agent._check_critical_failures_risk(risk_data) is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestRiskScoring -v`

Expected: FAIL with "CIOAgent has no attribute '_score_risk_dimension'"

**Step 3: Write minimal implementation**

Add to `hrp/agents/cio.py` (inside CIOAgent class):

```python
    def _score_max_drawdown(self, max_dd: float) -> float:
        """
        Score max drawdown: 30%->0, 20%->0.5, 10%->1.0.

        Lower is better for drawdown.

        Args:
            max_dd: Maximum drawdown (as decimal, e.g., 0.20 for 20%)

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.30, 0.20, 0.10
        if max_dd >= bad:
            return 0.0
        if max_dd <= good:
            return 1.0
        return (bad - max_dd) / (bad - good)

    def _score_volatility(self, vol: float) -> float:
        """
        Score annual volatility: 25%->0, 15%->0.5, 10%->1.0.

        Lower is better for volatility.

        Args:
            vol: Annualized volatility (as decimal)

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.25, 0.15, 0.10
        if vol >= bad:
            return 0.0
        if vol <= good:
            return 1.0
        return (bad - vol) / (bad - good)

    def _score_regime_stability(self, stable: bool) -> float:
        """
        Score regime stability (binary).

        Args:
            stable: Whether >= 2/3 regimes are profitable

        Returns:
            1.0 if stable, 0.0 if not
        """
        return 1.0 if stable else 0.0

    def _score_sharpe_decay(self, decay: float) -> float:
        """
        Score Sharpe decay (binary).

        Decay <= 50% is good (no overfitting).

        Args:
            decay: Sharpe decay ratio (train_sharpe - test_sharpe) / train_sharpe

        Returns:
            1.0 if decay <= 0.50, 0.0 if > 0.50
        """
        limit = self.thresholds["sharpe_decay_limit"]
        return 1.0 if decay <= limit else 0.0

    def _check_critical_failures_risk(self, risk_data: dict) -> bool:
        """
        Check for critical failures in risk dimension.

        Critical:
        - Max drawdown > 35%
        - Sharpe decay > 75%

        Args:
            risk_data: Dict with max_drawdown, sharpe_decay

        Returns:
            True if critical failure detected
        """
        max_dd = risk_data.get("max_drawdown", 0)
        sharpe_decay = risk_data.get("sharpe_decay", 0)

        if max_dd > self.thresholds["critical_max_drawdown"]:
            return True
        if sharpe_decay > self.thresholds["critical_sharpe_decay"]:
            return True

        return False

    def _score_risk_dimension(self, hypothesis_id: str, risk_data: dict) -> float:
        """
        Score risk profile dimension.

        Averages scores for: Max DD, volatility, regime stability, Sharpe decay.

        Args:
            hypothesis_id: The hypothesis being scored
            risk_data: Dict with max_drawdown, volatility, regime_stable, sharpe_decay_ok

        Returns:
            Score 0-1
        """
        max_dd_score = self._score_max_drawdown(risk_data.get("max_drawdown", 0.30))
        vol_score = self._score_volatility(risk_data.get("volatility", 0.25))
        regime_score = self._score_regime_stability(risk_data.get("regime_stable", False))
        decay_score = self._score_sharpe_decay(risk_data.get("sharpe_decay", 0.60))

        return (max_dd_score + vol_score + regime_score + decay_score) / 4
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestRiskScoring -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/cio.py tests/test_agents/test_cio_scoring.py
git commit -m "feat(cio): implement risk dimension scoring

Adds linear scoring for max drawdown and volatility.
Binary scoring for regime stability and Sharpe decay.
Critical failure detection for extreme risk metrics.

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 5: Implement Cost Dimension Scoring

**Files:**
- Modify: `hrp/agents/cio.py`
- Modify: `tests/test_agents/test_cio_scoring.py`

**Step 1: Write the failing test**

Add to `tests/test_agents/test_cio_scoring.py`:

```python
class TestCostScoring:
    """Test cost realism dimension scoring."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent()

    def test_score_cost_perfect(self, agent):
        """Test perfect cost score (all metrics at target)."""
        cost_data = {
            "slippage_survival": "stable",  # Survives 2x slippage
            "turnover": 0.20,  # Below 50% target (good)
            "capacity": "high",  # >$10M (good)
            "execution_complexity": "low",  # Low complexity (good)
        }

        score = agent._score_cost_dimension("HYP-2026-001", cost_data)

        assert score >= 0.9

    def test_score_cost_linear_turnover(self, agent):
        """Test turnover scoring: 100%->0, 50%->0.5, 20%->1.0."""
        # At bad threshold (100%)
        assert agent._score_turnover(1.00) == pytest.approx(0.0, abs=0.01)
        # At target (50%)
        assert agent._score_turnover(0.50) == pytest.approx(0.5, abs=0.01)
        # At good threshold (20%)
        assert agent._score_turnover(0.20) == pytest.approx(1.0, abs=0.01)

    def test_score_cost_ordinal_capacity(self, agent):
        """Test capacity scoring is ordinal."""
        assert agent._score_capacity("low") == pytest.approx(0.0, abs=0.01)   # <$1M
        assert agent._score_capacity("medium") == pytest.approx(0.5, abs=0.01)  # $1-10M
        assert agent._score_capacity("high") == pytest.approx(1.0, abs=0.01)    # >$10M

    def test_score_cost_ordinal_slippage(self, agent):
        """Test slippage survival scoring is ordinal."""
        assert agent._score_slippage_survival("collapse") == pytest.approx(0.0, abs=0.01)
        assert agent._score_slippage_survival("degraded") == pytest.approx(0.5, abs=0.01)
        assert agent._score_slippage_survival("stable") == pytest.approx(1.0, abs=0.01)

    def test_score_cost_ordinal_complexity(self, agent):
        """Test execution complexity scoring is ordinal."""
        assert agent._score_execution_complexity("high") == pytest.approx(0.0, abs=0.01)
        assert agent._score_execution_complexity("medium") == pytest.approx(0.5, abs=0.01)
        assert agent._score_execution_complexity("low") == pytest.approx(1.0, abs=0.01)

    def test_score_cost_dimension(self, agent):
        """Test full cost dimension scoring."""
        cost_data = {
            "slippage_survival": "stable",
            "turnover": 0.35,  # Between target and good
            "capacity": "medium",
            "execution_complexity": "medium",
        }

        score = agent._score_cost_dimension("HYP-2026-001", cost_data)

        # Should be middle score
        assert 0.4 <= score <= 0.7
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestCostScoring -v`

Expected: FAIL with "CIOAgent has no attribute '_score_cost_dimension'"

**Step 3: Write minimal implementation**

Add to `hrp/agents/cio.py` (inside CIOAgent class):

```python
    def _score_turnover(self, turnover: float) -> float:
        """
        Score annual turnover: 100%->0, 50%->0.5, 20%->1.0.

        Lower is better for turnover (less trading = lower costs).

        Args:
            turnover: Annual turnover rate (as decimal, e.g., 0.50 for 50%)

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 1.00, 0.50, 0.20
        if turnover >= bad:
            return 0.0
        if turnover <= good:
            return 1.0
        return (bad - turnover) / (bad - good)

    def _score_capacity(self, capacity: str) -> float:
        """
        Score capacity estimate (ordinal).

        Args:
            capacity: One of "low" (<$1M), "medium" ($1-10M), "high" (>$10M)

        Returns:
            Score 0-1
        """
        scores = {"low": 0.0, "medium": 0.5, "high": 1.0}
        return scores.get(capacity.lower(), 0.5)

    def _score_slippage_survival(self, survival: str) -> float:
        """
        Score slippage survival (ordinal).

        Args:
            survival: One of "collapse", "degraded", "stable"

        Returns:
            Score 0-1
        """
        scores = {"collapse": 0.0, "degraded": 0.5, "stable": 1.0}
        return scores.get(survival.lower(), 0.5)

    def _score_execution_complexity(self, complexity: str) -> float:
        """
        Score execution complexity (ordinal).

        Args:
            complexity: One of "high", "medium", "low"

        Returns:
            Score 0-1
        """
        scores = {"high": 0.0, "medium": 0.5, "low": 1.0}
        return scores.get(complexity.lower(), 0.5)

    def _score_cost_dimension(self, hypothesis_id: str, cost_data: dict) -> float:
        """
        Score cost realism dimension.

        Averages scores for: Slippage survival, turnover, capacity, complexity.

        Args:
            hypothesis_id: The hypothesis being scored
            cost_data: Dict with slippage_survival, turnover, capacity, execution_complexity

        Returns:
            Score 0-1
        """
        slippage_score = self._score_slippage_survival(
            cost_data.get("slippage_survival", "degraded")
        )
        turnover_score = self._score_turnover(cost_data.get("turnover", 1.00))
        capacity_score = self._score_capacity(cost_data.get("capacity", "medium"))
        complexity_score = self._score_execution_complexity(
            cost_data.get("execution_complexity", "medium")
        )

        return (slippage_score + turnover_score + capacity_score + complexity_score) / 4
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestCostScoring -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/cio.py tests/test_agents/test_cio_scoring.py
git commit -m "feat(cio): implement cost dimension scoring

Adds linear scoring for turnover.
Ordinal scoring for slippage survival, capacity, execution complexity.

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 6: Implement Economic Dimension Scoring (Claude API)

**Files:**
- Modify: `hrp/agents/cio.py`
- Modify: `tests/test_agents/test_cio_scoring.py`

**Step 1: Write the failing test**

Add to `tests/test_agents/test_cio_scoring.py`:

```python
class TestEconomicScoring:
    """Test economic rationale dimension scoring."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent()

    def test_score_economic_dimension(self, agent):
        """Test economic dimension with mocked Claude API."""
        economic_data = {
            "thesis_strength": "strong",
            "regime_alignment": "aligned",
            "feature_interpretability": 2,  # < 3 black-box features
            "uniqueness": "novel",
        }

        with patch.object(agent, "_assess_thesis_with_claude") as mock_claude:
            mock_claude.return_value = {
                "thesis_strength": "strong",
                "regime_alignment": "aligned",
            }

            score = agent._score_economic_dimension("HYP-2026-001", economic_data)

            # All strong should give high score
            assert score >= 0.8

    def test_score_economic_ordinal_thesis(self, agent):
        """Test thesis strength scoring is ordinal."""
        assert agent._score_thesis_strength("weak") == pytest.approx(0.0, abs=0.01)
        assert agent._score_thesis_strength("moderate") == pytest.approx(0.5, abs=0.01)
        assert agent._score_thesis_strength("strong") == pytest.approx(1.0, abs=0.01)

    def test_score_economic_ordinal_regime(self, agent):
        """Test regime alignment scoring is ordinal."""
        assert agent._score_regime_alignment("mismatch") == pytest.approx(0.0, abs=0.01)
        assert agent._score_regime_alignment("neutral") == pytest.approx(0.5, abs=0.01)
        assert agent._score_regime_alignment("aligned") == pytest.approx(1.0, abs=0.01)

    def test_score_economic_linear_interpretability(self, agent):
        """Test feature interpretability: >5->0, 3-5->0.5, <3->1.0."""
        assert agent._score_feature_interpretability(7) == pytest.approx(0.0, abs=0.01)
        assert agent._score_feature_interpretability(4) == pytest.approx(0.5, abs=0.01)
        assert agent._score_feature_interpretability(2) == pytest.approx(1.0, abs=0.01)

    def test_score_economic_ordinal_uniqueness(self, agent):
        """Test uniqueness scoring is ordinal."""
        assert agent._score_uniqueness("duplicate") == pytest.approx(0.0, abs=0.01)
        assert agent._score_uniqueness("related") == pytest.approx(0.5, abs=0.01)
        assert agent._score_uniqueness("novel") == pytest.approx(1.0, abs=0.01)

    def test_assess_thesis_with_claude(self, agent):
        """Test Claude API assessment of thesis strength and regime."""
        # Mock the Claude client
        mock_client = Mock()
        agent.anthropic_client = mock_client

        # Mock API response
        mock_response = Mock()
        mock_response.content = [Mock(text='{"thesis_strength": "strong", "regime_alignment": "aligned"}')]
        mock_client.messages.create.return_value = mock_response

        result = agent._assess_thesis_with_claude(
            hypothesis_id="HYP-2026-001",
            thesis="Momentum predicts returns",
            agent_reports={"alpha_researcher": "Strong thesis..."},
            current_regime="Bull Market",
        )

        assert result["thesis_strength"] == "strong"
        assert result["regime_alignment"] == "aligned"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestEconomicScoring -v`

Expected: FAIL with "CIOAgent has no attribute '_score_economic_dimension'"

**Step 3: Write minimal implementation**

Add to `hrp/agents/cio.py` (inside CIOAgent class):

```python
    def _score_thesis_strength(self, strength: str) -> float:
        """
        Score thesis strength (ordinal).

        Args:
            strength: One of "weak", "moderate", "strong"

        Returns:
            Score 0-1
        """
        scores = {"weak": 0.0, "moderate": 0.5, "strong": 1.0}
        return scores.get(strength.lower(), 0.5)

    def _score_regime_alignment(self, alignment: str) -> float:
        """
        Score regime alignment (ordinal).

        Args:
            alignment: One of "mismatch", "neutral", "aligned"

        Returns:
            Score 0-1
        """
        scores = {"mismatch": 0.0, "neutral": 0.5, "aligned": 1.0}
        return scores.get(alignment.lower(), 0.5)

    def _score_feature_interpretability(self, black_box_count: int) -> float:
        """
        Score feature interpretability: >5->0, 3-5->0.5, <3->1.0.

        Fewer black-box features = more interpretable = better score.

        Args:
            black_box_count: Number of black-box features (e.g., neural net outputs)

        Returns:
            Score 0-1
        """
        if black_box_count > 5:
            return 0.0
        if black_box_count < 3:
            return 1.0
        return 0.5  # 3-5 features

    def _score_uniqueness(self, uniqueness: str) -> float:
        """
        Score uniqueness (ordinal).

        Args:
            uniqueness: One of "duplicate", "related", "novel"

        Returns:
            Score 0-1
        """
        scores = {"duplicate": 0.0, "related": 0.5, "novel": 1.0}
        return scores.get(uniqueness.lower(), 0.5)

    def _assess_thesis_with_claude(
        self,
        hypothesis_id: str,
        thesis: str,
        agent_reports: dict,
        current_regime: str,
    ) -> dict:
        """
        Use Claude API to assess thesis strength and regime alignment.

        Args:
            hypothesis_id: The hypothesis being assessed
            thesis: The thesis statement
            agent_reports: Dict of agent report content
            current_regime: Current market regime (e.g., "Bull Market")

        Returns:
            Dict with thesis_strength and regime_alignment
        """
        if self.anthropic_client is None:
            # Fallback to moderate scores if no Claude client
            return {"thesis_strength": "moderate", "regime_alignment": "neutral"}

        # Build prompt for Claude
        prompt = f"""Assess this trading hypothesis for economic rationale:

Hypothesis: {thesis}

Current Market Regime: {current_regime}

Agent Reports:
{chr(10).join(f'- {k}: {v[:200]}...' for k, v in agent_reports.items())}

Respond ONLY with valid JSON:
{{
    "thesis_strength": "weak" | "moderate" | "strong",
    "regime_alignment": "mismatch" | "neutral" | "aligned"
}}

Criteria:
- thesis_strength: Is the economic logic sound? Does it have a clear edge?
- regime_alignment: Does this strategy suit the current market regime?
"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            import json

            content = response.content[0].text
            return json.loads(content)

        except Exception as e:
            # Fallback on any error
            return {"thesis_strength": "moderate", "regime_alignment": "neutral"}

    def _score_economic_dimension(
        self,
        hypothesis_id: str,
        economic_data: dict,
    ) -> float:
        """
        Score economic rationale dimension.

        Averages scores for: Thesis strength, regime alignment, interpretability, uniqueness.

        Args:
            hypothesis_id: The hypothesis being scored
            economic_data: Dict with thesis, regime, black_box_count, uniqueness, agent_reports

        Returns:
            Score 0-1
        """
        # Use Claude to assess thesis and regime if not provided
        if "thesis_strength" not in economic_data or "regime_alignment" not in economic_data:
            claude_assessment = self._assess_thesis_with_claude(
                hypothesis_id=hypothesis_id,
                thesis=economic_data.get("thesis", ""),
                agent_reports=economic_data.get("agent_reports", {}),
                current_regime=economic_data.get("current_regime", "Unknown"),
            )
            economic_data = {**economic_data, **claude_assessment}

        thesis_score = self._score_thesis_strength(economic_data.get("thesis_strength", "moderate"))
        regime_score = self._score_regime_alignment(economic_data.get("regime_alignment", "neutral"))
        interpretability_score = self._score_feature_interpretability(
            economic_data.get("black_box_count", 4)
        )
        uniqueness_score = self._score_uniqueness(economic_data.get("uniqueness", "related"))

        return (thesis_score + regime_score + interpretability_score + uniqueness_score) / 4
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestEconomicScoring -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/cio.py tests/test_agents/test_cio_scoring.py
git commit -m "feat(cio): implement economic dimension scoring

Adds ordinal scoring for thesis strength and regime alignment.
Linear scoring for feature interpretability.
Claude API integration for thesis assessment with fallback.

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 7: Implement Full Scoring Orchestration

**Files:**
- Modify: `hrp/agents/cio.py`
- Modify: `tests/test_agents/test_cio_scoring.py`

**Step 1: Write the failing test**

Add to `tests/test_agents/test_cio_scoring.py`:

```python
class TestFullScoring:
    """Test complete scoring orchestration."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent()

    def test_score_hypothesis_continue(self, agent):
        """Test scoring a hypothesis that should get CONTINUE."""
        # Mock data for a strong hypothesis
        experiment_data = {
            "sharpe": 1.5,
            "stability_score": 0.6,
            "mean_ic": 0.045,
            "fold_cv": 1.2,
        }
        risk_data = {
            "max_drawdown": 0.12,
            "volatility": 0.11,
            "regime_stable": True,
            "sharpe_decay": 0.30,
        }
        economic_data = {
            "thesis": "Strong momentum effect persists",
            "current_regime": "Bull Market",
            "black_box_count": 2,
            "uniqueness": "novel",
            "agent_reports": {},
        }
        cost_data = {
            "slippage_survival": "stable",
            "turnover": 0.25,
            "capacity": "high",
            "execution_complexity": "low",
        }

        with patch.object(agent, "_assess_thesis_with_claude") as mock_claude:
            mock_claude.return_value = {
                "thesis_strength": "strong",
                "regime_alignment": "aligned",
            }

            score = agent.score_hypothesis(
                hypothesis_id="HYP-2026-001",
                experiment_data=experiment_data,
                risk_data=risk_data,
                economic_data=economic_data,
                cost_data=cost_data,
            )

            # Should get CONTINUE decision
            assert score.decision == "CONTINUE"
            assert score.total >= 0.75

    def test_score_hypothesis_kill(self, agent):
        """Test scoring a hypothesis that should get KILL."""
        experiment_data = {
            "sharpe": 0.4,
            "stability_score": 2.5,
            "mean_ic": 0.01,
            "fold_cv": 3.2,
        }
        risk_data = {
            "max_drawdown": 0.28,
            "volatility": 0.22,
            "regime_stable": False,
            "sharpe_decay": 0.55,
        }
        economic_data = {
            "thesis": "Weak effect without clear mechanism",
            "current_regime": "Bull Market",
            "black_box_count": 7,
            "uniqueness": "duplicate",
            "agent_reports": {},
        }
        cost_data = {
            "slippage_survival": "degraded",
            "turnover": 0.90,
            "capacity": "low",
            "execution_complexity": "high",
        }

        with patch.object(agent, "_assess_thesis_with_claude") as mock_claude:
            mock_claude.return_value = {
                "thesis_strength": "weak",
                "regime_alignment": "mismatch",
            }

            score = agent.score_hypothesis(
                hypothesis_id="HYP-2026-003",
                experiment_data=experiment_data,
                risk_data=risk_data,
                economic_data=economic_data,
                cost_data=cost_data,
            )

            # Should get KILL decision
            assert score.decision == "KILL"
            assert score.total < 0.50

    def test_score_hypothesis_pivot_critical_failure(self, agent):
        """Test PIVOT decision on critical failure (target leakage)."""
        # Good scores but with critical failure
        experiment_data = {
            "sharpe": 1.5,
            "stability_score": 0.6,
            "mean_ic": 0.045,
            "fold_cv": 1.2,
        }
        risk_data = {
            "max_drawdown": 0.40,  # CRITICAL: > 35%
            "volatility": 0.12,
            "regime_stable": True,
            "sharpe_decay": 0.30,
        }
        economic_data = {"thesis": "...", "current_regime": "...", "agent_reports": {}}
        cost_data = {"slippage_survival": "stable", "turnover": 0.30, "capacity": "high"}

        with patch.object(agent, "_assess_thesis_with_claude"):
            score = agent.score_hypothesis(
                hypothesis_id="HYP-2026-005",
                experiment_data=experiment_data,
                risk_data=risk_data,
                economic_data=economic_data,
                cost_data=cost_data,
            )

            # Should get PIVOT despite good score
            assert score.decision == "PIVOT"
            assert score.critical_failure is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestFullScoring -v`

Expected: FAIL with "CIOAgent has no attribute 'score_hypothesis'"

**Step 3: Write minimal implementation**

Add to `hrp/agents/cio.py` (inside CIOAgent class):

```python
    def score_hypothesis(
        self,
        hypothesis_id: str,
        experiment_data: dict,
        risk_data: dict,
        economic_data: dict,
        cost_data: dict,
    ) -> CIOScore:
        """
        Score a hypothesis across all 4 dimensions.

        Args:
            hypothesis_id: The hypothesis to score
            experiment_data: Statistical metrics from MLflow
            risk_data: Risk metrics (max_dd, volatility, etc.)
            economic_data: Economic rationale data (thesis, regime, etc.)
            cost_data: Cost realism data (turnover, capacity, etc.)

        Returns:
            CIOScore with all dimension scores and decision
        """
        # Score each dimension
        statistical = self._score_statistical_dimension(hypothesis_id, experiment_data)
        risk = self._score_risk_dimension(hypothesis_id, risk_data)
        economic = self._score_economic_dimension(hypothesis_id, economic_data)
        cost = self._score_cost_dimension(hypothesis_id, cost_data)

        # Check for critical failures
        critical_failure = self._check_critical_failures_risk(risk_data)

        # Create score object
        return CIOScore(
            hypothesis_id=hypothesis_id,
            statistical=statistical,
            risk=risk,
            economic=economic,
            cost=cost,
            critical_failure=critical_failure,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_cio_scoring.py::TestFullScoring -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/cio.py tests/test_agents/test_cio_scoring.py
git commit -m "feat(cio): implement full scoring orchestration

Adds score_hypothesis method that orchestrates all 4 dimensions.
Returns CIOScore with decision logic.
Handles critical failure detection for PIVOT decisions.

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 8: Add Database Schema for Paper Portfolio

**Files:**
- Create: `hrp/data/migrations/add_cio_tables.sql`
- Test: `tests/test_data/test_cio_schema.py`

**Step 1: Verify current schema structure**

Run: `ls hrp/data/migrations/`

Expected: Should see existing migration files

**Step 2: Create migration file**

Create `hrp/data/migrations/add_cio_tables.sql`:

```sql
-- CIO Agent: Paper Portfolio and Decision Tracking Tables
-- Migration: 2026-01-27-cio-agent

-- Paper portfolio current allocations
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id INTEGER PRIMARY KEY,
    hypothesis_id VARCHAR NOT NULL UNIQUE,
    weight DECIMAL(5, 4),  -- Position weight (0-1)
    entry_price DECIMAL(10, 4),
    entry_date DATE,
    current_price DECIMAL(10, 4),
    unrealized_pnl DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

-- Daily portfolio history
CREATE TABLE IF NOT EXISTS paper_portfolio_history (
    id INTEGER PRIMARY KEY,
    as_of_date DATE UNIQUE,
    nav DECIMAL(12, 2),  -- Net asset value
    cash DECIMAL(12, 2),
    total_positions INTEGER,
    sharpe_ratio DECIMAL(5, 2),
    max_drawdown DECIMAL(5, 3),
    returns_daily DECIMAL(8, 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Simulated trade log
CREATE TABLE IF NOT EXISTS paper_portfolio_trades (
    id INTEGER PRIMARY KEY,
    hypothesis_id VARCHAR,
    action VARCHAR,  -- 'ADD', 'REMOVE', 'REBALANCE'
    weight_before DECIMAL(5, 4),
    weight_after DECIMAL(5, 4),
    price DECIMAL(10, 4),
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

-- CIO decisions
CREATE TABLE IF NOT EXISTS cio_decisions (
    id INTEGER PRIMARY KEY,
    decision_id VARCHAR UNIQUE,
    report_date DATE,
    hypothesis_id VARCHAR,
    decision VARCHAR,  -- CONTINUE, CONDITIONAL, KILL, PIVOT
    score_total DECIMAL(4, 2),
    score_statistical DECIMAL(4, 2),
    score_risk DECIMAL(4, 2),
    score_economic DECIMAL(4, 2),
    score_cost DECIMAL(4, 2),
    rationale TEXT,
    approved BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR,
    approved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

-- Model cemetery for killed strategies
CREATE TABLE IF NOT EXISTS model_cemetery (
    id INTEGER PRIMARY KEY,
    hypothesis_id VARCHAR UNIQUE,
    killed_date DATE,
    reason TEXT,
    final_score DECIMAL(4, 2),
    experiment_count INTEGER,
    archived_by VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

-- Adaptive threshold tracking
CREATE TABLE IF NOT EXISTS cio_threshold_history (
    id INTEGER PRIMARY KEY,
    threshold_name VARCHAR,
    old_value DECIMAL(10, 4),
    new_value DECIMAL(10, 4),
    reason TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Step 3: Write test to verify tables exist**

Create `tests/test_data/test_cio_schema.py`:

```python
"""Tests for CIO Agent database schema."""

import pytest
from hrp.data.db import DatabaseManager


class TestCIOSchema:
    """Test CIO tables are created correctly."""

    @pytest.fixture
    def db(self):
        """Get database connection."""
        return DatabaseManager()

    def test_paper_portfolio_table_exists(self, db):
        """Test paper_portfolio table exists."""
        result = db.execute_query("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'paper_portfolio'
        """)
        assert result[0]["count"] == 1

    def test_cio_decisions_table_exists(self, db):
        """Test cio_decisions table exists."""
        result = db.execute_query("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'cio_decisions'
        """)
        assert result[0]["count"] == 1

    def test_model_cemetery_table_exists(self, db):
        """Test model_cemetery table exists."""
        result = db.execute_query("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'model_cemetery'
        """)
        assert result[0]["count"] == 1

    def test_paper_portfolio_history_table_exists(self, db):
        """Test paper_portfolio_history table exists."""
        result = db.execute_query("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'paper_portfolio_history'
        """)
        assert result[0]["count"] == 1

    def test_paper_portfolio_trades_table_exists(self, db):
        """Test paper_portfolio_trades table exists."""
        result = db.execute_query("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'paper_portfolio_trades'
        """)
        assert result[0]["count"] == 1
```

**Step 4: Run test to verify it fails**

Run: `pytest tests/test_data/test_cio_schema.py -v`

Expected: FAIL (tables don't exist yet)

**Step 5: Run migration**

Run: `python -c "from hrp.data.db import DatabaseManager; db = DatabaseManager(); migration = open('hrp/data/migrations/add_cio_tables.sql').read(); db.execute_script(migration)"`

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_data/test_cio_schema.py -v`

Expected: PASS

**Step 7: Commit**

```bash
git add hrp/data/migrations/add_cio_tables.sql tests/test_data/test_cio_schema.py
git commit -m "feat(cio): add database schema for paper portfolio

Adds 5 tables:
- paper_portfolio: Current allocations
- paper_portfolio_history: Daily NAV tracking
- paper_portfolio_trades: Simulated trade log
- cio_decisions: Decision records with approval status
- model_cemetery: Killed strategies archive

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 9: Implement Paper Portfolio Allocation

**Files:**
- Modify: `hrp/agents/cio.py`
- Test: `tests/test_agents/test_cio_portfolio.py`

**Step 1: Write the failing test**

Create `tests/test_agents/test_cio_portfolio.py`:

```python
"""Tests for paper portfolio management."""

import pytest
from unittest.mock import Mock, patch
from datetime import date

from hrp.agents.cio import CIOAgent


class TestPortfolioAllocation:
    """Test portfolio allocation calculations."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent()

    def test_calculate_position_weights_equal_risk(self, agent):
        """Test equal-risk position weighting."""
        # 3 strategies with different volatilities
        strategies = [
            {"hypothesis_id": "HYP-001", "volatility": 0.10},
            {"hypothesis_id": "HYP-002", "volatility": 0.15},
            {"hypothesis_id": "HYP-003", "volatility": 0.20},
        ]

        weights = agent._calculate_position_weights(
            strategies=strategies,
            target_risk_contribution=0.03,  # 3% risk per position
            max_weight_cap=0.05,  # 5% max weight
        )

        # Lower volatility = higher weight
        assert weights["HYP-001"] > weights["HYP-002"]
        assert weights["HYP-002"] > weights["HYP-003"]

        # All weights capped at 5%
        for weight in weights.values():
            assert weight <= 0.05

    def test_calculate_position_weights_respects_max_weight(self, agent):
        """Test that max weight cap is enforced."""
        strategies = [
            {"hypothesis_id": "HYP-001", "volatility": 0.05},  # Very low vol
        ]

        weights = agent._calculate_position_weights(
            strategies=strategies,
            target_risk_contribution=0.03,
            max_weight_cap=0.05,  # 5% cap
        )

        # Should be capped at 5% even though vol is very low
        assert weights["HYP-001"] == 0.05

    def test_calculate_position_weights_empty(self, agent):
        """Test empty strategy list returns empty weights."""
        weights = agent._calculate_position_weights(
            strategies=[],
            target_risk_contribution=0.03,
            max_weight_cap=0.05,
        )

        assert len(weights) == 0

    def test_check_portfolio_constraints_pass(self, agent):
        """Test portfolio constraints pass when valid."""
        portfolio_state = {
            "total_weight": 0.95,  # Below 100%
            "max_sector_weight": 0.25,  # Below 30%
            "turnover": 0.40,  # Below 50%
            "max_drawdown": 0.12,  # Below 15%
        }

        violations = agent._check_portfolio_constraints(portfolio_state)

        assert len(violations) == 0

    def test_check_portfolio_constraints_fail(self, agent):
        """Test portfolio constraints detect violations."""
        portfolio_state = {
            "total_weight": 1.10,  # Over 100%
            "max_sector_weight": 0.35,  # Over 30%
            "turnover": 0.60,  # Over 50%
            "max_drawdown": 0.18,  # Over 15%
        }

        violations = agent._check_portfolio_constraints(portfolio_state)

        assert len(violations) == 4
        assert any("total_weight" in v for v in violations)
        assert any("sector" in v for v in violations)
        assert any("turnover" in v for v in violations)
        assert any("drawdown" in v for v in violations)


class TestPortfolioOperations:
    """Test portfolio database operations."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent()

    def test_add_position_to_portfolio(self, agent):
        """Test adding a position to paper portfolio."""
        with patch.object(agent.api.db, "execute_insert") as mock_insert:
            agent._add_paper_position(
                hypothesis_id="HYP-001",
                weight=0.042,
                entry_price=150.0,
            )

            mock_insert.assert_called_once()

    def test_remove_position_from_portfolio(self, agent):
        """Test removing a position from paper portfolio."""
        with patch.object(agent.api.db, "execute_update") as mock_update:
            agent._remove_paper_position(hypothesis_id="HYP-001")

            mock_update.assert_called_once()

    def test_log_paper_trade(self, agent):
        """Test logging a simulated trade."""
        with patch.object(agent.api.db, "execute_insert") as mock_insert:
            agent._log_paper_trade(
                hypothesis_id="HYP-001",
                action="ADD",
                weight_before=0.0,
                weight_after=0.042,
                price=150.0,
            )

            mock_insert.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio_portfolio.py -v`

Expected: FAIL with methods not implemented

**Step 3: Write minimal implementation**

Add to `hrp/agents/cio.py` (inside CIOAgent class):

```python
    # Portfolio constants
    PORTFOLIO_CAPITAL = 1_000_000  # $1M hypothetical
    MAX_POSITIONS = 20
    MAX_WEIGHT_PER_POSITION = 0.05  # 5%
    MIN_WEIGHT_THRESHOLD = 0.01  # 1%

    # Portfolio constraints
    MAX_GROSS_EXPOSURE = 1.0  # 100% long-only
    MAX_SECTOR_CONCENTRATION = 0.30  # 30%
    MAX_TURNOVER = 0.50  # 50% annual
    MAX_DRAWDOWN_LIMIT = 0.15  # 15%

    def _calculate_position_weights(
        self,
        strategies: list[dict],
        target_risk_contribution: float,
        max_weight_cap: float,
    ) -> dict[str, float]:
        """
        Calculate equal-risk position weights.

        weight = min(target_risk_contribution / volatility, max_weight_cap)

        Args:
            strategies: List of dicts with hypothesis_id and volatility
            target_risk_contribution: Target risk per position (e.g., 0.03 for 3%)
            max_weight_cap: Maximum weight per position (e.g., 0.05 for 5%)

        Returns:
            Dict mapping hypothesis_id to weight (0-1)
        """
        weights = {}
        for strategy in strategies:
            hypothesis_id = strategy["hypothesis_id"]
            volatility = strategy.get("volatility", 0.15)  # Default 15%

            # Equal-risk: weight = target_risk / vol
            weight = target_risk_contribution / volatility

            # Apply max weight cap
            weight = min(weight, max_weight_cap)

            weights[hypothesis_id] = weight

        return weights

    def _check_portfolio_constraints(self, portfolio_state: dict) -> list[str]:
        """
        Check portfolio state against constraints.

        Args:
            portfolio_state: Dict with total_weight, max_sector_weight, turnover, max_drawdown

        Returns:
            List of constraint violation messages (empty if all pass)
        """
        violations = []

        if portfolio_state.get("total_weight", 0) > self.MAX_GROSS_EXPOSURE:
            violations.append(f"total_weight {portfolio_state['total_weight']:.1%} > {self.MAX_GROSS_EXPOSURE:.0%}")

        if portfolio_state.get("max_sector_weight", 0) > self.MAX_SECTOR_CONCENTRATION:
            violations.append(f"sector concentration {portfolio_state['max_sector_weight']:.1%} > {self.MAX_SECTOR_CONCENTRATION:.0%}")

        if portfolio_state.get("turnover", 0) > self.MAX_TURNOVER:
            violations.append(f"turnover {portfolio_state['turnover']:.1%} > {self.MAX_TURNOVER:.0%}")

        if portfolio_state.get("max_drawdown", 0) > self.MAX_DRAWDOWN_LIMIT:
            violations.append(f"drawdown {portfolio_state['max_drawdown']:.1%} > {self.MAX_DRAWDOWN_LIMIT:.0%}")

        return violations

    def _add_paper_position(
        self,
        hypothesis_id: str,
        weight: float,
        entry_price: float,
    ):
        """
        Add a position to the paper portfolio.

        Args:
            hypothesis_id: The hypothesis being added
            weight: Position weight (0-1)
            entry_price: Entry price per share
        """
        from datetime import date

        self.api.db.execute_insert(
            """
            INSERT INTO paper_portfolio
            (hypothesis_id, weight, entry_price, entry_date, current_price, unrealized_pnl)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (hypothesis_id, weight, entry_price, date.today(), entry_price),
        )

    def _remove_paper_position(self, hypothesis_id: str):
        """
        Remove a position from the paper portfolio.

        Args:
            hypothesis_id: The hypothesis being removed
        """
        self.api.db.execute_update(
            "DELETE FROM paper_portfolio WHERE hypothesis_id = ?",
            (hypothesis_id,),
        )

    def _log_paper_trade(
        self,
        hypothesis_id: str,
        action: str,
        weight_before: float,
        weight_after: float,
        price: float,
    ):
        """
        Log a simulated paper trade.

        Args:
            hypothesis_id: The hypothesis being traded
            action: One of 'ADD', 'REMOVE', 'REBALANCE'
            weight_before: Weight before trade
            weight_after: Weight after trade
            price: Execution price
        """
        self.api.db.execute_insert(
            """
            INSERT INTO paper_portfolio_trades
            (hypothesis_id, action, weight_before, weight_after, price)
            VALUES (?, ?, ?, ?, ?)
            """,
            (hypothesis_id, action, weight_before, weight_after, price),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_cio_portfolio.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/cio.py tests/test_agents/test_cio_portfolio.py
git commit -m "feat(cio): implement paper portfolio allocation

Adds equal-risk position weighting calculation.
Portfolio constraint checking (exposure, sector, turnover, drawdown).
Database operations for adding/removing positions and logging trades.

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 10: Export CIOAgent from package

**Files:**
- Modify: `hrp/agents/__init__.py`
- Test: `tests/test_agents/test_cio_exports.py`

**Step 1: Write the failing test**

Create `tests/test_agents/test_cio_exports.py`:

```python
"""Tests for CIO Agent package exports."""

import pytest


def test_cio_agent_importable():
    """Test CIOAgent can be imported from hrp.agents."""
    from hrp.agents import CIOAgent

    assert CIOAgent is not None
    assert CIOAgent.agent_name == "cio"


def test_cio_dataclasses_importable():
    """Test CIO dataclasses can be imported."""
    from hrp.agents import CIOScore, CIODecision, CIOReport

    assert CIOScore is not None
    assert CIODecision is not None
    assert CIOReport is not None


def test_cio_agent_in_all():
    """Test CIOAgent is in hrp.agents.__all__."""
    from hrp.agents import __all__

    assert "CIOAgent" in __all__
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio_exports.py -v`

Expected: FAIL with "Cannot import CIOAgent"

**Step 3: Write minimal implementation**

Modify `hrp/agents/__init__.py` to add:

```python
from hrp.agents.cio import CIOAgent, CIOScore, CIODecision, CIOReport

__all__ = [
    # ... existing exports ...
    "CIOAgent",
    "CIOScore",
    "CIODecision",
    "CIOReport",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_cio_exports.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/__init__.py tests/test_agents/test_cio_exports.py
git commit -m "feat(cio): export CIOAgent from package

Adds CIOAgent, CIOScore, CIODecision, CIOReport to hrp.agents exports.

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Task 11: Run Full Test Suite

**Step 1: Run all CIO Agent tests**

Run: `pytest tests/test_agents/test_cio*.py -v`

Expected: All tests PASS

**Step 2: Run regression tests**

Run: `pytest tests/ -v --tb=short`

Expected: All existing tests still PASS

**Step 3: Verify import works**

Run: `python -c "from hrp.agents import CIOAgent; print(CIOAgent.agent_name)"`

Expected: Output: `cio`

**Step 4: Commit (if all pass)**

```bash
git add tests/
git commit -m "test(cio): all CIO Agent tests passing

Core implementation complete:
- 4-dimension scoring framework
- Decision logic (CONTINUE/CONDITIONAL/KILL/PIVOT)
- Paper portfolio allocation
- Database schema

Next: Report generation, scheduler integration, full workflow

Co-Authored-By: Claude (glm-4.7) <noreply@anthropic.com>"
```

---

## Summary

This implementation plan covers the **CIO Agent Phase 1** core functionality:

### Completed Tasks (11)

1. ✅ CIO Agent dataclasses (CIOScore, CIODecision, CIOReport)
2. ✅ CIOAgent class skeleton with SDKAgent inheritance
3. ✅ Statistical dimension scoring (Sharpe, stability, IC, fold CV)
4. ✅ Risk dimension scoring (max DD, volatility, regime, Sharpe decay)
5. ✅ Cost dimension scoring (slippage, turnover, capacity, complexity)
6. ✅ Economic dimension scoring (thesis, regime, interpretability, uniqueness)
7. ✅ Full scoring orchestration with decision logic
8. ✅ Database schema for 5 new tables
9. ✅ Paper portfolio allocation (equal-risk weighting)
10. ✅ Package exports
11. ✅ Full test suite

### Deferred to Phase 2

- Report generation (markdown + email)
- Scheduler integration (weekly reviews)
- `run_weekly_review()` orchestration method
- MCP server integration
- Event-driven alert integration
- Adaptive threshold tuning

### Key Design Decisions

- **TDD approach**: Failing test → minimal implementation → passing test → commit
- **Bite-sized tasks**: Each task is 2-5 minutes of work
- **Frequent commits**: Every task commits independently
- **Mock dependencies**: PlatformAPI, Claude client mocked in tests
- **Linear scoring**: Most metrics use linear interpolation between bad/target/good
- **Binary scoring**: Regime stability, Sharpe decay are pass/fail
- **Ordinal scoring**: Thesis strength, regime alignment are weak/moderate/strong
- **Equal-risk weighting**: Lower volatility = higher position weight
- **Portfolio constraints**: Hard limits on exposure, sector, turnover, drawdown

### Test Coverage

- Unit tests for each scoring dimension
- Unit tests for decision logic (CONTINUE/CONDITIONAL/KILL/PIVOT)
- Unit tests for portfolio allocation algorithms
- Unit tests for constraint checking
- Integration tests for full scoring workflow
- Export tests for package visibility

---

**End of Implementation Plan**
