# Validation Analyst Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-01-26)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Validation Analyst - a hybrid agent that stress tests validated hypotheses through parameter sensitivity, time stability, regime analysis, and execution cost estimation before final deployment approval.

**Architecture:** Hybrid agent extending `ResearchAgent` with optional Claude API calls for edge case reasoning. Uses existing robustness framework (`hrp/risk/robustness.py`) for deterministic tests, invokes Claude for borderline cases requiring judgment.

**Tech Stack:** Python 3.11+, pytest, existing robustness module, MLflow, lineage system

---

## Agent Definition

### Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Validation Analyst |
| **Actor ID** | `agent:validation-analyst` |
| **Type** | Hybrid (deterministic tests + Claude reasoning for edge cases) |
| **Role** | Stress testing, parameter sensitivity, regime analysis, execution realism |
| **Trigger** | Lineage event (after ML Quality Sentinel) + MCP on-demand |
| **Upstream** | ML Quality Sentinel (produces audited experiments) |
| **Downstream** | User/CIO (final deployment decision) |

### Purpose

Final validation gate before hypotheses can be considered for deployment. The Validation Analyst:

1. **Parameter sensitivity** - Tests if strategy degrades gracefully with parameter changes
2. **Time stability** - Verifies strategy works across multiple time periods
3. **Regime analysis** - Ensures strategy performs across market regimes (bull/bear/sideways)
4. **Execution cost estimation** - Calculates realistic transaction costs and slippage impact
5. **Promotes or flags** hypotheses based on validation results

---

## Task 1: Add EventType for Validation Analyst

**Files:**
- Modify: `hrp/research/lineage.py:21-39`
- Test: `tests/test_research/test_lineage.py`

**Step 1: Write the failing test**

```python
# In tests/test_research/test_lineage.py - add to existing file

def test_validation_analyst_event_type_exists():
    """VALIDATION_ANALYST_REVIEW event type is defined."""
    from hrp.research.lineage import EventType

    assert hasattr(EventType, "VALIDATION_ANALYST_REVIEW")
    assert EventType.VALIDATION_ANALYST_REVIEW.value == "validation_analyst_review"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_research/test_lineage.py::test_validation_analyst_event_type_exists -v`
Expected: FAIL with "AttributeError: VALIDATION_ANALYST_REVIEW"

**Step 3: Write minimal implementation**

Add to `EventType` enum in `hrp/research/lineage.py`:

```python
class EventType(str, Enum):
    """Supported lineage event types."""
    # ... existing types ...
    ALPHA_RESEARCHER_REVIEW = "alpha_researcher_review"
    VALIDATION_ANALYST_REVIEW = "validation_analyst_review"  # Add this line
    DATA_INGESTION = "data_ingestion"
    # ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_research/test_lineage.py::test_validation_analyst_event_type_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/research/lineage.py tests/test_research/test_lineage.py
git commit -m "feat(lineage): add VALIDATION_ANALYST_REVIEW event type"
```

---

## Task 2: Create Validation Analyst Dataclasses

**Files:**
- Modify: `hrp/agents/research_agents.py` (add after MLQualitySentinel class)
- Test: `tests/test_agents/test_validation_analyst.py` (new file)

**Step 1: Write the failing test**

Create new test file `tests/test_agents/test_validation_analyst.py`:

```python
"""
Tests for Validation Analyst research agent.
"""

from datetime import date

import pytest

from hrp.agents.research_agents import (
    ValidationCheck,
    ValidationSeverity,
    HypothesisValidation,
    ValidationAnalystReport,
)


class TestValidationDataclasses:
    """Tests for Validation Analyst dataclasses."""

    def test_validation_severity_enum(self):
        """ValidationSeverity has expected values."""
        assert ValidationSeverity.NONE.value == "none"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.CRITICAL.value == "critical"

    def test_validation_check_creation(self):
        """ValidationCheck can be created with all fields."""
        check = ValidationCheck(
            name="parameter_sensitivity",
            passed=True,
            severity=ValidationSeverity.NONE,
            details={"baseline_sharpe": 1.2, "variations": {}},
            message="Parameters are stable",
        )
        assert check.name == "parameter_sensitivity"
        assert check.passed is True
        assert check.severity == ValidationSeverity.NONE

    def test_hypothesis_validation_properties(self):
        """HypothesisValidation computes properties correctly."""
        validation = HypothesisValidation(
            hypothesis_id="HYP-2026-001",
            experiment_id="exp_123",
            validation_date=date(2026, 1, 26),
        )

        # Add checks
        validation.add_check(ValidationCheck(
            name="test1",
            passed=True,
            severity=ValidationSeverity.NONE,
            details={},
            message="OK",
        ))
        validation.add_check(ValidationCheck(
            name="test2",
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            details={},
            message="Failed",
        ))

        assert validation.overall_passed is False
        assert validation.critical_count == 1
        assert validation.warning_count == 0
        assert validation.has_critical_issues is True

    def test_validation_analyst_report(self):
        """ValidationAnalystReport aggregates correctly."""
        report = ValidationAnalystReport(
            report_date=date(2026, 1, 26),
            hypotheses_validated=5,
            hypotheses_passed=3,
            hypotheses_failed=2,
            validations=[],
            duration_seconds=45.2,
        )
        assert report.hypotheses_validated == 5
        assert report.hypotheses_passed == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_validation_analyst.py::TestValidationDataclasses -v`
Expected: FAIL with "ImportError: cannot import name 'ValidationCheck'"

**Step 3: Write minimal implementation**

Add to `hrp/agents/research_agents.py` after the MLQualitySentinel class:

```python
# Re-use AuditSeverity as ValidationSeverity for consistency
ValidationSeverity = AuditSeverity


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    severity: ValidationSeverity
    details: dict[str, Any]
    message: str


@dataclass
class HypothesisValidation:
    """Complete validation of a single hypothesis."""

    hypothesis_id: str
    experiment_id: str
    validation_date: date
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        """Check if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def critical_count(self) -> int:
        """Count critical failures."""
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Count warnings."""
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.WARNING)

    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues found."""
        return self.critical_count > 0

    def add_check(self, check: ValidationCheck) -> None:
        """Add a check result to the validation."""
        self.checks.append(check)


@dataclass
class ValidationAnalystReport:
    """Complete Validation Analyst run report."""

    report_date: date
    hypotheses_validated: int
    hypotheses_passed: int
    hypotheses_failed: int
    validations: list[HypothesisValidation]
    duration_seconds: float
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_validation_analyst.py::TestValidationDataclasses -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/research_agents.py tests/test_agents/test_validation_analyst.py
git commit -m "feat(agents): add ValidationAnalyst dataclasses"
```

---

## Task 3: Create ValidationAnalyst Class Skeleton

**Files:**
- Modify: `hrp/agents/research_agents.py`
- Test: `tests/test_agents/test_validation_analyst.py`

**Step 1: Write the failing test**

Add to `tests/test_agents/test_validation_analyst.py`:

```python
from hrp.agents.research_agents import ValidationAnalyst


class TestValidationAnalystInit:
    """Tests for ValidationAnalyst initialization."""

    def test_default_initialization(self):
        """ValidationAnalyst initializes with defaults."""
        agent = ValidationAnalyst()
        assert agent.ACTOR == "agent:validation-analyst"
        assert agent.actor == "agent:validation-analyst"
        assert agent.job_id == "validation_analyst_review"

    def test_custom_hypothesis_ids(self):
        """ValidationAnalyst accepts hypothesis filter."""
        agent = ValidationAnalyst(hypothesis_ids=["HYP-2026-001"])
        assert agent.hypothesis_ids == ["HYP-2026-001"]

    def test_custom_thresholds(self):
        """ValidationAnalyst accepts custom thresholds."""
        agent = ValidationAnalyst(
            param_sensitivity_threshold=0.6,
            min_profitable_periods=0.5,
            min_profitable_regimes=1,
        )
        assert agent.param_sensitivity_threshold == 0.6
        assert agent.min_profitable_periods == 0.5
        assert agent.min_profitable_regimes == 1

    def test_send_alerts_flag(self):
        """ValidationAnalyst accepts send_alerts flag."""
        agent = ValidationAnalyst(send_alerts=False)
        assert agent.send_alerts is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_validation_analyst.py::TestValidationAnalystInit -v`
Expected: FAIL with "ImportError: cannot import name 'ValidationAnalyst'"

**Step 3: Write minimal implementation**

Add to `hrp/agents/research_agents.py`:

```python
class ValidationAnalyst(ResearchAgent):
    """
    Stress tests validated hypotheses before deployment approval.

    Performs:
    1. Parameter sensitivity - Tests stability under parameter changes
    2. Time stability - Verifies consistent performance across periods
    3. Regime analysis - Checks performance in bull/bear/sideways markets
    4. Execution cost estimation - Calculates realistic transaction costs

    Type: Hybrid (deterministic tests + Claude reasoning for edge cases)
    """

    DEFAULT_JOB_ID = "validation_analyst_review"
    ACTOR = "agent:validation-analyst"

    # Default thresholds
    DEFAULT_PARAM_SENSITIVITY_THRESHOLD = 0.5  # Min ratio of varied/baseline Sharpe
    DEFAULT_MIN_PROFITABLE_PERIODS = 0.67  # 2/3 of periods must be profitable
    DEFAULT_MIN_PROFITABLE_REGIMES = 2  # At least 2 of 3 regimes profitable

    # Transaction cost assumptions
    DEFAULT_COMMISSION_BPS = 5  # 5 basis points per trade
    DEFAULT_SLIPPAGE_BPS = 10  # 10 basis points slippage

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        param_sensitivity_threshold: float | None = None,
        min_profitable_periods: float | None = None,
        min_profitable_regimes: int | None = None,
        commission_bps: float | None = None,
        slippage_bps: float | None = None,
        include_claude_reasoning: bool = True,
        send_alerts: bool = True,
    ):
        """
        Initialize the Validation Analyst.

        Args:
            hypothesis_ids: Specific hypotheses to validate (None = all audited)
            param_sensitivity_threshold: Min ratio for parameter sensitivity
            min_profitable_periods: Min ratio of profitable time periods
            min_profitable_regimes: Min number of profitable regimes
            commission_bps: Commission in basis points
            slippage_bps: Slippage in basis points
            include_claude_reasoning: Use Claude for edge case analysis
            send_alerts: Send email alerts on failures
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],  # Triggered by lineage events
        )
        self.hypothesis_ids = hypothesis_ids
        self.param_sensitivity_threshold = (
            param_sensitivity_threshold or self.DEFAULT_PARAM_SENSITIVITY_THRESHOLD
        )
        self.min_profitable_periods = (
            min_profitable_periods or self.DEFAULT_MIN_PROFITABLE_PERIODS
        )
        self.min_profitable_regimes = (
            min_profitable_regimes or self.DEFAULT_MIN_PROFITABLE_REGIMES
        )
        self.commission_bps = commission_bps or self.DEFAULT_COMMISSION_BPS
        self.slippage_bps = slippage_bps or self.DEFAULT_SLIPPAGE_BPS
        self.include_claude_reasoning = include_claude_reasoning
        self.send_alerts = send_alerts

    def execute(self) -> dict[str, Any]:
        """Run validation on hypotheses. Implemented in Task 5."""
        raise NotImplementedError("ValidationAnalyst.execute() not yet implemented")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_validation_analyst.py::TestValidationAnalystInit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/research_agents.py tests/test_agents/test_validation_analyst.py
git commit -m "feat(agents): add ValidationAnalyst class skeleton"
```

---

## Task 4: Implement Validation Check Methods

**Files:**
- Modify: `hrp/agents/research_agents.py`
- Test: `tests/test_agents/test_validation_analyst.py`

**Step 1: Write the failing tests**

Add to `tests/test_agents/test_validation_analyst.py`:

```python
from unittest.mock import MagicMock, patch
import numpy as np


class TestParameterSensitivity:
    """Tests for parameter sensitivity check."""

    def test_stable_parameters_pass(self):
        """Stable parameters pass the check."""
        agent = ValidationAnalyst()
        experiments = {
            "baseline": {"sharpe": 1.0, "params": {"lookback": 20}},
            "var_1": {"sharpe": 0.8, "params": {"lookback": 16}},
            "var_2": {"sharpe": 0.9, "params": {"lookback": 24}},
        }
        check = agent._check_parameter_sensitivity(experiments, "baseline")
        assert check.passed is True
        assert check.severity == ValidationSeverity.NONE

    def test_unstable_parameters_fail(self):
        """Unstable parameters fail the check."""
        agent = ValidationAnalyst()
        experiments = {
            "baseline": {"sharpe": 1.0, "params": {"lookback": 20}},
            "var_1": {"sharpe": 0.2, "params": {"lookback": 16}},  # 20% of baseline
        }
        check = agent._check_parameter_sensitivity(experiments, "baseline")
        assert check.passed is False
        assert check.severity == ValidationSeverity.CRITICAL


class TestTimeStability:
    """Tests for time stability check."""

    def test_stable_periods_pass(self):
        """Profitable across periods passes."""
        agent = ValidationAnalyst()
        period_metrics = [
            {"period": "2020", "sharpe": 1.0, "profitable": True},
            {"period": "2021", "sharpe": 0.8, "profitable": True},
            {"period": "2022", "sharpe": 0.5, "profitable": True},
        ]
        check = agent._check_time_stability(period_metrics)
        assert check.passed is True

    def test_unstable_periods_fail(self):
        """Unprofitable in most periods fails."""
        agent = ValidationAnalyst()
        period_metrics = [
            {"period": "2020", "sharpe": 1.0, "profitable": True},
            {"period": "2021", "sharpe": -0.5, "profitable": False},
            {"period": "2022", "sharpe": -0.3, "profitable": False},
        ]
        check = agent._check_time_stability(period_metrics)
        assert check.passed is False
        assert check.severity == ValidationSeverity.CRITICAL


class TestRegimeStability:
    """Tests for regime stability check."""

    def test_regime_stable_pass(self):
        """Profitable in multiple regimes passes."""
        agent = ValidationAnalyst()
        regime_metrics = {
            "bull": {"sharpe": 1.5, "profitable": True},
            "bear": {"sharpe": 0.3, "profitable": True},
            "sideways": {"sharpe": -0.2, "profitable": False},
        }
        check = agent._check_regime_stability(regime_metrics)
        assert check.passed is True

    def test_regime_unstable_fail(self):
        """Profitable in only 1 regime fails."""
        agent = ValidationAnalyst()
        regime_metrics = {
            "bull": {"sharpe": 1.5, "profitable": True},
            "bear": {"sharpe": -0.5, "profitable": False},
            "sideways": {"sharpe": -0.3, "profitable": False},
        }
        check = agent._check_regime_stability(regime_metrics)
        assert check.passed is False


class TestExecutionCost:
    """Tests for execution cost estimation."""

    def test_execution_cost_calculation(self):
        """Execution costs calculated correctly."""
        agent = ValidationAnalyst(commission_bps=5, slippage_bps=10)

        # 100 trades, $10,000 average trade size
        check = agent._estimate_execution_costs(
            num_trades=100,
            avg_trade_value=10000,
            gross_return=0.15,  # 15% gross return
        )

        assert check.passed is True
        assert "net_return" in check.details
        assert "total_cost_bps" in check.details
        # Total cost: 15 bps * 100 trades = 1500 bps = 15%
        # Net return should be ~0% (gross 15% - costs 15%)

    def test_high_costs_warning(self):
        """High costs relative to return trigger warning."""
        agent = ValidationAnalyst(commission_bps=50, slippage_bps=50)

        check = agent._estimate_execution_costs(
            num_trades=100,
            avg_trade_value=10000,
            gross_return=0.10,  # 10% gross
        )

        # Costs exceed 50% of return
        assert check.severity in [ValidationSeverity.WARNING, ValidationSeverity.CRITICAL]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agents/test_validation_analyst.py -k "TestParameterSensitivity or TestTimeStability or TestRegimeStability or TestExecutionCost" -v`
Expected: FAIL with "AttributeError: 'ValidationAnalyst' object has no attribute '_check_parameter_sensitivity'"

**Step 3: Write minimal implementation**

Add methods to `ValidationAnalyst` class:

```python
    def _check_parameter_sensitivity(
        self,
        experiments: dict[str, dict[str, Any]],
        baseline_key: str,
    ) -> ValidationCheck:
        """
        Check parameter sensitivity using existing robustness module.

        Args:
            experiments: Dict mapping experiment name to metrics
            baseline_key: Key for baseline experiment

        Returns:
            ValidationCheck with sensitivity results
        """
        from hrp.risk.robustness import check_parameter_sensitivity

        result = check_parameter_sensitivity(
            experiments=experiments,
            baseline_key=baseline_key,
            threshold=self.param_sensitivity_threshold,
        )

        if not result.passed:
            return ValidationCheck(
                name="parameter_sensitivity",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=result.checks.get("parameter_sensitivity", {}),
                message=f"Parameter sensitivity failed: {'; '.join(result.failures)}",
            )

        return ValidationCheck(
            name="parameter_sensitivity",
            passed=True,
            severity=ValidationSeverity.NONE,
            details=result.checks.get("parameter_sensitivity", {}),
            message="Parameters are stable under variation",
        )

    def _check_time_stability(
        self,
        period_metrics: list[dict[str, Any]],
    ) -> ValidationCheck:
        """
        Check time period stability using existing robustness module.

        Args:
            period_metrics: List of period metrics with 'sharpe', 'profitable'

        Returns:
            ValidationCheck with stability results
        """
        from hrp.risk.robustness import check_time_stability

        result = check_time_stability(
            period_metrics=period_metrics,
            min_profitable_ratio=self.min_profitable_periods,
        )

        if not result.passed:
            return ValidationCheck(
                name="time_stability",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=result.checks.get("time_stability", {}),
                message=f"Time stability failed: {'; '.join(result.failures)}",
            )

        return ValidationCheck(
            name="time_stability",
            passed=True,
            severity=ValidationSeverity.NONE,
            details=result.checks.get("time_stability", {}),
            message="Strategy is stable across time periods",
        )

    def _check_regime_stability(
        self,
        regime_metrics: dict[str, dict[str, Any]],
    ) -> ValidationCheck:
        """
        Check market regime stability using existing robustness module.

        Args:
            regime_metrics: Dict mapping regime name to metrics

        Returns:
            ValidationCheck with regime results
        """
        from hrp.risk.robustness import check_regime_stability

        result = check_regime_stability(
            regime_metrics=regime_metrics,
            min_regimes_profitable=self.min_profitable_regimes,
        )

        if not result.passed:
            return ValidationCheck(
                name="regime_stability",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=result.checks.get("regime_stability", {}),
                message=f"Regime stability failed: {'; '.join(result.failures)}",
            )

        return ValidationCheck(
            name="regime_stability",
            passed=True,
            severity=ValidationSeverity.NONE,
            details=result.checks.get("regime_stability", {}),
            message="Strategy works across market regimes",
        )

    def _estimate_execution_costs(
        self,
        num_trades: int,
        avg_trade_value: float,
        gross_return: float,
    ) -> ValidationCheck:
        """
        Estimate realistic execution costs and net return.

        Args:
            num_trades: Number of round-trip trades
            avg_trade_value: Average trade value in dollars
            gross_return: Gross return before costs

        Returns:
            ValidationCheck with cost analysis
        """
        # Calculate total cost in basis points
        cost_per_trade_bps = self.commission_bps + self.slippage_bps
        total_cost_bps = cost_per_trade_bps * num_trades

        # Convert to decimal
        total_cost_decimal = total_cost_bps / 10000

        # Net return
        net_return = gross_return - total_cost_decimal

        # Cost as percentage of gross return
        cost_ratio = total_cost_decimal / gross_return if gross_return > 0 else float("inf")

        details = {
            "num_trades": num_trades,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "total_cost_bps": total_cost_bps,
            "total_cost_decimal": total_cost_decimal,
            "gross_return": gross_return,
            "net_return": net_return,
            "cost_ratio": cost_ratio,
        }

        # Determine severity
        if net_return < 0:
            return ValidationCheck(
                name="execution_costs",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=details,
                message=f"Net return negative after costs: {net_return:.2%}",
            )
        elif cost_ratio > 0.5:  # Costs exceed 50% of gross return
            return ValidationCheck(
                name="execution_costs",
                passed=True,
                severity=ValidationSeverity.WARNING,
                details=details,
                message=f"High execution costs: {cost_ratio:.1%} of gross return",
            )
        else:
            return ValidationCheck(
                name="execution_costs",
                passed=True,
                severity=ValidationSeverity.NONE,
                details=details,
                message=f"Execution costs acceptable: net return {net_return:.2%}",
            )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agents/test_validation_analyst.py -k "TestParameterSensitivity or TestTimeStability or TestRegimeStability or TestExecutionCost" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/research_agents.py tests/test_agents/test_validation_analyst.py
git commit -m "feat(agents): implement ValidationAnalyst check methods"
```

---

## Task 5: Implement execute() Method

**Files:**
- Modify: `hrp/agents/research_agents.py`
- Test: `tests/test_agents/test_validation_analyst.py`

**Step 1: Write the failing test**

Add to `tests/test_agents/test_validation_analyst.py`:

```python
class TestValidationAnalystExecute:
    """Tests for ValidationAnalyst execute method."""

    @patch.object(ValidationAnalyst, "_get_hypotheses_to_validate")
    @patch.object(ValidationAnalyst, "_validate_hypothesis")
    @patch.object(ValidationAnalyst, "_write_research_note")
    @patch.object(ValidationAnalyst, "_log_agent_event")
    def test_execute_processes_hypotheses(
        self,
        mock_log,
        mock_write_note,
        mock_validate,
        mock_get_hypotheses,
    ):
        """Execute processes all hypotheses and returns report."""
        # Setup mocks
        mock_get_hypotheses.return_value = [
            {"hypothesis_id": "HYP-2026-001", "experiment_id": "exp_1"},
            {"hypothesis_id": "HYP-2026-002", "experiment_id": "exp_2"},
        ]
        mock_validate.return_value = HypothesisValidation(
            hypothesis_id="HYP-2026-001",
            experiment_id="exp_1",
            validation_date=date.today(),
            checks=[ValidationCheck(
                name="test",
                passed=True,
                severity=ValidationSeverity.NONE,
                details={},
                message="OK",
            )],
        )

        agent = ValidationAnalyst(send_alerts=False)
        result = agent.execute()

        assert result["hypotheses_validated"] == 2
        assert mock_validate.call_count == 2
        assert mock_write_note.called
        assert mock_log.called

    @patch.object(ValidationAnalyst, "_get_hypotheses_to_validate")
    def test_execute_no_hypotheses(self, mock_get_hypotheses):
        """Execute handles no hypotheses gracefully."""
        mock_get_hypotheses.return_value = []

        agent = ValidationAnalyst(send_alerts=False)
        result = agent.execute()

        assert result["hypotheses_validated"] == 0
        assert result["hypotheses_passed"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_validation_analyst.py::TestValidationAnalystExecute -v`
Expected: FAIL with "NotImplementedError"

**Step 3: Write implementation**

Replace the `execute()` method in `ValidationAnalyst`:

```python
    def execute(self) -> dict[str, Any]:
        """
        Run validation on hypotheses ready for final review.

        Returns:
            Dict with validation results summary
        """
        start_time = time.time()

        # 1. Get hypotheses to validate
        hypotheses = self._get_hypotheses_to_validate()

        # 2. Validate each hypothesis
        validations: list[HypothesisValidation] = []
        passed_count = 0
        failed_count = 0

        for hypothesis in hypotheses:
            validation = self._validate_hypothesis(hypothesis)
            validations.append(validation)

            if validation.overall_passed:
                passed_count += 1
                self._update_hypothesis_status(
                    validation.hypothesis_id,
                    "validated",
                    validation,
                )
            else:
                failed_count += 1
                self._update_hypothesis_status(
                    validation.hypothesis_id,
                    "validation_failed",
                    validation,
                )

            # Log per-hypothesis validation event
            self._log_agent_event(
                event_type=EventType.VALIDATION_ANALYST_REVIEW,
                hypothesis_id=validation.hypothesis_id,
                experiment_id=validation.experiment_id,
                details={
                    "overall_passed": validation.overall_passed,
                    "critical_count": validation.critical_count,
                    "warning_count": validation.warning_count,
                    "checks": [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "severity": c.severity.value,
                        }
                        for c in validation.checks
                    ],
                },
            )

        # 3. Log completion event
        duration = time.time() - start_time
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "hypotheses_validated": len(validations),
                "hypotheses_passed": passed_count,
                "hypotheses_failed": failed_count,
                "duration_seconds": duration,
            },
        )

        # 4. Write research note
        self._write_research_note(validations, duration)

        # 5. Send alerts if failures
        if self.send_alerts and failed_count > 0:
            self._send_alert_email(validations)

        # 6. Build report
        report = ValidationAnalystReport(
            report_date=date.today(),
            hypotheses_validated=len(validations),
            hypotheses_passed=passed_count,
            hypotheses_failed=failed_count,
            validations=validations,
            duration_seconds=duration,
        )

        return {
            "report_date": report.report_date.isoformat(),
            "hypotheses_validated": report.hypotheses_validated,
            "hypotheses_passed": report.hypotheses_passed,
            "hypotheses_failed": report.hypotheses_failed,
            "duration_seconds": report.duration_seconds,
        }

    def _get_hypotheses_to_validate(self) -> list[dict[str, Any]]:
        """
        Get hypotheses ready for validation.

        Returns hypotheses that:
        - Passed ML Quality Sentinel audit (no critical issues)
        - Are in 'testing' or 'audited' status
        """
        if self.hypothesis_ids:
            # Specific hypotheses requested
            return [
                self.api.get_hypothesis(hid)
                for hid in self.hypothesis_ids
                if self.api.get_hypothesis(hid) is not None
            ]

        # Get hypotheses that passed quality audit
        # Look for recent ML_QUALITY_SENTINEL_AUDIT events with overall_passed=True
        db = get_db()
        result = db.fetchall(
            """
            SELECT DISTINCT l.hypothesis_id
            FROM lineage l
            WHERE l.event_type = ?
              AND l.timestamp > datetime('now', '-7 days')
              AND json_extract(l.details, '$.overall_passed') = true
            """,
            (EventType.ML_QUALITY_SENTINEL_AUDIT.value,),
        )

        hypothesis_ids = [row[0] for row in result if row[0]]
        return [
            self.api.get_hypothesis(hid)
            for hid in hypothesis_ids
            if self.api.get_hypothesis(hid) is not None
        ]

    def _validate_hypothesis(
        self,
        hypothesis: dict[str, Any],
    ) -> HypothesisValidation:
        """
        Run all validation checks on a hypothesis.

        Args:
            hypothesis: Hypothesis dict with metadata

        Returns:
            HypothesisValidation with all check results
        """
        hypothesis_id = hypothesis.get("hypothesis_id", hypothesis.get("id", "unknown"))
        experiment_id = hypothesis.get("metadata", {}).get("experiment_id", "unknown")

        validation = HypothesisValidation(
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            validation_date=date.today(),
        )

        # Get experiment data for this hypothesis
        experiment_data = self._get_experiment_data(hypothesis)

        # 1. Parameter sensitivity check
        if "param_experiments" in experiment_data:
            check = self._check_parameter_sensitivity(
                experiment_data["param_experiments"],
                "baseline",
            )
            validation.add_check(check)

        # 2. Time stability check
        if "period_metrics" in experiment_data:
            check = self._check_time_stability(experiment_data["period_metrics"])
            validation.add_check(check)

        # 3. Regime stability check
        if "regime_metrics" in experiment_data:
            check = self._check_regime_stability(experiment_data["regime_metrics"])
            validation.add_check(check)

        # 4. Execution cost estimation
        if all(k in experiment_data for k in ["num_trades", "avg_trade_value", "gross_return"]):
            check = self._estimate_execution_costs(
                experiment_data["num_trades"],
                experiment_data["avg_trade_value"],
                experiment_data["gross_return"],
            )
            validation.add_check(check)

        return validation

    def _get_experiment_data(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """
        Gather experiment data needed for validation checks.

        This is a placeholder - actual implementation would query MLflow
        and run additional backtests for parameter sensitivity.
        """
        # In a full implementation, this would:
        # 1. Query MLflow for the hypothesis's experiments
        # 2. Run parameter variations if not already done
        # 3. Split returns into time periods
        # 4. Detect regimes and calculate regime metrics
        # 5. Extract trade statistics

        # For now, return data from hypothesis metadata if available
        metadata = hypothesis.get("metadata", {})
        return {
            "param_experiments": metadata.get("param_experiments", {}),
            "period_metrics": metadata.get("period_metrics", []),
            "regime_metrics": metadata.get("regime_metrics", {}),
            "num_trades": metadata.get("num_trades", 0),
            "avg_trade_value": metadata.get("avg_trade_value", 0),
            "gross_return": metadata.get("gross_return", 0),
        }

    def _update_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: str,
        validation: HypothesisValidation,
    ) -> None:
        """Update hypothesis status and metadata with validation results."""
        try:
            self.api.update_hypothesis(
                hypothesis_id=hypothesis_id,
                status=new_status,
                metadata={
                    "validation_analyst_review": {
                        "date": validation.validation_date.isoformat(),
                        "passed": validation.overall_passed,
                        "critical_count": validation.critical_count,
                        "warning_count": validation.warning_count,
                        "checks": [c.name for c in validation.checks],
                    }
                },
                actor=self.ACTOR,
            )
        except Exception as e:
            logger.warning(f"Failed to update hypothesis {hypothesis_id}: {e}")

    def _write_research_note(
        self,
        validations: list[HypothesisValidation],
        duration: float,
    ) -> None:
        """Write per-run validation report to docs/research/."""
        from pathlib import Path

        report_date = date.today().isoformat()
        filename = f"{report_date}-validation-analyst.md"
        filepath = Path("docs/research") / filename

        lines = [
            f"# Validation Analyst Report - {report_date}",
            "",
            "## Summary",
            f"- Hypotheses validated: {len(validations)}",
            f"- Passed: {sum(1 for v in validations if v.overall_passed)}",
            f"- Failed: {sum(1 for v in validations if not v.overall_passed)}",
            f"- Duration: {duration:.1f}s",
            "",
        ]

        for validation in validations:
            status = "PASSED" if validation.overall_passed else "FAILED"
            lines.extend([
                f"## {validation.hypothesis_id}: {status}",
                "",
                "| Check | Passed | Severity | Message |",
                "|-------|--------|----------|---------|",
            ])
            for check in validation.checks:
                passed_str = "Yes" if check.passed else "No"
                lines.append(
                    f"| {check.name} | {passed_str} | {check.severity.value} | {check.message} |"
                )
            lines.append("")

        lines.extend([
            "---",
            f"*Generated by Validation Analyst ({self.ACTOR})*",
        ])

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(lines))
        logger.info(f"Research note written to {filepath}")

    def _send_alert_email(self, validations: list[HypothesisValidation]) -> None:
        """Send alert email for validation failures."""
        try:
            failed = [v for v in validations if not v.overall_passed]
            if not failed:
                return

            notifier = EmailNotifier()
            subject = f"[HRP] Validation Analyst - {len(failed)} Hypothesis Validation Failures"

            body_lines = [
                "Validation Analyst detected hypothesis validation failures:",
                "",
            ]
            for v in failed:
                body_lines.append(f"- {v.hypothesis_id}: {v.critical_count} critical, {v.warning_count} warnings")

            notifier.send_notification(
                subject=subject,
                body="\n".join(body_lines),
            )
        except Exception as e:
            logger.warning(f"Failed to send alert email: {e}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agents/test_validation_analyst.py::TestValidationAnalystExecute -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/research_agents.py tests/test_agents/test_validation_analyst.py
git commit -m "feat(agents): implement ValidationAnalyst execute method"
```

---

## Task 6: Export ValidationAnalyst from Module

**Files:**
- Modify: `hrp/agents/__init__.py`
- Test: `tests/test_agents/test_validation_analyst.py`

**Step 1: Write the failing test**

Add to `tests/test_agents/test_validation_analyst.py`:

```python
def test_validation_analyst_exported():
    """ValidationAnalyst is exported from hrp.agents module."""
    from hrp.agents import (
        ValidationAnalyst,
        ValidationCheck,
        ValidationSeverity,
        HypothesisValidation,
        ValidationAnalystReport,
    )

    assert ValidationAnalyst is not None
    assert ValidationCheck is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_validation_analyst.py::test_validation_analyst_exported -v`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

Update `hrp/agents/__init__.py`:

```python
from hrp.agents.research_agents import (
    # ... existing imports ...
    ValidationAnalyst,
    ValidationCheck,
    ValidationSeverity,
    HypothesisValidation,
    ValidationAnalystReport,
)

__all__ = [
    # ... existing exports ...
    # Validation Analyst
    "ValidationAnalyst",
    "ValidationCheck",
    "ValidationSeverity",
    "HypothesisValidation",
    "ValidationAnalystReport",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_validation_analyst.py::test_validation_analyst_exported -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/__init__.py tests/test_agents/test_validation_analyst.py
git commit -m "feat(agents): export ValidationAnalyst from module"
```

---

## Task 7: Add Scheduler Integration

**Files:**
- Modify: `hrp/agents/scheduler.py`
- Test: `tests/test_agents/test_scheduler.py`

**Step 1: Write the failing test**

Add to `tests/test_agents/test_scheduler.py`:

```python
def test_setup_validation_analyst_trigger():
    """Scheduler can set up ValidationAnalyst lineage trigger."""
    scheduler = IngestionScheduler()
    scheduler.setup_research_agent_triggers()

    # Should have trigger for validation analyst
    triggers = [t for t in scheduler._lineage_triggers
                if t.callback_name == "_trigger_validation_analyst"]
    assert len(triggers) == 1
    assert triggers[0].actor_filter == "agent:ml-quality-sentinel"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_scheduler.py::test_setup_validation_analyst_trigger -v`
Expected: FAIL

**Step 3: Write implementation**

Add to `IngestionScheduler.setup_research_agent_triggers()` in `hrp/agents/scheduler.py`:

```python
    # ML Quality Sentinel → Validation Analyst
    self.register_lineage_trigger(
        event_type=EventType.AGENT_RUN_COMPLETE.value,
        actor_filter="agent:ml-quality-sentinel",
        callback=self._trigger_validation_analyst,
        callback_name="_trigger_validation_analyst",
    )

def _trigger_validation_analyst(self, event: dict) -> None:
    """Callback when ML Quality Sentinel completes."""
    from hrp.agents.research_agents import ValidationAnalyst

    logger.info("Triggering ValidationAnalyst after ML Quality Sentinel")

    # Get hypothesis IDs from the sentinel's completion event
    details = event.get("details", {})

    agent = ValidationAnalyst(
        hypothesis_ids=None,  # Validate all recently audited
        send_alerts=True,
    )

    try:
        agent.run()
    except Exception as e:
        logger.error(f"ValidationAnalyst failed: {e}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_scheduler.py::test_setup_validation_analyst_trigger -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/scheduler.py tests/test_agents/test_scheduler.py
git commit -m "feat(scheduler): add ValidationAnalyst lineage trigger"
```

---

## Task 8: Update CLAUDE.md and Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/plans/Project-Status.md`

**Step 1: Update CLAUDE.md**

Add to the "Common Tasks" section:

```markdown
### Run Validation Analyst for stress testing
```python
from hrp.agents import ValidationAnalyst

# Validate hypotheses that passed ML Quality Sentinel audit
analyst = ValidationAnalyst(
    param_sensitivity_threshold=0.5,  # Min ratio of varied/baseline Sharpe
    min_profitable_periods=0.67,      # 2/3 periods must be profitable
    min_profitable_regimes=2,         # At least 2 regimes profitable
    send_alerts=True,
)
result = analyst.run()

print(f"Validated: {result['hypotheses_validated']}")
print(f"Passed: {result['hypotheses_passed']}")
print(f"Failed: {result['hypotheses_failed']}")
# Research note written to docs/research/YYYY-MM-DD-validation-analyst.md
```

**Step 2: Update Project-Status.md**

Update the Research Agents table:

```markdown
| **Validation Analyst** | ✅ | Stress testing, parameter sensitivity (`hrp/agents/research_agents.py`) |
```

Update the pipeline diagram:

```
Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel → Validation Analyst
```

**Step 3: Commit**

```bash
git add CLAUDE.md docs/plans/Project-Status.md
git commit -m "docs: add ValidationAnalyst usage examples and update status"
```

---

## Task 9: Run Full Test Suite

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass

**Step 2: Commit any fixes if needed**

```bash
git add -A
git commit -m "fix: address test failures from ValidationAnalyst integration"
```

---

## Task 10: Create Agent Definition Document

**Files:**
- Create: `docs/plans/2026-01-26-validation-analyst-agent.md`

**Step 1: Create the definition file**

The file should follow the same format as other agent definitions. Content provided above in the "Agent Definition" section.

**Step 2: Commit**

```bash
git add docs/plans/2026-01-26-validation-analyst-agent.md
git commit -m "docs: add ValidationAnalyst agent definition"
```

---

## Verification Checklist

- [ ] `pytest tests/test_agents/test_validation_analyst.py -v` passes
- [ ] `pytest tests/ -v` all tests pass
- [ ] ValidationAnalyst imported from `hrp.agents`
- [ ] Scheduler triggers ValidationAnalyst after ML Quality Sentinel
- [ ] CLAUDE.md has usage example
- [ ] Project-Status.md shows ValidationAnalyst as done
- [ ] Agent definition document created

---

## Summary

This plan implements the Validation Analyst agent with:

1. **4 validation checks**: Parameter sensitivity, time stability, regime stability, execution costs
2. **Hybrid architecture**: Deterministic tests using existing `hrp/risk/robustness.py`
3. **Event-driven trigger**: Automatically runs after ML Quality Sentinel
4. **Research notes**: Writes markdown reports to `docs/research/`
5. **Email alerts**: Notifies on validation failures
6. **Full test coverage**: ~25 new tests
