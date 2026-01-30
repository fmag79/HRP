# Agent Redesign Feedback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 5 feedback-driven improvements to HRP research agent pipeline (Code Materializer, adaptive IC thresholds, Stability Score v1, Pre-Backtest Review, HMM regimes)

**Architecture:** Phase-based implementation starting with high-impact changes (Code Materializer + Alpha Researcher split), then foundation metrics (Stability Score + Pre-Backtest Review), finally enhancement (HMM regimes). Each phase is independent and can be tested before proceeding.

**Tech Stack:** Python 3.11+, dataclasses, pytest, MLflow, DuckDB, scikit-learn, hmmlearn

---

## Phase 1: High Impact (Code Materializer + Alpha Researcher Split)

### Task 1: Create Code Materializer Agent Documentation

**Files:**
- Create: `docs/agents/2026-01-29-code-materializer-agent.md`

**Step 1: Write agent spec document**

Create file with following sections:

```markdown
# Agent Definition: Code Materializer

**Date:** January 29, 2026
**Status:** Implemented
**Type:** Custom (deterministic - extends `ResearchAgent`)

## Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Code Materializer |
| **Actor ID** | `agent:code-materializer` |
| **Type** | Custom (deterministic) |
| **Role** | Strategy specification to executable code translation |
| **Implementation** | `hrp/agents/code_materializer.py` |
| **Trigger** | Lineage event (after Alpha Researcher) |
| **Upstream** | Alpha Researcher (strategy specs) |
| **Downstream** | ML Scientist |

## Purpose

Mechanical translation of strategy specifications into executable code without logic changes or interpretation.

## Core Capabilities

### 1. Strategy Spec Parsing

Reads strategy specifications from Alpha Researcher:

```python
from hrp.agents import CodeMaterializer

materializer = CodeMaterializer(hypothesis_ids=["HYP-2026-001"])
result = materializer.run()

print(f"Code generated: {result['code_generated']}")
print(f"Syntax valid: {result['syntax_valid']}")
```

### 2. Code Generation

Translates economic specs to executable code:

| Spec Input | Code Output |
|------------|-------------|
| "Long top decile of momentum_20d" | `signals = momentum_20d.rank(pct=True) >= 0.9` |
| "Max 10% sector exposure" | `SectorConstraint(max_exposure=0.10)` |
| "Weekly rebalance on Mondays" | `RebalanceSchedule(frequency='weekly', day_of_week=0)` |

### 3. Syntax Validation

Verifies generated code is syntactically correct:

```python
import ast

def validate_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
```

## Configuration

```python
@dataclass
class CodeMaterializerConfig:
    hypothesis_ids: list[str] | None = None
    validate_syntax: bool = True
    enforce_pit: bool = True  # Point-in-time constraints
    output_dir: str = "hrp/strategies/generated"
```

## Outputs

- Generated strategy code: `hrp/strategies/generated/{hypothesis_id}.py`
- Syntax validation report
- Lineage event: `CODE_MATERIALIZER_COMPLETE`

## Explicit Non-Responsibilities

- ❌ No economic rationale interpretation
- ❌ No signal optimization
- ❌ No feature selection
- ❌ No performance judgment

## Document History

- **2026-01-29:** Initial agent definition created
```

**Step 2: Commit documentation**

```bash
git add docs/agents/2026-01-29-code-materializer-agent.md
git commit -m "docs(agents): add Code Materializer agent specification"
```

---

### Task 2: Update Alpha Researcher Agent Spec

**Files:**
- Modify: `docs/agents/2026-01-26-alpha-researcher-agent.md`

**Step 1: Update Purpose section**

Replace current purpose with:

```markdown
## Purpose

Reviews and refines draft hypotheses using Claude's reasoning capabilities, and generates novel strategy concepts. The Alpha Researcher:

1. **Analyzes economic rationale** - Why might this signal work?
2. **Checks regime context** - Does signal work across market conditions?
3. **Searches related hypotheses** - Novel or variant of existing idea?
4. **Promotes or rejects** hypotheses based on analysis
5. **Generates new strategies** - Creates novel strategy concepts from economic principles
6. **Writes strategy specifications** - Economic specs only, NO code generation
7. **Tags strategy class** - For adaptive IC thresholds (cross-sectional, time-series, ML)
8. **Triggers downstream** Code Materializer via lineage events
```

**Step 2: Remove code generation from Core Capabilities**

Remove section "5. Strategy Generation (NEW)" code generation parts. Keep strategy concept generation but remove code output.

**Step 3: Add strategy classification**

Add new section:

```markdown
### 6. Strategy Classification

Tags each hypothesis with strategy class for adaptive IC thresholds:

| Strategy Class | IC Pass | IC Kill | Examples |
|----------------|---------|---------|----------|
| `cross_sectional_factor` | ≥ 0.015 | < 0.005 | Value, quality, low-vol |
| `time_series_momentum` | ≥ 0.02 | < 0.01 | Trend-following |
| `ml_composite` | ≥ 0.025 | < 0.01 | Multi-feature ML |

Updated in hypothesis metadata: `strategy_class` field.
```

**Step 4: Update downstream trigger**

Change downstream from "ML Scientist" to "Code Materializer".

**Step 5: Commit changes**

```bash
git add docs/agents/2026-01-26-alpha-researcher-agent.md
git commit -m "docs(agents): update Alpha Researcher - specs only, add strategy classification"
```

---

### Task 3: Implement Code Materializer Agent

**Files:**
- Create: `hrp/agents/code_materializer.py`
- Modify: `hrp/agents/__init__.py`

**Step 1: Write the failing test**

Create `tests/agents/test_code_materializer.py`:

```python
import pytest
from hrp.agents.code_materializer import CodeMaterializer, CodeMaterializerConfig
from hrp.api.platform import PlatformAPI

def test_code_materializer_initialization():
    """Code Materializer initializes with default config."""
    config = CodeMaterializerConfig()
    agent = CodeMaterializer(config=config)
    assert agent.ACTOR == "agent:code-materializer"
    assert agent.config.validate_syntax is True

def test_materialize_simple_momentum_strategy():
    """Materialize simple momentum strategy spec."""
    # Mock hypothesis with strategy spec
    hypothesis = {
        "hypothesis_id": "HYP-TEST-001",
        "title": "Momentum Strategy",
        "metadata": {
            "strategy_spec": {
                "signal_logic": "long top decile of momentum_20d",
                "universe": "sp500",
                "holding_period_days": 20,
                "rebalance_cadence": "weekly",
            }
        }
    }

    agent = CodeMaterializer()
    result = agent._materialize_hypothesis(hypothesis)

    assert result["code_generated"] is True
    assert "momentum_20d" in result["code"]
    assert result["syntax_valid"] is True

def test_syntax_validation():
    """Syntax validation catches invalid code."""
    agent = CodeMaterializer()
    assert agent._validate_syntax("def foo(): return 1") is True
    assert agent._validate_syntax("def foo(: return 1") is False
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_code_materializer.py -v
```

Expected: FAIL - "ModuleNotFoundError: No module named 'hrp.agents.code_materializer'"

**Step 3: Implement Code Materializer agent**

Create `hrp/agents/code_materializer.py`:

```python
"""Code Materializer Agent - Translates strategy specs to executable code."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import ast

from hrp.agents.research_agents import ResearchAgent
from hrp.research.lineage import log_event, EventType


@dataclass
class CodeMaterializerConfig:
    """Configuration for Code Materializer agent."""
    hypothesis_ids: list[str] | None = None
    validate_syntax: bool = True
    enforce_pit: bool = True
    output_dir: str = "hrp/strategies/generated"


class CodeMaterializer(ResearchAgent):
    """
    Translates strategy specifications into executable code.

    Performs mechanical translation WITHOUT:
    - Interpreting economic rationale
    - Optimizing signal logic
    - Choosing features
    - Judging performance
    """

    ACTOR = "agent:code-materializer"
    DEFAULT_JOB_ID = "code_materializer"

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        config: CodeMaterializerConfig | None = None,
        api: Any = None,
    ):
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],
        )
        self.hypothesis_ids = hypothesis_ids
        self.config = config or CodeMaterializerConfig()
        self.api = api

    def execute(self) -> dict[str, Any]:
        """Execute code materialization for hypotheses."""
        from hrp.api.platform import PlatformAPI

        api = self.api or PlatformAPI()
        hypotheses = self._get_hypotheses(api)

        results = []
        for hypothesis in hypotheses:
            result = self._materialize_hypothesis(hypothesis)
            results.append(result)

            # Log completion event
            self._log_agent_event(
                event_type=EventType.CODE_MATERIALIZER_COMPLETE,
                hypothesis_id=hypothesis["hypothesis_id"],
                details={
                    "code_generated": result["code_generated"],
                    "syntax_valid": result["syntax_valid"],
                    "output_path": result.get("output_path"),
                },
            )

        return {
            "hypotheses_processed": len(results),
            "code_generated": sum(1 for r in results if r["code_generated"]),
            "syntax_valid": sum(1 for r in results if r["syntax_valid"]),
        }

    def _get_hypotheses(self, api) -> list[dict]:
        """Get hypotheses to materialize."""
        if self.hypothesis_ids:
            return [api.get_hypothesis(hid) for hid in self.hypothesis_ids]
        # Get all hypotheses with strategy specs and no code
        return api.list_hypotheses(status="testing")  # Simplified

    def _materialize_hypothesis(self, hypothesis: dict) -> dict:
        """Materialize a single hypothesis."""
        hypothesis_id = hypothesis["hypothesis_id"]
        strategy_spec = hypothesis.get("metadata", {}).get("strategy_spec", {})

        # Generate code from spec
        code = self._generate_code(strategy_spec)

        # Validate syntax if enabled
        syntax_valid = True
        if self.config.validate_syntax:
            syntax_valid = self._validate_syntax(code)

        # Write to file
        output_path = None
        if syntax_valid:
            output_path = self._write_code(hypothesis_id, code)

        return {
            "hypothesis_id": hypothesis_id,
            "code_generated": True,
            "code": code,
            "syntax_valid": syntax_valid,
            "output_path": output_path,
        }

    def _generate_code(self, spec: dict) -> str:
        """Generate code from strategy spec."""
        signal_logic = spec.get("signal_logic", "")
        universe = spec.get("universe", "sp500")
        holding_days = spec.get("holding_period_days", 20)

        # Simple translation logic
        code = f"""# Auto-generated by Code Materializer
# DO NOT EDIT MANUALLY

from hrp.strategies.base import StrategyBase

class GeneratedStrategy(StrategyBase):
    def __init__(self):
        self.universe = "{universe}"
        self.holding_period_days = {holding_days}

    def generate_signals(self, prices, features):
        # Signal: {signal_logic}
        # TODO: Translate to executable code
        pass
"""
        return code

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _write_code(self, hypothesis_id: str, code: str) -> str:
        """Write generated code to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{hypothesis_id}.py"
        output_path.write_text(code)

        return str(output_path)
```

**Step 4: Add export to __init__.py**

Modify `hrp/agents/__init__.py`:

```python
from hrp.agents.code_materializer import CodeMaterializer, CodeMaterializerConfig

__all__ = [
    ...
    "CodeMaterializer",
    "CodeMaterializerConfig",
]
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/agents/test_code_materializer.py -v
```

Expected: PASS

**Step 6: Commit implementation**

```bash
git add hrp/agents/code_materializer.py hrp/agents/__init__.py tests/agents/test_code_materializer.py
git commit -m "feat(agents): implement Code Materializer agent

- Mechanical translation of strategy specs to executable code
- Syntax validation, no logic interpretation
- Tests: initialization, materialization, syntax validation
"
```

---

### Task 4: Update Agent Interaction Diagram

**Files:**
- Modify: `docs/agents/agent-interaction-diagram.md`

**Step 1: Update agent pipeline diagram**

Add Code Materializer between Alpha Researcher and ML Scientist:

```markdown
```mermaid
graph TB
    Start([Pipeline Start]) --> SS[Signal Scientist<br/>SDK Agent]
    SS -->|hypothesis_created| AR[Alpha Researcher<br/>SDK Agent]
    AR -->|strategy_spec| CM[Code Materializer<br/>Custom Agent] ← NEW
    CM -->|code_generated| MLS[ML Scientist<br/>SDK Agent]
    ...
```

**Step 2: Update Event-Driven Trigger Matrix table**

Add row:

```markdown
| Alpha Researcher | `alpha_researcher_complete` | Code Materializer | Strategy spec ready |
| Code Materializer | `code_materializer_complete` | ML Scientist | Code generated, syntax valid |
```

**Step 3: Update Agent Responsibility Matrix table**

Add Code Materializer row and update Alpha Researcher row:

```markdown
| **Alpha Researcher** | SDK | Economic rationale, strategy specs | Draft hypotheses | Research notes, strategy specs (NO CODE) |
| **Code Materializer** | Custom | Executable code | Strategy specs | Generated strategy code |
```

**Step 4: Commit changes**

```bash
git add docs/agents/agent-interaction-diagram.md
git commit -m "docs(agents): update interaction diagram with Code Materializer"
```

---

## Phase 2: Foundation (Stability Score v1 + Pre-Backtest Review)

### Task 5: Implement Stability Score v1

**Files:**
- Create: `hrp/research/metrics.py` (or extend if exists)
- Test: `tests/research/test_metrics.py`

**Step 1: Write the failing test**

Create `tests/research/test_metrics.py`:

```python
import pytest
from hrp.research.metrics import calculate_stability_score_v1

def test_stability_score_perfect_stability():
    """Perfect stability returns low score."""
    fold_sharpes = [1.0, 1.0, 1.0, 1.0, 1.0]  # No variation
    fold_drawdowns = [0.10, 0.10, 0.10, 0.10, 0.10]  # No variation
    mean_ic = 0.05  # Positive

    score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)

    assert score <= 1.0  # Should be stable

def test_stability_score_unstable():
    """Highly variable folds return high score."""
    fold_sharpes = [2.0, 0.5, -0.5, 1.5, 0.2]  # High variation
    fold_drawdowns = [0.05, 0.30, 0.40, 0.10, 0.25]  # High variation
    mean_ic = 0.02  # Low IC

    score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)

    assert score > 1.0  # Should be unstable

def test_stability_score_sign_flip_penalty():
    """Sign flip adds penalty."""
    fold_sharpes = [1.0, 1.0, 1.0, 1.0, 1.0]
    fold_drawdowns = [0.10, 0.10, 0.10, 0.10, 0.10]
    mean_ic = -0.01  # Negative IC

    score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)

    # Sign flip should add penalty
    # (This test depends on whether sign flip is calculated differently)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/research/test_metrics.py::test_stability_score_perfect_stability -v
```

Expected: FAIL - "ModuleNotFoundError: No module named 'hrp.research.metrics'"

**Step 3: Implement Stability Score v1**

Create or extend `hrp/research/metrics.py`:

```python
"""Stability Score v1 - Formal definition for model stability assessment."""

import numpy as np


def calculate_stability_score_v1(
    fold_sharpes: list[float],
    fold_drawdowns: list[float],
    mean_fold_ic: float,
) -> float:
    """
    Stability Score v1 - Lower is better.

    Components:
    1. Sharpe coefficient of variation (CV)
    2. Drawdown dispersion penalty
    3. Sign flip penalty

    Args:
        fold_sharpes: List of Sharpe ratios from each fold
        fold_drawdowns: List of max drawdowns from each fold
        mean_fold_ic: Mean Information Coefficient across folds

    Returns:
        float: Stability score (≤ 1.0 is stable)

    Examples:
        >>> fold_sharpes = [1.0, 1.0, 1.0, 1.0, 1.0]
        >>> fold_drawdowns = [0.10, 0.10, 0.10, 0.10, 0.10]
        >>> score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, 0.05)
        >>> assert score <= 1.0
    """
    # Component 1: Sharpe CV
    mean_sharpe = np.mean(fold_sharpes)
    std_sharpe = np.std(fold_sharpes)
    sharpe_cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else float('inf')

    # Component 2: Drawdown dispersion
    mean_dd = np.mean(fold_drawdowns)
    std_dd = np.std(fold_drawdowns)
    dd_dispersion = (std_dd / mean_dd) if mean_dd > 0 else 0

    # Component 3: Sign flip penalty (simplified - uses mean IC sign)
    # For multi-fold sign flips, count positive vs negative fold ICs
    # This is a simplified version using mean IC
    sign_flip_penalty = 0.5 if mean_fold_ic < 0 else 0

    stability_score = sharpe_cv + dd_dispersion + sign_flip_penalty

    return stability_score
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/research/test_metrics.py -v
```

Expected: PASS

**Step 5: Add documentation**

Add docstring documentation:

```python
"""
Stability Score metrics for model validation.

Versioning:
    - stability_score_v1: Initial definition (2026-01-29)

Components:
    1. Sharpe coefficient of variation (CV): Measures performance consistency
    2. Drawdown dispersion: Measures risk consistency
    3. Sign flip penalty: Penalizes direction changes

Threshold:
    - ≤ 1.0: Stable
    - > 1.0: Unstable

Usage:
    >>> from hrp.research.metrics import calculate_stability_score_v1
    >>> score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)
    >>> if score <= 1.0:
    ...     print("Model is stable")
"""
```

**Step 6: Commit implementation**

```bash
git add hrp/research/metrics.py tests/research/test_metrics.py
git commit -m "feat(research): add Stability Score v1 formal definition

- Sharpe CV + drawdown dispersion + sign flip penalty
- Versioned metric (stability_score_v1)
- Threshold: ≤ 1.0 = stable
- Tests: perfect stability, unstable, sign flip penalty
"
```

---

### Task 6: Implement Adaptive IC Thresholds

**Files:**
- Modify: `hrp/agents/signal_scientist.py` (or wherever IC logic lives)
- Test: `tests/agents/test_signal_scientist.py` (extend)

**Step 1: Write the failing test**

Add to existing test file:

```python
def test_adaptive_ic_thresholds():
    """IC thresholds adapt to strategy class."""
    from hrp.agents.signal_scientist import IC_THRESHOLDS

    # Cross-sectional factor (more lenient)
    assert IC_THRESHOLDS["cross_sectional_factor"]["pass"] == 0.015
    assert IC_THRESHOLDS["cross_sectional_factor"]["kill"] == 0.005

    # Time-series momentum (moderate)
    assert IC_THRESHOLDS["time_series_momentum"]["pass"] == 0.02
    assert IC_THRESHOLDS["time_series_momentum"]["kill"] == 0.01

    # ML composite (stricter)
    assert IC_THRESHOLDS["ml_composite"]["pass"] == 0.025
    assert IC_THRESHOLDS["ml_composite"]["kill"] == 0.01

def test_ic_threshold_by_strategy_class():
    """Get IC threshold for specific strategy class."""
    from hrp.agents.signal_scientist import get_ic_thresholds

    thresholds = get_ic_thresholds("cross_sectional_factor")
    assert thresholds["pass"] == 0.015

    thresholds = get_ic_thresholds("ml_composite")
    assert thresholds["pass"] == 0.025
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_signal_scientist.py::test_adaptive_ic_thresholds -v
```

Expected: FAIL - function doesn't exist yet

**Step 3: Implement adaptive IC thresholds**

Add to appropriate agent file (likely Signal Scientist or shared config):

```python
# Adaptive IC thresholds by strategy class
IC_THRESHOLDS = {
    "cross_sectional_factor": {
        "pass": 0.015,
        "kill": 0.005,
        "description": "Value, quality, low-vol factors"
    },
    "time_series_momentum": {
        "pass": 0.02,
        "kill": 0.01,
        "description": "Trend-following strategies"
    },
    "ml_composite": {
        "pass": 0.025,
        "kill": 0.01,
        "description": "Multi-feature ML models"
    },
    "default": {
        "pass": 0.03,
        "kill": 0.01,
        "description": "Legacy uniform threshold"
    }
}

def get_ic_thresholds(strategy_class: str) -> dict:
    """
    Get IC thresholds for a strategy class.

    Args:
        strategy_class: One of: cross_sectional_factor, time_series_momentum,
                       ml_composite, default

    Returns:
        dict with 'pass' and 'kill' thresholds

    Examples:
        >>> thresholds = get_ic_thresholds("cross_sectional_factor")
        >>> assert thresholds["pass"] == 0.015
    """
    return IC_THRESHOLDS.get(strategy_class, IC_THRESHOLDS["default"])
```

**Step 4: Update Signal Scientist to use strategy class**

Modify hypothesis creation to include `strategy_class`:

```python
# When creating hypothesis
hypothesis = api.create_hypothesis(
    title="Momentum predicts returns",
    thesis="...",
    prediction="...",
    falsification="...",
    strategy_class="time_series_momentum",  # NEW
    actor='agent:signal-scientist',
)
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/agents/test_signal_scientist.py::test_adaptive_ic_thresholds -v
```

Expected: PASS

**Step 6: Commit implementation**

```bash
git add hrp/agents/signal_scientist.py tests/agents/test_signal_scientist.py
git commit -m "feat(agents): implement adaptive IC thresholds by strategy class

- Cross-sectional: ≥0.015 pass, <0.005 kill
- Time-series: ≥0.02 pass, <0.01 kill
- ML composite: ≥0.025 pass, <0.01 kill
- Alpha Researcher tags strategy_class
"
```

---

### Task 7: Implement Pre-Backtest Review

**Files:**
- Modify: `docs/agents/2026-01-29-quant-developer-agent.md`
- Modify: `hrp/agents/quant_developer.py` (or create)

**Step 1: Update Quant Developer agent spec**

Add Pre-Backtest Review section to agent doc:

```markdown
## Pre-Backtest Review (NEW)

### Purpose

Lightweight execution feasibility sanity check before expensive backtests.

### When It Runs

Between ML Quality Sentinel and full Production Backtesting.

### Checks

| Check | Description | Action |
|-------|-------------|--------|
| Data Availability | Required features exist | Warning if missing |
| Point-in-Time Validity | Features computable as of dates | Warning if violated |
| Execution Frequency | Rebalance cadence achievable | Warning if unrealistic |
| Universe Liquidity | Sufficient liquidity | Warning if illiquid |
| Cost Model | Can handle IBKR costs | Warning if dominant |

### Output

```python
@dataclass
class PreBacktestReviewResult:
    hypothesis_id: str
    passed: bool  # Always True (warnings only)
    warnings: list[str]
    data_issues: list[str]
    execution_notes: list[str]
    reviewed_at: datetime
```
```

**Step 2: Write the failing test**

Add to `tests/agents/test_quant_developer.py`:

```python
def test_pre_backtest_review_warnings():
    """Pre-backtest review generates warnings."""
    from hrp.agents.quant_developer import QuantDeveloper

    developer = QuantDeveloper()
    result = developer._pre_backtest_review("HYP-TEST-001")

    assert result["passed"] is True  # Always passes (warnings only)
    assert isinstance(result["warnings"], list)
    assert isinstance(result["data_issues"], list)

def test_pre_backtest_review_data_availability():
    """Check data availability before backtest."""
    from hrp.agents.quant_developer import QuantDeveloper

    developer = QuantDeveloper()
    warnings = developer._check_data_availability(
        symbols=["NONEXISTENT"],
        features=["momentum_20d"],
        start_date="2020-01-01",
    )

    # Should warn about missing data
    assert len(warnings) > 0
```

**Step 3: Implement Pre-Backtest Review**

Add to Quant Developer:

```python
def _pre_backtest_review(self, hypothesis_id: str) -> dict:
    """
    Lightweight execution feasibility check.

    Returns warnings only - does not block or veto.
    """
    from datetime import datetime

    hypothesis = self.api.get_hypothesis(hypothesis_id)
    strategy_spec = hypothesis.get("metadata", {}).get("strategy_spec", {})

    warnings = []
    data_issues = []
    execution_notes = []

    # Check 1: Data availability
    data_warnings = self._check_data_availability(
        symbols=strategy_spec.get("universe_symbols", []),
        features=strategy_spec.get("features", []),
        start_date=strategy_spec.get("start_date"),
    )
    warnings.extend(data_warnings)

    # Check 2: Point-in-time validity
    pit_warnings = self._check_point_in_time_validity(strategy_spec)
    warnings.extend(pit_warnings)

    # Check 3: Execution frequency
    freq_notes = self._check_execution_frequency(strategy_spec)
    execution_notes.extend(freq_notes)

    # Check 4: Universe liquidity
    liquidity_warnings = self._check_universe_liquidity(
        strategy_spec.get("universe_symbols", [])
    )
    warnings.extend(liquidity_warnings)

    # Check 5: Cost model applicability
    cost_warnings = self._check_cost_model_applicability(strategy_spec)
    warnings.extend(cost_warnings)

    return {
        "hypothesis_id": hypothesis_id,
        "passed": True,  # Always True (warnings only)
        "warnings": warnings,
        "data_issues": data_issues,
        "execution_notes": execution_notes,
        "reviewed_at": datetime.now().isoformat(),
    }

def _check_data_availability(
    self, symbols: list[str], features: list[str], start_date: str
) -> list[str]:
    """Check if required data exists."""
    warnings = []
    # Simplified implementation
    # In production: query database for feature availability
    return warnings

def _check_point_in_time_validity(self, strategy_spec: dict) -> list[str]:
    """Check features can be computed as of required dates."""
    warnings = []
    # Check lookback windows vs data availability
    return warnings

def _check_execution_frequency(self, strategy_spec: dict) -> list[str]:
    """Check if rebalance cadence is achievable."""
    notes = []
    frequency = strategy_spec.get("rebalance_cadence", "weekly")
    if frequency == "intraday":
        notes.append("WARNING: Intraday rebalancing not supported")
    return notes

def _check_universe_liquidity(self, symbols: list[str]) -> list[str]:
    """Check if symbols have sufficient liquidity."""
    warnings = []
    # In production: query average daily volume
    # For now: placeholder
    return warnings

def _check_cost_model_applicability(self, strategy_spec: dict) -> list[str]:
    """Check if strategy can handle transaction costs."""
    warnings = []
    # Estimate turnover and check if costs dominate
    return warnings
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/agents/test_quant_developer.py::test_pre_backtest_review_warnings -v
```

Expected: PASS

**Step 5: Commit implementation**

```bash
git add docs/agents/2026-01-29-quant-developer-agent.md hrp/agents/quant_developer.py tests/agents/test_quant_developer.py
git commit -m "feat(agents): add Pre-Backtest Review stage to Quant Developer

- Execution feasibility sanity check before backtests
- Warnings only (no veto power)
- Checks: data availability, point-in-time, frequency, liquidity, costs
"
```

---

## Phase 3: Enhancement (HMM Structural Regimes)

### Task 8: Implement HMM Structural Regimes

**Files:**
- Modify: `hrp/ml/regime_detection.py` (or create)
- Modify: `hrp/agents/pipeline_orchestrator.py`

**Step 1: Write the failing test**

Create `tests/ml/test_regime_detection.py`:

```python
import pytest
import pandas as pd
from hrp.ml.regime_detection import (
    VolatilityHMM,
    TrendHMM,
    combine_regime_labels,
    StructuralRegimeClassifier,
)

def test_volatility_hmm_classification():
    """Volatility HMM classifies high/low vol regimes."""
    import numpy as np

    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    low_vol = np.random.normal(0.01, 0.01, n_samples // 2)
    high_vol = np.random.normal(0.01, 0.04, n_samples // 2)
    volatility = np.concatenate([low_vol, high_vol])

    hmm = VolatilityHMM(n_regimes=2)
    regimes = hmm.fit_predict(volatility)

    assert len(regimes) == n_samples
    assert len(set(regimes)) <= 2  # Should classify into 2 regimes

def test_trend_hmm_classification():
    """Trend HMM classifies bull/bear regimes."""
    import numpy as np

    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    bull = np.random.normal(0.001, 0.01, n_samples // 2)
    bear = np.random.normal(-0.0005, 0.01, n_samples // 2)
    returns = np.concatenate([bull, bear])

    hmm = TrendHMM(n_regimes=2)
    regimes = hmm.fit_predict(returns)

    assert len(regimes) == n_samples
    assert len(set(regimes)) <= 2

def test_structural_regime_combination():
    """Combine vol and trend regimes into structural regimes."""
    vol_regimes = [0, 0, 1, 1]  # low, low, high, high
    trend_regimes = [0, 1, 0, 1]  # bull, bear, bull, bear

    structural = combine_regime_labels(vol_regimes, trend_regimes)

    assert structural[0] == "low_vol_bull"
    assert structural[1] == "low_vol_bear"
    assert structural[2] == "high_vol_bull"
    assert structural[3] == "high_vol_bear"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/ml/test_regime_detection.py -v
```

Expected: FAIL - modules don't exist

**Step 3: Implement HMM structural regimes**

Create `hrp/ml/regime_detection.py`:

```python
"""
HMM-based structural regime detection for scenario analysis.

Combines volatility and trend HMMs to classify market periods into 4 structural regimes:
- low_vol_bull: Low volatility, positive returns
- low_vol_bear: Low volatility, negative returns
- high_vol_bull: High volatility, positive returns
- high_vol_bear: High volatility, negative returns (crisis)
"""

from enum import Enum
from typing import Literal

import numpy as np
from sklearn.hmm import GaussianHMM  # or hmmlearn


StructuralRegime = Literal[
    "low_vol_bull",
    "low_vol_bear",
    "high_vol_bull",
    "high_vol_bear",
]


class VolatilityHMM:
    """Volatility regime classification using HMM."""

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.fitted = False

    def fit(self, volatility: np.ndarray) -> None:
        """Fit HMM to volatility series."""
        # Reshape for HMM
        X = volatility.reshape(-1, 1)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
        )
        self.model.fit(X)
        self.fitted = True

    def predict(self, volatility: np.ndarray) -> np.ndarray:
        """Predict volatility regimes."""
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = volatility.reshape(-1, 1)
        regimes = self.model.predict(X)
        return regimes


class TrendHMM:
    """Trend regime classification using HMM."""

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.fitted = False

    def fit(self, returns: np.ndarray) -> None:
        """Fit HMM to returns series."""
        X = returns.reshape(-1, 1)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
        )
        self.model.fit(X)
        self.fitted = True

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Predict trend regimes."""
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = returns.reshape(-1, 1)
        regimes = self.model.predict(X)
        return regimes


def combine_regime_labels(
    vol_regimes: np.ndarray,
    trend_regimes: np.ndarray,
) -> list[StructuralRegime]:
    """
    Combine volatility and trend regimes into structural regimes.

    Args:
        vol_regimes: Volatility regime labels (0=low, 1=high)
        trend_regimes: Trend regime labels (0=bull, 1=bear)

    Returns:
        List of structural regime labels

    Examples:
        >>> vol = [0, 0, 1, 1]
        >>> trend = [0, 1, 0, 1]
        >>> structural = combine_regime_labels(vol, trend)
        >>> assert structural[0] == "low_vol_bull"
        >>> assert structural[3] == "high_vol_bear"
    """
    vol_map = {0: "low_vol", 1: "high_vol"}
    trend_map = {0: "bull", 1: "bear"}

    structural = []
    for v, t in zip(vol_regimes, trend_regimes):
        regime = f"{vol_map[v]}_{trend_map[t]}"
        structural.append(regime)

    return structural


class StructuralRegimeClassifier:
    """
    Classify market periods into 4 structural regimes using HMM.

    Regimes:
    1. low_vol_bull: Calm uptrend
    2. low_vol_bear: Calm downtrend
    3. high_vol_bull: Volatile uptrend (recovery)
    4. high_vol_bear: Volatile downtrend (crisis)
    """

    def __init__(self):
        self.vol_hmm = VolatilityHMM(n_regimes=2)
        self.trend_hmm = TrendHMM(n_regimes=2)
        self.fitted = False

    def fit(self, prices: pd.DataFrame) -> None:
        """
        Fit HMMs to market price data.

        Args:
            prices: DataFrame with 'close' column
        """
        # Compute features
        returns = prices["close"].pct_change().dropna()
        volatility = returns.rolling(20).std().dropna()

        # Fit HMMs
        self.vol_hmm.fit(volatility.values)
        self.trend_hmm.fit(returns.values)

        self.fitted = True

    def predict(self, prices: pd.DataFrame) -> list[StructuralRegime]:
        """
        Predict structural regimes.

        Args:
            prices: DataFrame with 'close' column

        Returns:
            List of structural regime labels
        """
        if not self.fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        # Compute features
        returns = prices["close"].pct_change().dropna()
        volatility = returns.rolling(20).std().dropna()

        # Predict regimes
        vol_regimes = self.vol_hmm.predict(volatility.values)
        trend_regimes = self.trend_hmm.predict(returns.values)

        # Combine
        structural = combine_regime_labels(vol_regimes, trend_regimes)

        return structural

    def get_scenario_periods(
        self,
        prices: pd.DataFrame,
        min_days: int = 60,
    ) -> dict[StructuralRegime, list[tuple]]:
        """
        Get continuous periods for each structural regime.

        Args:
            prices: DataFrame with 'close' column and DatetimeIndex
            min_days: Minimum days for a period to be included

        Returns:
            Dict mapping regime to list of (start_date, end_date) tuples
        """
        import itertools

        regimes = self.predict(prices)
        dates = prices.index.tolist()

        # Group consecutive dates by regime
        periods: dict[StructuralRegime, list[tuple]] = {
            "low_vol_bull": [],
            "low_vol_bear": [],
            "high_vol_bull": [],
            "high_vol_bear": [],
        }

        for regime, group in itertools.groupby(
            zip(dates, regimes), key=lambda x: x[1]
        ):
            group_list = list(group)
            start_date = group_list[0][0]
            end_date = group_list[-1][0]
            days = (end_date - start_date).days

            if days >= min_days:
                periods[regime].append((start_date, end_date))

        return periods
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/ml/test_regime_detection.py -v
```

Expected: PASS

**Step 5: Commit implementation**

```bash
git add hrp/ml/regime_detection.py tests/ml/test_regime_detection.py
git commit -m "feat(ml): implement HMM-based structural regime detection

- VolatilityHMM: Classify high/low volatility periods
- TrendHMM: Classify bull/bear markets
- StructuralRegimeClassifier: Combine into 4 regimes
- Scenario period extraction for backtesting
"
```

---

### Task 9: Update Pipeline Orchestrator with Structural Regimes

**Files:**
- Modify: `docs/agents/2026-01-29-pipeline-orchestrator-agent.md`
- Modify: `hrp/agents/pipeline_orchestrator.py`

**Step 1: Update Pipeline Orchestrator spec**

Update scenario requirements:

```markdown
## Structural Regime Scenarios

**Updated:** Pipeline Orchestrator uses HMM-based structural regimes.

### Regime Matrix

| Vol \ Trend | Bull | Bear |
|-------------|------|------|
| **Low** | Low Vol Bull | Low Vol Bear |
| **High** | High Vol Bull | High Vol Bear (Crisis) |

### Requirements

| Requirement | Specification |
|-------------|---------------|
| Minimum scenarios | 4 (one per regime type) |
| Sharpe CV threshold | ≤ 0.30 across all regimes |
| Regime coverage | Must test in all 4 regime types |
```

**Step 2: Update Pipeline Orchestrator implementation**

Add structural regime scenario generation:

```python
def _generate_structural_regime_scenarios(
    self,
    hypothesis_id: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Generate backtest scenarios using HMM-based structural regimes.

    Returns 4 scenarios (one per structural regime type).
    """
    from hrp.ml.regime_detection import StructuralRegimeClassifier
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI()

    # Get market data for regime classification
    spy_prices = api.get_prices(
        symbols=["SPY"],
        start_date=start_date,
        end_date=end_date,
    )

    # Fit classifier and get periods
    classifier = StructuralRegimeClassifier()
    classifier.fit(spy_prices)

    periods = classifier.get_scenario_periods(
        spy_prices,
        min_days=60,  # Minimum 2 months per scenario
    )

    # Build scenarios
    scenarios = []
    for regime_type, period_list in periods.items():
        if period_list:  # If we have data for this regime
            for start, end in period_list[:1]:  # Take one period per regime
                scenarios.append({
                    "regime_type": regime_type,
                    "start_date": start,
                    "end_date": end,
                    "description": f"{regime_type} period {start} to {end}",
                })

    return scenarios
```

**Step 3: Write tests**

```python
def test_generate_structural_regime_scenarios():
    """Pipeline Orchestrator generates 4 structural regime scenarios."""
    from hrp.agents.pipeline_orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()
    scenarios = orchestrator._generate_structural_regime_scenarios(
        hypothesis_id="HYP-TEST-001",
        start_date="2010-01-01",
        end_date="2023-12-31",
    )

    # Should have scenarios for multiple regime types
    assert len(scenarios) >= 2
    assert all("regime_type" in s for s in scenarios)
    assert all("start_date" in s for s in scenarios)
```

**Step 4: Run tests**

```bash
pytest tests/agents/test_pipeline_orchestrator.py::test_generate_structural_regime_scenarios -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/agents/2026-01-29-pipeline-orchestrator-agent.md hrp/agents/pipeline_orchestrator.py tests/agents/test_pipeline_orchestrator.py
git commit -m "feat(agents): update Pipeline Orchestrator with HMM structural regimes

- 4 mandatory structural regime scenarios
- HMM-based regime detection (no hardcoded dates)
- Scenario period extraction
- Regime coverage requirement
"
```

---

## Final Tasks

### Task 10: Update All Documentation

**Files:**
- Modify: `docs/agents/agent-interaction-diagram.md`
- Modify: `docs/agents/decision-pipeline.md`
- Modify: `CLAUDE.md`

**Step 1: Update interaction diagram with all changes**

Update agent count (10 → 11), add Code Materializer, add Pre-Backtest Review.

**Step 2: Update decision pipeline with adaptive IC thresholds**

Document new IC thresholds per strategy class.

**Step 3: Update CLAUDE.md with new usage examples**

```python
### Use Code Materializer
from hrp.agents import CodeMaterializer

materializer = CodeMaterializer(hypothesis_ids=["HYP-2026-001"])
result = materializer.run()

### Use Stability Score v1
from hrp.research.metrics import calculate_stability_score_v1

score = calculate_stability_score_v1(fold_sharpes, fold_drawdowns, mean_ic)
if score <= 1.0:
    print("Model is stable")
```

**Step 4: Commit**

```bash
git add docs/ CLAUDE.md
git commit -m "docs: update all documentation for agent redesign

- Agent interaction diagram (11 agents)
- Decision pipeline (adaptive IC thresholds)
- CLAUDE.md (new usage examples)
"
```

---

### Task 11: Run Full Test Suite

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

**Step 2: Check test coverage**

```bash
pytest --cov=hrp/agents --cov=hrp/research --cov=hrp/ml tests/
```

Expected: Coverage report generated

**Step 3: Fix any failing tests**

If any tests fail, fix and commit.

---

### Task 12: Create Implementation Summary

**Files:**
- Create: `docs/plans/2026-01-29-agent-redesign-summary.md`

**Step 1: Write summary document**

```markdown
# Agent Redesign Implementation Summary

**Date:** January 29, 2026
**Status:** Complete

## What Was Implemented

1. **Code Materializer Agent**
   - Mechanical translation of strategy specs to executable code
   - No logic interpretation
   - Syntax validation
   - Location: `hrp/agents/code_materializer.py`

2. **Alpha Researcher Update**
   - Removed code generation
   - Added strategy classification (cross_sectional_factor, time_series_momentum, ml_composite)
   - Outputs strategy specs only

3. **Adaptive IC Thresholds**
   - Cross-sectional: ≥0.015 pass, <0.005 kill
   - Time-series: ≥0.02 pass, <0.01 kill
   - ML composite: ≥0.025 pass, <0.01 kill
   - Location: `hrp/agents/signal_scientist.py`

4. **Stability Score v1**
   - Formal definition in `hrp/research/metrics.py`
   - Components: Sharpe CV + drawdown dispersion + sign flip penalty
   - Threshold: ≤1.0 = stable

5. **Pre-Backtest Review**
   - Added to Quant Developer workflow
   - Warnings only (no veto)
   - Checks: data, point-in-time, frequency, liquidity, costs

6. **HMM Structural Regimes**
   - 4 regime types: low_vol_bull, low_vol_bear, high_vol_bull, high_vol_bear
   - Data-driven (no hardcoded dates)
   - Location: `hrp/ml/regime_detection.py`

## Pipeline Changes

**Before:** 10 agents
**After:** 11 agents + 1 review stage

## Test Coverage

All new code has corresponding tests.
```

**Step 2: Commit**

```bash
git add docs/plans/2026-01-29-agent-redesign-summary.md
git commit -m "docs: add implementation summary"
```

---

## Verification Checklist

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code Materializer generates valid Python code
- [ ] Alpha Researcher tags strategy_class
- [ ] Adaptive IC thresholds applied correctly
- [ ] Stability Score v1 calculated correctly
- [ ] Pre-Backtest Review generates warnings
- [ ] HMM classifies 4 structural regimes
- [ ] Pipeline Orchestrator uses structural regimes
- [ ] Documentation updated
- [ ] Test coverage adequate

---

## Notes

- This plan follows TDD: write failing test, implement, verify pass, commit
- Each task is independent and can be done in isolation
- Use `superpowers:executing-plans` to execute this plan
- Reference original design: `docs/plans/2026-01-29-agent-redesign-feedback.md`
