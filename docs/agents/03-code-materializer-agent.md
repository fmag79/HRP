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
