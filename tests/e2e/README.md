# E2E Tests for HRP Agent Workflows

Comprehensive end-to-end testing suite for the HRP (Hedge Fund Research Platform) agent workflows.

## Overview

This test suite provides complete coverage of the agent pipeline, from hypothesis creation through deployment decision-making. Tests are organized into logical suites covering different aspects of the system.

## Test Suites

### 1. Complete Event-Driven Pipeline (`TestEventDrivenPipeline`)
Tests the full agent workflow chain:
- Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel
- → Quant Developer → Pipeline Orchestrator → Validation Analyst

**Key Tests:**
- ✅ Complete pipeline flow with all agents
- ✅ Early kill gates for resource optimization
- ✅ Error recovery and failure handling

### 2. Event Coordination (`TestEventCoordination`)
Tests lineage event propagation between agents.

**Key Tests:**
- ✅ Event propagation through agent chain
- ✅ Multiple agents triggered by single event
- ✅ Event ordering and timing

### 3. Concurrency & Resource Management (`TestConcurrency`)
Tests parallel execution and resource limits.

**Key Tests:**
- ✅ Parallel experiment execution
- ✅ Resource cleanup on failure
- ✅ Concurrent access handling

### 4. Scheduled Workflows (`TestScheduledWorkflows`)
Tests scheduled job execution and integration.

**Key Tests:**
- ✅ Daily ingestion workflow (prices → universe → features)
- ✅ Weekly research workflow (signal scan → Alpha → ML)
- ✅ Scheduler integration

### 5. Cross-Agent Dependencies (`TestCrossAgentDependencies`)
Tests data flow and dependencies between agents.

**Key Tests:**
- ✅ Hypothesis status lifecycle
- ✅ Experiment-to-hypothesis linkage
- ✅ Feature dependencies

### 6. Performance & Scalability (`TestPerformanceScalability`)
Tests system behavior under load.

**Key Tests:**
- ✅ Large batch processing
- ✅ Memory efficiency with large datasets

### 7. Error Handling Edge Cases (`TestErrorHandlingEdgeCases`)
Tests error handling in edge cases.

**Key Tests:**
- ✅ Agent timeout handling
- ✅ Database connection recovery
- ✅ Malformed data handling

### 8. External Service Integration (`TestExternalServiceIntegration`)
Tests integration with MLflow and external APIs.

**Key Tests:**
- ✅ MLflow logging
- ✅ External API failure handling

## Quick Start

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-mock pytest-cov pytest-asyncio

# Ensure HRP is installed
pip install -e .
```

### Run All E2E Tests

```bash
# Run all E2E tests
pytest tests/e2e/ -v

# Run with coverage
pytest tests/e2e/ -v --cov=hrp.agents --cov-report=html

# Run only E2E marked tests
pytest -m e2e -v
```

### Run Specific Test Suites

```bash
# Event-driven pipeline tests
pytest tests/e2e/test_agent_workflows_e2e.py::TestEventDrivenPipeline -v

# Concurrency tests
pytest tests/e2e/test_agent_workflows_e2e.py::TestConcurrency -v

# Scheduled workflow tests
pytest tests/e2e/test_agent_workflows_e2e.py::TestScheduledWorkflows -v
```

### Run Specific Tests

```bash
# Run complete pipeline flow test
pytest tests/e2e/test_agent_workflows_e2e.py::TestEventDrivenPipeline::test_complete_pipeline_flow -v

# Run kill gates test
pytest tests/e2e/test_agent_workflows_e2e.py::TestEventDrivenPipeline::test_pipeline_with_kill_gates -v
```

### Run with Different Verbosity

```bash
# Minimal output
pytest tests/e2e/ -q

# Verbose output
pytest tests/e2e/ -vv

# Show print statements
pytest tests/e2e/ -v -s
```

## Test Configuration

### Environment Variables

```bash
# Use test database (optional, defaults to temp directory)
export HRP_DB_PATH=/tmp/test_hrp.duckdb

# Enable debug logging
export HRP_LOG_LEVEL=DEBUG

# Disable external API calls during testing
export HRP_TEST_MODE=1
```

### Pytest Configuration

Create `pytest.ini` or `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "e2e: End-to-end tests",
    "benchmark: Performance benchmarks",
    "integration: Integration tests",
]
addopts = "-v --tb=short --strict-markers"
```

## Test Fixtures

Key fixtures available in `tests/e2e/conftest.py`:

| Fixture | Description |
|---------|-------------|
| `test_environment` | Sets up temporary test environment |
| `clean_test_db` | Provides clean database for each test |
| `mock_platform_api_factory` | Creates configured PlatformAPI mocks |
| `sample_hypothesis_data` | Sample hypothesis for testing |
| `sample_experiment_data` | Sample experiment for testing |
| `sample_symbols` | Symbol sets for testing |
| `sample_price_data` | Sample price data DataFrame |
| `sample_feature_data` | Sample feature data DataFrame |
| `test_context` | Context manager for test tracking |
| `performance_monitor` | Monitor test performance |

## Writing New E2E Tests

### Template

```python
import pytest
from datetime import date

class TestNewFeature:
    """Test new feature end-to-end."""

    def test_basic_functionality(self, mock_platform_api):
        """Test basic functionality works."""
        # Arrange
        hypothesis_id = 'HYP-2026-NEW-001'

        # Act
        result = mock_platform_api.create_hypothesis(
            title="Test",
            thesis="Test thesis",
            prediction="Test prediction",
            falsification="Sharpe < 0.5",
            actor='test:e2e',
        )

        # Assert
        assert result is not None
        assert result.startswith('HYP-')

    @pytest.mark.slow
    def test_with_real_data(self):
        """Test with real data (marked as slow)."""
        # Test implementation
        pass
```

### Best Practices

1. **Use Descriptive Names**: Test names should describe what is being tested
2. **Follow AAA Pattern**: Arrange, Act, Assert
3. **Mock External Dependencies**: Use mocks for PlatformAPI, external APIs
4. **Clean Up Resources**: Use fixtures for setup/teardown
5. **Test Edge Cases**: Include negative test cases
6. **Mark Tests Appropriately**: Use markers for slow/integration/benchmark tests

## Continuous Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-mock pytest-cov
      - name: Run E2E tests
        run: pytest tests/e2e/ -v --cov=hrp.agents
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

**Issue:** Tests fail with import errors
```bash
# Solution: Ensure HRP is installed in development mode
pip install -e .
```

**Issue:** Database lock errors
```bash
# Solution: Use test database or run tests sequentially
pytest tests/e2e/ -v --forked
```

**Issue:** External API rate limiting
```bash
# Solution: Set test mode to disable external calls
export HRP_TEST_MODE=1
pytest tests/e2e/ -v
```

**Issue:** Timeout errors
```bash
# Solution: Increase timeout for slow tests
pytest tests/e2e/ -v --timeout=300
```

## Coverage Goals

Target coverage for agent workflows:
- **Overall**: >80% code coverage
- **Critical Paths**: >90% coverage
- **Error Handling**: >70% coverage

Current coverage can be viewed with:
```bash
pytest tests/e2e/ --cov=hrp.agents --cov-report=html
open htmlcov/index.html
```

## Contributing

When adding new agents or workflows:

1. **Add E2E tests** alongside the implementation
2. **Update this README** with new test suite documentation
3. **Run full E2E suite** before committing
4. **Ensure all tests pass** in CI/CD pipeline

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Project overview and conventions
- [tests/README.md](../README.md) - General testing documentation
- [hrp/agents/README.md](../../hrp/agents/README.md) - Agent documentation

## Support

For issues or questions:
1. Check existing test cases for examples
2. Review pytest documentation: https://docs.pytest.org/
3. Open an issue on GitHub
