# Platform API Test Suite Design

**Date**: 2025-01-22
**Status**: Approved
**Goal**: 70%+ code coverage on Platform API with comprehensive unit and integration tests

## Overview

This design covers a comprehensive test suite for `hrp/api/platform.py` including:
- Input validation implementation (to fix 20 failing tests)
- Additional unit tests for uncovered methods
- Integration test for full backtest flow
- Synthetic data generator for test fixtures
- Database migration tests

## Component 1: Input Validation in platform.py

The existing test file has 20 failing tests expecting validation that doesn't exist. Add validation for:

| Method | Validation Needed |
|--------|-------------------|
| `get_prices()` | Future dates, invalid date range (start > end), invalid symbols not in universe |
| `get_features()` | Future dates, invalid symbols not in universe at specified date |
| `get_universe()` | Future dates |
| `create_hypothesis()` | Empty/whitespace strings for title, thesis, prediction, falsification, actor |
| `update_hypothesis()` | Empty hypothesis_id, status, actor |
| `list_hypotheses()` | Non-positive limit |
| `get_hypothesis()` | Empty hypothesis_id |

**Implementation pattern**: Add validation at the start of each method, raising `ValueError` with descriptive messages matching test expectations.

## Component 2: Additional Unit Tests

Methods needing test coverage:

| Method | Tests to Add |
|--------|--------------|
| `run_backtest()` | Success flow (mocked), signal generation, MLflow logging, hypothesis linking |
| `get_experiment()` | Found case, not found case, MLflow error handling |
| `compare_experiments()` | Default metrics list, partial experiment data |
| `adjust_prices_for_splits()` | Empty df, no splits, single split, multiple splits per symbol |
| `get_corporate_actions()` | Empty symbols validation, date range, data returned |

**Test classes to add**:
- `TestPlatformAPIBacktest` - Mocked MLflow tests
- `TestPlatformAPISplitAdjustment` - Price adjustment tests
- `TestPlatformAPICorporateActions` - Corporate actions tests

## Component 3: Integration Test for Full Backtest Flow

Test the complete flow: hypothesis → experiment → lineage

**Test**: `test_full_backtest_flow_with_lineage`

Steps:
1. Create hypothesis via API
2. Run backtest linked to hypothesis (mocked MLflow returns run_id)
3. Verify experiment linked to hypothesis
4. Verify lineage trail contains all events (hypothesis_created, experiment_run)
5. Update hypothesis to validated
6. Verify final lineage state

**Mocking strategy**:
- Mock `mlflow.start_run()`, `mlflow.log_params()`, `mlflow.log_metrics()`
- Mock `hrp.research.backtest.run_backtest` to return `BacktestResult` with known metrics
- Mock `hrp.research.backtest.get_price_data` to use test fixture data

## Component 4: Synthetic Data Generator

Create `tests/fixtures/synthetic.py` with generators:

```python
def generate_prices(symbols, start, end, seed=42) -> pd.DataFrame
def generate_features(symbols, feature_names, dates, seed=42) -> pd.DataFrame
def generate_corporate_actions(symbols, start, end, action_types=['split', 'dividend']) -> pd.DataFrame
def generate_universe(symbols, dates) -> pd.DataFrame
```

**Design principles**:
- Deterministic with seed parameter for reproducibility
- Realistic price movements (random walk with drift)
- Proper split factors (2:1, 3:1, etc.)
- Consistent across related data (prices adjust around split dates)

## Component 5: Database Migration Tests

Create `tests/test_data/test_migrations.py`:

| Test | Purpose |
|------|---------|
| `test_create_tables_fresh_db` | Schema creates successfully on empty database |
| `test_create_tables_idempotent` | Running create_tables twice doesn't error |
| `test_all_tables_exist` | All expected tables present after migration |
| `test_foreign_key_constraints` | FK constraints enforced |
| `test_unique_constraints` | Unique constraints work |

## Implementation Order

1. Add input validation to `platform.py` (fixes 20 failing tests)
2. Create synthetic data generator
3. Add unit tests for uncovered methods
4. Add database migration tests
5. Add integration test for backtest flow
6. Verify 70%+ coverage achieved

## Acceptance Criteria Mapping

- [x] 70%+ code coverage on Platform API → Components 1-3
- [x] Synthetic data generator for test fixtures → Component 4
- [x] Integration test for full backtest flow → Component 3
- [x] Tests pass in CI environment → All components use temp databases
- [x] Database migration tests → Component 5
