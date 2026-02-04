# Bayesian Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace custom grid/random search with Optuna for Bayesian hyperparameter optimization.

**Architecture:** Optuna manages the optimization loop with TPE sampler. HyperparameterTrialCounter remains the authority for trial limits and audit logging. Studies persist to SQLite for resume capability.

**Tech Stack:** Optuna 3.4+, SQLite, existing walk-forward validation infrastructure.

---

## Task 1: Create Optuna Storage Directory

**Files:**
- Modify: `hrp/utils/config.py`

**Step 1: Add OPTUNA_DIR constant**

In `hrp/utils/config.py`, add after line ~20 (near other directory constants):

```python
OPTUNA_DIR = DATA_DIR / "optuna"
```

**Step 2: Ensure directory creation**

Add to the `ensure_directories()` function (or create if doesn't exist):

```python
def ensure_directories() -> None:
    """Ensure all required data directories exist."""
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
```

**Step 3: Verify manually**

Run: `python -c "from hrp.utils.config import OPTUNA_DIR, ensure_directories; ensure_directories(); print(OPTUNA_DIR)"`

Expected: `~/hrp-data/optuna` printed, directory exists.

**Step 4: Commit**

```bash
git add hrp/utils/config.py
git commit -m "feat(config): add OPTUNA_DIR for study persistence"
```

---

## Task 2: Update OptimizationConfig Dataclass

**Files:**
- Modify: `hrp/ml/optimization.py:47-120`
- Test: `tests/test_ml/test_optimization.py`

**Step 1: Write failing test for new config**

Add to `tests/test_ml/test_optimization.py`:

```python
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution


class TestOptimizationConfigNew:
    """Tests for new Optuna-based OptimizationConfig."""

    def test_config_with_param_space(self):
        """Test creating config with param_space distributions."""
        config = OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            param_space={
                "alpha": FloatDistribution(0.01, 100.0, log=True),
            },
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert "alpha" in config.param_space
        assert config.sampler == "tpe"
        assert config.n_trials == 50
        assert config.enable_pruning is True

    def test_config_rejects_invalid_sampler(self):
        """Test config rejects invalid sampler."""
        with pytest.raises(ValueError, match="Unknown sampler"):
            OptimizationConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                param_space={"alpha": FloatDistribution(0.1, 10.0)},
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                sampler="invalid",
            )

    def test_config_rejects_empty_param_space(self):
        """Test config rejects empty param_space."""
        with pytest.raises(ValueError, match="param_space cannot be empty"):
            OptimizationConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                param_space={},
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_optimization.py::TestOptimizationConfigNew -v`

Expected: FAIL - old config doesn't have `param_space`.

**Step 3: Update OptimizationConfig**

Replace the dataclass in `hrp/ml/optimization.py`:

```python
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution, CategoricalDistribution

VALID_SAMPLERS = {"grid", "random", "tpe", "cmaes"}


@dataclass
class OptimizationConfig:
    """
    Configuration for cross-validated optimization with Optuna.

    Attributes:
        model_type: Type of model (must be in SUPPORTED_MODELS)
        target: Target variable name (e.g., 'returns_20d')
        features: List of feature names from feature store
        param_space: Dict mapping param names to Optuna distributions
        start_date: Start of the entire date range
        end_date: End of the entire date range
        n_folds: Number of cross-validation folds (default 5)
        window_type: 'expanding' or 'rolling' (default 'expanding')
        scoring_metric: Metric to optimize (default 'ic')
        sampler: Optuna sampler type: 'grid', 'random', 'tpe', 'cmaes' (default 'tpe')
        n_trials: Maximum number of trials (default 50)
        hypothesis_id: Optional hypothesis ID for tracking
        enable_pruning: Enable MedianPruner for early stopping (default True)
        early_stop_decay_threshold: Sharpe decay threshold for pruning
        min_train_periods: Minimum training samples required
        feature_selection: Enable automatic feature selection
        max_features: Maximum features after selection
    """

    model_type: str
    target: str
    features: list[str]
    param_space: dict[str, BaseDistribution]
    start_date: date
    end_date: date
    n_folds: int = 5
    window_type: str = "expanding"
    scoring_metric: str = "ic"
    sampler: str = "tpe"
    n_trials: int = 50
    hypothesis_id: str | None = None
    enable_pruning: bool = True
    early_stop_decay_threshold: float = 0.5
    min_train_periods: int = 252
    feature_selection: bool = True
    max_features: int = 20

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.model_type not in SUPPORTED_MODELS:
            available = ", ".join(sorted(SUPPORTED_MODELS.keys()))
            raise ValueError(
                f"Unsupported model type: '{self.model_type}'. "
                f"Available: {available}"
            )
        if self.window_type not in ("expanding", "rolling"):
            raise ValueError(
                f"window_type must be 'expanding' or 'rolling', "
                f"got '{self.window_type}'"
            )
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.scoring_metric not in SCORING_METRICS:
            available = ", ".join(sorted(SCORING_METRICS.keys()))
            raise ValueError(
                f"Unsupported scoring_metric: '{self.scoring_metric}'. "
                f"Available: {available}"
            )
        if not self.param_space:
            raise ValueError("param_space cannot be empty")
        if self.sampler not in VALID_SAMPLERS:
            raise ValueError(
                f"Unknown sampler: '{self.sampler}'. "
                f"Options: {', '.join(sorted(VALID_SAMPLERS))}"
            )

        logger.debug(
            f"OptimizationConfig created: {self.model_type}, "
            f"{self.n_folds} folds, sampler={self.sampler}"
        )
```

**Step 4: Update imports at top of file**

Add to imports in `hrp/ml/optimization.py`:

```python
from optuna.distributions import (
    BaseDistribution,
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_ml/test_optimization.py::TestOptimizationConfigNew -v`

Expected: PASS

**Step 6: Commit**

```bash
git add hrp/ml/optimization.py tests/test_ml/test_optimization.py
git commit -m "feat(optimization): update OptimizationConfig for Optuna

- Replace param_grid with param_space (Optuna distributions)
- Replace search_type with sampler (grid, random, tpe, cmaes)
- Replace max_trials with n_trials
- Add enable_pruning flag

BREAKING CHANGE: param_grid removed, use param_space with Optuna distributions"
```

---

## Task 3: Implement Sampler Selection

**Files:**
- Modify: `hrp/ml/optimization.py`
- Test: `tests/test_ml/test_optimization.py`

**Step 1: Write failing test for sampler selection**

Add to `tests/test_ml/test_optimization.py`:

```python
import optuna
from hrp.ml.optimization import _get_sampler


class TestGetSampler:
    """Tests for _get_sampler function."""

    def test_tpe_sampler(self):
        """Test TPE sampler creation."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}
        sampler = _get_sampler("tpe", param_space)
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_random_sampler(self):
        """Test random sampler creation."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}
        sampler = _get_sampler("random", param_space)
        assert isinstance(sampler, optuna.samplers.RandomSampler)

    def test_grid_sampler_with_categorical(self):
        """Test grid sampler with categorical distribution."""
        param_space = {"solver": CategoricalDistribution(["lbfgs", "saga"])}
        sampler = _get_sampler("grid", param_space)
        assert isinstance(sampler, optuna.samplers.GridSampler)

    def test_grid_sampler_requires_step_for_float(self):
        """Test grid sampler requires step for float distributions."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}  # No step
        with pytest.raises(ValueError, match="requires step"):
            _get_sampler("grid", param_space)

    def test_cmaes_sampler(self):
        """Test CMA-ES sampler creation."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}
        sampler = _get_sampler("cmaes", param_space)
        assert isinstance(sampler, optuna.samplers.CmaEsSampler)

    def test_invalid_sampler_raises(self):
        """Test invalid sampler raises ValueError."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}
        with pytest.raises(ValueError, match="Unknown sampler"):
            _get_sampler("invalid", param_space)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_optimization.py::TestGetSampler -v`

Expected: FAIL - `_get_sampler` not defined.

**Step 3: Implement _get_sampler**

Add to `hrp/ml/optimization.py`:

```python
import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler, CmaEsSampler


def _get_sampler(
    sampler_name: str,
    param_space: dict[str, BaseDistribution],
) -> optuna.samplers.BaseSampler:
    """
    Create Optuna sampler from name.

    Args:
        sampler_name: One of 'grid', 'random', 'tpe', 'cmaes'
        param_space: Parameter space (needed for GridSampler)

    Returns:
        Configured Optuna sampler

    Raises:
        ValueError: If sampler_name invalid or grid requires step for floats
    """
    if sampler_name == "grid":
        search_space = _distributions_to_grid(param_space)
        return GridSampler(search_space)
    elif sampler_name == "random":
        return RandomSampler(seed=42)
    elif sampler_name == "tpe":
        return TPESampler(seed=42)
    elif sampler_name == "cmaes":
        return CmaEsSampler(seed=42)
    else:
        raise ValueError(
            f"Unknown sampler: '{sampler_name}'. "
            f"Options: grid, random, tpe, cmaes"
        )


def _distributions_to_grid(
    param_space: dict[str, BaseDistribution],
) -> dict[str, list]:
    """
    Convert Optuna distributions to explicit grid for GridSampler.

    Args:
        param_space: Dict of param name to distribution

    Returns:
        Dict of param name to list of values

    Raises:
        ValueError: If FloatDistribution lacks step
    """
    grid = {}
    for name, dist in param_space.items():
        if isinstance(dist, CategoricalDistribution):
            grid[name] = list(dist.choices)
        elif isinstance(dist, IntDistribution):
            step = dist.step if dist.step else 1
            grid[name] = list(range(dist.low, dist.high + 1, step))
        elif isinstance(dist, FloatDistribution):
            if dist.step:
                values = []
                v = dist.low
                while v <= dist.high:
                    values.append(v)
                    v += dist.step
                grid[name] = values
            else:
                raise ValueError(
                    f"Grid sampler requires step for float param '{name}'. "
                    f"Use FloatDistribution({dist.low}, {dist.high}, step=X)"
                )
        else:
            raise ValueError(f"Unsupported distribution type for '{name}': {type(dist)}")
    return grid
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml/test_optimization.py::TestGetSampler -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ml/optimization.py tests/test_ml/test_optimization.py
git commit -m "feat(optimization): add Optuna sampler selection

- _get_sampler() maps sampler names to Optuna samplers
- _distributions_to_grid() converts distributions for GridSampler
- Supports: grid, random, tpe, cmaes"
```

---

## Task 4: Implement Fold Evaluation with Pruning

**Files:**
- Modify: `hrp/ml/optimization.py`
- Test: `tests/test_ml/test_optimization.py`

**Step 1: Write failing test for pruning**

Add to `tests/test_ml/test_optimization.py`:

```python
class TestEvaluateWithPruning:
    """Tests for _evaluate_with_pruning function."""

    @pytest.fixture
    def mock_trial(self):
        """Create mock Optuna trial."""
        trial = MagicMock()
        trial.should_prune.return_value = False
        return trial

    @pytest.fixture
    def sample_config(self):
        """Create sample config."""
        return OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            param_space={"alpha": FloatDistribution(0.1, 10.0)},
            start_date=date(2015, 1, 1),
            end_date=date(2020, 12, 31),
            n_folds=3,
            feature_selection=False,
        )

    @pytest.fixture
    def mock_data(self):
        """Create mock data."""
        dates = pd.date_range("2015-01-01", "2020-12-31", freq="B")
        symbols = ["AAPL"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        np.random.seed(42)
        n = len(index)
        return pd.DataFrame(
            {
                "momentum_20d": np.random.randn(n) * 0.1,
                "returns_20d": np.random.randn(n) * 0.05,
            },
            index=index,
        )

    def test_reports_intermediate_values(self, mock_trial, sample_config, mock_data):
        """Test that intermediate values are reported to trial."""
        from hrp.ml.optimization import _evaluate_with_pruning
        from hrp.ml.validation import generate_folds, WalkForwardConfig

        wf_config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=sample_config.start_date,
            end_date=sample_config.end_date,
            n_folds=3,
        )
        dates = sorted(mock_data.index.get_level_values("date").unique())
        dates = [d.date() for d in dates]
        folds = generate_folds(wf_config, dates)

        _evaluate_with_pruning(
            trial=mock_trial,
            params={"alpha": 1.0},
            config=sample_config,
            all_data=mock_data,
            folds=folds,
        )

        # Should have reported once per fold
        assert mock_trial.report.call_count == len(folds)

    def test_prunes_on_should_prune(self, mock_trial, sample_config, mock_data):
        """Test that trial is pruned when should_prune returns True."""
        from hrp.ml.optimization import _evaluate_with_pruning
        from hrp.ml.validation import generate_folds, WalkForwardConfig

        mock_trial.should_prune.return_value = True

        wf_config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=sample_config.start_date,
            end_date=sample_config.end_date,
            n_folds=3,
        )
        dates = sorted(mock_data.index.get_level_values("date").unique())
        dates = [d.date() for d in dates]
        folds = generate_folds(wf_config, dates)

        with pytest.raises(optuna.TrialPruned):
            _evaluate_with_pruning(
                trial=mock_trial,
                params={"alpha": 1.0},
                config=sample_config,
                all_data=mock_data,
                folds=folds,
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_optimization.py::TestEvaluateWithPruning -v`

Expected: FAIL - `_evaluate_with_pruning` not defined or wrong signature.

**Step 3: Implement _evaluate_with_pruning**

Replace `_evaluate_params` in `hrp/ml/optimization.py` with:

```python
def _evaluate_with_pruning(
    trial: optuna.Trial,
    params: dict[str, Any],
    config: OptimizationConfig,
    all_data: pd.DataFrame,
    folds: list[tuple[date, date, date, date]],
) -> tuple[float, list[FoldResult]]:
    """
    Evaluate parameters across all folds with Optuna pruning support.

    Reports intermediate values after each fold for MedianPruner.
    Raises TrialPruned if should_prune() or Sharpe decay exceeded.

    Args:
        trial: Optuna trial for reporting intermediate values
        params: Hyperparameter values to evaluate
        config: Optimization configuration
        all_data: Full dataset
        folds: List of (train_start, train_end, test_start, test_end) tuples

    Returns:
        Tuple of (mean_score, fold_results)

    Raises:
        optuna.TrialPruned: If trial should be pruned
    """
    fold_results = []
    fold_scores = []
    train_sharpes = []
    test_sharpes = []

    decay_monitor = SharpeDecayMonitor(max_decay_ratio=config.early_stop_decay_threshold)
    higher_is_better = SCORING_METRICS[config.scoring_metric]

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        # Split data
        dates = all_data.index.get_level_values("date")
        if hasattr(dates[0], "date"):
            train_mask = (dates.date >= train_start) & (dates.date <= train_end)
            test_mask = (dates.date >= test_start) & (dates.date <= test_end)
        else:
            train_mask = (dates >= pd.Timestamp(train_start)) & (
                dates <= pd.Timestamp(train_end)
            )
            test_mask = (dates >= pd.Timestamp(test_start)) & (
                dates <= pd.Timestamp(test_end)
            )

        train_data = all_data.loc[train_mask]
        test_data = all_data.loc[test_mask]

        if len(train_data) == 0 or len(test_data) == 0:
            continue

        # Feature selection
        selected_features = list(config.features)
        if config.feature_selection and len(config.features) > config.max_features:
            selected_features = select_features(
                train_data[config.features],
                train_data[config.target],
                config.max_features,
            )

        X_train = train_data[selected_features]
        y_train = train_data[config.target]
        X_test = test_data[selected_features]
        y_test = test_data[config.target]

        # Train model
        model = get_model(config.model_type, params)
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics
        train_metrics = compute_fold_metrics(y_train, y_train_pred)
        test_metrics = compute_fold_metrics(y_test, y_test_pred)

        # Track Sharpe proxy
        train_sharpes.append(train_metrics.get("ic", 0.0))
        test_sharpes.append(test_metrics.get("ic", 0.0))

        # Extract score
        score = test_metrics.get(config.scoring_metric, float("nan"))
        if not np.isnan(score):
            fold_scores.append(score)

        fold_result = FoldResult(
            fold_index=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            metrics=test_metrics,
            model=model,
            n_train_samples=len(X_train),
            n_test_samples=len(X_test),
        )
        fold_results.append(fold_result)

        # Report intermediate value for pruning
        if fold_scores:
            intermediate_score = float(np.mean(fold_scores))
            trial.report(intermediate_score, fold_idx)

            # Check if should prune
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at fold {fold_idx}")

        # Check Sharpe decay
        if train_sharpes and test_sharpes:
            mean_train = float(np.mean(train_sharpes))
            mean_test = float(np.mean(test_sharpes))
            decay_result = decay_monitor.check(mean_train, mean_test)
            if not decay_result.passed:
                raise optuna.TrialPruned(f"Sharpe decay: {decay_result.message}")

    if not fold_scores:
        mean_score = float("-inf") if higher_is_better else float("inf")
    else:
        mean_score = float(np.mean(fold_scores))

    return mean_score, fold_results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml/test_optimization.py::TestEvaluateWithPruning -v`

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ml/optimization.py tests/test_ml/test_optimization.py
git commit -m "feat(optimization): add fold evaluation with Optuna pruning

- _evaluate_with_pruning() reports intermediate values per fold
- Prunes on MedianPruner signal or Sharpe decay threshold
- Replaces _evaluate_params()"
```

---

## Task 5: Implement Main Optimization Loop

**Files:**
- Modify: `hrp/ml/optimization.py`
- Test: `tests/test_ml/test_optimization.py`

**Step 1: Write failing test for new optimize function**

Update existing test in `tests/test_ml/test_optimization.py`:

```python
class TestCrossValidatedOptimizeNew:
    """Tests for Optuna-based cross_validated_optimize."""

    @pytest.fixture
    def sample_config(self):
        """Create sample config with new API."""
        return OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            param_space={
                "alpha": FloatDistribution(0.1, 10.0, log=True),
            },
            start_date=date(2015, 1, 1),
            end_date=date(2020, 12, 31),
            n_folds=3,
            sampler="tpe",
            n_trials=5,
            feature_selection=False,
            enable_pruning=False,  # Disable for predictable test
        )

    @pytest.fixture
    def mock_features_df(self):
        """Create mock features DataFrame."""
        dates = pd.date_range("2015-01-01", "2020-12-31", freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        np.random.seed(42)
        n = len(index)
        momentum = np.random.randn(n) * 0.1
        volatility = np.abs(np.random.randn(n)) * 0.2
        target = 0.1 * momentum + np.random.randn(n) * 0.05
        return pd.DataFrame(
            {
                "momentum_20d": momentum,
                "volatility_20d": volatility,
                "returns_20d": target,
            },
            index=index,
        )

    def test_returns_optimization_result(self, sample_config, mock_features_df):
        """Test returns OptimizationResult with best params."""
        with patch("hrp.ml.optimization._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            result = cross_validated_optimize(
                config=sample_config,
                symbols=["AAPL", "MSFT"],
                log_to_mlflow=False,
            )

        assert isinstance(result, OptimizationResult)
        assert "alpha" in result.best_params
        assert result.n_trials_completed > 0
        assert result.n_trials_completed <= 5

    def test_uses_tpe_sampler(self, sample_config, mock_features_df):
        """Test TPE sampler is used when specified."""
        with patch("hrp.ml.optimization._fetch_features") as mock_fetch, \
             patch("hrp.ml.optimization.optuna.create_study") as mock_create:
            mock_fetch.return_value = mock_features_df
            mock_study = MagicMock()
            mock_study.best_trial.params = {"alpha": 1.0}
            mock_study.best_trial.value = 0.05
            mock_study.trials = []
            mock_create.return_value = mock_study

            cross_validated_optimize(
                config=sample_config,
                symbols=["AAPL", "MSFT"],
                log_to_mlflow=False,
            )

        # Verify TPESampler was passed
        call_kwargs = mock_create.call_args.kwargs
        assert isinstance(call_kwargs["sampler"], TPESampler)

    def test_respects_trial_counter(self, mock_features_df):
        """Test integrates with HyperparameterTrialCounter."""
        config = OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            param_space={"alpha": FloatDistribution(0.1, 10.0)},
            start_date=date(2015, 1, 1),
            end_date=date(2020, 12, 31),
            n_folds=3,
            hypothesis_id="HYP-TEST-001",
            n_trials=10,
            feature_selection=False,
            enable_pruning=False,
        )

        with patch("hrp.ml.optimization._fetch_features") as mock_fetch, \
             patch("hrp.ml.optimization.HyperparameterTrialCounter") as mock_counter:
            mock_fetch.return_value = mock_features_df
            mock_instance = MagicMock()
            mock_instance.remaining_trials = 3
            mock_instance.can_try.return_value = True
            mock_counter.return_value = mock_instance

            result = cross_validated_optimize(
                config=config,
                symbols=["AAPL"],
                log_to_mlflow=False,
            )

        mock_counter.assert_called_once_with(
            hypothesis_id="HYP-TEST-001",
            max_trials=10,
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_optimization.py::TestCrossValidatedOptimizeNew -v`

Expected: FAIL - old implementation doesn't match new API.

**Step 3: Implement new cross_validated_optimize**

Replace `cross_validated_optimize` in `hrp/ml/optimization.py`:

```python
from pathlib import Path
from optuna.pruners import MedianPruner

from hrp.utils.config import DATA_DIR


def cross_validated_optimize(
    config: OptimizationConfig,
    symbols: list[str],
    log_to_mlflow: bool = True,
) -> OptimizationResult:
    """
    Run cross-validated parameter optimization using Optuna.

    Uses Bayesian optimization (TPE) by default with MedianPruner for
    early stopping of unpromising trials. Integrates with existing
    HyperparameterTrialCounter for audit trail.

    Args:
        config: Optimization configuration with param_space
        symbols: List of symbols to use
        log_to_mlflow: Whether to log results to MLflow

    Returns:
        OptimizationResult with best parameters and trial history

    Raises:
        ValueError: If no data found for symbols
    """
    logger.info(
        f"Starting optimization: {config.model_type}, "
        f"sampler={config.sampler}, n_trials={config.n_trials}"
    )

    start_time = time.perf_counter()

    # Initialize trial counter if hypothesis provided
    trial_counter = None
    max_trials = config.n_trials
    if config.hypothesis_id:
        trial_counter = HyperparameterTrialCounter(
            hypothesis_id=config.hypothesis_id,
            max_trials=config.n_trials,
        )
        max_trials = trial_counter.remaining_trials
        logger.info(f"Trial counter: {max_trials} trials remaining")

    # Fetch data once
    all_data = _fetch_features(
        symbols=symbols,
        features=config.features,
        start_date=config.start_date,
        end_date=config.end_date,
        target=config.target,
    )

    if all_data.empty:
        raise ValueError(f"No data found for symbols {symbols}")

    all_data = all_data.dropna()

    # Generate folds
    available_dates = sorted(all_data.index.get_level_values("date").unique())
    available_dates = [d.date() if hasattr(d, "date") else d for d in available_dates]

    wf_config = WalkForwardConfig(
        model_type=config.model_type,
        target=config.target,
        features=config.features,
        start_date=config.start_date,
        end_date=config.end_date,
        n_folds=config.n_folds,
        window_type=config.window_type,
        min_train_periods=config.min_train_periods,
    )
    folds = generate_folds(wf_config, available_dates)

    # Create Optuna study
    higher_is_better = SCORING_METRICS[config.scoring_metric]
    direction = "maximize" if higher_is_better else "minimize"

    # Storage for persistence
    storage = None
    study_name = None
    if config.hypothesis_id:
        optuna_dir = DATA_DIR / "optuna"
        optuna_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{optuna_dir}/{config.hypothesis_id}.db"
        study_name = config.hypothesis_id

    # Pruner
    pruner = MedianPruner() if config.enable_pruning else None

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=_get_sampler(config.sampler, config.param_space),
        pruner=pruner,
    )

    # Track all trials for result
    all_trials = []
    early_stopped = False
    early_stop_reason = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal early_stopped, early_stop_reason

        # Check trial counter
        if trial_counter and not trial_counter.can_try():
            early_stopped = True
            early_stop_reason = f"Trial limit ({config.n_trials}) exceeded"
            raise optuna.TrialPruned(early_stop_reason)

        # Sample parameters
        params = {}
        for name, dist in config.param_space.items():
            if isinstance(dist, FloatDistribution):
                params[name] = trial.suggest_float(
                    name, dist.low, dist.high,
                    log=dist.log, step=dist.step
                )
            elif isinstance(dist, IntDistribution):
                params[name] = trial.suggest_int(
                    name, dist.low, dist.high, step=dist.step or 1
                )
            elif isinstance(dist, CategoricalDistribution):
                params[name] = trial.suggest_categorical(name, dist.choices)

        logger.debug(f"Trial {trial.number}: {params}")

        # Evaluate
        try:
            score, fold_results = _evaluate_with_pruning(
                trial=trial,
                params=params,
                config=config,
                all_data=all_data,
                folds=folds,
            )
        except optuna.TrialPruned:
            raise

        # Log to trial counter
        if trial_counter:
            try:
                trial_counter.log_trial(
                    model_type=config.model_type,
                    hyperparameters=params,
                    metric_name=config.scoring_metric,
                    metric_value=score,
                )
            except OverfittingError as e:
                early_stopped = True
                early_stop_reason = str(e)
                raise optuna.TrialPruned(early_stop_reason)

        # Track trial
        all_trials.append({
            "trial_idx": trial.number,
            "params": params,
            "mean_score": score,
        })

        return score

    # Run optimization
    try:
        study.optimize(objective, n_trials=max_trials, show_progress_bar=False)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")

    # Build results
    if study.best_trial:
        best_params = study.best_trial.params
        best_score = study.best_trial.value
    else:
        best_params = {}
        best_score = float("-inf") if higher_is_better else float("inf")

    # Re-evaluate best params for fold results
    if best_params:
        # Create a dummy trial for final evaluation
        final_trial = MagicMock()
        final_trial.should_prune.return_value = False
        final_trial.report = MagicMock()
        _, best_fold_results = _evaluate_with_pruning(
            trial=final_trial,
            params=best_params,
            config=config,
            all_data=all_data,
            folds=folds,
        )
    else:
        best_fold_results = []

    # Build CV results DataFrame
    cv_results_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {"mean_score": trial.value}
            row.update({f"param_{k}": v for k, v in trial.params.items()})
            cv_results_data.append(row)

    cv_results = pd.DataFrame(cv_results_data) if cv_results_data else pd.DataFrame()

    execution_time = time.perf_counter() - start_time

    result = OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        cv_results=cv_results,
        fold_results=best_fold_results,
        all_trials=all_trials,
        hypothesis_id=config.hypothesis_id,
        n_trials_completed=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
        execution_time_seconds=execution_time,
    )

    logger.info(
        f"Optimization complete: best_score={best_score:.4f}, "
        f"best_params={best_params}, trials={result.n_trials_completed}"
    )

    if log_to_mlflow:
        _log_optimization_to_mlflow(result, config)

    return result
```

**Step 4: Add MagicMock import for final evaluation**

Add at top of file:

```python
from unittest.mock import MagicMock
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_ml/test_optimization.py::TestCrossValidatedOptimizeNew -v`

Expected: PASS

**Step 6: Commit**

```bash
git add hrp/ml/optimization.py tests/test_ml/test_optimization.py
git commit -m "feat(optimization): implement Optuna-based optimization loop

- Uses Optuna study with configurable sampler
- Persists studies to SQLite for resume capability
- Integrates with HyperparameterTrialCounter
- MedianPruner for early stopping"
```

---

## Task 6: Update MLflow Logging

**Files:**
- Modify: `hrp/ml/optimization.py`

**Step 1: Update _log_optimization_to_mlflow**

Update function in `hrp/ml/optimization.py`:

```python
def _log_optimization_to_mlflow(
    result: OptimizationResult,
    config: OptimizationConfig,
) -> None:
    """Log optimization results to MLflow."""
    try:
        import mlflow

        experiment_name = f"hrp_optimization_{config.model_type}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"opt_{config.sampler}_{config.scoring_metric}"):
            # Log config
            mlflow.log_param("model_type", config.model_type)
            mlflow.log_param("n_folds", config.n_folds)
            mlflow.log_param("window_type", config.window_type)
            mlflow.log_param("scoring_metric", config.scoring_metric)
            mlflow.log_param("sampler", config.sampler)
            mlflow.log_param("n_trials", result.n_trials_completed)
            mlflow.log_param("enable_pruning", config.enable_pruning)

            # Log best params
            for k, v in result.best_params.items():
                mlflow.log_param(f"best_{k}", v)

            # Log metrics
            mlflow.log_metric("best_score", result.best_score)
            mlflow.log_metric("execution_time_seconds", result.execution_time_seconds)

            if result.early_stopped:
                mlflow.log_param("early_stopped", True)
                mlflow.log_param("early_stop_reason", result.early_stop_reason)

        logger.info(f"Logged optimization results to MLflow: {experiment_name}")

    except ImportError:
        logger.warning("MLflow not installed, skipping logging")
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
```

**Step 2: Commit**

```bash
git add hrp/ml/optimization.py
git commit -m "refactor(optimization): update MLflow logging for new config fields"
```

---

## Task 7: Clean Up Old Code

**Files:**
- Modify: `hrp/ml/optimization.py`

**Step 1: Remove deprecated functions and code**

Delete from `hrp/ml/optimization.py`:
- `_generate_param_combinations()` function (replaced by Optuna samplers)
- `_evaluate_params()` function (replaced by `_evaluate_with_pruning()`)
- Any references to `param_grid`, `search_type`, `n_random_samples`, `max_trials`

**Step 2: Remove unused imports**

Remove:
```python
import itertools  # No longer needed
```

**Step 3: Verify no regressions**

Run: `pytest tests/test_ml/test_optimization.py -v`

Expected: All new tests pass (old tests will fail - we'll update them next).

**Step 4: Commit**

```bash
git add hrp/ml/optimization.py
git commit -m "refactor(optimization): remove deprecated grid/random search code

- Remove _generate_param_combinations()
- Remove _evaluate_params()
- Clean up unused imports"
```

---

## Task 8: Update Existing Tests

**Files:**
- Modify: `tests/test_ml/test_optimization.py`

**Step 1: Remove old test classes**

Delete these test classes from `tests/test_ml/test_optimization.py`:
- `TestOptimizationConfig` (old API)
- `TestGenerateParamCombinations` (deleted function)
- `TestCrossValidatedOptimize` (old API)

Keep:
- `TestOptimizationResult`
- `TestScoringMetrics`
- `TestModuleExports`

And the new test classes we added.

**Step 2: Update TestModuleExports**

```python
class TestModuleExports:
    """Test that optimization module is properly exported."""

    def test_import_from_ml_module(self):
        """Test importing from hrp.ml."""
        from hrp.ml.optimization import (
            OptimizationConfig,
            OptimizationResult,
            cross_validated_optimize,
        )
        from optuna.distributions import FloatDistribution

        assert OptimizationConfig is not None
        assert OptimizationResult is not None
        assert cross_validated_optimize is not None
        assert FloatDistribution is not None
```

**Step 3: Run all tests**

Run: `pytest tests/test_ml/test_optimization.py -v`

Expected: All tests pass.

**Step 4: Commit**

```bash
git add tests/test_ml/test_optimization.py
git commit -m "test(optimization): update tests for new Optuna API

- Remove tests for deprecated param_grid API
- Keep result and scoring metric tests
- Update module export tests"
```

---

## Task 9: Update Module Exports

**Files:**
- Modify: `hrp/ml/__init__.py`

**Step 1: Add distribution re-exports**

Update `hrp/ml/__init__.py`:

```python
from hrp.ml.optimization import (
    OptimizationConfig,
    OptimizationResult,
    cross_validated_optimize,
    SCORING_METRICS,
)
# Re-export Optuna distributions for convenience
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)
```

**Step 2: Update __all__**

Add to `__all__`:

```python
    # Optimization
    "OptimizationConfig",
    "OptimizationResult",
    "cross_validated_optimize",
    "SCORING_METRICS",
    "FloatDistribution",
    "IntDistribution",
    "CategoricalDistribution",
```

**Step 3: Verify imports work**

Run: `python -c "from hrp.ml import OptimizationConfig, FloatDistribution; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add hrp/ml/__init__.py
git commit -m "feat(ml): re-export Optuna distributions for convenience"
```

---

## Task 10: Run Full Test Suite

**Step 1: Run all optimization tests**

Run: `pytest tests/test_ml/test_optimization.py -v`

Expected: All tests pass.

**Step 2: Run full ML test suite**

Run: `pytest tests/test_ml/ -v`

Expected: All tests pass (or only pre-existing failures).

**Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short 2>&1 | tail -30`

Expected: Same baseline as before (2688 passed, 17 pre-existing failures).

**Step 4: Final commit if needed**

```bash
git status
# If any uncommitted changes, commit them
```

---

## Summary

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Create OPTUNA_DIR | `feat(config): add OPTUNA_DIR` |
| 2 | Update OptimizationConfig | `feat(optimization): update OptimizationConfig for Optuna` |
| 3 | Implement sampler selection | `feat(optimization): add Optuna sampler selection` |
| 4 | Implement fold evaluation with pruning | `feat(optimization): add fold evaluation with pruning` |
| 5 | Implement main optimization loop | `feat(optimization): implement Optuna-based optimization loop` |
| 6 | Update MLflow logging | `refactor(optimization): update MLflow logging` |
| 7 | Clean up old code | `refactor(optimization): remove deprecated code` |
| 8 | Update existing tests | `test(optimization): update tests for new API` |
| 9 | Update module exports | `feat(ml): re-export Optuna distributions` |
| 10 | Run full test suite | Verification only |
