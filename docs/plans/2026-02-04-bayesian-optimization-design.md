# Design: Bayesian Optimization with Optuna

## Overview

Replace custom grid/random search in `hrp/ml/optimization.py` with Optuna, enabling Bayesian hyperparameter optimization via TPE sampler.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration | Replace grid/random entirely | Single backend, less code to maintain |
| Param definition | Optuna-native distributions | No translation layer, full Optuna power |
| Trial tracking | HyperparameterTrialCounter authority | Preserves existing audit trail in DuckDB |
| Pruning | MedianPruner enabled | Saves compute on unpromising trials |
| Persistence | SQLite in `~/hrp-data/optuna/` | Resumable optimizations, history analysis |
| Sampler names | `grid`, `random`, `tpe`, `cmaes` | Precise naming, extensible |

## New API

```python
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

config = OptimizationConfig(
    model_type="ridge",
    target="returns_20d",
    features=["momentum_20d", "volatility_20d"],
    param_space={
        "alpha": FloatDistribution(0.01, 100.0, log=True),
    },
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    sampler="tpe",  # "grid", "random", "tpe", "cmaes"
    n_trials=50,
    enable_pruning=True,
    hypothesis_id="HYP-2026-001",
)

result = cross_validated_optimize(config, symbols=["AAPL", "MSFT"])
```

## Breaking Changes

| Old | New |
|-----|-----|
| `param_grid={"alpha": [0.1, 1.0]}` | `param_space={"alpha": FloatDistribution(0.1, 1.0, log=True)}` |
| `search_type="grid"` | `sampler="grid"` |
| `search_type="random"` | `sampler="random"` |
| `n_random_samples=20` | Removed (use `n_trials`) |
| `max_trials=50` | `n_trials=50` |

## Implementation Structure

### Core Flow

```python
def cross_validated_optimize(config, symbols, ...):
    # 1. Initialize trial counter (existing overfitting guard)
    trial_counter = HyperparameterTrialCounter(config.hypothesis_id, config.n_trials)

    # 2. Create Optuna study with SQLite persistence
    storage = f"sqlite:///{Path.home()}/hrp-data/optuna/{config.hypothesis_id}.db"
    study = optuna.create_study(
        study_name=config.hypothesis_id,
        storage=storage,
        load_if_exists=True,
        direction="maximize" if higher_is_better else "minimize",
        sampler=_get_sampler(config.sampler, config.param_space),
        pruner=MedianPruner() if config.enable_pruning else None,
    )

    # 3. Fetch data once
    all_data = _fetch_features(...)
    folds = generate_folds(...)

    # 4. Define objective with pruning
    def objective(trial):
        if not trial_counter.can_try():
            raise TrialPruned("Trial limit exceeded")

        params = {name: trial._suggest(name, dist) for name, dist in config.param_space.items()}
        score = _evaluate_with_pruning(trial, params, ...)

        trial_counter.log_trial(...)
        return score

    # 5. Run optimization
    study.optimize(objective, n_trials=trial_counter.remaining_trials)

    return _build_result(study, config)
```

### Fold Evaluation with Pruning

```python
def _evaluate_with_pruning(trial, params, config, all_data, folds):
    fold_scores = []
    decay_monitor = SharpeDecayMonitor(config.early_stop_decay_threshold)

    for fold_idx, fold in enumerate(folds):
        # Train and evaluate
        score = _evaluate_fold(params, config, all_data, fold)
        fold_scores.append(score)

        # Report intermediate for pruning
        trial.report(np.mean(fold_scores), fold_idx)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # Check Sharpe decay
        if not decay_monitor.check(train_sharpe, test_sharpe).passed:
            raise optuna.TrialPruned("Sharpe decay exceeded")

    return np.mean(fold_scores)
```

### Sampler Selection

```python
def _get_sampler(sampler_name, param_space):
    if sampler_name == "grid":
        return GridSampler(_distributions_to_grid(param_space))
    elif sampler_name == "random":
        return RandomSampler(seed=42)
    elif sampler_name == "tpe":
        return TPESampler(seed=42)
    elif sampler_name == "cmaes":
        return CmaEsSampler(seed=42)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")
```

## Files to Modify

| File | Changes |
|------|---------|
| `hrp/ml/optimization.py` | Replace implementation (~200 lines rewritten) |
| `tests/test_ml/test_optimization.py` | Update for new API |
| Callers using old API | Update param_grid → param_space |

## New Artifacts

- `~/hrp-data/optuna/` directory for study SQLite files
- One `.db` file per hypothesis_id

## Benefits

1. **~150 lines deleted** - Custom optimization loop replaced by Optuna
2. **Pruning** - Early-stop bad trials mid-fold, saving compute
3. **Resumable** - Interrupted optimizations can continue
4. **Better search** - TPE finds better hyperparameters with fewer trials
5. **Extensible** - CMA-ES available for continuous optimization
6. **Visualization** - Optuna's built-in plotting for analysis
