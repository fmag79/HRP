"""
Trading Strategy Signal Generators.

Provides signal generation functions for various trading strategies,
including multi-factor and ML-predicted strategies.

Usage:
    from hrp.research.strategies import (
        generate_multifactor_signals,
        generate_ml_predicted_signals,
        STRATEGY_REGISTRY,
    )

    # Multi-factor strategy
    signals = generate_multifactor_signals(
        prices=prices,
        feature_weights={"momentum_20d": 1.0, "volatility_60d": -0.5},
        top_n=10,
    )

    # ML-predicted strategy
    signals = generate_ml_predicted_signals(
        prices=prices,
        model_type="ridge",
        features=["momentum_20d", "volatility_60d"],
        signal_method="rank",
        top_pct=0.1,
    )
"""

from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from hrp.data.features.computation import FeatureComputer
from hrp.ml.models import get_model, SUPPORTED_MODELS
from hrp.ml.signals import predictions_to_signals


def generate_multifactor_signals(
    prices: pd.DataFrame,
    feature_weights: dict[str, float],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Multi-factor strategy combining features with configurable weights.

    For each date:
    1. Fetch features for all symbols
    2. Z-score normalize each factor cross-sectionally
    3. Compute composite = sum(weight * factor)
    4. Rank by composite, select top N

    Args:
        prices: Price data with 'close' level (MultiIndex columns: symbol, field)
        feature_weights: Dict mapping feature names to weights.
                        Positive weights prefer higher values.
                        Negative weights prefer lower values (e.g., volatility).
                        e.g., {"momentum_20d": 1.0, "volatility_60d": -0.5}
        top_n: Number of top stocks to hold

    Returns:
        Signal DataFrame (1 = long, 0 = no position)
        Index = dates, columns = symbols
    """
    if not feature_weights:
        raise ValueError("feature_weights cannot be empty")

    # Extract symbols and date range from prices
    close = prices['close'] if 'close' in prices.columns.get_level_values(0) else prices
    symbols = list(close.columns)
    dates = close.index

    logger.info(
        f"Generating multi-factor signals: {len(feature_weights)} factors, "
        f"top_n={top_n}, {len(symbols)} symbols"
    )

    # Initialize feature computer
    computer = FeatureComputer()

    # Get list of features to compute
    feature_names = list(feature_weights.keys())

    # Load features for all dates
    try:
        features_df = computer.get_stored_features(
            symbols=symbols,
            dates=dates,
            feature_names=feature_names,
        )
    except ValueError as e:
        logger.warning(f"Could not load stored features: {e}. Computing on-the-fly.")
        # Compute features if not stored
        features_df = computer.compute_features(
            symbols=symbols,
            dates=dates,
            feature_names=feature_names,
        )

    if features_df.empty:
        logger.warning("No features available, returning empty signals")
        return pd.DataFrame(0.0, index=dates, columns=symbols)

    # Initialize signals DataFrame
    signals = pd.DataFrame(0.0, index=dates, columns=symbols)

    # Process each date
    for date_idx in dates:
        try:
            # Get features for this date
            date_features = features_df.loc[date_idx]

            if date_features.empty:
                continue

            # Z-score normalize each factor cross-sectionally
            normalized = pd.DataFrame(index=date_features.index)

            for feature_name in feature_names:
                if feature_name in date_features.columns:
                    values = date_features[feature_name]
                    mean = values.mean()
                    std = values.std()

                    if std > 0:
                        normalized[feature_name] = (values - mean) / std
                    else:
                        normalized[feature_name] = 0.0
                else:
                    normalized[feature_name] = 0.0

            # Compute weighted composite score
            composite = pd.Series(0.0, index=normalized.index)
            for feature_name, weight in feature_weights.items():
                if feature_name in normalized.columns:
                    composite += weight * normalized[feature_name]

            # Skip dates with all NaN composites
            valid_composite = composite.dropna()
            if len(valid_composite) == 0:
                continue

            # Select top N symbols by composite score
            n_select = min(top_n, len(valid_composite))
            top_symbols = valid_composite.nlargest(n_select).index.tolist()

            # Set signals for top symbols
            for symbol in top_symbols:
                if symbol in signals.columns:
                    signals.loc[date_idx, symbol] = 1.0

        except KeyError:
            # Date not in features_df
            continue

    logger.info(
        f"Multi-factor signals generated: {signals.sum().sum():.0f} total positions"
    )

    return signals


def generate_ml_predicted_signals(
    prices: pd.DataFrame,
    model_type: str = "ridge",
    features: list[str] | None = None,
    signal_method: str = "rank",
    top_pct: float = 0.1,
    threshold: float = 0.0,
    train_lookback: int = 252,
    retrain_frequency: int = 21,
) -> pd.DataFrame:
    """
    ML-predicted strategy using trained model.

    For each rebalance date:
    1. Train model on past `train_lookback` days of data
    2. Generate predictions for all symbols
    3. Convert to signals via predictions_to_signals()

    Args:
        prices: Price data with 'close' level (MultiIndex columns: symbol, field)
        model_type: Model type from SUPPORTED_MODELS
                   ("ridge", "lasso", "random_forest", "lightgbm", "xgboost")
        features: List of feature names to use. If None, uses default features.
        signal_method: Signal generation method ("rank", "threshold", "zscore")
        top_pct: For rank method, fraction of symbols to select (default 0.1)
        threshold: For threshold method, minimum prediction value
        train_lookback: Days of historical data for training (default 252)
        retrain_frequency: Days between model retraining (default 21 = monthly)

    Returns:
        Signal DataFrame (1 = long, 0 = no position for rank/threshold)
        Index = dates, columns = symbols
    """
    if model_type not in SUPPORTED_MODELS:
        available = ", ".join(sorted(SUPPORTED_MODELS.keys()))
        raise ValueError(f"Unsupported model_type: '{model_type}'. Available: {available}")

    # Extract symbols and date range from prices
    close = prices['close'] if 'close' in prices.columns.get_level_values(0) else prices
    symbols = list(close.columns)
    dates = close.index

    # Default features
    if features is None:
        features = ["momentum_20d", "volatility_60d"]

    logger.info(
        f"Generating ML-predicted signals: model={model_type}, "
        f"{len(features)} features, method={signal_method}, {len(symbols)} symbols"
    )

    # Initialize feature computer
    computer = FeatureComputer()

    # Target: forward 20-day returns (for training)
    # We'll compute returns from prices directly
    target_horizon = 20
    returns_20d = close.pct_change(target_horizon).shift(-target_horizon)  # Forward returns

    # Initialize predictions DataFrame
    predictions = pd.DataFrame(np.nan, index=dates, columns=symbols)

    # Determine rebalance dates (every retrain_frequency days)
    rebalance_indices = list(range(train_lookback, len(dates), retrain_frequency))

    logger.info(f"Processing {len(rebalance_indices)} rebalance dates")

    for idx in rebalance_indices:
        current_date = dates[idx]

        # Training window: from idx - train_lookback to idx - 1 - target_horizon
        # We shift train_end back by target_horizon because the forward return
        # target for date t requires knowing prices at t + target_horizon.
        # Without this shift, training uses targets that peek into the future
        # relative to the prediction date (idx).
        train_start_idx = idx - train_lookback
        train_end_idx = idx - 1 - target_horizon

        if train_start_idx < 0 or train_end_idx < train_start_idx:
            continue

        train_dates = dates[train_start_idx:train_end_idx + 1]

        try:
            # Load features for training period
            train_features = computer.get_stored_features(
                symbols=symbols,
                dates=train_dates,
                feature_names=features,
            )
        except ValueError:
            # Try computing if not stored
            try:
                train_features = computer.compute_features(
                    symbols=symbols,
                    dates=train_dates,
                    feature_names=features,
                )
            except Exception as e:
                logger.warning(f"Could not compute features for {current_date}: {e}")
                continue

        if train_features.empty:
            continue

        # Get target (forward returns) for training period
        train_targets = returns_20d.loc[train_dates]

        # Prepare training data (flatten multi-index)
        X_train = train_features.reset_index()
        y_train_list = []

        for _, row in X_train.iterrows():
            date_val = row['date']
            symbol = row['symbol']
            if hasattr(date_val, 'strftime'):
                # Already datetime
                pass
            else:
                date_val = pd.to_datetime(date_val)

            if date_val in train_targets.index and symbol in train_targets.columns:
                y_train_list.append(train_targets.loc[date_val, symbol])
            else:
                y_train_list.append(np.nan)

        X_train = X_train.drop(columns=['date', 'symbol'], errors='ignore')
        y_train = pd.Series(y_train_list)

        # Drop NaN rows
        valid_mask = ~y_train.isna() & X_train.notna().all(axis=1)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        if len(X_train) < 50:  # Minimum samples for training
            continue

        # Train model
        model = get_model(model_type)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"Model training failed at {current_date}: {e}")
            continue

        # Load features for prediction (current date)
        try:
            pred_features = computer.get_stored_features(
                symbols=symbols,
                dates=[current_date],
                feature_names=features,
            )
        except ValueError:
            try:
                pred_features = computer.compute_features(
                    symbols=symbols,
                    dates=[current_date],
                    feature_names=features,
                )
            except Exception:
                continue

        if pred_features.empty:
            continue

        # Generate predictions
        X_pred = pred_features.reset_index()
        pred_symbols = X_pred['symbol'].tolist()
        X_pred = X_pred.drop(columns=['date', 'symbol'], errors='ignore')

        # Handle missing features
        valid_pred_mask = X_pred.notna().all(axis=1)
        X_pred_valid = X_pred[valid_pred_mask]
        pred_symbols_valid = [s for s, v in zip(pred_symbols, valid_pred_mask) if v]

        if len(X_pred_valid) == 0:
            continue

        try:
            preds = model.predict(X_pred_valid)
        except Exception as e:
            logger.warning(f"Prediction failed at {current_date}: {e}")
            continue

        # Store predictions
        for symbol, pred in zip(pred_symbols_valid, preds):
            if symbol in predictions.columns:
                predictions.loc[current_date, symbol] = pred

        # Forward fill predictions until next rebalance
        next_rebal_idx = min(idx + retrain_frequency, len(dates))
        for future_idx in range(idx + 1, next_rebal_idx):
            future_date = dates[future_idx]
            for symbol, pred in zip(pred_symbols_valid, preds):
                if symbol in predictions.columns:
                    predictions.loc[future_date, symbol] = pred

    # Convert predictions to signals
    valid_predictions = predictions.dropna(how='all')

    if valid_predictions.empty:
        logger.warning("No valid predictions generated, returning empty signals")
        return pd.DataFrame(0.0, index=dates, columns=symbols)

    signals = predictions_to_signals(
        predictions=valid_predictions,
        method=signal_method,
        top_pct=top_pct,
        threshold=threshold,
    )

    # Reindex to original dates and fill with 0
    signals = signals.reindex(dates).fillna(0.0)

    logger.info(
        f"ML-predicted signals generated: {signals.sum().sum():.0f} total positions"
    )

    return signals


def generate_regime_switching_signals(
    prices: pd.DataFrame,
    regime_config: dict | None = None,
    bull_weights: dict[str, float] | None = None,
    bear_weights: dict[str, float] | None = None,
    sideways_weights: dict[str, float] | None = None,
    top_n: int = 10,
    lookback: int = 252,
    retrain_frequency: int = 63,
) -> pd.DataFrame:
    """
    Regime-switching strategy that adapts factor weights based on HMM regime detection.

    1. Fit HMM to detect current regime (bull/bear/sideways)
    2. Get regime probabilities for smooth blending
    3. Apply regime-specific factor weights to generate composite signal
    4. Select top-N stocks by blended score

    Default weights:
    - Bull: momentum_60d=1.0, volume_ratio=0.5, trend=0.5
    - Bear: volatility_60d=-1.0, dividend_yield=1.0, pb_ratio=-0.5 (defensive)
    - Sideways: rsi_14d=-1.0, price_to_sma_20d=-1.0, bb_width_20d=-0.5 (mean-reversion)

    Args:
        prices: Price data with 'close' level (MultiIndex columns: symbol, field)
        regime_config: Optional config dict for HMM (n_regimes, features, etc.)
        bull_weights: Factor weights for bull regime (default: momentum-focused)
        bear_weights: Factor weights for bear regime (default: defensive)
        sideways_weights: Factor weights for sideways regime (default: mean-reversion)
        top_n: Number of top stocks to hold
        lookback: Lookback period for HMM training (default: 252 days = 1 year)
        retrain_frequency: How often to refit HMM (default: 63 days = quarterly)

    Returns:
        Signal DataFrame (1 = long, 0 = no position)
        Index = dates, columns = symbols
    """
    from hrp.ml.regime import RegimeDetector, HMMConfig

    # Set default weights if not provided
    if bull_weights is None:
        bull_weights = {
            "momentum_60d": 1.0,
            "volume_ratio": 0.5,
            "trend": 0.5,
        }

    if bear_weights is None:
        bear_weights = {
            "volatility_60d": -1.0,
            "dividend_yield": 1.0,
            "pb_ratio": -0.5,
        }

    if sideways_weights is None:
        sideways_weights = {
            "rsi_14d": -1.0,
            "price_to_sma_20d": -1.0,
            "bb_width_20d": -0.5,
        }

    # Set default regime config if not provided
    if regime_config is None:
        regime_config = {
            "n_regimes": 3,
            "features": ["returns_20d", "volatility_20d"],
            "covariance_type": "full",
        }

    # Extract symbols and date range from prices
    close = prices['close'] if 'close' in prices.columns.get_level_values(0) else prices
    symbols = list(close.columns)
    dates = close.index

    logger.info(
        f"Generating regime-switching signals: {len(symbols)} symbols, "
        f"top_n={top_n}, lookback={lookback}"
    )

    # Initialize regime detector
    hmm_config = HMMConfig(**regime_config)
    detector = RegimeDetector(hmm_config)

    # Determine rebalance dates for HMM refitting
    rebalance_indices = list(range(lookback, len(dates), retrain_frequency))

    logger.info(f"Regime refitting at {len(rebalance_indices)} dates")

    # Initialize signals DataFrame
    signals = pd.DataFrame(0.0, index=dates, columns=symbols)

    # Process each rebalance period
    for idx in rebalance_indices:
        current_date = dates[idx]

        # Training window for HMM
        train_start_idx = max(0, idx - lookback)
        train_dates = dates[train_start_idx:idx]

        # Fit HMM on historical prices
        try:
            detector.fit(close.loc[train_dates])
        except Exception as e:
            logger.warning(f"HMM fit failed at {current_date}: {e}")
            continue

        # Get regime probabilities for prediction period
        pred_start_idx = idx
        pred_end_idx = min(idx + retrain_frequency, len(dates))
        pred_dates = dates[pred_start_idx:pred_end_idx]

        try:
            regime_proba = detector.predict_proba(close.loc[pred_dates])
        except Exception as e:
            logger.warning(f"Regime prediction failed at {current_date}: {e}")
            continue

        # Generate signals for each date in prediction period
        for date in pred_dates:
            try:
                # Get current regime probabilities
                if date not in regime_proba.index:
                    continue

                probs = regime_proba.loc[date]

                # Get regime-specific signals
                regime_signals = {}

                for regime, weights in [
                    ("bull", bull_weights),
                    ("bear", bear_weights),
                    ("sideways", sideways_weights),
                ]:
                    if regime in probs and probs[regime] > 0:
                        # Generate signals for this regime
                        regime_signal = generate_multifactor_signals(
                            prices=close.loc[:date],
                            feature_weights=weights,
                            top_n=top_n,
                        )

                        # Get signal for this date (most recent)
                        if not regime_signal.empty:
                            regime_signals[regime] = regime_signal.tail(1).iloc[0]

                # Blend signals by regime probability
                if regime_signals:
                    blended_signal = pd.Series(0.0, index=symbols)

                    for regime, signal in regime_signals.items():
                        if regime in probs:
                            blended_signal += signal * probs[regime]

                    # Select top N by blended score
                    top_stocks = blended_signal.nlargest(top_n).index
                    signals.loc[date, top_stocks] = 1.0

            except Exception as e:
                logger.warning(f"Signal generation failed at {date}: {e}")
                continue

    logger.info(
        f"Regime-switching signals generated: {signals.sum().sum():.0f} total positions"
    )

    return signals


# Strategy registry for dashboard integration
STRATEGY_REGISTRY: dict[str, dict] = {
    "momentum": {
        "name": "Momentum",
        "description": "Long top N stocks by trailing return",
        "generator": "generate_momentum_signals",  # From backtest.py
        "params": ["lookback", "top_n"],
    },
    "multifactor": {
        "name": "Multi-Factor",
        "description": "Combine multiple factors with configurable weights",
        "generator": "generate_multifactor_signals",
        "params": ["feature_weights", "top_n"],
    },
    "ml_predicted": {
        "name": "ML-Predicted",
        "description": "Use trained ML model predictions as signals",
        "generator": "generate_ml_predicted_signals",
        "params": [
            "model_type",
            "features",
            "signal_method",
            "top_pct",
            "train_lookback",
            "retrain_frequency",
        ],
    },
    "regime_switching": {
        "name": "Regime Switching",
        "description": "Adaptive strategy that blends signals based on HMM regime detection",
        "generator": "generate_regime_switching_signals",
        "params": [
            "regime_config",
            "bull_weights",
            "bear_weights",
            "sideways_weights",
            "top_n",
            "lookback",
            "retrain_frequency",
        ],
    },
}


# Named presets for common strategy configurations
PRESET_STRATEGIES: dict[str, dict] = {
    "mean_reversion": {
        "name": "Mean Reversion",
        "description": "Buy oversold stocks expecting bounce-back",
        "feature_weights": {
            "rsi_14d": -1.0,
            "price_to_sma_20d": -1.0,
            "bb_width_20d": 1.0,
        },
        "default_top_n": 10,
    },
    "trend_following": {
        "name": "Trend Following",
        "description": "Ride established trends with strong momentum",
        "feature_weights": {
            "trend": 1.0,
            "adx_14d": 1.0,
            "macd_histogram": 1.0,
        },
        "default_top_n": 10,
    },
    "quality_momentum": {
        "name": "Quality Momentum",
        "description": "Momentum filtered by low volatility",
        "feature_weights": {
            "momentum_60d": 1.0,
            "volatility_60d": -1.0,
            "atr_14d": -0.5,
        },
        "default_top_n": 10,
    },
    "volume_breakout": {
        "name": "Volume Breakout",
        "description": "Detect accumulation via volume surge",
        "feature_weights": {
            "volume_ratio": 1.0,
            "obv": 1.0,
            "momentum_20d": 0.5,
        },
        "default_top_n": 10,
    },
    "regime_adaptive": {
        "name": "Regime Adaptive",
        "description": "Blends momentum, defensive, and mean-reversion based on market regime",
        "bull_weights": {"momentum_60d": 1.0, "volume_ratio": 0.5, "trend": 0.5},
        "bear_weights": {"volatility_60d": -1.0, "dividend_yield": 1.0, "pb_ratio": -0.5},
        "sideways_weights": {"rsi_14d": -1.0, "price_to_sma_20d": -1.0, "bb_width_20d": -0.5},
        "top_n": 10,
        "lookback": 252,
        "retrain_frequency": 63,
    },
}


def get_preset_strategy(preset_name: str) -> dict:
    """
    Get preset strategy configuration.

    Args:
        preset_name: One of the keys in PRESET_STRATEGIES

    Returns:
        Dictionary with feature_weights and top_n

    Raises:
        ValueError: If preset_name not in registry
    """
    if preset_name not in PRESET_STRATEGIES:
        available = ", ".join(sorted(PRESET_STRATEGIES.keys()))
        raise ValueError(f"Unknown preset: '{preset_name}'. Available: {available}")

    preset = PRESET_STRATEGIES[preset_name]
    return {
        "feature_weights": preset["feature_weights"].copy(),
        "top_n": preset["default_top_n"],
    }


def get_strategy_generator(strategy_type: str) -> Callable:
    """
    Get the signal generator function for a strategy type.

    Args:
        strategy_type: One of the keys in STRATEGY_REGISTRY

    Returns:
        Signal generator function

    Raises:
        ValueError: If strategy_type not in registry
    """
    from hrp.research.backtest import generate_momentum_signals

    generators = {
        "momentum": generate_momentum_signals,
        "multifactor": generate_multifactor_signals,
        "ml_predicted": generate_ml_predicted_signals,
    }

    if strategy_type not in generators:
        available = ", ".join(sorted(generators.keys()))
        raise ValueError(f"Unknown strategy: '{strategy_type}'. Available: {available}")

    return generators[strategy_type]
