"""
Signal Generation from ML Predictions.

Converts raw ML predictions to trading signals using various methods:
- rank: Cross-sectional rank, go long top percentile
- threshold: Go long if prediction >= threshold
- zscore: Normalize predictions cross-sectionally (continuous signal)

Usage:
    from hrp.ml.signals import predictions_to_signals

    # Rank-based signals (top 10% go long)
    signals = predictions_to_signals(predictions, method="rank", top_pct=0.1)

    # Threshold-based signals
    signals = predictions_to_signals(predictions, method="threshold", threshold=0.02)

    # Z-score normalized continuous signals
    signals = predictions_to_signals(predictions, method="zscore")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def _rank_signals(predictions: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    """
    Generate binary signals based on cross-sectional rank.

    For each row (date), ranks all predictions and assigns 1.0 to the top
    percentile and 0.0 to the rest.

    Args:
        predictions: DataFrame with dates as index and symbols as columns
        top_pct: Fraction of top-ranked symbols to go long (e.g., 0.1 for top 10%)

    Returns:
        DataFrame with same shape, containing 1.0 for selected symbols, 0.0 otherwise
    """
    signals = pd.DataFrame(0.0, index=predictions.index, columns=predictions.columns)

    for date_idx in predictions.index:
        row = predictions.loc[date_idx]
        n_symbols = len(row)
        n_select = max(1, int(np.ceil(n_symbols * top_pct)))

        # Get indices of top n_select predictions
        top_symbols = row.nlargest(n_select).index
        signals.loc[date_idx, top_symbols] = 1.0

    logger.debug(f"Rank signals generated: top_pct={top_pct}, n_dates={len(predictions)}")
    return signals


def _threshold_signals(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Generate binary signals based on a threshold.

    Assigns 1.0 to symbols with predictions >= threshold, 0.0 otherwise.

    Args:
        predictions: DataFrame with dates as index and symbols as columns
        threshold: Minimum prediction value to go long

    Returns:
        DataFrame with same shape, containing 1.0 or 0.0
    """
    signals = (predictions >= threshold).astype(float)
    logger.debug(f"Threshold signals generated: threshold={threshold}, n_dates={len(predictions)}")
    return signals


def _zscore_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Generate continuous signals by cross-sectionally normalizing predictions.

    For each row (date), computes z-scores so that mean=0 and std=1.
    This produces a continuous signal useful for position sizing.

    Args:
        predictions: DataFrame with dates as index and symbols as columns

    Returns:
        DataFrame with same shape, containing z-score normalized values
    """
    signals = pd.DataFrame(index=predictions.index, columns=predictions.columns, dtype=float)

    for date_idx in predictions.index:
        row = predictions.loc[date_idx]
        mean = row.mean()
        std = row.std()

        if std > 0:
            signals.loc[date_idx] = (row - mean) / std
        else:
            # If all predictions are the same, z-scores are 0
            signals.loc[date_idx] = 0.0

    logger.debug(f"Z-score signals generated: n_dates={len(predictions)}")
    return signals


def predictions_to_signals(
    predictions: pd.DataFrame,
    method: str = "rank",
    top_pct: float = 0.1,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Convert ML predictions to trading signals.

    Args:
        predictions: DataFrame with dates as index and symbols as columns.
                     Values are predicted returns or scores.
        method: Signal generation method:
                - "rank": Cross-sectional rank, go long top percentile
                - "threshold": Go long if prediction >= threshold
                - "zscore": Normalize predictions cross-sectionally
        top_pct: For "rank" method, fraction of symbols to select (default 0.1)
        threshold: For "threshold" method, minimum value to go long (default 0.0)

    Returns:
        DataFrame with same shape as predictions containing signals:
        - For "rank": 1.0 for selected symbols, 0.0 otherwise
        - For "threshold": 1.0 if >= threshold, 0.0 otherwise
        - For "zscore": Continuous z-score values

    Raises:
        ValueError: If method is not one of "rank", "threshold", "zscore"

    Example:
        >>> predictions = pd.DataFrame({
        ...     "AAPL": [0.05, 0.02],
        ...     "MSFT": [0.03, 0.06],
        ... }, index=pd.date_range("2023-01-01", periods=2))
        >>> signals = predictions_to_signals(predictions, method="rank", top_pct=0.5)
        >>> signals.loc["2023-01-01", "AAPL"]  # Highest prediction
        1.0
    """
    valid_methods = {"rank", "threshold", "zscore"}

    if method not in valid_methods:
        raise ValueError(
            f"Unknown method: '{method}'. " f"Valid methods: {', '.join(sorted(valid_methods))}"
        )

    logger.info(f"Generating signals: method={method}, shape={predictions.shape}")

    if method == "rank":
        return _rank_signals(predictions, top_pct)
    elif method == "threshold":
        return _threshold_signals(predictions, threshold)
    else:  # zscore
        return _zscore_signals(predictions)
