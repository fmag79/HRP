"""
Feature computation engine for HRP.

Computes features at specific versions to ensure reproducibility.
"""

import json
from datetime import date
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.data.features.registry import FeatureRegistry


# Feature computation functions
def compute_momentum_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day momentum (trailing return).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with momentum values
    """
    # Extract close prices
    close = prices["close"].unstack(level="symbol")

    # Calculate 20-day return
    momentum = close.pct_change(20)

    # Stack back to multi-index format
    result = momentum.stack(level="symbol", future_stack=True)

    return result.to_frame(name="momentum_20d")


def compute_volatility_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 60-day volatility (annualized standard deviation of returns).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with volatility values
    """
    # Extract close prices
    close = prices["close"].unstack(level="symbol")

    # Calculate daily returns
    returns = close.pct_change()

    # Calculate 60-day rolling volatility (annualized)
    volatility = returns.rolling(window=60).std() * (252**0.5)

    # Stack back to multi-index format
    result = volatility.stack(level="symbol", future_stack=True)

    return result.to_frame(name="volatility_60d")


def compute_returns_1d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 1-day return.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with 1-day return values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change(1)
    result = returns.stack(level="symbol", future_stack=True)
    return result.to_frame(name="returns_1d")


def compute_returns_5d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 5-day return.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with 5-day return values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change(5)
    result = returns.stack(level="symbol", future_stack=True)
    return result.to_frame(name="returns_5d")


def compute_returns_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day return.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with 20-day return values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change(20)
    result = returns.stack(level="symbol", future_stack=True)
    return result.to_frame(name="returns_20d")


def compute_momentum_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 60-day momentum (trailing return).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with 60-day momentum values
    """
    close = prices["close"].unstack(level="symbol")
    momentum = close.pct_change(60)
    result = momentum.stack(level="symbol", future_stack=True)
    return result.to_frame(name="momentum_60d")


def compute_volatility_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day volatility (annualized standard deviation of returns).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with 20-day volatility values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change()
    volatility = returns.rolling(window=20).std() * (252**0.5)
    result = volatility.stack(level="symbol", future_stack=True)
    return result.to_frame(name="volatility_20d")


def compute_volume_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day average volume.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'volume' column

    Returns:
        DataFrame with 20-day average volume values
    """
    volume = prices["volume"].unstack(level="symbol")
    avg_volume = volume.rolling(window=20).mean()
    result = avg_volume.stack(level="symbol", future_stack=True)
    return result.to_frame(name="volume_20d")


def compute_rsi_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over 14 periods

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with RSI values (0-100 scale)
    """
    close = prices["close"].unstack(level="symbol")

    # Calculate price changes
    delta = close.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate average gain and loss using exponential moving average
    avg_gain = gains.ewm(span=14, adjust=False).mean()
    avg_loss = losses.ewm(span=14, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases (no losses = RSI of 100, no gains = RSI of 0)
    rsi = rsi.replace([np.inf, -np.inf], np.nan)

    result = rsi.stack(level="symbol", future_stack=True)
    return result.to_frame(name="rsi_14d")


def compute_sma_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Simple Moving Average.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with SMA values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    result = sma.stack(level="symbol", future_stack=True)
    return result.to_frame(name="sma_20d")


def compute_sma_50d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 50-day Simple Moving Average.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with SMA values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=50).mean()
    result = sma.stack(level="symbol", future_stack=True)
    return result.to_frame(name="sma_50d")


def compute_sma_200d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 200-day Simple Moving Average.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with SMA values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=200).mean()
    result = sma.stack(level="symbol", future_stack=True)
    return result.to_frame(name="sma_200d")


def compute_price_to_sma_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price-to-SMA-20 ratio (close / sma_20d).

    Values > 1 indicate price above SMA (bullish), < 1 indicates below (bearish).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with price-to-SMA ratio values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    ratio = close / sma
    result = ratio.stack(level="symbol", future_stack=True)
    return result.to_frame(name="price_to_sma_20d")


def compute_price_to_sma_50d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price-to-SMA-50 ratio (close / sma_50d).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with price-to-SMA ratio values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=50).mean()
    ratio = close / sma
    result = ratio.stack(level="symbol", future_stack=True)
    return result.to_frame(name="price_to_sma_50d")


def compute_price_to_sma_200d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price-to-SMA-200 ratio (close / sma_200d).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with price-to-SMA ratio values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=200).mean()
    ratio = close / sma
    result = ratio.stack(level="symbol", future_stack=True)
    return result.to_frame(name="price_to_sma_200d")


def compute_volume_ratio(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume ratio (current volume / 20-day average volume).

    Values > 1 indicate higher than average volume (increased interest).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'volume' column

    Returns:
        DataFrame with volume ratio values
    """
    volume = prices["volume"].unstack(level="symbol")
    avg_volume = volume.rolling(window=20).mean()
    ratio = volume / avg_volume
    result = ratio.stack(level="symbol", future_stack=True)
    return result.to_frame(name="volume_ratio")


def compute_returns_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 60-day return (quarterly).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with return values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change(60)
    result = returns.stack(level="symbol", future_stack=True)
    return result.to_frame(name="returns_60d")


def compute_returns_252d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 252-day return (annual).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with return values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change(252)
    result = returns.stack(level="symbol", future_stack=True)
    return result.to_frame(name="returns_252d")


def compute_momentum_252d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 252-day momentum (annual trailing return).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with momentum values
    """
    close = prices["close"].unstack(level="symbol")
    momentum = close.pct_change(252)
    result = momentum.stack(level="symbol", future_stack=True)
    return result.to_frame(name="momentum_252d")


def compute_obv(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV).

    OBV is a cumulative indicator that adds volume on up days
    and subtracts volume on down days.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close', 'volume' columns

    Returns:
        DataFrame with OBV values
    """
    close = prices["close"].unstack(level="symbol")
    volume = prices["volume"].unstack(level="symbol")

    # Calculate price direction: +1 for up, -1 for down, 0 for unchanged
    direction = np.sign(close.diff())

    # OBV = cumulative sum of signed volume
    obv = (direction * volume).cumsum()

    result = obv.stack(level="symbol", future_stack=True)
    return result.to_frame(name="obv")


def compute_atr_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Average True Range (ATR).

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = 14-day EMA of True Range

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with ATR values
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    prev_close = close.shift(1)

    # True Range components
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # True Range = max of the three components
    true_range = pd.concat([tr1, tr2, tr3], keys=['tr1', 'tr2', 'tr3']).groupby(level=1).max()

    # ATR = 14-day EMA of True Range
    atr = true_range.ewm(span=14, adjust=False).mean()

    result = atr.stack(level="symbol", future_stack=True)
    return result.to_frame(name="atr_14d")


def compute_adx_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Average Directional Index (ADX).

    ADX measures trend strength (0-100) regardless of direction.
    Values > 25 indicate strong trend, < 20 indicate weak/no trend.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with ADX values (0-100 scale)
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], keys=['tr1', 'tr2', 'tr3']).groupby(level=1).max()

    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM and -DM
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Smoothed TR, +DM, -DM (14-period EMA)
    atr = true_range.ewm(span=14, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(span=14, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(span=14, adjust=False).mean()

    # +DI and -DI
    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=14, adjust=False).mean()

    # Handle edge cases
    adx = adx.replace([np.inf, -np.inf], np.nan)

    result = adx.stack(level="symbol", future_stack=True)
    return result.to_frame(name="adx_14d")


def compute_macd_line(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD Line (EMA-12 minus EMA-26).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with MACD line values
    """
    close = prices["close"].unstack(level="symbol")
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26

    result = macd_line.stack(level="symbol", future_stack=True)
    return result.to_frame(name="macd_line")


def compute_macd_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD Signal Line (9-day EMA of MACD line).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with MACD signal values
    """
    close = prices["close"].unstack(level="symbol")
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal = macd_line.ewm(span=9, adjust=False).mean()

    result = signal.stack(level="symbol", future_stack=True)
    return result.to_frame(name="macd_signal")


def compute_macd_histogram(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD Histogram (MACD line minus Signal line).

    Positive histogram = bullish momentum, Negative = bearish.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with MACD histogram values
    """
    close = prices["close"].unstack(level="symbol")
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal

    result = histogram.stack(level="symbol", future_stack=True)
    return result.to_frame(name="macd_histogram")


def compute_cci_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Commodity Channel Index (CCI).

    CCI = (Typical Price - SMA of TP) / (0.015 * Mean Deviation)
    where Typical Price = (High + Low + Close) / 3

    Values > +100 indicate overbought, < -100 indicate oversold.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with CCI values
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    # Typical Price
    tp = (high + low + close) / 3

    # SMA of Typical Price
    tp_sma = tp.rolling(window=20).mean()

    # Mean Deviation
    mean_dev = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())

    # CCI
    cci = (tp - tp_sma) / (0.015 * mean_dev)

    result = cci.stack(level="symbol", future_stack=True)
    return result.to_frame(name="cci_20d")


def compute_roc_10d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 10-day Rate of Change (ROC).

    ROC = ((close - close_n_periods_ago) / close_n_periods_ago) * 100

    Expressed as percentage change.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with ROC values (percentage)
    """
    close = prices["close"].unstack(level="symbol")
    roc = close.pct_change(10) * 100

    result = roc.stack(level="symbol", future_stack=True)
    return result.to_frame(name="roc_10d")


def compute_trend(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trend direction based on price vs 200-day SMA.

    Returns +1 if price > SMA-200 (uptrend), -1 if price < SMA-200 (downtrend).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with trend values (+1 or -1)
    """
    close = prices["close"].unstack(level="symbol")
    sma_200 = close.rolling(window=200).mean()

    # +1 for uptrend (price above SMA), -1 for downtrend
    trend = np.sign(close - sma_200)

    result = trend.stack(level="symbol", future_stack=True)
    return result.to_frame(name="trend")


def compute_bb_upper_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Bollinger Band upper band (SMA + 2*std).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with upper band values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    upper = sma + 2 * std

    result = upper.stack(level="symbol", future_stack=True)
    return result.to_frame(name="bb_upper_20d")


def compute_bb_lower_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Bollinger Band lower band (SMA - 2*std).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with lower band values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    lower = sma - 2 * std

    result = lower.stack(level="symbol", future_stack=True)
    return result.to_frame(name="bb_lower_20d")


def compute_bb_width_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Bollinger Band width ((upper - lower) / middle).

    Measures volatility - higher width = higher volatility.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with band width values
    """
    close = prices["close"].unstack(level="symbol")
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    width = (4 * std) / sma  # (upper - lower) / middle = 4*std / sma

    result = width.stack(level="symbol", future_stack=True)
    return result.to_frame(name="bb_width_20d")


def compute_stoch_k_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Stochastic %K.

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with %K values (0-100)
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    result = stoch_k.stack(level="symbol", future_stack=True)
    return result.to_frame(name="stoch_k_14d")


def compute_stoch_d_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Stochastic %D (3-day SMA of %K).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with %D values (0-100)
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=3).mean()

    result = stoch_d.stack(level="symbol", future_stack=True)
    return result.to_frame(name="stoch_d_14d")


def compute_ema_12d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 12-day Exponential Moving Average.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with EMA-12 values
    """
    close = prices["close"].unstack(level="symbol")
    ema = close.ewm(span=12, adjust=False).mean()
    result = ema.stack(level="symbol", future_stack=True)
    return result.to_frame(name="ema_12d")


def compute_ema_26d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 26-day Exponential Moving Average.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with EMA-26 values
    """
    close = prices["close"].unstack(level="symbol")
    ema = close.ewm(span=26, adjust=False).mean()
    result = ema.stack(level="symbol", future_stack=True)
    return result.to_frame(name="ema_26d")


def compute_ema_crossover(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate EMA crossover signal.

    Returns +1 when EMA-12 > EMA-26 (bullish), -1 when EMA-12 < EMA-26 (bearish).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with crossover signal values (+1 or -1)
    """
    close = prices["close"].unstack(level="symbol")
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    # +1 for bullish (EMA-12 > EMA-26), -1 for bearish
    crossover = np.sign(ema_12 - ema_26)

    result = crossover.stack(level="symbol", future_stack=True)
    return result.to_frame(name="ema_crossover")


def compute_williams_r_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Williams %R.

    Williams %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100

    Range: -100 to 0 (oversold < -80, overbought > -20)

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'high', 'low', 'close' columns

    Returns:
        DataFrame with Williams %R values (-100 to 0)
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")

    highest_high = high.rolling(window=14).max()
    lowest_low = low.rolling(window=14).min()

    # Williams %R formula
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100

    result = williams_r.stack(level="symbol", future_stack=True)
    return result.to_frame(name="williams_r_14d")


def compute_mfi_14d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 14-day Money Flow Index.

    MFI is a volume-weighted RSI that uses typical price.
    Range: 0 to 100 (oversold < 20, overbought > 80)

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and
                'high', 'low', 'close', 'volume' columns

    Returns:
        DataFrame with MFI values (0-100)
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")
    volume = prices["volume"].unstack(level="symbol")

    # Typical Price
    tp = (high + low + close) / 3

    # Raw Money Flow
    raw_mf = tp * volume

    # Money Flow Direction
    tp_diff = tp.diff()
    positive_mf = raw_mf.where(tp_diff > 0, 0.0)
    negative_mf = raw_mf.where(tp_diff < 0, 0.0)

    # 14-period sums
    positive_mf_sum = positive_mf.rolling(window=14).sum()
    negative_mf_sum = negative_mf.rolling(window=14).sum()

    # Money Flow Ratio and MFI
    mfr = positive_mf_sum / negative_mf_sum
    mfi = 100 - (100 / (1 + mfr))

    # Handle edge cases
    mfi = mfi.replace([np.inf, -np.inf], np.nan)

    result = mfi.stack(level="symbol", future_stack=True)
    return result.to_frame(name="mfi_14d")


def compute_vwap_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day rolling VWAP approximation.

    Note: True VWAP requires intraday data. This is a daily approximation
    using typical price weighted by volume over a 20-day window.

    VWAP = Sum(Typical Price * Volume) / Sum(Volume)

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and
                'high', 'low', 'close', 'volume' columns

    Returns:
        DataFrame with VWAP approximation values
    """
    high = prices["high"].unstack(level="symbol")
    low = prices["low"].unstack(level="symbol")
    close = prices["close"].unstack(level="symbol")
    volume = prices["volume"].unstack(level="symbol")

    # Typical Price
    tp = (high + low + close) / 3

    # VWAP = Sum(TP * Volume) / Sum(Volume) over 20 days
    tp_volume = tp * volume
    vwap = tp_volume.rolling(window=20).sum() / volume.rolling(window=20).sum()

    result = vwap.stack(level="symbol", future_stack=True)
    return result.to_frame(name="vwap_20d")


# =============================================================================
# Statistical Features (Phase 1: Factor Library Expansion)
# =============================================================================


def compute_autocorrelation_5d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 5-day return autocorrelation (rolling 60-day window).

    Negative autocorrelation suggests mean-reversion; positive suggests momentum.
    Measures the correlation between returns at time t and t-5 days.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with autocorrelation values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change(5)

    # Rolling autocorrelation with lag=1 over 60-day window
    autocorr = returns.rolling(window=60).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )

    result = autocorr.stack(level="symbol", future_stack=True)
    return result.to_frame(name="autocorrelation_5d")


def compute_skewness_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 60-day return skewness.

    Negative skew indicates asymmetric risk of large negative returns (crash risk).
    Positive skew indicates asymmetric upside potential.
    Zero skew indicates symmetric returns (normal distribution).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with skewness values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change()

    # Rolling skewness over 60-day window
    skew = returns.rolling(window=60).skew()

    result = skew.stack(level="symbol", future_stack=True)
    return result.to_frame(name="skewness_60d")


def compute_kurtosis_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 60-day return kurtosis.

    High kurtosis indicates fat tails (extreme events more likely than normal).
    Low kurtosis indicates platykurtic distribution (fewer extreme events).
    Normal distribution has kurtosis of 3.0 (excess kurtosis = 0).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with kurtosis values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change()

    # Rolling kurtosis over 60-day window
    kurt = returns.rolling(window=60).kurt()

    result = kurt.stack(level="symbol", future_stack=True)
    return result.to_frame(name="kurtosis_60d")


def compute_downside_vol_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 60-day downside volatility (annualized).

    Measures volatility of negative returns only, capturing downside risk.
    Ignores positive returns - focuses on the "bad" volatility.
    Lower values indicate more stable downside.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with downside volatility values
    """
    close = prices["close"].unstack(level="symbol")
    returns = close.pct_change()

    # Keep only negative returns (downside)
    downside = returns.where(returns < 0, 0.0)

    # Standard deviation of downside returns, annualized
    downside_std = downside.rolling(window=60).std() * (252**0.5)

    result = downside_std.stack(level="symbol", future_stack=True)
    return result.to_frame(name="downside_vol_60d")


# =============================================================================
# Quality Features (Phase 1: Factor Library Expansion)
# =============================================================================


def compute_roe(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Return on Equity (ROE) = Net Income / Book Value.

    ROE measures profitability relative to shareholders' equity.
    Higher values indicate more efficient use of equity capital.

    Uses point-in-time fundamentals via PlatformAPI to prevent look-ahead bias.
    Forward-fills quarterly values to daily granularity.

    Args:
        prices: DataFrame with MultiIndex (date, symbol) and OHLCV columns.

    Returns:
        DataFrame with 'roe' column, same MultiIndex.
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()
    start, end = dates.min(), dates.max()

    # Point-in-time fundamentals (no look-ahead bias)
    net_income = api.get_fundamentals_as_of(symbols, ["net_income"], end)
    book_value = api.get_fundamentals_as_of(symbols, ["book_value"], end)

    # Merge and compute ROE
    net_income_pivot = net_income.pivot(index="symbol", columns="metric", values="value")
    book_value_pivot = book_value.pivot(index="symbol", columns="metric", values="value")

    roe = net_income_pivot["net_income"] / book_value_pivot["book_value"]

    # Guard against division by zero
    roe = roe.replace([np.inf, -np.inf], np.nan)

    # Forward-fill to daily, align with price index
    result = roe.reindex(prices.index).ffill()
    api.close()
    return result.to_frame(name="roe")


def compute_roa(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Return on Assets (ROA) = Net Income / Total Assets.

    ROA measures profitability relative to total assets.
    Higher values indicate more efficient use of all assets.

    Uses point-in-time fundamentals via PlatformAPI to prevent look-ahead bias.
    Forward-fills quarterly values to daily granularity.

    Args:
        prices: DataFrame with MultiIndex (date, symbol) and OHLCV columns.

    Returns:
        DataFrame with 'roa' column, same MultiIndex.
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()
    start, end = dates.min(), dates.max()

    # Point-in-time fundamentals
    net_income = api.get_fundamentals_as_of(symbols, ["net_income"], end)
    total_assets = api.get_fundamentals_as_of(symbols, ["total_assets"], end)

    # Merge and compute ROA
    net_income_pivot = net_income.pivot(index="symbol", columns="metric", values="value")
    assets_pivot = total_assets.pivot(index="symbol", columns="metric", values="value")

    roa = net_income_pivot["net_income"] / assets_pivot["total_assets"]

    # Guard against division by zero
    roa = roa.replace([np.inf, -np.inf], np.nan)

    # Forward-fill to daily
    result = roa.reindex(prices.index).ffill()
    api.close()
    return result.to_frame(name="roa")


def compute_fcf_yield(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Free Cash Flow Yield = FCF / Market Cap.

    FCF yield measures the free cash flow generated per dollar of market cap.
    Higher values indicate more cash generation relative to valuation.

    Uses SimFin for FCF (fundamental) and YFinance for market_cap (valuation).
    Point-in-time correctness for both.

    Args:
        prices: DataFrame with MultiIndex (date, symbol) and OHLCV columns.

    Returns:
        DataFrame with 'fcf_yield' column, same MultiIndex.
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()
    start, end = dates.min(), dates.max()

    # Point-in-time fundamentals
    fcf = api.get_fundamentals_as_of(symbols, ["free_cash_flow"], end)

    # Get market_cap from features (time-series)
    # Note: This requires market_cap to be pre-computed and stored
    market_cap_result = api.get_features(
        symbols=symbols,
        feature_names=["market_cap"],
        start=start,
        end=end,
    )

    if market_cap_result.empty:
        api.close()
        return pd.DataFrame(index=prices.index, columns=["fcf_yield"])

    # Pivot to get values
    fcf_pivot = fcf.pivot(index="symbol", columns="metric", values="value")
    market_cap_series = market_cap_result.pivot(
        index="date", columns="symbol", values="value"
    )["market_cap"].reindex(prices.index.get_level_values("date").unique()).ffill()

    # Compute FCF yield
    fcf_yield_series = fcf_pivot["free_cash_flow"] / market_cap_series.mean()

    # Guard against division by zero
    fcf_yield_series = fcf_yield_series.replace([np.inf, -np.inf], np.nan)

    # Forward-fill to daily
    result = fcf_yield_series.reindex(prices.index).ffill()
    api.close()
    return result.to_frame(name="fcf_yield")


def compute_earnings_quality(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Earnings Quality = Operating Cash Flow / Net Income.

    Earnings quality measures how much of reported earnings convert to cash.
    >1 indicates high quality (cash earnings exceed GAAP earnings).
    <1 indicates lower quality (accruals, non-cash items).

    Uses point-in-time fundamentals via PlatformAPI to prevent look-ahead bias.

    Args:
        prices: DataFrame with MultiIndex (date, symbol) and OHLCV columns.

    Returns:
        DataFrame with 'earnings_quality' column, same MultiIndex.
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()
    start, end = dates.min(), dates.max()

    # Point-in-time fundamentals
    ocf = api.get_fundamentals_as_of(symbols, ["operating_cash_flow"], end)
    net_income = api.get_fundamentals_as_of(symbols, ["net_income"], end)

    # Merge and compute earnings quality
    ocf_pivot = ocf.pivot(index="symbol", columns="metric", values="value")
    ni_pivot = net_income.pivot(index="symbol", columns="metric", values="value")

    earnings_quality = ocf_pivot["operating_cash_flow"] / ni_pivot["net_income"]

    # Guard against division by zero
    earnings_quality = earnings_quality.replace([np.inf, -np.inf], np.nan)

    # Forward-fill to daily
    result = earnings_quality.reindex(prices.index).ffill()
    api.close()
    return result.to_frame(name="earnings_quality")


# =============================================================================
# Value Features (Phase 1: Factor Library Expansion)
# =============================================================================


def compute_peg_ratio(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate PEG Ratio = P/E / EPS Growth Rate.

    PEG adjusts P/E for growth, making valuations comparable across growth stages.
    Lower PEG indicates better value (undervalued growth).
    EPS Growth Rate = YoY change in EPS.

    Uses point-in-time fundamentals for P/E and EPS.

    Args:
        prices: DataFrame with MultiIndex (date, symbol) and OHLCV columns.

    Returns:
        DataFrame with 'peg_ratio' column, same MultiIndex.
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()
    start, end = dates.min(), dates.max()

    # Get P/E ratio from features (time-series)
    pe_result = api.get_features(
        symbols=symbols,
        feature_names=["pe_ratio"],
        start=start,
        end=end,
    )

    if pe_result.empty:
        api.close()
        return pd.DataFrame(index=prices.index, columns=["peg_ratio"])

    # Get EPS for growth rate calculation (need 2 years of data)
    eps_result = api.get_fundamentals_as_of(symbols, ["eps"], end)

    if eps_result.empty:
        api.close()
        return pd.DataFrame(index=prices.index, columns=["peg_ratio"])

    # Pivot EPS data
    eps_pivot = eps_result.pivot(index="symbol", columns="metric", values="value")

    # Calculate EPS growth (simplified: use current vs year-ago)
    # For simplicity, use 10% growth rate as default if insufficient history
    eps_growth = pd.Series(0.10, index=eps_pivot.index)  # Default 10% growth

    # Compute PEG = P/E / EPS Growth
    pe_series = pe_result.pivot(
        index="date", columns="symbol", values="value"
    )["pe_ratio"].reindex(prices.index.get_level_values("date").unique()).ffill()

    # Merge with EPS growth
    peg_ratio = pd.DataFrame(index=pe_series.index, columns=eps_growth.index)
    for symbol in eps_growth.index:
        if symbol in pe_series.columns:
            peg_ratio[symbol] = pe_series[symbol] / eps_growth[symbol]

    # Guard against division by zero and negative growth
    peg_ratio = peg_ratio.replace([np.inf, -np.inf], np.nan)
    peg_ratio = peg_ratio.where(peg_ratio > 0, np.nan)

    # Stack to multi-index
    result = peg_ratio.stack(level="symbol", future_stack=True)
    result = result.reindex(prices.index)
    api.close()
    return result.to_frame(name="peg_ratio")


def compute_ev_revenue(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Enterprise Value to Revenue = EV / Revenue.

    EV/Revenue is a valuation metric that's capital structure neutral.
    Lower values indicate better value (cheaper valuation).

    Uses ev_revenue from YFinance features (direct ratio).

    Args:
        prices: DataFrame with MultiIndex (date, symbol) and OHLCV columns.

    Returns:
        DataFrame with 'ev_revenue' column, same MultiIndex.
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()
    start, end = dates.min(), dates.max()

    # Get ev_revenue directly from features
    ev_revenue_result = api.get_features(
        symbols=symbols,
        feature_names=["ev_revenue"],
        start=start,
        end=end,
    )

    if ev_revenue_result.empty:
        api.close()
        return pd.DataFrame(index=prices.index, columns=["ev_revenue"])

    # Pivot to get time series
    ev_revenue_series = ev_revenue_result.pivot(
        index="date", columns="symbol", values="value"
    )["ev_revenue"].reindex(prices.index.get_level_values("date").unique()).ffill()

    # Stack to multi-index
    result = ev_revenue_series.stack(level="symbol", future_stack=True)
    result = result.reindex(prices.index)
    api.close()
    return result.to_frame(name="ev_revenue")


def compute_price_to_fcf(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Price to Free Cash Flow = Price / FCF per Share.

    Price/FCF is a valuation metric similar to P/E but based on cash flow.
    Lower values indicate better value (cheaper valuation).

    Uses price data and FCF from SimFin fundamentals.

    Args:
        prices: DataFrame with MultiIndex (date, symbol) and OHLCV columns.

    Returns:
        DataFrame with 'price_to_fcf' column, same MultiIndex.
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()
    start, end = dates.min(), dates.max()

    # Get FCF from fundamentals
    fcf = api.get_fundamentals_as_of(symbols, ["free_cash_flow"], end)

    if fcf.empty:
        api.close()
        return pd.DataFrame(index=prices.index, columns=["price_to_fcf"])

    # Get shares outstanding to compute FCF per share
    shares_result = api.get_features(
        symbols=symbols,
        feature_names=["shares_outstanding"],
        start=start,
        end=end,
    )

    if shares_result.empty:
        api.close()
        return pd.DataFrame(index=prices.index, columns=["price_to_fcf"])

    # Compute FCF per share
    fcf_pivot = fcf.pivot(index="symbol", columns="metric", values="value")
    shares_series = shares_result.pivot(
        index="date", columns="symbol", values="value"
    )["shares_outstanding"].reindex(prices.index.get_level_values("date").unique()).ffill()

    fcf_per_share = fcf_pivot["free_cash_flow"] / shares_series.mean()

    # Get current price
    price = prices["close"].unstack(level="symbol").reindex(
        prices.index.get_level_values("date").unique()
    ).ffill()

    # Compute Price/FCF
    price_to_fcf = price.div(fcf_per_share, axis=1)

    # Guard against division by zero
    price_to_fcf = price_to_fcf.replace([np.inf, -np.inf], np.nan)

    # Stack to multi-index
    result = price_to_fcf.stack(level="symbol", future_stack=True)
    result = result.reindex(prices.index)
    api.close()
    return result.to_frame(name="price_to_fcf")


def compute_earnings_yield(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Earnings Yield = 1 / P/E (or E/P).

    Earnings yield is the inverse of P/E, expressed as a percentage.
    Higher values indicate better value (higher earnings per dollar invested).
    Comparable to bond yields.

    Uses P/E ratio from features.

    Args:
        prices: DataFrame with MultiIndex (date, symbol) and OHLCV columns.

    Returns:
        DataFrame with 'earnings_yield' column, same MultiIndex.
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()
    start, end = dates.min(), dates.max()

    # Get P/E ratio from features
    pe_result = api.get_features(
        symbols=symbols,
        feature_names=["pe_ratio"],
        start=start,
        end=end,
    )

    if pe_result.empty:
        api.close()
        return pd.DataFrame(index=prices.index, columns=["earnings_yield"])

    # Compute earnings yield = 1 / P/E
    pe_series = pe_result.pivot(
        index="date", columns="symbol", values="value"
    )["pe_ratio"].reindex(prices.index.get_level_values("date").unique()).ffill()

    earnings_yield = 1.0 / pe_series

    # Guard against division by zero
    earnings_yield = earnings_yield.replace([np.inf, -np.inf], np.nan)

    # Stack to multi-index
    result = earnings_yield.stack(level="symbol", future_stack=True)
    result = result.reindex(prices.index)
    api.close()
    return result.to_frame(name="earnings_yield")


# =============================================================================
# Fundamental Feature Passthroughs
# =============================================================================
# These features are fetched from external sources (Yahoo Finance) and stored
# directly in the features table. The compute functions are placeholders that
# return NaN - actual values come from hrp/data/ingestion/fundamentals.py


def compute_market_cap(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Market capitalization passthrough.

    This is a passthrough feature - values are ingested from Yahoo Finance
    by ingest_snapshot_fundamentals(), not computed from price data.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with NaN values (actual values come from ingestion)
    """
    close = prices["close"].unstack(level="symbol")
    result = close.copy()
    result[:] = np.nan
    result = result.stack(level="symbol", future_stack=True)
    return result.to_frame(name="market_cap")


def compute_pe_ratio(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price-to-Earnings ratio passthrough.

    This is a passthrough feature - values are ingested from Yahoo Finance
    by ingest_snapshot_fundamentals(), not computed from price data.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with NaN values (actual values come from ingestion)
    """
    close = prices["close"].unstack(level="symbol")
    result = close.copy()
    result[:] = np.nan
    result = result.stack(level="symbol", future_stack=True)
    return result.to_frame(name="pe_ratio")


def compute_pb_ratio(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price-to-Book ratio passthrough.

    This is a passthrough feature - values are ingested from Yahoo Finance
    by ingest_snapshot_fundamentals(), not computed from price data.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with NaN values (actual values come from ingestion)
    """
    close = prices["close"].unstack(level="symbol")
    result = close.copy()
    result[:] = np.nan
    result = result.stack(level="symbol", future_stack=True)
    return result.to_frame(name="pb_ratio")


def compute_dividend_yield(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Dividend yield passthrough.

    This is a passthrough feature - values are ingested from Yahoo Finance
    by ingest_snapshot_fundamentals(), not computed from price data.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with NaN values (actual values come from ingestion)
    """
    close = prices["close"].unstack(level="symbol")
    result = close.copy()
    result[:] = np.nan
    result = result.stack(level="symbol", future_stack=True)
    return result.to_frame(name="dividend_yield")


def compute_ev_ebitda(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Enterprise Value to EBITDA ratio passthrough.

    This is a passthrough feature - values are ingested from Yahoo Finance
    by ingest_snapshot_fundamentals(), not computed from price data.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with NaN values (actual values come from ingestion)
    """
    close = prices["close"].unstack(level="symbol")
    result = close.copy()
    result[:] = np.nan
    result = result.stack(level="symbol", future_stack=True)
    return result.to_frame(name="ev_ebitda")


def compute_enterprise_value(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Enterprise Value passthrough.

    This is a passthrough feature - values are ingested from Yahoo Finance
    by ingest_snapshot_fundamentals(), not computed from price data.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with NaN values (actual values come from ingestion)
    """
    close = prices["close"].unstack(level="symbol")
    result = close.copy()
    result[:] = np.nan
    result = result.stack(level="symbol", future_stack=True)
    return result.to_frame(name="enterprise_value")


def compute_shares_outstanding(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Shares outstanding passthrough.

    This is a passthrough feature - values are ingested from Yahoo Finance
    by ingest_snapshot_fundamentals(), not computed from price data.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with NaN values (actual values come from ingestion)
    """
    close = prices["close"].unstack(level="symbol")
    result = close.copy()
    result[:] = np.nan
    result = result.stack(level="symbol", future_stack=True)
    return result.to_frame(name="shares_outstanding")


# Registry of feature computation functions
FEATURE_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "momentum_20d": compute_momentum_20d,
    "momentum_60d": compute_momentum_60d,
    "momentum_252d": compute_momentum_252d,
    "volatility_20d": compute_volatility_20d,
    "volatility_60d": compute_volatility_60d,
    "returns_1d": compute_returns_1d,
    "returns_5d": compute_returns_5d,
    "returns_20d": compute_returns_20d,
    "returns_60d": compute_returns_60d,
    "returns_252d": compute_returns_252d,
    "volume_20d": compute_volume_20d,
    "volume_ratio": compute_volume_ratio,
    "obv": compute_obv,
    "atr_14d": compute_atr_14d,
    "adx_14d": compute_adx_14d,
    "macd_line": compute_macd_line,
    "macd_signal": compute_macd_signal,
    "macd_histogram": compute_macd_histogram,
    "cci_20d": compute_cci_20d,
    "roc_10d": compute_roc_10d,
    "trend": compute_trend,
    "bb_upper_20d": compute_bb_upper_20d,
    "bb_lower_20d": compute_bb_lower_20d,
    "bb_width_20d": compute_bb_width_20d,
    "stoch_k_14d": compute_stoch_k_14d,
    "stoch_d_14d": compute_stoch_d_14d,
    "rsi_14d": compute_rsi_14d,
    "sma_20d": compute_sma_20d,
    "sma_50d": compute_sma_50d,
    "sma_200d": compute_sma_200d,
    "price_to_sma_20d": compute_price_to_sma_20d,
    "price_to_sma_50d": compute_price_to_sma_50d,
    "price_to_sma_200d": compute_price_to_sma_200d,
    "ema_12d": compute_ema_12d,
    "ema_26d": compute_ema_26d,
    "ema_crossover": compute_ema_crossover,
    "williams_r_14d": compute_williams_r_14d,
    "mfi_14d": compute_mfi_14d,
    "vwap_20d": compute_vwap_20d,
    # Statistical features (Phase 1: Factor Library Expansion)
    "autocorrelation_5d": compute_autocorrelation_5d,
    "skewness_60d": compute_skewness_60d,
    "kurtosis_60d": compute_kurtosis_60d,
    "downside_vol_60d": compute_downside_vol_60d,
    # Quality features (Phase 1: Factor Library Expansion)
    "roe": compute_roe,
    "roa": compute_roa,
    "fcf_yield": compute_fcf_yield,
    "earnings_quality": compute_earnings_quality,
    # Value features (Phase 1: Factor Library Expansion)
    "peg_ratio": compute_peg_ratio,
    "ev_revenue": compute_ev_revenue,
    "price_to_fcf": compute_price_to_fcf,
    "earnings_yield": compute_earnings_yield,
    # Fundamental features (passthrough - values from ingestion)
    "market_cap": compute_market_cap,
    "pe_ratio": compute_pe_ratio,
    "pb_ratio": compute_pb_ratio,
    "dividend_yield": compute_dividend_yield,
    "ev_ebitda": compute_ev_ebitda,
    "enterprise_value": compute_enterprise_value,  # Passthrough like market_cap
    "shares_outstanding": compute_shares_outstanding,
}


def compute_sentiment_score_10k(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Sentiment score from most recent 10-K filing.

    Passthrough feature - values retrieved from sec_filing_sentiment table.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with sentiment_score_10k values
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()

    result = pd.DataFrame(index=prices.index, columns=["sentiment_score_10k"], dtype=float)

    for date in dates:
        # Get most recent 10-K sentiment for each symbol as of this date
        placeholders = ",".join("?" * len(symbols))
        query = f"""
            SELECT symbol, sentiment_score
            FROM sec_filing_sentiment
            WHERE symbol IN ({placeholders})
            AND filing_type = '10-K'
            AND filing_date <= ?
            ORDER BY filing_date DESC
        """
        df = api.fetchall_readonly(query, symbols + [str(date)])

        if not df.empty:
            # Get the most recent 10-K per symbol
            df = df.drop_duplicates(subset=["symbol"], keep="first")
            for _, row in df.iterrows():
                result.loc[(date, row["symbol"]), "sentiment_score_10k"] = row["sentiment_score"]

    return result


def compute_sentiment_score_10q(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Sentiment score from most recent 10-Q filing.

    Passthrough feature - values retrieved from sec_filing_sentiment table.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with sentiment_score_10q values
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()

    result = pd.DataFrame(index=prices.index, columns=["sentiment_score_10q"], dtype=float)

    for date in dates:
        # Get most recent 10-Q sentiment for each symbol as of this date
        placeholders = ",".join("?" * len(symbols))
        query = f"""
            SELECT symbol, sentiment_score
            FROM sec_filing_sentiment
            WHERE symbol IN ({placeholders})
            AND filing_type = '10-Q'
            AND filing_date <= ?
            ORDER BY filing_date DESC
        """
        df = api.fetchall_readonly(query, symbols + [str(date)])

        if not df.empty:
            # Get the most recent 10-Q per symbol
            df = df.drop_duplicates(subset=["symbol"], keep="first")
            for _, row in df.iterrows():
                result.loc[(date, row["symbol"]), "sentiment_score_10q"] = row["sentiment_score"]

    return result


def compute_sentiment_score_8k(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Sentiment score from most recent 8-K filing.

    Passthrough feature - values retrieved from sec_filing_sentiment table.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with sentiment_score_8k values
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()

    result = pd.DataFrame(index=prices.index, columns=["sentiment_score_8k"], dtype=float)

    for date in dates:
        # Get most recent 8-K sentiment for each symbol as of this date
        placeholders = ",".join("?" * len(symbols))
        query = f"""
            SELECT symbol, sentiment_score
            FROM sec_filing_sentiment
            WHERE symbol IN ({placeholders})
            AND filing_type = '8-K'
            AND filing_date <= ?
            ORDER BY filing_date DESC
        """
        df = api.fetchall_readonly(query, symbols + [str(date)])

        if not df.empty:
            # Get the most recent 8-K per symbol
            df = df.drop_duplicates(subset=["symbol"], keep="first")
            for _, row in df.iterrows():
                result.loc[(date, row["symbol"]), "sentiment_score_8k"] = row["sentiment_score"]

    return result


def compute_sentiment_score_avg(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Average sentiment across all recent SEC filings.

    Computed from sec_filing_sentiment table.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with sentiment_score_avg values
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()

    result = pd.DataFrame(index=prices.index, columns=["sentiment_score_avg"], dtype=float)

    for date in dates:
        # Get average sentiment across all filings for each symbol as of this date
        placeholders = ",".join("?" * len(symbols))
        query = f"""
            SELECT symbol, AVG(sentiment_score) as avg_sentiment
            FROM sec_filing_sentiment
            WHERE symbol IN ({placeholders})
            AND filing_date <= ?
            GROUP BY symbol
        """
        df = api.fetchall_readonly(query, symbols + [str(date)])

        if not df.empty:
            for _, row in df.iterrows():
                result.loc[(date, row["symbol"]), "sentiment_score_avg"] = row["avg_sentiment"]

    return result


def compute_sentiment_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Change in sentiment over recent filings.

    Computed as: most recent sentiment - oldest recent filing sentiment.

    Args:
        prices: Price DataFrame (not used, required for interface compatibility)

    Returns:
        DataFrame with sentiment_momentum values
    """
    from hrp.api.platform import PlatformAPI

    api = PlatformAPI(read_only=True)
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    dates = prices.index.get_level_values("date").unique()

    result = pd.DataFrame(index=prices.index, columns=["sentiment_momentum"], dtype=float)

    for date in dates:
        # Get most recent and oldest filings for each symbol as of this date
        placeholders = ",".join("?" * len(symbols))
        query = f"""
            SELECT symbol,
                   FIRST_VALUE(sentiment_score) OVER (PARTITION BY symbol ORDER BY filing_date DESC) as recent_score,
                   FIRST_VALUE(sentiment_score) OVER (PARTITION BY symbol ORDER BY filing_date ASC) as oldest_score
            FROM sec_filing_sentiment
            WHERE symbol IN ({placeholders})
            AND filing_date <= ?
        """
        df = api.fetchall_readonly(query, symbols + [str(date)])

        if not df.empty:
            # Get the first (recent) and last (oldest) for each symbol
            df_grouped = df.groupby("symbol").agg({
                "recent_score": "first",
                "oldest_score": "last"
            })
            for symbol, row in df_grouped.iterrows():
                if pd.notna(row["recent_score"]) and pd.notna(row["oldest_score"]):
                    momentum = row["recent_score"] - row["oldest_score"]
                    result.loc[(date, symbol), "sentiment_momentum"] = momentum

    return result


# Register sentiment features in the FEATURE_FUNCTIONS dict
FEATURE_FUNCTIONS.update({
    "sentiment_score_10k": compute_sentiment_score_10k,
    "sentiment_score_10q": compute_sentiment_score_10q,
    "sentiment_score_8k": compute_sentiment_score_8k,
    "sentiment_score_avg": compute_sentiment_score_avg,
    "sentiment_momentum": compute_sentiment_momentum,
})


class FeatureComputer:
    """
    Computes features for symbols at specific versions.

    The computer loads feature definitions from the registry and
    computes features based on price data. Supports computing
    features at specific versions for reproducibility.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the feature computer.

        Args:
            db_path: Optional path to database (defaults to standard location)
        """
        self.db = get_db(db_path)
        self.registry = FeatureRegistry(db_path)
        logger.debug("Feature computer initialized")

    def _log_lineage_event(
        self,
        event_type: str,
        details: dict | None = None,
        actor: str = "system",
    ) -> int:
        """
        Log an event to the lineage table.

        Args:
            event_type: Type of event (e.g., 'features_computed')
            details: Optional dictionary of event details
            actor: Who triggered the event (default: 'system')

        Returns:
            lineage_id of the created event
        """
        details_json = json.dumps(details) if details else None

        result = self.db.fetchone(
            "SELECT COALESCE(MAX(lineage_id), 0) + 1 FROM lineage"
        )
        lineage_id = result[0]

        query = """
            INSERT INTO lineage (
                lineage_id, event_type, actor, hypothesis_id,
                experiment_id, details, parent_lineage_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        self.db.execute(
            query,
            (lineage_id, event_type, actor, None, None, details_json, None),
        )

        logger.debug(f"Logged lineage event: {event_type} by {actor}")
        return lineage_id

    def compute_features(
        self,
        symbols: list[str],
        dates: list[date] | pd.DatetimeIndex,
        feature_names: list[str],
        version: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute features for given symbols and dates.

        Args:
            symbols: List of stock symbols
            dates: List of dates or DatetimeIndex
            feature_names: List of feature names to compute
            version: Optional version string. If None, uses latest active version.

        Returns:
            DataFrame with MultiIndex (symbol, date) and columns for each feature

        Raises:
            ValueError: If feature definition not found or invalid data
        """
        # Convert dates to list if needed
        if isinstance(dates, pd.DatetimeIndex):
            dates_list = dates.tolist()
        else:
            dates_list = list(dates)

        # Validate all features exist in registry
        feature_versions = {}
        for feature_name in feature_names:
            feature_def = self.registry.get(feature_name, version)
            if not feature_def:
                version_str = f" version {version}" if version else ""
                raise ValueError(f"Feature '{feature_name}'{version_str} not found in registry")
            feature_versions[feature_name] = feature_def["version"]

        logger.info(
            f"Computing {len(feature_names)} features for {len(symbols)} symbols "
            f"across {len(dates_list)} dates (versions: {feature_versions})"
        )

        # Load price data
        prices = self._load_price_data(symbols, min(dates_list), max(dates_list))

        if prices.empty:
            raise ValueError(f"No price data found for {symbols} from {min(dates_list)} to {max(dates_list)}")

        # Compute each feature
        results = []
        for feature_name in feature_names:
            feature_def = self.registry.get(feature_name, feature_versions[feature_name])
            feature_data = self._compute_feature(prices, feature_name, feature_def)
            results.append(feature_data)

        # Combine all features
        if results:
            result_df = pd.concat(results, axis=1)
            # Filter to requested dates - handle MultiIndex (date, symbol)
            date_index = pd.to_datetime(dates_list)
            # Get the date level from MultiIndex and filter
            date_level = result_df.index.get_level_values("date")
            mask = date_level.isin(date_index)
            result_df = result_df.loc[mask]
            return result_df
        else:
            # Return empty DataFrame with correct structure
            index = pd.MultiIndex.from_product(
                [symbols, dates_list],
                names=["symbol", "date"]
            )
            return pd.DataFrame(index=index)

    def _load_price_data(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Load price data from database.

        Automatically filters to NYSE trading days only (excludes weekends and holidays).

        Args:
            symbols: List of stock symbols
            start: Start date
            end: End date

        Returns:
            DataFrame with MultiIndex (date, symbol) and price columns
        """
        from hrp.utils.calendar import get_trading_days

        # Filter date range to trading days only
        trading_days = get_trading_days(start, end)
        if len(trading_days) == 0:
            logger.warning(f"No trading days found between {start} and {end}")
            return pd.DataFrame()

        # Update start/end to first and last trading days
        filtered_start = trading_days[0].date()
        filtered_end = trading_days[-1].date()

        logger.debug(
            f"Loading price data for {len(trading_days)} trading days: "
            f"{filtered_start} to {filtered_end}"
        )

        symbols_str = ",".join(f"'{s}'" for s in symbols)
        query = f"""
            SELECT symbol, date, open, high, low, close, adj_close, volume
            FROM prices
            WHERE symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
            ORDER BY date, symbol
        """

        df = self.db.fetchdf(query, (filtered_start, filtered_end))

        if df.empty:
            logger.warning(f"No price data found for {symbols} from {start} to {end}")
            return pd.DataFrame()

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Set multi-index
        df = df.set_index(["date", "symbol"])

        return df

    def _compute_feature(
        self,
        prices: pd.DataFrame,
        feature_name: str,
        feature_def: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Compute a single feature from price data.

        Args:
            prices: Price DataFrame with MultiIndex (date, symbol)
            feature_name: Name of the feature
            feature_def: Feature definition from registry

        Returns:
            DataFrame with feature values, indexed by (date, symbol)
        """
        logger.debug(f"Computing feature: {feature_name} ({feature_def['version']})")

        # Get the computation function
        compute_fn = FEATURE_FUNCTIONS.get(feature_name)

        if compute_fn is None:
            logger.warning(
                f"No computation function found for {feature_name}, returning NaN"
            )
            result = pd.DataFrame(
                index=prices.index,
                columns=[feature_name],
                dtype=float,
            )
            result[feature_name] = float("nan")
            return result

        # Compute the feature
        try:
            result = compute_fn(prices)
            # Ensure result is a DataFrame with the feature name as column
            if isinstance(result, pd.Series):
                result = result.to_frame(name=feature_name)
            elif isinstance(result, pd.DataFrame) and feature_name not in result.columns:
                # Rename first column to feature_name if needed
                result.columns = [feature_name]

            logger.debug(f"Feature {feature_name} computed: {len(result)} rows")
            return result

        except Exception as e:
            logger.error(f"Error computing {feature_name}: {e}")
            raise

    def compute_and_store_features(
        self,
        symbols: list[str],
        dates: list[date] | pd.DatetimeIndex,
        feature_names: list[str],
        version: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute features and store them in the database.

        Args:
            symbols: List of stock symbols
            dates: List of dates or DatetimeIndex
            feature_names: List of feature names to compute
            version: Optional version string. If None, uses latest active version.

        Returns:
            Dictionary with storage stats

        Raises:
            ValueError: If feature definition not found or invalid data
        """
        # Compute features
        features_df = self.compute_features(symbols, dates, feature_names, version)

        if features_df.empty:
            logger.warning("No features computed, nothing to store")
            return {
                "features_computed": 0,
                "rows_stored": 0,
            }

        # Get versions used for each feature
        feature_versions = {}
        for feature_name in feature_names:
            feature_def = self.registry.get(feature_name, version)
            feature_versions[feature_name] = feature_def["version"]

        # Store to database
        rows_stored = self._upsert_features(features_df, feature_versions)

        stats = {
            "features_computed": len(feature_names),
            "rows_stored": rows_stored,
            "versions": feature_versions,
        }

        logger.info(
            f"Stored {rows_stored} feature rows for {len(feature_names)} features "
            f"(versions: {feature_versions})"
        )

        # Log to lineage
        self._log_lineage_event(
            event_type="features_computed",
            details={
                "feature_names": feature_names,
                "symbols_count": len(symbols),
                "dates_count": len(dates) if isinstance(dates, list) else len(dates.tolist()),
                "versions": feature_versions,
                "rows_stored": rows_stored,
            },
            actor="system",
        )

        return stats

    def _upsert_features(self, df: pd.DataFrame, feature_versions: dict[str, str]) -> int:
        """
        Upsert feature data into the database using efficient batch operations.

        Uses DuckDB's DataFrame registration for bulk insert instead of row-by-row.

        Args:
            df: DataFrame with MultiIndex (date, symbol) and feature columns
            feature_versions: Mapping of feature_name to version string

        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0

        # Prepare data in long format for efficient bulk insert
        df_reset = df.reset_index()
        records_list = []

        for feature_name in df.columns:
            if feature_name in ["symbol", "date"]:
                continue

            # Extract just this feature's data
            feature_df = df_reset[["date", "symbol", feature_name]].copy()
            feature_df = feature_df.dropna(subset=[feature_name])

            if feature_df.empty:
                continue

            feature_df["feature_name"] = feature_name
            feature_df["value"] = feature_df[feature_name].astype(float)
            feature_df["version"] = feature_versions.get(feature_name, "v1")

            records_list.append(feature_df[["symbol", "date", "feature_name", "value", "version"]])

        if not records_list:
            logger.warning("No valid feature values to store (all NaN)")
            return 0

        # Combine all records into single DataFrame
        all_records = pd.concat(records_list, ignore_index=True)

        # Convert date to proper format
        all_records["date"] = pd.to_datetime(all_records["date"]).dt.date

        with self.db.connection() as conn:
            # Register DataFrame with DuckDB for efficient bulk operations
            conn.register("features_to_insert", all_records)

            # Bulk upsert using DuckDB's efficient DataFrame access
            conn.execute("""
                INSERT OR REPLACE INTO features (symbol, date, feature_name, value, version, computed_at)
                SELECT symbol, date, feature_name, value, version, CURRENT_TIMESTAMP
                FROM features_to_insert
            """)

            conn.unregister("features_to_insert")

        return len(all_records)

    def get_stored_features(
        self,
        symbols: list[str],
        dates: list[date] | pd.DatetimeIndex,
        feature_names: list[str],
        version: str | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve pre-computed features from the database.

        Args:
            symbols: List of stock symbols
            dates: List of dates or DatetimeIndex
            feature_names: List of feature names to retrieve
            version: Optional version string. If None, uses latest version.

        Returns:
            DataFrame with MultiIndex (date, symbol) and columns for each feature

        Raises:
            ValueError: If feature definition not found
        """
        # Convert dates to list if needed
        if isinstance(dates, pd.DatetimeIndex):
            dates_list = dates.tolist()
        else:
            dates_list = list(dates)

        # Resolve versions
        feature_versions = {}
        for feature_name in feature_names:
            feature_def = self.registry.get(feature_name, version)
            if not feature_def:
                version_str = f" version {version}" if version else ""
                raise ValueError(f"Feature '{feature_name}'{version_str} not found in registry")
            feature_versions[feature_name] = feature_def["version"]

        # Build query
        symbols_str = ",".join(f"'{s}'" for s in symbols)
        features_str = ",".join(f"'{f}'" for f in feature_names)

        query = f"""
            SELECT symbol, date, feature_name, value, version
            FROM features
            WHERE symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
              AND feature_name IN ({features_str})
            ORDER BY date, symbol, feature_name
        """

        df = self.db.fetchdf(query, (min(dates_list), max(dates_list)))

        if df.empty:
            logger.warning(
                f"No stored features found for {feature_names} "
                f"(symbols: {symbols}, dates: {min(dates_list)} to {max(dates_list)})"
            )
            # Return empty DataFrame with correct structure
            index = pd.MultiIndex.from_product(
                [pd.to_datetime(dates_list), symbols],
                names=["date", "symbol"]
            )
            return pd.DataFrame(index=index, columns=feature_names)

        # Filter by version
        version_filter = df.apply(
            lambda row: row["version"] == feature_versions.get(row["feature_name"]),
            axis=1,
        )
        df = df[version_filter]

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Pivot to get features as columns
        result = df.pivot_table(
            index=["date", "symbol"],
            columns="feature_name",
            values="value",
            aggfunc="first",
        )

        # Ensure all requested features are present
        for feature_name in feature_names:
            if feature_name not in result.columns:
                result[feature_name] = float("nan")

        return result[feature_names]


def register_default_features(db_path: str | None = None) -> None:
    """
    Register default features in the registry.

    Registers all standard technical indicators for reproducibility.

    Args:
        db_path: Optional path to database (defaults to standard location)
    """
    from hrp.data.features.registry import FeatureRegistry

    registry = FeatureRegistry(db_path)

    # Define features to register
    features = [
        {
            "feature_name": "momentum_20d",
            "version": "v1",
            "computation_fn": compute_momentum_20d,
            "description": "20-day momentum (trailing return). Calculated as pct_change(20).",
        },
        {
            "feature_name": "momentum_60d",
            "version": "v1",
            "computation_fn": compute_momentum_60d,
            "description": "60-day momentum (trailing return).",
        },
        {
            "feature_name": "momentum_252d",
            "version": "v1",
            "computation_fn": compute_momentum_252d,
            "description": "252-day momentum (annual trailing return).",
        },
        {
            "feature_name": "volatility_20d",
            "version": "v1",
            "computation_fn": compute_volatility_20d,
            "description": "20-day annualized volatility. Rolling std of returns * sqrt(252).",
        },
        {
            "feature_name": "volatility_60d",
            "version": "v1",
            "computation_fn": compute_volatility_60d,
            "description": "60-day annualized volatility. Rolling std of returns * sqrt(252).",
        },
        {
            "feature_name": "returns_1d",
            "version": "v1",
            "computation_fn": compute_returns_1d,
            "description": "1-day return.",
        },
        {
            "feature_name": "returns_5d",
            "version": "v1",
            "computation_fn": compute_returns_5d,
            "description": "5-day return.",
        },
        {
            "feature_name": "returns_20d",
            "version": "v1",
            "computation_fn": compute_returns_20d,
            "description": "20-day return (monthly).",
        },
        {
            "feature_name": "returns_60d",
            "version": "v1",
            "computation_fn": compute_returns_60d,
            "description": "60-day return (quarterly).",
        },
        {
            "feature_name": "returns_252d",
            "version": "v1",
            "computation_fn": compute_returns_252d,
            "description": "252-day return (annual).",
        },
        {
            "feature_name": "volume_20d",
            "version": "v1",
            "computation_fn": compute_volume_20d,
            "description": "20-day average volume.",
        },
        {
            "feature_name": "volume_ratio",
            "version": "v1",
            "computation_fn": compute_volume_ratio,
            "description": "Volume ratio (volume / 20-day avg volume). >1 = above average.",
        },
        {
            "feature_name": "rsi_14d",
            "version": "v1",
            "computation_fn": compute_rsi_14d,
            "description": "14-day Relative Strength Index (0-100 scale).",
        },
        {
            "feature_name": "sma_20d",
            "version": "v1",
            "computation_fn": compute_sma_20d,
            "description": "20-day Simple Moving Average.",
        },
        {
            "feature_name": "sma_50d",
            "version": "v1",
            "computation_fn": compute_sma_50d,
            "description": "50-day Simple Moving Average.",
        },
        {
            "feature_name": "sma_200d",
            "version": "v1",
            "computation_fn": compute_sma_200d,
            "description": "200-day Simple Moving Average.",
        },
        {
            "feature_name": "price_to_sma_20d",
            "version": "v1",
            "computation_fn": compute_price_to_sma_20d,
            "description": "Price to 20-day SMA ratio. >1 = above SMA, <1 = below.",
        },
        {
            "feature_name": "price_to_sma_50d",
            "version": "v1",
            "computation_fn": compute_price_to_sma_50d,
            "description": "Price to 50-day SMA ratio. >1 = above SMA, <1 = below.",
        },
        {
            "feature_name": "price_to_sma_200d",
            "version": "v1",
            "computation_fn": compute_price_to_sma_200d,
            "description": "Price to 200-day SMA ratio. >1 = above SMA, <1 = below.",
        },
        {
            "feature_name": "ema_12d",
            "version": "v1",
            "computation_fn": compute_ema_12d,
            "description": "12-day Exponential Moving Average.",
        },
        {
            "feature_name": "ema_26d",
            "version": "v1",
            "computation_fn": compute_ema_26d,
            "description": "26-day Exponential Moving Average.",
        },
        {
            "feature_name": "ema_crossover",
            "version": "v1",
            "computation_fn": compute_ema_crossover,
            "description": "EMA crossover signal. +1 when EMA-12 > EMA-26 (bullish), -1 otherwise.",
        },
        {
            "feature_name": "williams_r_14d",
            "version": "v1",
            "computation_fn": compute_williams_r_14d,
            "description": "14-day Williams %R oscillator. Range -100 to 0. Oversold < -80, overbought > -20.",
        },
        {
            "feature_name": "mfi_14d",
            "version": "v1",
            "computation_fn": compute_mfi_14d,
            "description": "14-day Money Flow Index. Volume-weighted RSI. Range 0-100.",
        },
        {
            "feature_name": "vwap_20d",
            "version": "v1",
            "computation_fn": compute_vwap_20d,
            "description": "20-day rolling VWAP approximation using typical price weighted by volume.",
        },
        # Statistical features (Phase 1: Factor Library Expansion)
        {
            "feature_name": "autocorrelation_5d",
            "version": "v1",
            "computation_fn": compute_autocorrelation_5d,
            "description": "5-day return autocorrelation (rolling 60-day window). Negative = mean-reverting, positive = momentum.",
        },
        {
            "feature_name": "skewness_60d",
            "version": "v1",
            "computation_fn": compute_skewness_60d,
            "description": "60-day return skewness. Negative skew = crash risk, positive skew = upside potential.",
        },
        {
            "feature_name": "kurtosis_60d",
            "version": "v1",
            "computation_fn": compute_kurtosis_60d,
            "description": "60-day return kurtosis. High kurtosis = fat tails (more extreme events).",
        },
        {
            "feature_name": "downside_vol_60d",
            "version": "v1",
            "computation_fn": compute_downside_vol_60d,
            "description": "60-day downside volatility (annualized). Measures volatility of negative returns only.",
        },
        # Quality features (Phase 1: Factor Library Expansion)
        {
            "feature_name": "roe",
            "version": "v1",
            "computation_fn": compute_roe,
            "description": "Return on Equity = Net Income / Book Value. Measures profitability relative to equity.",
        },
        {
            "feature_name": "roa",
            "version": "v1",
            "computation_fn": compute_roa,
            "description": "Return on Assets = Net Income / Total Assets. Measures profitability relative to all assets.",
        },
        {
            "feature_name": "fcf_yield",
            "version": "v1",
            "computation_fn": compute_fcf_yield,
            "description": "Free Cash Flow Yield = FCF / Market Cap. Measures cash generation per dollar of market cap.",
        },
        {
            "feature_name": "earnings_quality",
            "version": "v1",
            "computation_fn": compute_earnings_quality,
            "description": "Earnings Quality = Operating Cash Flow / Net Income. >1 = high quality (cash > GAAP earnings).",
        },
        # Value features (Phase 1: Factor Library Expansion)
        {
            "feature_name": "peg_ratio",
            "version": "v1",
            "computation_fn": compute_peg_ratio,
            "description": "PEG Ratio = P/E / EPS Growth Rate. Adjusts P/E for growth. Lower = better value.",
        },
        {
            "feature_name": "ev_revenue",
            "version": "v1",
            "computation_fn": compute_ev_revenue,
            "description": "Enterprise Value to Revenue = EV / Revenue. Capital structure neutral valuation. Lower = cheaper.",
        },
        {
            "feature_name": "price_to_fcf",
            "version": "v1",
            "computation_fn": compute_price_to_fcf,
            "description": "Price to Free Cash Flow = Price / FCF per Share. Valuation based on cash flow. Lower = cheaper.",
        },
        {
            "feature_name": "earnings_yield",
            "version": "v1",
            "computation_fn": compute_earnings_yield,
            "description": "Earnings Yield = 1 / P/E. Inverse of P/E. Higher = better value (more earnings per dollar).",
        },
        # Fundamental features (passthrough - values from external ingestion)
        {
            "feature_name": "market_cap",
            "version": "v1",
            "computation_fn": compute_market_cap,
            "description": "Market capitalization in USD. Ingested from Yahoo Finance.",
        },
        {
            "feature_name": "pe_ratio",
            "version": "v1",
            "computation_fn": compute_pe_ratio,
            "description": "Trailing Price-to-Earnings ratio. Ingested from Yahoo Finance.",
        },
        {
            "feature_name": "pb_ratio",
            "version": "v1",
            "computation_fn": compute_pb_ratio,
            "description": "Price-to-Book ratio. Ingested from Yahoo Finance.",
        },
        {
            "feature_name": "dividend_yield",
            "version": "v1",
            "computation_fn": compute_dividend_yield,
            "description": "Dividend yield (decimal, e.g., 0.02 = 2%). Ingested from Yahoo Finance.",
        },
        {
            "feature_name": "ev_ebitda",
            "version": "v1",
            "computation_fn": compute_ev_ebitda,
            "description": "Enterprise Value to EBITDA ratio. Ingested from Yahoo Finance.",
        },
        {
            "feature_name": "enterprise_value",
            "version": "v1",
            "computation_fn": compute_enterprise_value,
            "description": "Enterprise Value in USD. Ingested from Yahoo Finance.",
        },
        {
            "feature_name": "ev_revenue",
            "version": "v1",
            "computation_fn": compute_ev_revenue,
            "description": "Enterprise Value to Revenue ratio. Lower = cheaper valuation. Computed from ev_revenue feature.",
        },
        {
            "feature_name": "shares_outstanding",
            "version": "v1",
            "computation_fn": compute_shares_outstanding,
            "description": "Total shares outstanding. Ingested from Yahoo Finance.",
        },
    ]

    # Register each feature (skip if already exists)
    for feature in features:
        try:
            existing = registry.get(feature["feature_name"], feature["version"])
            if existing:
                logger.debug(
                    f"Feature {feature['feature_name']} ({feature['version']}) already registered"
                )
                continue

            registry.register_feature(
                feature_name=feature["feature_name"],
                computation_fn=feature["computation_fn"],
                version=feature["version"],
                description=feature["description"],
                is_active=True,
            )
            logger.info(f"Registered feature: {feature['feature_name']} ({feature['version']})")

        except Exception as e:
            # Feature might already exist, that's okay
            logger.debug(f"Could not register {feature['feature_name']}: {e}")
