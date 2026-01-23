"""
Synthetic data generators for deterministic test fixtures.

All generators use a seed parameter for reproducibility.
"""

from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


def generate_prices(
    symbols: List[str],
    start: date,
    end: date,
    seed: int = 42,
    base_prices: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Generate synthetic price data with realistic random walk.

    Args:
        symbols: List of ticker symbols
        start: Start date
        end: End date
        seed: Random seed for reproducibility
        base_prices: Optional dict of symbol -> base price

    Returns:
        DataFrame with columns: symbol, date, open, high, low, close, adj_close, volume
    """
    np.random.seed(seed)

    if base_prices is None:
        base_prices = {s: 100.0 + i * 50 for i, s in enumerate(symbols)}

    dates = pd.date_range(start, end, freq="B")
    data = []

    for symbol in symbols:
        price = base_prices.get(symbol, 100.0)

        for d in dates:
            # Random walk with slight upward drift
            daily_return = np.random.normal(0.0005, 0.02)
            price = price * (1 + daily_return)

            # Generate OHLC from close
            intraday_vol = abs(np.random.normal(0, 0.01))
            high = price * (1 + intraday_vol)
            low = price * (1 - intraday_vol)
            open_price = low + (high - low) * np.random.random()

            volume = int(np.random.uniform(500000, 5000000))

            data.append({
                "symbol": symbol,
                "date": d.date(),
                "open": round(open_price, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(price, 4),
                "adj_close": round(price, 4),
                "volume": volume,
            })

    return pd.DataFrame(data)


def generate_features(
    symbols: List[str],
    feature_names: List[str],
    as_of_date: date,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic feature data.

    Args:
        symbols: List of ticker symbols
        feature_names: List of feature names
        as_of_date: Date for features
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: symbol, date, feature_name, value, version
    """
    np.random.seed(seed)

    data = []
    for symbol in symbols:
        for feature in feature_names:
            # Generate realistic ranges based on feature name
            if "momentum" in feature.lower():
                value = np.random.uniform(-0.3, 0.5)
            elif "volatility" in feature.lower():
                value = np.random.uniform(0.1, 0.6)
            elif "rsi" in feature.lower():
                value = np.random.uniform(20, 80)
            else:
                value = np.random.uniform(-1, 1)

            data.append({
                "symbol": symbol,
                "date": as_of_date,
                "feature_name": feature,
                "value": round(value, 6),
                "version": "v1",
            })

    return pd.DataFrame(data)


def generate_corporate_actions(
    symbols: List[str],
    start: date,
    end: date,
    action_types: List[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic corporate actions.

    Args:
        symbols: List of ticker symbols
        start: Start date
        end: End date
        action_types: Types to generate (default: ['split', 'dividend'])
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: symbol, date, action_type, factor, source
    """
    np.random.seed(seed)

    if action_types is None:
        action_types = ["split", "dividend"]

    data = []
    days_range = (end - start).days

    for symbol in symbols:
        # Each symbol has 0-2 corporate actions
        n_actions = np.random.randint(0, 3)

        for _ in range(n_actions):
            action_date = start + timedelta(days=np.random.randint(0, days_range))
            action_type = np.random.choice(action_types)

            if action_type == "split":
                # Common split ratios
                factor = np.random.choice([2.0, 3.0, 4.0, 0.5, 0.25])
            else:
                # Dividend as percentage of price (1-5%)
                factor = round(np.random.uniform(0.01, 0.05), 4)

            data.append({
                "symbol": symbol,
                "date": action_date,
                "action_type": action_type,
                "factor": factor,
                "source": "synthetic",
            })

    return pd.DataFrame(data)


def generate_universe(
    symbols: List[str],
    dates: List[date],
    sectors: Optional[dict] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic universe membership data.

    Args:
        symbols: List of ticker symbols
        dates: List of dates for universe snapshots
        sectors: Optional dict of symbol -> sector
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: symbol, date, in_universe, sector, market_cap
    """
    np.random.seed(seed)

    if sectors is None:
        default_sectors = ["Technology", "Healthcare", "Consumer", "Industrial", "Energy"]
        sectors = {s: np.random.choice(default_sectors) for s in symbols}

    data = []
    for symbol in symbols:
        base_market_cap = np.random.uniform(10e9, 500e9)

        for d in dates:
            # Small random variation in market cap
            market_cap = base_market_cap * np.random.uniform(0.9, 1.1)

            data.append({
                "symbol": symbol,
                "date": d,
                "in_universe": True,
                "sector": sectors.get(symbol, "Unknown"),
                "market_cap": round(market_cap, 2),
            })

    return pd.DataFrame(data)
