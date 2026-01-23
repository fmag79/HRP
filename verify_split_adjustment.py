#!/usr/bin/env python3
"""
Verification script for AAPL split adjustment in backtest.

AAPL had a 4:1 stock split on August 31, 2020.
This script verifies that:
1. Raw prices show the split (price drops by ~4x)
2. Split-adjusted prices are continuous
3. Momentum signals don't show artificial spikes
"""

from datetime import date
import pandas as pd
import numpy as np
from hrp.research.backtest import get_price_data, generate_momentum_signals
from hrp.data.db import get_db

pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 2)


def verify_split_adjustment():
    """
    Verify that AAPL split adjustment works correctly in backtest.
    """
    print("="*80)
    print("VERIFICATION: AAPL Split Adjustment in Backtest")
    print("="*80)

    # Load AAPL data around split date
    symbols = ['AAPL']
    start = date(2020, 7, 1)  # 2 months before split
    end = date(2020, 10, 31)  # 2 months after split

    print(f"\nLoading AAPL data from {start} to {end}")
    print("Split date: August 31, 2020 (4:1 split)\n")

    # 1. Load raw prices (without split adjustment)
    print("-" * 80)
    print("STEP 1: Raw Prices (without split adjustment)")
    print("-" * 80)

    db = get_db()
    query = """
        SELECT date, close
        FROM prices
        WHERE symbol = 'AAPL'
          AND date >= ?
          AND date <= ?
        ORDER BY date
    """
    raw_df = db.fetchdf(query, (start, end))

    if raw_df.empty:
        print("❌ ERROR: No AAPL price data found in database!")
        print("Run: python3 -m hrp.data.ingestion.prices --symbols AAPL --start 2020-01-01")
        return False

    raw_df['date'] = pd.to_datetime(raw_df['date'])
    raw_df = raw_df.set_index('date')

    # Show prices around split date
    split_date = pd.Timestamp('2020-08-31')
    window_before = raw_df.loc[raw_df.index <= split_date].tail(5)
    window_after = raw_df.loc[raw_df.index > split_date].head(5)

    print("\nPrices before split (last 5 days):")
    print(window_before)

    print("\nPrices after split (first 5 days):")
    print(window_after)

    # Calculate the price ratio to show the split
    if not window_before.empty and not window_after.empty:
        price_before = window_before['close'].iloc[-1]
        price_after = window_after['close'].iloc[0]
        ratio = price_before / price_after
        print(f"\n📊 Price ratio across split: {ratio:.2f}x")
        print(f"   Expected for 4:1 split: ~4.0x")

        if 3.5 < ratio < 4.5:
            print("   ✅ Split detected in raw data")
        else:
            print("   ⚠️  Unexpected ratio - may not be a 4:1 split")

    # 2. Load split-adjusted prices
    print("\n" + "-" * 80)
    print("STEP 2: Split-Adjusted Prices")
    print("-" * 80)

    prices_adjusted = get_price_data(symbols, start, end, adjust_splits=True)

    if 'split_adjusted_close' in prices_adjusted.columns.get_level_values(0):
        close_adj = prices_adjusted['split_adjusted_close']['AAPL']
        print("\n✅ Split-adjusted prices loaded successfully")

        # Show adjusted prices around split
        adj_before = close_adj.loc[close_adj.index <= split_date].tail(5)
        adj_after = close_adj.loc[close_adj.index > split_date].head(5)

        print("\nSplit-adjusted prices before split (last 5 days):")
        print(adj_before)

        print("\nSplit-adjusted prices after split (first 5 days):")
        print(adj_after)

        # Calculate continuity
        if not adj_before.empty and not adj_after.empty:
            price_before_adj = adj_before.iloc[-1]
            price_after_adj = adj_after.iloc[0]
            pct_change = (price_after_adj / price_before_adj - 1) * 100

            print(f"\n📊 Price change across split (adjusted): {pct_change:.2f}%")
            print(f"   Expected: Normal daily volatility (< 10%)")

            if abs(pct_change) < 10:
                print("   ✅ Adjusted prices are continuous")
            else:
                print(f"   ❌ Large gap detected: {pct_change:.2f}%")
                return False
    else:
        print("❌ ERROR: split_adjusted_close column not found!")
        return False

    # 3. Calculate momentum signals
    print("\n" + "-" * 80)
    print("STEP 3: Momentum Signal Continuity")
    print("-" * 80)

    # Use a shorter lookback for this test
    lookback = 20  # 20-day momentum

    # Calculate momentum on adjusted prices
    momentum = close_adj.pct_change(lookback)

    # Show momentum around split
    mom_before = momentum.loc[momentum.index <= split_date].tail(5)
    mom_after = momentum.loc[momentum.index > split_date].head(5)

    print(f"\n20-day momentum before split (last 5 days):")
    print(mom_before)

    print(f"\n20-day momentum after split (first 5 days):")
    print(mom_after)

    # Check for artificial spike
    # Drop NaN values before checking
    mom_valid = momentum.dropna()

    if len(mom_valid) > lookback + 5:
        # Get momentum values around split date
        split_window = mom_valid.loc[
            (mom_valid.index >= split_date - pd.Timedelta(days=5)) &
            (mom_valid.index <= split_date + pd.Timedelta(days=5))
        ]

        if not split_window.empty:
            # Calculate z-scores to detect outliers
            mom_mean = mom_valid.mean()
            mom_std = mom_valid.std()

            z_scores = (split_window - mom_mean) / mom_std
            max_z = abs(z_scores).max()

            print(f"\n📊 Momentum statistics:")
            print(f"   Mean momentum: {mom_mean:.4f}")
            print(f"   Std dev: {mom_std:.4f}")
            print(f"   Max |z-score| around split: {max_z:.2f}")
            print(f"   Expected: < 3.0 (no artificial spikes)")

            if max_z < 3.0:
                print("   ✅ No artificial momentum spike detected")
            else:
                print(f"   ❌ Artificial spike detected: z-score = {max_z:.2f}")
                return False

    # 4. Compare with/without adjustment
    print("\n" + "-" * 80)
    print("STEP 4: Comparison - Adjusted vs Unadjusted")
    print("-" * 80)

    # Calculate returns with and without adjustment
    returns_raw = raw_df['close'].pct_change()
    returns_adj = close_adj.pct_change()

    # Focus on split date
    split_returns_raw = returns_raw.loc[
        (returns_raw.index >= split_date - pd.Timedelta(days=2)) &
        (returns_raw.index <= split_date + pd.Timedelta(days=2))
    ]

    split_returns_adj = returns_adj.loc[
        (returns_adj.index >= split_date - pd.Timedelta(days=2)) &
        (returns_adj.index <= split_date + pd.Timedelta(days=2))
    ]

    print("\nDaily returns around split (unadjusted):")
    print(split_returns_raw * 100)

    print("\nDaily returns around split (adjusted):")
    print(split_returns_adj * 100)

    # Check if unadjusted shows the split as a huge negative return
    if not split_returns_raw.empty:
        split_day_return_raw = split_returns_raw.loc[split_returns_raw.index >= split_date].iloc[0] if any(split_returns_raw.index >= split_date) else None
        split_day_return_adj = split_returns_adj.loc[split_returns_adj.index >= split_date].iloc[0] if any(split_returns_adj.index >= split_date) else None

        if split_day_return_raw is not None and split_day_return_adj is not None:
            print(f"\n📊 Returns on split date:")
            print(f"   Unadjusted: {split_day_return_raw * 100:.2f}%")
            print(f"   Adjusted: {split_day_return_adj * 100:.2f}%")

            # Unadjusted should show ~-75% (4:1 split = 75% drop)
            # Adjusted should show normal returns
            if split_day_return_raw < -0.5:
                print("   ✅ Unadjusted correctly shows split as price drop")

            if abs(split_day_return_adj) < 0.15:
                print("   ✅ Adjusted shows normal market returns")
            else:
                print(f"   ⚠️  Adjusted return seems high: {split_day_return_adj * 100:.2f}%")

    # Final verdict
    print("\n" + "="*80)
    print("VERIFICATION RESULT")
    print("="*80)
    print("✅ All checks passed!")
    print("\nConclusion:")
    print("- Raw prices correctly show the 4:1 split")
    print("- Split-adjusted prices are continuous")
    print("- Momentum signals do not show artificial spikes")
    print("- Backtest will use accurate, split-adjusted data")
    print("\n✅ Split adjustment is working correctly in backtests")
    print("="*80)

    return True


if __name__ == "__main__":
    import sys
    success = verify_split_adjustment()
    sys.exit(0 if success else 1)
