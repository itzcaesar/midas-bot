"""
Generate synthetic XAUUSD data for training and dry-run testing.

This creates realistic-looking gold price data with:
  - Trending and ranging regimes
  - Volatility clustering (GARCH-like)
  - Session-based volume patterns
  - Realistic price levels (~$1800-$2400 range)

Output: data/XAU_1h_data.csv (same format as the Kaggle dataset)

Usage: python scripts/generate_synthetic_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd


def generate_xauusd_synthetic(
    n_bars: int = 50_000,
    timeframe_minutes: int = 60,
    start_price: float = 1800.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic XAUUSD OHLCV data."""
    rng = np.random.default_rng(seed)

    # --- Price generation with regime switching ---
    prices = np.zeros(n_bars)
    prices[0] = start_price

    # Volatility (GARCH-like clustering)
    base_vol = 0.0008  # ~0.08% per hour
    vol = np.zeros(n_bars)
    vol[0] = base_vol

    # Regime: 0=ranging, 1=trending up, 2=trending down
    regime = np.zeros(n_bars, dtype=int)
    regime[0] = 0
    regime_duration = 0

    for i in range(1, n_bars):
        # Regime switching (average regime lasts ~200 bars)
        regime_duration += 1
        if rng.random() < 0.005 or regime_duration > 500:
            regime[i] = rng.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
            regime_duration = 0
        else:
            regime[i] = regime[i - 1]

        # Volatility clustering
        vol[i] = 0.9 * vol[i - 1] + 0.1 * base_vol + rng.exponential(base_vol * 0.3)
        vol[i] = np.clip(vol[i], base_vol * 0.3, base_vol * 5.0)

        # Drift based on regime
        if regime[i] == 1:  # Trending up
            drift = vol[i] * 0.3
        elif regime[i] == 2:  # Trending down
            drift = -vol[i] * 0.3
        else:  # Ranging
            drift = -0.0001 * (prices[i - 1] - start_price) / start_price  # Mean reversion

        # Price step
        shock = rng.normal(0, vol[i])
        prices[i] = prices[i - 1] * (1 + drift + shock)

    # --- Generate OHLCV from close prices ---
    # Intrabar volatility
    intrabar_vol = vol * 0.6

    high = prices * (1 + np.abs(rng.normal(0, intrabar_vol)))
    low = prices * (1 - np.abs(rng.normal(0, intrabar_vol)))
    open_prices = prices * (1 + rng.normal(0, intrabar_vol * 0.3))

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_prices, prices))
    low = np.minimum(low, np.minimum(open_prices, prices))

    # Volume with session patterns (higher during London/NY)
    hour_of_day = np.arange(n_bars) % 24
    session_volume = np.where(
        (hour_of_day >= 8) & (hour_of_day <= 16), 1.5,  # London
        np.where((hour_of_day >= 13) & (hour_of_day <= 21), 1.3, 0.5)  # NY
    )
    volume = (rng.exponential(1000, n_bars) * session_volume * (1 + vol / base_vol)).astype(int)

    # --- Build DataFrame ---
    start_date = pd.Timestamp("2004-01-01")
    timestamps = pd.date_range(start=start_date, periods=n_bars, freq=f"{timeframe_minutes}min")

    df = pd.DataFrame({
        "time": timestamps,
        "open": np.round(open_prices, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(prices, 2),
        "volume": volume,
    })

    return df


def main():
    print("Generating synthetic XAUUSD data...")

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate 1h data (primary training timeframe)
    df_1h = generate_xauusd_synthetic(n_bars=50_000, timeframe_minutes=60, seed=42)
    path_1h = data_dir / "XAU_1h_data.csv"
    df_1h.to_csv(path_1h, sep=";", index=False)
    print(f"  1h: {len(df_1h)} bars -> {path_1h}")
    print(f"      Range: {df_1h['close'].min():.2f} - {df_1h['close'].max():.2f}")
    print(f"      Period: {df_1h['time'].iloc[0]} to {df_1h['time'].iloc[-1]}")

    # Generate 15m data (intraday)
    df_15m = generate_xauusd_synthetic(n_bars=100_000, timeframe_minutes=15, seed=7)
    path_15m = data_dir / "XAU_15m_data.csv"
    df_15m.to_csv(path_15m, sep=";", index=False)
    print(f"  15m: {len(df_15m)} bars -> {path_15m}")

    # Generate 4h data (swing/position)
    df_4h = generate_xauusd_synthetic(n_bars=15_000, timeframe_minutes=240, seed=99)
    path_4h = data_dir / "XAU_4h_data.csv"
    df_4h.to_csv(path_4h, sep=";", index=False)
    print(f"  4h: {len(df_4h)} bars -> {path_4h}")

    print("\nDone! Data ready for training.")


if __name__ == "__main__":
    main()
