"""REQ-P2-11: Regime classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _trending_data(n=200):
    """Generate data with a clear uptrend (high ADX)."""
    rng = np.random.default_rng(10)
    close = 1900 + np.cumsum(np.ones(n) * 0.5 + rng.normal(0, 0.1, n))
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 1.0,
        "low": close - 0.5,
        "close": close,
    })


def _ranging_data(n=200):
    """Generate mean-reverting data (low ADX)."""
    rng = np.random.default_rng(20)
    close = 1900 + rng.normal(0, 0.3, n).cumsum() * 0.1
    # Force mean reversion
    close = 1900 + (close - close.mean()) * 0.1
    return pd.DataFrame({
        "open": close - 0.05,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
    })


def test_trending_regime_detected() -> None:
    from analysis.regime import classify_regime, MarketRegime

    df = _trending_data(200)
    regime, details = classify_regime(df)
    assert regime in (MarketRegime.TRENDING_HIGH_VOL, MarketRegime.TRENDING_LOW_VOL)
    assert details["is_trending"] is True


def test_add_regime_features_columns() -> None:
    from analysis.regime import add_regime_features

    df = _trending_data(200)
    result = add_regime_features(df)

    expected_cols = ["regime_adx", "regime_vol_pct", "regime_trending", "regime_high_vol", "regime_label"]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"

    # regime_label should be 0-3
    valid = result["regime_label"].dropna()
    assert valid.isin([0, 1, 2, 3]).all()
