"""
Market Regime Classifier (REQ-P2-11).

Classifies the current market state into one of several regimes so that
models and strategies can adapt their behavior. Regimes are determined by
a combination of ADX (trend strength) and ATR percentile (volatility level).

Regime labels:
  - TRENDING_HIGH_VOL: Strong trend + high volatility (momentum strategies)
  - TRENDING_LOW_VOL: Strong trend + low volatility (trend-following)
  - RANGING_HIGH_VOL: No trend + high volatility (mean-reversion or sit out)
  - RANGING_LOW_VOL: No trend + low volatility (breakout watch)

These labels can be used as:
  1. A categorical feature for ML models
  2. A filter to select which strategy/style to deploy
  3. A gate to reduce position size in unfavorable regimes
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.logger import get_logger

logger = get_logger("midas.regime")


class MarketRegime(Enum):
    TRENDING_HIGH_VOL = "trending_high_vol"
    TRENDING_LOW_VOL = "trending_low_vol"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    UNKNOWN = "unknown"


# Thresholds (can be tuned or made configurable)
ADX_TREND_THRESHOLD = 25.0  # ADX > 25 = trending
VOL_PERCENTILE_THRESHOLD = 50  # Above median = high vol


def classify_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    atr_period: int = 14,
    vol_lookback: int = 100,
) -> Tuple[MarketRegime, dict]:
    """Classify the current market regime from OHLCV data.

    Args:
        df: DataFrame with at least 'high', 'low', 'close' columns.
        adx_period: Period for ADX calculation.
        atr_period: Period for ATR calculation.
        vol_lookback: Window for computing ATR percentile rank.

    Returns:
        Tuple of (regime, details_dict).
    """
    if len(df) < max(adx_period, atr_period, vol_lookback) + 10:
        return MarketRegime.UNKNOWN, {"reason": "insufficient data"}

    # --- ADX for trend strength ---
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = tr.rolling(window=adx_period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=adx_period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=adx_period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=adx_period).mean()

    current_adx = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0

    # --- ATR percentile for volatility ---
    atr_series = tr.rolling(window=atr_period).mean()
    current_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

    # Percentile rank of current ATR within the lookback window
    atr_window = atr_series.iloc[-vol_lookback:]
    if len(atr_window.dropna()) > 10:
        vol_percentile = float((atr_window < current_atr).sum() / len(atr_window.dropna()) * 100)
    else:
        vol_percentile = 50.0

    # --- Classify ---
    is_trending = current_adx >= ADX_TREND_THRESHOLD
    is_high_vol = vol_percentile >= VOL_PERCENTILE_THRESHOLD

    if is_trending and is_high_vol:
        regime = MarketRegime.TRENDING_HIGH_VOL
    elif is_trending and not is_high_vol:
        regime = MarketRegime.TRENDING_LOW_VOL
    elif not is_trending and is_high_vol:
        regime = MarketRegime.RANGING_HIGH_VOL
    else:
        regime = MarketRegime.RANGING_LOW_VOL

    details = {
        "adx": round(current_adx, 2),
        "atr": round(current_atr, 4),
        "vol_percentile": round(vol_percentile, 1),
        "is_trending": is_trending,
        "is_high_vol": is_high_vol,
    }

    logger.debug(f"Regime: {regime.value} (ADX={current_adx:.1f}, vol_pct={vol_percentile:.0f}%)")
    return regime, details


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime-related features to a DataFrame for ML consumption.

    Adds columns:
      - regime_adx: raw ADX value
      - regime_vol_pct: ATR percentile rank
      - regime_trending: binary (1 if ADX > threshold)
      - regime_high_vol: binary (1 if vol percentile > threshold)
      - regime_label: integer encoding of the 4 regimes (0-3)
    """
    df = df.copy()

    # ADX
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
    atr = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr
    minus_di = 100 * minus_dm.rolling(14).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14).mean()

    df["regime_adx"] = adx

    # Volatility percentile (rolling rank)
    atr_series = tr.rolling(14).mean()
    df["regime_vol_pct"] = atr_series.rolling(100).rank(pct=True) * 100

    # Binary flags
    df["regime_trending"] = (df["regime_adx"] >= ADX_TREND_THRESHOLD).astype(int)
    df["regime_high_vol"] = (df["regime_vol_pct"] >= VOL_PERCENTILE_THRESHOLD).astype(int)

    # Integer label: 0=ranging_low, 1=ranging_high, 2=trending_low, 3=trending_high
    df["regime_label"] = df["regime_trending"] * 2 + df["regime_high_vol"]

    return df
