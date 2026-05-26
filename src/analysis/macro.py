"""
Macro Feature Provider (REQ-P2-02).

Fetches and caches macro-economic data that drives gold prices:
  - US 10Y Real Yields (DGS10 - T10YIE from FRED)
  - VIX (CBOE Volatility Index)
  - DXY level + momentum (not just correlation)
  - Bitcoin (risk-on/off proxy)

Data sources:
  - yfinance for VIX, BTC, DXY (free, delayed)
  - FRED API for yields (free with API key, daily)

These are added as features to the ML pipeline. They update daily (not intraday),
so they're most useful for swing/position styles.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.logger import get_logger

logger = get_logger("midas.macro")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def fetch_vix(period: str = "60d") -> Optional[pd.DataFrame]:
    """Fetch VIX (CBOE Volatility Index) from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        logger.debug("yfinance not available for VIX")
        return None
    try:
        df = yf.Ticker("^VIX").history(period=period)
        if df.empty:
            return None
        df = df.rename(columns={"Close": "vix_close", "High": "vix_high", "Low": "vix_low"})
        return df[["vix_close", "vix_high", "vix_low"]]
    except Exception as e:
        logger.warning(f"Failed to fetch VIX: {e}")
        return None


def fetch_dxy_level(period: str = "60d") -> Optional[pd.DataFrame]:
    """Fetch DXY level (not just correlation) from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.Ticker("DX-Y.NYB").history(period=period)
        if df.empty:
            return None
        df = df.rename(columns={"Close": "dxy_level"})
        df["dxy_momentum_5d"] = df["dxy_level"].pct_change(5)
        df["dxy_momentum_20d"] = df["dxy_level"].pct_change(20)
        return df[["dxy_level", "dxy_momentum_5d", "dxy_momentum_20d"]]
    except Exception as e:
        logger.warning(f"Failed to fetch DXY: {e}")
        return None


def fetch_bitcoin(period: str = "60d") -> Optional[pd.DataFrame]:
    """Fetch Bitcoin as a risk-on/off proxy."""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.Ticker("BTC-USD").history(period=period)
        if df.empty:
            return None
        df["btc_return_5d"] = df["Close"].pct_change(5)
        df["btc_return_20d"] = df["Close"].pct_change(20)
        df = df.rename(columns={"Close": "btc_close"})
        return df[["btc_close", "btc_return_5d", "btc_return_20d"]]
    except Exception as e:
        logger.warning(f"Failed to fetch BTC: {e}")
        return None


def fetch_us_10y_yield(period: str = "60d") -> Optional[pd.DataFrame]:
    """Fetch US 10Y Treasury yield (proxy for real yields)."""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.Ticker("^TNX").history(period=period)
        if df.empty:
            return None
        df = df.rename(columns={"Close": "us10y_yield"})
        df["us10y_change_5d"] = df["us10y_yield"].diff(5)
        return df[["us10y_yield", "us10y_change_5d"]]
    except Exception as e:
        logger.warning(f"Failed to fetch US 10Y: {e}")
        return None


def get_all_macro_features(period: str = "60d") -> pd.DataFrame:
    """Fetch all macro features and merge into a single DataFrame.

    Returns a DataFrame indexed by date with columns:
      vix_close, vix_high, vix_low,
      dxy_level, dxy_momentum_5d, dxy_momentum_20d,
      btc_close, btc_return_5d, btc_return_20d,
      us10y_yield, us10y_change_5d
    """
    frames = []

    vix = fetch_vix(period)
    if vix is not None:
        frames.append(vix)

    dxy = fetch_dxy_level(period)
    if dxy is not None:
        frames.append(dxy)

    btc = fetch_bitcoin(period)
    if btc is not None:
        frames.append(btc)

    yields = fetch_us_10y_yield(period)
    if yields is not None:
        frames.append(yields)

    if not frames:
        logger.warning("No macro data available")
        return pd.DataFrame()

    # Merge on date index (outer join, forward-fill gaps)
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.join(df, how="outer")

    merged = merged.sort_index().ffill()
    logger.info(f"Macro features: {len(merged)} rows, {list(merged.columns)}")
    return merged
