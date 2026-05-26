"""
Multi-Asset Portfolio with Covariance-Aware Sizing (REQ-P3-04).

When trading multiple correlated assets (XAU, XAG, DXY, BTC), naive per-asset
risk sizing can lead to concentrated exposure. This module computes portfolio-
level position sizes that respect a total-portfolio risk budget.

Approach:
  1. Estimate the covariance matrix of asset returns
  2. For a candidate set of positions, compute portfolio variance
  3. Scale each position so total portfolio risk <= budget

This is a simplified version of risk-parity / mean-variance optimization
suitable for a small number of assets (2-5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.logger import get_logger

logger = get_logger("midas.portfolio")


@dataclass
class AssetAllocation:
    """Proposed allocation for a single asset."""
    symbol: str
    direction: str  # BUY or SELL
    raw_lot: float  # Before portfolio adjustment
    adjusted_lot: float  # After covariance scaling
    weight: float  # Portfolio weight (0-1)
    marginal_risk_contribution: float = 0.0


class PortfolioSizer:
    """Covariance-aware position sizer for multi-asset portfolios.

    Args:
        max_portfolio_risk_pct: Maximum total portfolio risk as % of equity.
        lookback_days: Window for covariance estimation.
    """

    def __init__(
        self,
        max_portfolio_risk_pct: float = 2.0,
        lookback_days: int = 60,
    ) -> None:
        self.max_risk_pct = max_portfolio_risk_pct
        self.lookback = lookback_days
        self._cov_matrix: Optional[np.ndarray] = None
        self._symbols: List[str] = []

    def update_covariance(self, returns_df) -> None:
        """Update the covariance matrix from a DataFrame of daily returns.

        Args:
            returns_df: DataFrame with columns = asset symbols, rows = daily returns.
        """
        if len(returns_df) < 20:
            logger.warning("Insufficient data for covariance estimation")
            return

        recent = returns_df.tail(self.lookback)
        self._cov_matrix = recent.cov().values
        self._symbols = list(returns_df.columns)
        logger.debug(f"Covariance updated: {self._symbols}, shape={self._cov_matrix.shape}")

    def size_portfolio(
        self,
        signals: Dict[str, dict],
        equity: float,
        per_asset_risk_pct: float = 1.0,
    ) -> List[AssetAllocation]:
        """Compute portfolio-aware position sizes.

        Args:
            signals: Dict of {symbol: {'direction': str, 'lot': float, 'sl_distance': float}}.
            equity: Current account equity.
            per_asset_risk_pct: Naive per-asset risk % (before portfolio adjustment).

        Returns:
            List of AssetAllocation with adjusted lots.
        """
        if not signals:
            return []

        symbols = list(signals.keys())
        n = len(symbols)

        # If no covariance data or single asset, use naive sizing
        if self._cov_matrix is None or n == 1:
            return [
                AssetAllocation(
                    symbol=sym,
                    direction=sig["direction"],
                    raw_lot=sig["lot"],
                    adjusted_lot=sig["lot"],
                    weight=1.0 / n,
                )
                for sym, sig in signals.items()
            ]

        # Build weight vector (equal-weight starting point)
        weights = np.ones(n) / n

        # Get covariance sub-matrix for the active symbols
        idx_map = {s: i for i, s in enumerate(self._symbols)}
        active_idx = [idx_map[s] for s in symbols if s in idx_map]

        if len(active_idx) != n:
            # Some symbols not in covariance matrix — fall back to naive
            return [
                AssetAllocation(
                    symbol=sym,
                    direction=sig["direction"],
                    raw_lot=sig["lot"],
                    adjusted_lot=sig["lot"],
                    weight=1.0 / n,
                )
                for sym, sig in signals.items()
            ]

        sub_cov = self._cov_matrix[np.ix_(active_idx, active_idx)]

        # Portfolio variance = w' * Cov * w
        port_var = weights @ sub_cov @ weights
        port_vol = np.sqrt(port_var) if port_var > 0 else 1e-6

        # Scale factor: target_risk / portfolio_vol
        target_risk = self.max_risk_pct / 100.0
        scale = target_risk / (port_vol * np.sqrt(252))  # Annualized
        scale = min(scale, 2.0)  # Cap at 2x to prevent extreme leverage

        allocations = []
        for i, sym in enumerate(symbols):
            sig = signals[sym]
            raw_lot = sig["lot"]
            adjusted = round(raw_lot * scale * weights[i] * n, 2)
            adjusted = max(0.01, adjusted)

            # Marginal risk contribution
            mrc = (sub_cov[i] @ weights) / port_vol if port_vol > 0 else 0.0

            allocations.append(AssetAllocation(
                symbol=sym,
                direction=sig["direction"],
                raw_lot=raw_lot,
                adjusted_lot=adjusted,
                weight=float(weights[i]),
                marginal_risk_contribution=float(mrc.sum()) if hasattr(mrc, 'sum') else float(mrc),
            ))

        logger.info(
            f"Portfolio sizing: {n} assets, port_vol={port_vol:.4f}, "
            f"scale={scale:.2f}"
        )
        return allocations
