"""
Walk-Forward Validation with Deflated Sharpe Ratio (REQ-P2-01).

Implements proper rolling walk-forward analysis:
  1. Train on window [t0, t1]
  2. Predict on [t1+embargo, t2]
  3. Slide forward
  4. Concatenate all OOS predictions
  5. Report Deflated Sharpe Ratio (Bailey & López de Prado 2014)
  6. Report Probability of Backtest Overfitting (PBO)

This replaces the fake walk-forward in engine.py that ran the same untrained
strategy on N windows without any in-sample optimization step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from scipy import stats

from core.logger import get_logger

logger = get_logger("midas.walk_forward")


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & López de Prado 2014)
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    returns_skew: float = 0.0,
    returns_kurtosis: float = 3.0,
    n_observations: int = 252,
    sr_benchmark: float = 0.0,
) -> float:
    """Compute the Deflated Sharpe Ratio.

    Adjusts the observed Sharpe Ratio for the number of trials (parameter
    combinations, strategy variants) tested. A DSR > 0.95 suggests the
    observed SR is unlikely to be due to chance alone.

    Args:
        observed_sr: The best observed Sharpe Ratio (annualized).
        n_trials: Number of independent trials/strategies tested.
        returns_skew: Skewness of the return series.
        returns_kurtosis: Kurtosis of the return series (excess=0 for normal).
        n_observations: Number of return observations.
        sr_benchmark: Benchmark SR to beat (default 0).

    Returns:
        Probability that the observed SR is genuine (0 to 1).
    """
    if n_trials <= 1:
        return 1.0

    # Expected maximum SR under the null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    z = stats.norm.ppf(1 - 1.0 / n_trials)
    expected_max_sr = (1 - euler_mascheroni) * z + euler_mascheroni * stats.norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )

    # Standard error of the SR estimate
    sr_std = np.sqrt(
        (1 - returns_skew * observed_sr + (returns_kurtosis - 1) / 4 * observed_sr**2)
        / (n_observations - 1)
    )

    if sr_std == 0:
        return 1.0

    # Test statistic
    test_stat = (observed_sr - expected_max_sr * sr_std) / sr_std

    # One-sided p-value (probability SR is genuine)
    return float(stats.norm.cdf(test_stat))


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (simplified CSCV approach)
# ---------------------------------------------------------------------------


def probability_of_overfitting(
    oos_sharpes: List[float],
    is_sharpes: List[float],
) -> float:
    """Simplified PBO: fraction of walk-forward windows where OOS SR < 0.

    A proper CSCV (Combinatorially Symmetric Cross-Validation) implementation
    requires combinatorial partitioning which is expensive. This simplified
    version uses the walk-forward OOS windows directly.

    Args:
        oos_sharpes: Out-of-sample Sharpe ratios per window.
        is_sharpes: In-sample Sharpe ratios per window.

    Returns:
        PBO estimate (0 to 1). Values > 0.5 indicate likely overfitting.
    """
    if not oos_sharpes:
        return 1.0

    # Fraction of OOS windows with negative performance
    n_negative = sum(1 for sr in oos_sharpes if sr <= 0)
    pbo = n_negative / len(oos_sharpes)

    # Also check rank correlation between IS and OOS
    if len(is_sharpes) == len(oos_sharpes) and len(is_sharpes) >= 3:
        corr, _ = stats.spearmanr(is_sharpes, oos_sharpes)
        # Negative correlation = strong overfitting signal
        if corr < 0:
            pbo = max(pbo, 0.5 + abs(corr) / 2)

    return min(1.0, pbo)


# ---------------------------------------------------------------------------
# Walk-Forward Engine
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardWindow:
    """Results from a single walk-forward window."""
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    oos_trades: int = 0
    oos_pnl: float = 0.0
    oos_win_rate: float = 0.0


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward validation results."""
    windows: List[WalkForwardWindow] = field(default_factory=list)
    n_trials: int = 1
    total_oos_trades: int = 0
    total_oos_pnl: float = 0.0
    aggregate_oos_sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    pbo: float = 0.0

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("WALK-FORWARD VALIDATION REPORT (REQ-P2-01)")
        print("=" * 60)
        print(f"Windows: {len(self.windows)}")
        print(f"Total OOS trades: {self.total_oos_trades}")
        print(f"Total OOS P/L: ${self.total_oos_pnl:+,.2f}")
        print(f"Aggregate OOS Sharpe: {self.aggregate_oos_sharpe:.3f}")
        print(f"Deflated Sharpe Ratio: {self.deflated_sharpe:.3f}")
        print(f"Probability of Overfitting: {self.pbo:.1%}")
        print("-" * 60)
        for w in self.windows:
            print(
                f"  Window {w.window_idx}: "
                f"IS_SR={w.is_sharpe:.2f} | OOS_SR={w.oos_sharpe:.2f} | "
                f"trades={w.oos_trades} | PnL=${w.oos_pnl:+.2f}"
            )
        print("=" * 60)

        if self.deflated_sharpe < 0.95:
            print(
                "\n⚠️  Deflated Sharpe < 0.95 — the observed performance may be "
                "due to chance given the number of trials tested."
            )
        if self.pbo > 0.5:
            print(
                "\n⚠️  PBO > 50% — high probability of backtest overfitting. "
                "Do NOT deploy this configuration with real capital."
            )


def run_walk_forward(
    data: pd.DataFrame,
    train_and_predict_fn: Callable,
    n_windows: int = 5,
    train_pct: float = 0.7,
    embargo_bars: int = 1,
    n_trials: int = 1,
) -> WalkForwardReport:
    """Run rolling walk-forward validation.

    Args:
        data: Full historical DataFrame with features + target columns.
        train_and_predict_fn: Callable(train_df, test_df) -> dict with keys:
            'is_sharpe': float, 'oos_sharpe': float, 'oos_trades': int,
            'oos_pnl': float, 'oos_win_rate': float.
        n_windows: Number of walk-forward windows.
        train_pct: Fraction of each window used for training.
        embargo_bars: Gap between train end and test start.
        n_trials: Total number of strategy variants tested (for DSR correction).

    Returns:
        WalkForwardReport with per-window and aggregate metrics.
    """
    n = len(data)
    window_size = n // n_windows
    report = WalkForwardReport(n_trials=n_trials)

    oos_returns_all = []

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, n)
        if end - start < 50:
            continue

        train_end_idx = start + int((end - start) * train_pct)
        test_start_idx = train_end_idx + embargo_bars
        if test_start_idx >= end:
            continue

        train_slice = data.iloc[start:train_end_idx]
        test_slice = data.iloc[test_start_idx:end]

        if len(train_slice) < 20 or len(test_slice) < 10:
            continue

        try:
            result = train_and_predict_fn(train_slice, test_slice)
        except Exception as exc:
            logger.warning(f"Walk-forward window {i} failed: {exc}")
            continue

        window = WalkForwardWindow(
            window_idx=i,
            train_start=start,
            train_end=train_end_idx,
            test_start=test_start_idx,
            test_end=end,
            is_sharpe=result.get("is_sharpe", 0.0),
            oos_sharpe=result.get("oos_sharpe", 0.0),
            oos_trades=result.get("oos_trades", 0),
            oos_pnl=result.get("oos_pnl", 0.0),
            oos_win_rate=result.get("oos_win_rate", 0.0),
        )
        report.windows.append(window)

        # Collect OOS returns for aggregate Sharpe
        oos_ret = result.get("oos_returns", [])
        if oos_ret:
            oos_returns_all.extend(oos_ret)

    # Aggregate metrics
    if report.windows:
        report.total_oos_trades = sum(w.oos_trades for w in report.windows)
        report.total_oos_pnl = sum(w.oos_pnl for w in report.windows)

        oos_sharpes = [w.oos_sharpe for w in report.windows]
        is_sharpes = [w.is_sharpe for w in report.windows]

        # Aggregate OOS Sharpe from concatenated returns
        if oos_returns_all:
            arr = np.array(oos_returns_all)
            mean_r = arr.mean()
            std_r = arr.std()
            report.aggregate_oos_sharpe = (
                (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
            )
        else:
            report.aggregate_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0

        # Deflated Sharpe
        skew = float(stats.skew(oos_returns_all)) if len(oos_returns_all) > 3 else 0.0
        kurt = float(stats.kurtosis(oos_returns_all, fisher=False)) if len(oos_returns_all) > 3 else 3.0
        report.deflated_sharpe = deflated_sharpe_ratio(
            observed_sr=report.aggregate_oos_sharpe,
            n_trials=n_trials,
            returns_skew=skew,
            returns_kurtosis=kurt,
            n_observations=len(oos_returns_all) or 252,
        )

        # PBO
        report.pbo = probability_of_overfitting(oos_sharpes, is_sharpes)

    return report
