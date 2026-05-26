"""REQ-P2-01: Walk-forward validation + Deflated Sharpe + PBO."""
from __future__ import annotations

import numpy as np
import pytest


def test_deflated_sharpe_ratio_basic() -> None:
    from backtesting.walk_forward import deflated_sharpe_ratio

    # Single trial — no correction needed
    dsr = deflated_sharpe_ratio(observed_sr=1.5, n_trials=1)
    assert dsr == 1.0

    # Many trials — DSR should be lower than 1.0 for a modest SR
    dsr = deflated_sharpe_ratio(observed_sr=0.5, n_trials=100, n_observations=252)
    assert 0.0 < dsr < 1.0

    # Very high SR with few trials — should still be high
    dsr = deflated_sharpe_ratio(observed_sr=3.0, n_trials=5, n_observations=500)
    assert dsr > 0.9


def test_probability_of_overfitting() -> None:
    from backtesting.walk_forward import probability_of_overfitting

    # All OOS windows positive — low PBO
    pbo = probability_of_overfitting(
        oos_sharpes=[0.5, 0.8, 1.2, 0.3, 0.6],
        is_sharpes=[1.0, 1.5, 2.0, 0.8, 1.2],
    )
    assert pbo == 0.0

    # All OOS windows negative — high PBO
    pbo = probability_of_overfitting(
        oos_sharpes=[-0.5, -0.8, -1.2, -0.3, -0.6],
        is_sharpes=[1.0, 1.5, 2.0, 0.8, 1.2],
    )
    assert pbo >= 0.5

    # Empty — worst case
    pbo = probability_of_overfitting([], [])
    assert pbo == 1.0


def test_walk_forward_runs_with_synthetic_data() -> None:
    from backtesting.walk_forward import run_walk_forward
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
        "target": rng.choice([-1, 0, 1], n),
    })

    def train_and_predict(train_df, test_df):
        # Trivial: random predictions
        n_test = len(test_df)
        preds = rng.choice([-1, 0, 1], n_test)
        returns = rng.normal(0, 0.01, n_test)
        return {
            "is_sharpe": 0.5,
            "oos_sharpe": float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)),
            "oos_trades": int((preds != 0).sum()),
            "oos_pnl": float(returns.sum() * 1000),
            "oos_win_rate": 0.5,
            "oos_returns": returns.tolist(),
        }

    report = run_walk_forward(df, train_and_predict, n_windows=4, n_trials=10)
    assert len(report.windows) >= 2
    assert report.deflated_sharpe >= 0.0
    assert 0.0 <= report.pbo <= 1.0
    # Should print without error
    report.print_summary()
