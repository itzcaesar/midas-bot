"""P0 quant correctness tests — look-ahead, purge gap, cost model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# REQ-P0-05: add_market_structure must be causal.
# ---------------------------------------------------------------------------


def _ohlc(seed: int = 1, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1900 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.7,
            "low": close - 0.7,
            "close": close,
        }
    )


def test_market_structure_no_lookahead() -> None:
    """Mutating future bars must not change current swing flags."""
    from ml.indicators import add_market_structure

    df = _ohlc()
    a = add_market_structure(df.copy())

    perturb = df.copy()
    cut = 150
    perturb.iloc[cut:, perturb.columns.get_loc("high")] += 100
    perturb.iloc[cut:, perturb.columns.get_loc("low")] += 100
    b = add_market_structure(perturb)

    for col in ("swing_high", "swing_low"):
        # Values at indices < cut must be identical (causal). The boundary
        # index `cut-1` previously depended on the bar at `cut` via shift(-1);
        # after the fix that dependency is shifted forward, so cut-1 itself is
        # also independent of post-cut data.
        for t in range(cut):
            assert a[col].iloc[t] == b[col].iloc[t], (
                f"{col} at t={t} changed when only rows >= {cut} were perturbed — look-ahead leak!"
            )


# ---------------------------------------------------------------------------
# REQ-P0-06: prepare_ml_data must purge the gap.
# ---------------------------------------------------------------------------


def test_prepare_ml_data_purge_gap() -> None:
    from ml.features import prepare_ml_data

    n = 100
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "open": rng.random(n),
            "high": rng.random(n),
            "low": rng.random(n),
            "close": rng.random(n),
            "volume": rng.random(n),
            "feat_a": rng.random(n),
            "feat_b": rng.random(n),
            "target_signal": rng.choice([-1, 0, 1], n),
        }
    )

    for purge_gap in (0, 1, 5, 10):
        X_train, X_test, *_ = prepare_ml_data(df, test_size=0.2, purge_gap=purge_gap)
        split = int(n * 0.8)
        assert len(X_train) == split - purge_gap
        assert len(X_test) == n - split


# ---------------------------------------------------------------------------
# REQ-P0-09: cost model must reduce PnL vs zero-cost.
# ---------------------------------------------------------------------------


class _FakeSig:
    def __init__(self, direction, sl, tp):
        self.direction = direction
        self.confidence = 1.0
        self.stop_loss = sl
        self.take_profit = tp
        self.reasons = []


class _FakeStrategy:
    def __init__(self):
        self.toggle = True

    def analyze(self, window, dxy_window):
        price = float(window["close"].iloc[-1])
        if self.toggle:
            self.toggle = False
            return _FakeSig("BUY", price - 5.0, price + 5.0)
        self.toggle = True
        return _FakeSig("SELL", price + 5.0, price - 5.0)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 300
    close = 1900 + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )


def test_cost_model_penalizes_pnl(synthetic_data) -> None:
    from backtesting.engine import BacktestEngine, CostModel, ZERO_COST_MODEL

    eng_zero = BacktestEngine(initial_balance=10_000.0, cost_model=ZERO_COST_MODEL)
    res_zero = eng_zero.run(synthetic_data, _FakeStrategy(), warmup_bars=10)

    eng_costed = BacktestEngine(
        initial_balance=10_000.0,
        cost_model=CostModel(
            spread_points=20.0, slippage_points_market=5.0, commission_per_lot=2.0
        ),
    )
    res_costed = eng_costed.run(synthetic_data, _FakeStrategy(), warmup_bars=10)

    assert res_zero.total_trades == res_costed.total_trades > 0
    assert res_costed.total_profit < res_zero.total_profit


def test_cost_model_zero_matches_legacy_no_cost(synthetic_data) -> None:
    """Sanity: the explicit ZERO_COST_MODEL behaves like a free-execution backtest."""
    from backtesting.engine import BacktestEngine, ZERO_COST_MODEL

    eng = BacktestEngine(initial_balance=10_000.0, cost_model=ZERO_COST_MODEL)
    res = eng.run(synthetic_data, _FakeStrategy(), warmup_bars=10)
    # All trades must hit a TP or SL exactly, so PnL is some integer multiple of 5 USD * lots
    for trade in res.trades:
        gross_per_lot = (trade.exit_price - trade.entry_price) * 10 * 10  # pip*pip_value
        if trade.direction == "SELL":
            gross_per_lot = -gross_per_lot
        # within rounding (lot is a float)
        assert abs(trade.profit - gross_per_lot * trade.lot_size) < 1e-6
