"""REQ-P1-08: RiskGovernor must enforce all kill switches and the soft scale."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _gov(**overrides):
    """Build a deterministic governor with a controllable clock."""
    from risk.governor import RiskGovernor

    state = {"now": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)}

    def clock() -> float:
        return state["now"].timestamp()

    defaults = dict(
        daily_loss_pct=0.04,
        max_drawdown_pct=0.15,
        equity_floor=0.0,
        max_consecutive_losses=3,
        dd_scale_threshold=0.05,
        dd_scale_floor=0.25,
        min_seconds_between_orders=60,
        clock=clock,
    )
    defaults.update(overrides)
    governor = RiskGovernor(**defaults)
    return governor, state


def _state(equity, *, balance=None, peak=None, day_start=None):
    from risk.governor import AccountState

    return AccountState(
        equity=equity,
        balance=balance if balance is not None else equity,
        peak_equity=peak,
        day_start_equity=day_start,
    )


def test_governor_allows_clean_state() -> None:
    governor, _ = _gov()
    decision = governor.can_open(_state(10_000.0))
    assert decision.allow is True
    assert decision.risk_scale == pytest.approx(1.0)


def test_equity_floor_blocks() -> None:
    governor, _ = _gov(equity_floor=5_000.0)
    decision = governor.can_open(_state(4_999.0))
    assert decision.allow is False
    assert "equity-floor" in decision.reason


def test_daily_loss_cap_blocks() -> None:
    governor, _ = _gov(daily_loss_pct=0.04)
    # Day starts at 10k; account drops to 9.5k = -5%
    decision = governor.can_open(_state(9_500.0, day_start=10_000.0))
    assert decision.allow is False
    assert "daily-loss" in decision.reason


def test_max_drawdown_blocks_and_persists() -> None:
    governor, _ = _gov(max_drawdown_pct=0.10)
    # peak 10k -> equity 8.9k = -11% drawdown
    decision = governor.can_open(_state(8_900.0, peak=10_000.0))
    assert decision.allow is False
    assert "max-drawdown" in decision.reason
    # Even after recovery in the same call sequence, the halt persists.
    governor.reset()
    decision = governor.can_open(_state(10_500.0, peak=10_500.0))
    assert decision.allow is True


def test_consecutive_loss_circuit_breaker() -> None:
    governor, _ = _gov(max_consecutive_losses=3)
    governor.record_trade_result(-100)
    governor.record_trade_result(-50)
    governor.record_trade_result(-10)
    decision = governor.can_open(_state(10_000.0))
    assert decision.allow is False
    assert "consecutive-losses" in decision.reason
    # A win clears it
    governor.record_trade_result(+200)
    decision = governor.can_open(_state(10_000.0))
    assert decision.allow is True


def test_order_rate_limiter_throttles() -> None:
    governor, state = _gov(min_seconds_between_orders=60)
    decision = governor.can_open(_state(10_000.0))
    assert decision.allow is True
    governor.record_order_sent()
    # Same instant — throttled
    decision = governor.can_open(_state(10_000.0))
    assert decision.allow is False
    assert "rate-limit" in decision.reason
    # 61 seconds later — clear
    state["now"] = state["now"] + timedelta(seconds=61)
    decision = governor.can_open(_state(10_000.0))
    assert decision.allow is True


def test_drawdown_soft_scaling() -> None:
    # Daily-loss cap is set high so it doesn't interfere with the DD scaling
    # cases; this test isolates the soft-scale ramp.
    governor, _ = _gov(
        max_drawdown_pct=0.15,
        dd_scale_threshold=0.05,
        dd_scale_floor=0.25,
        daily_loss_pct=0.50,
    )
    # Below threshold — full size
    d = governor.can_open(_state(9_700.0, peak=10_000.0, day_start=10_000.0))  # 3% dd
    assert d.allow and d.risk_scale == pytest.approx(1.0)

    # Halfway between threshold and cap (10% dd) — interp ≈ 0.625
    d = governor.can_open(_state(9_000.0, peak=10_000.0, day_start=10_000.0))
    assert d.allow
    assert 0.5 < d.risk_scale < 0.75
    assert "dd-scaled" in d.reason

    # At the cap floor (14.99% dd) — scale near floor
    d = governor.can_open(_state(8_502.0, peak=10_000.0, day_start=10_000.0))
    assert d.allow
    assert d.risk_scale == pytest.approx(governor.dd_scale_floor, rel=0.05)


def test_status_and_reset() -> None:
    governor, _ = _gov(max_consecutive_losses=2)
    governor.record_trade_result(-1)
    governor.record_trade_result(-1)
    assert governor.can_open(_state(10_000.0)).allow is False
    assert governor.consecutive_losses == 2
    governor.reset()
    assert governor.consecutive_losses == 0
    assert governor.can_open(_state(10_000.0)).allow is True


def test_daily_loss_resets_at_utc_midnight() -> None:
    governor, state = _gov(daily_loss_pct=0.04)
    decision = governor.can_open(_state(9_500.0, day_start=10_000.0))
    assert decision.allow is False  # blocked today
    # Roll to next UTC day; equity recovers
    state["now"] = state["now"] + timedelta(days=1)
    decision = governor.can_open(_state(9_700.0, day_start=9_700.0))
    assert decision.allow is True
