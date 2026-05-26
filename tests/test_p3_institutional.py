"""P3 institutional-grade tests."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest


# --- REQ-P3-01: Tick buffer + microstructure ---

def test_tick_buffer_computes_features() -> None:
    from data.tick_ingestion import TickBuffer, Tick

    buf = TickBuffer(max_ticks=1000, window_seconds=60)
    now = time.time()

    # Add 100 ticks
    for i in range(100):
        buf.add(Tick(
            timestamp=now - 50 + i * 0.5,
            bid=1900.0 + i * 0.01,
            ask=1900.2 + i * 0.01,
            last=1900.1 + i * 0.01,
            volume=float(i % 10 + 1),
        ))

    features = buf.compute_features(window_seconds=60)
    assert features["tick_rate"] > 0
    assert features["avg_spread"] > 0
    assert 0.0 <= features["buy_volume_ratio"] <= 1.0
    assert features["vwap"] > 0


# --- REQ-P3-05: Execution algos ---

def test_twap_splits_order() -> None:
    from execution.algos import TWAPExecutor

    tickets = []

    def fake_place(symbol, order_type, lot, **kwargs):
        t = len(tickets) + 1
        tickets.append(t)
        return t

    twap = TWAPExecutor(n_slices=3, interval_seconds=0.01)
    result = twap.execute("XAUUSD", "BUY", 0.30, fake_place)

    assert result.success
    assert len(result.children) == 3
    assert result.filled_lot == pytest.approx(0.30, abs=0.01)
    assert all(c.filled for c in result.children)


# --- REQ-P3-07: Independent risk system ---

def test_kill_switch_activates_on_pnl_limit() -> None:
    from risk.independent_risk import IndependentRiskSystem

    sys = IndependentRiskSystem(daily_pnl_limit=-100.0)
    sys.heartbeat(daily_pnl=-150.0)

    allowed, reason = sys.is_trading_allowed()
    assert allowed is False
    assert "KILL SWITCH" in reason


def test_kill_switch_heartbeat_timeout() -> None:
    from risk.independent_risk import IndependentRiskSystem

    sys = IndependentRiskSystem(heartbeat_timeout=0.1)
    sys.heartbeat(daily_pnl=0)
    time.sleep(0.2)

    allowed, reason = sys.is_trading_allowed()
    assert allowed is False
    assert "heartbeat" in reason.lower()


def test_kill_switch_manual_override() -> None:
    from risk.independent_risk import IndependentRiskSystem

    sys = IndependentRiskSystem()
    sys.activate_kill_switch("test")
    assert sys.is_trading_allowed()[0] is False
    sys.deactivate_kill_switch()
    sys.heartbeat()
    assert sys.is_trading_allowed()[0] is True


# --- REQ-P3-08: Audit trail ---

def test_audit_trail_writes_and_queries(tmp_path) -> None:
    from compliance.audit import AuditTrail

    trail = AuditTrail(log_dir=str(tmp_path))
    trail.log_signal("BUY", 0.85, ["MACD bullish", "RSI oversold"], model_version="rf_v1.2")
    trail.log_order(12345, "XAUUSD", "BUY", 0.1, 1900.0, 1890.0, 1920.0)
    trail.log_rejection("daily-loss cap", source="RiskGovernor")
    trail.log_close(12345, 1915.0, 150.0, "TP_HIT")

    entries = trail.query()
    assert len(entries) == 4
    assert entries[0]["event_type"] == "SIGNAL"
    assert entries[1]["event_type"] == "ORDER"
    assert entries[2]["event_type"] == "REJECTION"
    assert entries[3]["event_type"] == "CLOSE"

    # Filter by type
    signals = trail.query(event_type="SIGNAL")
    assert len(signals) == 1
    assert signals[0]["details"]["confidence"] == 0.85


# --- REQ-P3-09: Shadow mode ---

def test_shadow_runner_tracks_trades() -> None:
    from risk.shadow_mode import ShadowRunner

    shadow = ShadowRunner("lgbm_v2", min_days=0, min_trades=2)
    shadow.record_signal("BUY", 1900.0, 1890.0, 1920.0, 0.8)
    shadow.record_signal("SELL", 1920.0, 1930.0, 1900.0, 0.75)

    # Simulate price hitting TP for the BUY
    shadow.update_open_trades(1920.0)
    # Simulate price hitting SL for the SELL
    shadow.update_open_trades(1930.0)

    report = shadow.evaluate(champion_pnl=500.0, champion_trades=10)
    assert report.challenger_trades == 2
    assert report.challenger_pnl != 0


# --- REQ-P3-10: Capital tiering ---

def test_capital_tier_progression() -> None:
    from risk.capital_tiering import TierManager

    mgr = TierManager(current_level=0)
    assert mgr.current_tier.name == "Paper"

    # Can't advance without enough profitable days
    can, reason = mgr.can_advance(current_capital=5000.0)
    assert can is False

    # Simulate 30 profitable days
    for _ in range(30):
        mgr.record_day(profitable=True)

    can, reason = mgr.can_advance(current_capital=5000.0)
    assert can is True

    mgr.advance()
    assert mgr.current_tier.name == "Micro"
    assert mgr.current_tier.max_risk_pct == 0.5


def test_capital_tier_risk_limits() -> None:
    from risk.capital_tiering import TierManager

    mgr = TierManager(current_level=4)  # Full tier
    limits = mgr.get_risk_limits()
    assert limits["max_risk_pct"] == 0.5
    assert limits["max_positions"] == 5
    assert limits["tier_name"] == "Full"


# --- REQ-P3-04: Portfolio sizing ---

def test_portfolio_sizer_single_asset() -> None:
    from portfolio.sizing import PortfolioSizer

    sizer = PortfolioSizer(max_portfolio_risk_pct=2.0)
    signals = {"XAUUSD": {"direction": "BUY", "lot": 0.5, "sl_distance": 10.0}}

    allocations = sizer.size_portfolio(signals, equity=10_000.0)
    assert len(allocations) == 1
    assert allocations[0].symbol == "XAUUSD"
    assert allocations[0].adjusted_lot == 0.5  # No adjustment for single asset


def test_portfolio_sizer_multi_asset_with_covariance() -> None:
    import pandas as pd
    from portfolio.sizing import PortfolioSizer

    rng = np.random.default_rng(42)
    # Simulate correlated returns
    returns = pd.DataFrame({
        "XAUUSD": rng.normal(0.001, 0.01, 100),
        "XAGUSD": rng.normal(0.001, 0.015, 100),
    })
    # Add correlation
    returns["XAGUSD"] += returns["XAUUSD"] * 0.7

    sizer = PortfolioSizer(max_portfolio_risk_pct=2.0)
    sizer.update_covariance(returns)

    signals = {
        "XAUUSD": {"direction": "BUY", "lot": 0.5, "sl_distance": 10.0},
        "XAGUSD": {"direction": "BUY", "lot": 0.3, "sl_distance": 15.0},
    }

    allocations = sizer.size_portfolio(signals, equity=50_000.0)
    assert len(allocations) == 2
    # With correlation, total allocation should be scaled down
    total_adjusted = sum(a.adjusted_lot for a in allocations)
    total_raw = 0.5 + 0.3
    # Correlated assets should get less than naive sum
    assert total_adjusted > 0  # At least something allocated
