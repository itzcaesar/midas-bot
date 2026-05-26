"""P0 smoke tests — entrypoints, ticket generator, telegram stub, filters."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


# ---------------------------------------------------------------------------
# REQ-P0-01: every entrypoint must parse and contain `import sys`.
# ---------------------------------------------------------------------------

ENTRYPOINTS = [
    "src/main.py",
    "src/ml/train.py",
    "src/dashboard/app.py",
    "src/strategy/styled_signal_generator.py",
]


@pytest.mark.parametrize("relpath", ENTRYPOINTS)
def test_entrypoint_imports_sys(relpath: str) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / relpath).read_text(encoding="utf-8")
    assert "import sys" in text, f"{relpath} is missing `import sys`"


@pytest.mark.parametrize("relpath", ENTRYPOINTS)
def test_entrypoint_compiles(relpath: str) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / relpath).read_text(encoding="utf-8")
    compile(source, relpath, "exec")  # raises SyntaxError on failure


# ---------------------------------------------------------------------------
# REQ-P0-03: dry-run tickets must be unique and negative.
# ---------------------------------------------------------------------------


def _import_mt5_with_stub() -> ModuleType:
    """Import broker.mt5 with a fake MetaTrader5 module since it isn't on this box.

    The stub must look like a real module to other importers — torch's
    introspection walks ``sys.modules`` and crashes on modules without
    ``__file__``. We provide one and clean up after the test scope.
    """
    if "MetaTrader5" not in sys.modules:
        fake = ModuleType("MetaTrader5")
        fake.__file__ = "<stub>"
        fake.__spec__ = None  # type: ignore[attr-defined]
        # Anything that the module touches at import time returns 0
        fake.__getattr__ = lambda name: 0  # type: ignore[attr-defined]
        sys.modules["MetaTrader5"] = fake
    return importlib.import_module("broker.mt5")


@pytest.fixture(autouse=False)
def _clean_mt5_stub():
    """Remove the MT5 stub after the test that injected it, so other tests
    that import ``torch`` (which introspects ``sys.modules``) see a clean state.
    """
    yield
    sys.modules.pop("MetaTrader5", None)


def test_dry_run_ticket_uniqueness(_clean_mt5_stub) -> None:
    mt5_mod = _import_mt5_with_stub()
    tickets = [mt5_mod._make_dry_run_ticket() for _ in range(100)]
    assert len(set(tickets)) == 100, "dry-run tickets collided"
    assert all(t < 0 for t in tickets), "dry-run tickets must be negative to avoid colliding with real MT5 tickets"


# ---------------------------------------------------------------------------
# REQ-P0-04: TelegramNotifier import + safe no-ops when disabled.
# ---------------------------------------------------------------------------


def test_telegram_notifier_disabled_is_no_op() -> None:
    from notifications.telegram import TelegramNotifier

    notifier = TelegramNotifier(bot_token=None, chat_id=None)
    assert notifier.is_enabled is False
    # Every method must return False (not raise) when disabled
    assert notifier.send_startup_message(1234.56) is False
    assert notifier.send_trade_opened(1, "XAUUSD", "BUY", 0.1, 1900, 1890, 1920) is False
    assert notifier.send_trade_closed(1, "XAUUSD", "BUY", 1900, 1920, 200.0, "TP") is False
    assert notifier.send_error_alert("boom", "main") is False
    assert notifier.send_shutdown_message() is False


# ---------------------------------------------------------------------------
# REQ-P0-10: pre_trade filter helper.
# ---------------------------------------------------------------------------


class _BlockSession:
    def is_valid_session(self):
        return False, "weekend"


class _OpenSession:
    def is_valid_session(self):
        return True, "ok"


class _UnsafeNews:
    def is_safe_to_trade(self):
        return False, "NFP in 12 minutes"


class _SafeNews:
    def is_safe_to_trade(self):
        return True, "ok"


def test_pre_trade_filters_block_session(monkeypatch) -> None:
    from analysis.pre_trade import pre_trade_filters_passed
    from config import settings

    monkeypatch.setattr(settings, "session_filter_enabled", True, raising=False)
    monkeypatch.setattr(settings, "news_filter_enabled", True, raising=False)
    ok, reason = pre_trade_filters_passed(
        session_filter=_BlockSession(), news_filter=_SafeNews()
    )
    assert ok is False
    assert "weekend" in reason


def test_pre_trade_filters_block_news(monkeypatch) -> None:
    from analysis.pre_trade import pre_trade_filters_passed
    from config import settings

    monkeypatch.setattr(settings, "session_filter_enabled", True, raising=False)
    monkeypatch.setattr(settings, "news_filter_enabled", True, raising=False)
    ok, reason = pre_trade_filters_passed(
        session_filter=_OpenSession(), news_filter=_UnsafeNews()
    )
    assert ok is False
    assert "NFP" in reason


def test_pre_trade_filters_pass_when_disabled(monkeypatch) -> None:
    from analysis.pre_trade import pre_trade_filters_passed
    from config import settings

    monkeypatch.setattr(settings, "session_filter_enabled", False, raising=False)
    monkeypatch.setattr(settings, "news_filter_enabled", False, raising=False)
    ok, _ = pre_trade_filters_passed(
        session_filter=_BlockSession(), news_filter=_UnsafeNews()
    )
    assert ok is True
