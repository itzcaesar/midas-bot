"""REQ-P1-13: Pydantic v2 settings validation must actually run."""
from __future__ import annotations

import pytest


def test_settings_has_pydantic_v2_validation() -> None:
    """Validation kicks in: assigning out-of-range values raises."""
    from pydantic import ValidationError

    from config.settings import Settings

    # Default construction must succeed
    s = Settings(_env_file=None)  # type: ignore[arg-type]
    assert s.risk_percent == pytest.approx(1.0)

    # Out-of-range value must raise
    with pytest.raises(ValidationError):
        Settings(risk_percent=10.0, _env_file=None)  # type: ignore[arg-type]

    # macd_slow <= macd_fast must raise
    with pytest.raises(ValidationError):
        Settings(macd_fast=20, macd_slow=20, _env_file=None)  # type: ignore[arg-type]


def test_settings_placeholder_mt5_login_treated_as_none() -> None:
    """Placeholder strings like 'your_mt5_account_number' coerce to None instead of raising."""
    from config.settings import Settings

    s = Settings(mt5_login="your_mt5_account_number", _env_file=None)  # type: ignore[arg-type]
    assert s.mt5_login is None

    s = Settings(mt5_login="", _env_file=None)  # type: ignore[arg-type]
    assert s.mt5_login is None

    s = Settings(mt5_login="12345", _env_file=None)  # type: ignore[arg-type]
    assert s.mt5_login == 12345


def test_settings_uppercase_alias_is_dynamic() -> None:
    """`from config import MACD_FAST` must reflect runtime mutation of the singleton."""
    import config
    from config import settings

    original = settings.macd_fast
    assert config.MACD_FAST == original
    try:
        settings.macd_fast = 7
        assert config.MACD_FAST == 7
        assert settings.MACD_FAST == 7  # instance-level alias
    finally:
        settings.macd_fast = original


def test_settings_snapshot_restore_round_trips() -> None:
    """`snapshot()` and `restore()` must perfectly round-trip values."""
    from config import settings

    snap = settings.snapshot()
    settings.risk_percent = 2.5
    settings.macd_fast = 8
    settings.macd_slow = 30
    settings.restore(snap)

    assert settings.risk_percent == snap["risk_percent"]
    assert settings.macd_fast == snap["macd_fast"]
    assert settings.macd_slow == snap["macd_slow"]


def test_settings_unknown_session_rejected() -> None:
    from pydantic import ValidationError
    from config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(allowed_sessions="LONDON,FANTASY_SESSION", _env_file=None)  # type: ignore[arg-type]
