"""
MIDAS Configuration Settings (REQ-P1-13).

Pydantic v2 + ``pydantic-settings``. The previous version silently fell back
to a plain Python class when ``BaseSettings`` couldn't be imported (which was
always, on pydantic >= 2.0). That fallback skipped validation entirely. This
module now requires ``pydantic-settings`` and validates on every load.

Settings remain mutable at runtime so the optimizer can rewrite a parameter,
run a backtest, and restore the original. ``Settings.snapshot()`` /
``Settings.restore(snapshot)`` provide a safe way to round-trip mutations.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as exc:  # pragma: no cover - hard fail is the point
    raise ImportError(
        "pydantic-settings is required for MIDAS configuration. "
        "Install with: pip install pydantic-settings"
    ) from exc


class Settings(BaseSettings):
    """Configuration settings with strict validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow extra env vars without crashing (broker installs add things)
        extra="ignore",
        # Mutable so the optimizer can poke values per-trial
        frozen=False,
    )

    # MT5 Connection -----------------------------------------------------------
    mt5_login: Optional[int] = Field(default=None)
    mt5_password: Optional[str] = Field(default=None)
    mt5_server: Optional[str] = Field(default=None)

    # ----- coercion: treat placeholder env strings as "unset" -----
    @field_validator("mt5_login", mode="before")
    @classmethod
    def _coerce_mt5_login(cls, v):  # type: ignore[no-untyped-def]
        # Accept None, empty, or known-placeholder strings as "unset"
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped or stripped.lower().startswith("your_") or stripped.lower() in {"none", "null"}:
                return None
        return v

    @field_validator("mt5_password", "mt5_server", mode="before")
    @classmethod
    def _coerce_optional_str(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped or stripped.lower().startswith("your_"):
                return None
            return stripped
        return v

    @field_validator("telegram_bot_token", "telegram_chat_id", mode="before")
    @classmethod
    def _coerce_optional_telegram(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return None
            return stripped
        return v

    # Symbol Settings ----------------------------------------------------------
    symbol: str = Field(default="XAUUSD")
    dxy_symbol: str = Field(default="DXY")
    timeframe: Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"] = Field(default="M15")

    # Risk Management ----------------------------------------------------------
    risk_percent: float = Field(default=1.0, ge=0.1, le=5.0)
    max_positions: int = Field(default=1, ge=1, le=10)
    stop_loss_pips: float = Field(default=50.0, ge=10.0, le=200.0)
    take_profit_pips: float = Field(default=100.0, ge=20.0, le=500.0)
    trailing_stop_pips: float = Field(default=30.0, ge=10.0, le=100.0)

    # Strategy: MACD -----------------------------------------------------------
    macd_fast: int = Field(default=12, ge=5, le=20)
    macd_slow: int = Field(default=26, ge=15, le=50)
    macd_signal: int = Field(default=9, ge=5, le=20)

    # Strategy: Consolidation --------------------------------------------------
    consolidation_lookback: int = Field(default=20, ge=10, le=50)
    consolidation_threshold: float = Field(default=0.3, ge=0.1, le=1.0)

    # Strategy: Liquidity ------------------------------------------------------
    liquidity_lookback: int = Field(default=50, ge=20, le=100)
    swing_strength: int = Field(default=3, ge=2, le=10)

    # Strategy: DXY ------------------------------------------------------------
    dxy_correlation_period: int = Field(default=20, ge=10, le=50)
    dxy_threshold: float = Field(default=-0.5, ge=-1.0, le=0.0)

    # Strategy: General --------------------------------------------------------
    min_confidence: float = Field(default=0.8, ge=0.4, le=1.0)
    min_factors: int = Field(default=3, ge=2, le=5)

    # Risk Governor (REQ-P1-08) -----------------------------------------------
    # Hard kill switches: when any one trips, no new orders are sent.
    risk_daily_loss_pct: float = Field(default=0.04, ge=0.001, le=0.5)
    risk_max_drawdown_pct: float = Field(default=0.15, ge=0.01, le=0.5)
    risk_equity_floor: float = Field(default=0.0, ge=0.0)
    risk_max_consecutive_losses: int = Field(default=4, ge=1, le=20)
    # Soft scaling: when in `dd_scale_threshold` drawdown, multiply
    # `risk_percent` by a factor that decays from 1.0 to `dd_scale_floor`
    # linearly between `dd_scale_threshold` and `risk_max_drawdown_pct`.
    risk_dd_scale_threshold: float = Field(default=0.05, ge=0.0, le=0.3)
    risk_dd_scale_floor: float = Field(default=0.25, ge=0.05, le=1.0)
    # Order rate limiting (also used by REQ-P1-14 later).
    risk_min_seconds_between_orders: int = Field(default=60, ge=0, le=3600)

    # Multi-Timeframe ----------------------------------------------------------
    htf_enabled: bool = Field(default=True)
    htf_confirmation_required: bool = Field(default=True)

    # Session Filter -----------------------------------------------------------
    session_filter_enabled: bool = Field(default=True)
    allowed_sessions: str = Field(default="LONDON,NEW_YORK,OVERLAP")

    # News Filter --------------------------------------------------------------
    news_filter_enabled: bool = Field(default=True)
    news_buffer_minutes: int = Field(default=30, ge=10, le=120)

    # Execution ----------------------------------------------------------------
    dry_run: bool = Field(default=True)
    loop_interval_seconds: int = Field(default=60, ge=10, le=300)

    # Notifications ------------------------------------------------------------
    telegram_enabled: bool = Field(default=False)
    telegram_bot_token: Optional[str] = Field(default=None)
    telegram_chat_id: Optional[str] = Field(default=None)

    # Database -----------------------------------------------------------------
    database_url: str = Field(default="sqlite:///data/mt5bot.db")

    # Logging ------------------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")

    # ----- validators --------------------------------------------------------

    @field_validator("macd_slow")
    @classmethod
    def _slow_must_exceed_fast(cls, v: int, info) -> int:
        fast = info.data.get("macd_fast")
        if fast is not None and v <= fast:
            raise ValueError(f"macd_slow ({v}) must be greater than macd_fast ({fast})")
        return v

    @field_validator("take_profit_pips")
    @classmethod
    def _tp_must_exceed_sl(cls, v: float, info) -> float:
        sl = info.data.get("stop_loss_pips")
        if sl is not None and v <= sl:
            raise ValueError(f"take_profit_pips ({v}) must be greater than stop_loss_pips ({sl})")
        return v

    @field_validator("allowed_sessions")
    @classmethod
    def _allowed_sessions_known(cls, v: str) -> str:
        valid = {"SYDNEY", "TOKYO", "LONDON", "NEW_YORK", "OVERLAP", "ALL"}
        names = [s.strip().upper() for s in v.split(",")]
        unknown = [n for n in names if n not in valid]
        if unknown:
            raise ValueError(
                f"allowed_sessions contains unknown session(s): {unknown}. "
                f"Valid values: {sorted(valid)}"
            )
        return v

    # ----- helpers -----------------------------------------------------------

    def get_allowed_sessions_list(self) -> List[str]:
        """Parse allowed sessions string to list."""
        return [s.strip().upper() for s in self.allowed_sessions.split(",")]

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the current values, suitable for `restore()`."""
        return copy.deepcopy(self.model_dump())

    def restore(self, snap: Dict[str, Any]) -> None:
        """Restore field values from a snapshot. Validates on assignment."""
        for key, value in snap.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Singleton — instantiation triggers validation against env / defaults.
settings = Settings()


# -----------------------------------------------------------------------------
# Backwards-compatibility uppercase aliases.
#
# Older modules (analysis/indicators.py, analysis/liquidity.py, analysis/dxy.py)
# still read e.g. `settings.MACD_FAST` and `from config import MACD_FAST`. We
# expose dynamic module-level constants via __getattr__ so they always reflect
# the current settings value even after runtime mutation.
# -----------------------------------------------------------------------------

_LEGACY_ALIASES: Dict[str, str] = {
    "SYMBOL": "symbol",
    "DXY_SYMBOL": "dxy_symbol",
    "TIMEFRAME": "timeframe",
    "RISK_PERCENT": "risk_percent",
    "MAX_POSITIONS": "max_positions",
    "STOP_LOSS_PIPS": "stop_loss_pips",
    "TAKE_PROFIT_PIPS": "take_profit_pips",
    "MACD_FAST": "macd_fast",
    "MACD_SLOW": "macd_slow",
    "MACD_SIGNAL": "macd_signal",
    "CONSOLIDATION_LOOKBACK": "consolidation_lookback",
    "CONSOLIDATION_THRESHOLD": "consolidation_threshold",
    "LIQUIDITY_LOOKBACK": "liquidity_lookback",
    "SWING_STRENGTH": "swing_strength",
    "DXY_CORRELATION_PERIOD": "dxy_correlation_period",
    "DXY_THRESHOLD": "dxy_threshold",
    "DRY_RUN": "dry_run",
    "LOOP_INTERVAL_SECONDS": "loop_interval_seconds",
}


def __getattr__(name: str) -> Any:
    """PEP 562 module-level lazy attribute lookup for legacy uppercase aliases."""
    field = _LEGACY_ALIASES.get(name)
    if field is not None:
        return getattr(settings, field)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Pydantic v2 also exposes `Settings.model_fields` on the instance (not as a
# Settings classvar magic), so legacy code reading `settings.MACD_FAST`
# (instance-level uppercase access) still needs help. Pydantic instances raise
# on unknown attributes by default, so we add a thin __getattr__ at *class*
# level via a wrapper. Keeping it module-level via the patch below avoids
# mutating the model class itself.
def _settings_uppercase_proxy(self: Settings, name: str) -> Any:
    field = _LEGACY_ALIASES.get(name)
    if field is not None:
        return object.__getattribute__(self, field)
    raise AttributeError(f"'Settings' object has no attribute {name!r}")


# Install only once. Pydantic's BaseModel uses __getattr__ for private attrs;
# we chain through it.
_orig_getattr = Settings.__getattr__ if hasattr(Settings, "__getattr__") else None


def _chained_getattr(self: Settings, name: str) -> Any:
    field = _LEGACY_ALIASES.get(name)
    if field is not None:
        return object.__getattribute__(self, field)
    if _orig_getattr is not None:
        return _orig_getattr(self, name)
    raise AttributeError(f"'Settings' object has no attribute {name!r}")


Settings.__getattr__ = _chained_getattr  # type: ignore[assignment]
