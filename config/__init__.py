"""
MIDAS Configuration Module.

Re-exports the validated ``settings`` singleton plus dynamic uppercase aliases
for backwards compatibility (``from config import MACD_FAST`` etc.). The
aliases reflect runtime mutations of ``settings`` because they are resolved on
each attribute access via PEP 562 ``__getattr__``.
"""
from typing import Any

from .settings import Settings, settings, _LEGACY_ALIASES  # noqa: F401

__all__ = [
    "Settings",
    "settings",
    *list(_LEGACY_ALIASES.keys()),
]


def __getattr__(name: str) -> Any:
    """Resolve uppercase legacy constants dynamically against the live settings.

    Without this, ``from config import MACD_FAST`` would bind the value once at
    import time and never reflect later mutations done by the optimizer.
    """
    field = _LEGACY_ALIASES.get(name)
    if field is not None:
        return getattr(settings, field)
    raise AttributeError(f"module 'config' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
