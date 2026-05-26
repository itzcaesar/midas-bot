"""
Capital Tiering (REQ-P3-10).

Defines a progression of capital tiers with risk caps that scale sub-linearly.
Each tier has a gate: the bot must demonstrate profitable operation at the
current tier for a minimum period before advancing.

Tiers:
  1. Paper ($0 real)     — unlimited time, must show 30d profitable shadow
  2. Micro ($1,000)      — max 0.5% risk, max 1 position, 30d gate
  3. Small ($10,000)     — max 1.0% risk, max 2 positions, 60d gate
  4. Medium ($50,000)    — max 0.75% risk, max 3 positions, 90d gate
  5. Full ($100,000+)    — max 0.5% risk, max 5 positions, ongoing

Risk caps scale *down* as capital grows (sub-linear) because the absolute
dollar risk per trade grows while the percentage shrinks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.logger import get_logger

logger = get_logger("midas.capital_tier")


@dataclass
class CapitalTier:
    """Definition of a single capital tier."""
    name: str
    level: int
    min_capital: float
    max_risk_pct: float
    max_positions: int
    gate_days: int  # Minimum profitable days at this tier before advancing
    description: str = ""


# Pre-defined tier progression
TIERS: List[CapitalTier] = [
    CapitalTier(
        name="Paper",
        level=0,
        min_capital=0.0,
        max_risk_pct=2.0,
        max_positions=5,
        gate_days=30,
        description="Shadow/paper trading. No real capital at risk.",
    ),
    CapitalTier(
        name="Micro",
        level=1,
        min_capital=1_000.0,
        max_risk_pct=0.5,
        max_positions=1,
        gate_days=30,
        description="First real capital. Conservative sizing.",
    ),
    CapitalTier(
        name="Small",
        level=2,
        min_capital=10_000.0,
        max_risk_pct=1.0,
        max_positions=2,
        gate_days=60,
        description="Proven at micro. Slightly larger positions.",
    ),
    CapitalTier(
        name="Medium",
        level=3,
        min_capital=50_000.0,
        max_risk_pct=0.75,
        max_positions=3,
        gate_days=90,
        description="Institutional-lite. Sub-linear risk scaling.",
    ),
    CapitalTier(
        name="Full",
        level=4,
        min_capital=100_000.0,
        max_risk_pct=0.5,
        max_positions=5,
        gate_days=0,  # Ongoing — no advancement beyond this
        description="Full deployment. Tightest risk per trade.",
    ),
]


class TierManager:
    """Manages capital tier progression and enforces tier-specific limits."""

    def __init__(self, current_level: int = 0) -> None:
        self._level = current_level
        self._profitable_days = 0

    @property
    def current_tier(self) -> CapitalTier:
        return TIERS[min(self._level, len(TIERS) - 1)]

    @property
    def next_tier(self) -> Optional[CapitalTier]:
        if self._level >= len(TIERS) - 1:
            return None
        return TIERS[self._level + 1]

    @property
    def days_until_advance(self) -> int:
        gate = self.current_tier.gate_days
        return max(0, gate - self._profitable_days)

    def record_day(self, profitable: bool) -> None:
        """Record a trading day result for gate tracking."""
        if profitable:
            self._profitable_days += 1
        else:
            # Reset streak (strict gate)
            self._profitable_days = max(0, self._profitable_days - 1)

    def can_advance(self, current_capital: float) -> tuple[bool, str]:
        """Check if advancement to the next tier is allowed."""
        next_t = self.next_tier
        if next_t is None:
            return False, "Already at maximum tier"

        if self._profitable_days < self.current_tier.gate_days:
            return False, (
                f"Need {self.current_tier.gate_days - self._profitable_days} more "
                f"profitable days at {self.current_tier.name} tier"
            )

        if current_capital < next_t.min_capital:
            return False, (
                f"Capital ${current_capital:,.0f} < ${next_t.min_capital:,.0f} "
                f"required for {next_t.name} tier"
            )

        return True, f"Ready to advance to {next_t.name} tier"

    def advance(self) -> bool:
        """Advance to the next tier. Returns True if successful."""
        if self._level >= len(TIERS) - 1:
            return False
        self._level += 1
        self._profitable_days = 0
        logger.info(f"Advanced to tier: {self.current_tier.name}")
        return True

    def get_risk_limits(self) -> dict:
        """Get the current tier's risk limits for the governor."""
        tier = self.current_tier
        return {
            "tier_name": tier.name,
            "tier_level": tier.level,
            "max_risk_pct": tier.max_risk_pct,
            "max_positions": tier.max_positions,
            "gate_days_remaining": self.days_until_advance,
        }

    def status(self) -> dict:
        """Full status for dashboard/monitoring."""
        return {
            "current_tier": self.current_tier.name,
            "level": self._level,
            "profitable_days": self._profitable_days,
            "gate_days": self.current_tier.gate_days,
            "days_until_advance": self.days_until_advance,
            "max_risk_pct": self.current_tier.max_risk_pct,
            "max_positions": self.current_tier.max_positions,
            "next_tier": self.next_tier.name if self.next_tier else "N/A",
        }
