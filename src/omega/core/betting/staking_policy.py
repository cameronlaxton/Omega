"""Pluggable, versioned staking policies — the single shared staking path.

Historically each call site invoked :func:`omega.core.betting.kelly.recommend_stake`
directly, so there was no one place that owned "how big is this bet". This module
introduces a small policy object that owns that decision; ``recommend_stake``
now delegates to the default policy (:class:`FractionalKellyByTier`), so every
caller sizes through one path. Later phases swap the policy per (league, market)
via a registry and add portfolio-aware sizing — without touching call sites.

Bounded-autonomy note (AGENTS.md): staking is owned by the deterministic engine.
All sizing here is deterministic Python; the LLM never authors units / Kelly /
stake. A policy consumes engine-computed inputs and returns engine-computed sizes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from omega.core.betting.kelly import kelly_fraction

# Canonical defaults (relocated from kelly._TIER_MULTIPLIERS). Fraction of full
# Kelly to use per confidence tier: full Kelly is growth-optimal but volatile;
# fractional Kelly trades expected growth for lower variance.
DEFAULT_TIER_MULTIPLIERS: dict[str, float] = {
    "A": 0.50,  # high confidence: half Kelly
    "B": 0.25,  # medium confidence: quarter Kelly
    "C": 0.10,  # low confidence: tenth Kelly
}

# Per-bet unit cap. 1 unit == 1% of bankroll.
DEFAULT_UNIT_CAP: float = 5.0


@dataclass(frozen=True)
class StakingContext:
    """Inputs to a staking decision.

    The first four fields reproduce the legacy ``recommend_stake`` signature. The
    remaining fields are forward-compatible hooks for portfolio-aware sizing
    (later phases) and are ignored by the policies in this module today.
    """

    true_prob: float
    odds: float
    bankroll: float
    confidence_tier: str = "B"
    league: str | None = None
    market: str | None = None
    open_exposure: float = 0.0
    entity_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class StakingDecision:
    """A sizing decision plus audit metadata.

    ``units`` and ``kelly_fraction`` reproduce the legacy ``recommend_stake``
    return; ``stake_amount`` (dollars) and ``capped_by`` (which caps bound the
    size) are additive audit fields used by later phases.
    """

    units: float
    kelly_fraction: float
    stake_amount: float
    capped_by: tuple[str, ...] = ()
    policy_id: str = ""
    policy_version: int = 1

    def to_recommend_stake_dict(self) -> dict[str, float]:
        """Legacy dict shape consumed by existing call sites."""
        return {"units": self.units, "kelly_fraction": self.kelly_fraction}


def _units_and_stake(
    scaled_kelly: float, bankroll: float, unit_cap: float
) -> tuple[float, float, tuple[str, ...]]:
    """Convert a scaled Kelly fraction to (units, stake_amount, capped_by).

    1 unit == 1% of bankroll. Mirrors the legacy ``min(scaled * 100, cap)`` then
    ``round(..., 2)`` exactly, and records whether the unit cap bound the size.
    """
    units_raw = scaled_kelly * 100.0
    capped_by: tuple[str, ...] = ()
    if units_raw > unit_cap:
        units_raw = unit_cap
        capped_by = ("unit_cap",)
    units = round(units_raw, 2)
    stake_amount = round(units * bankroll / 100.0, 2)
    return units, stake_amount, capped_by


class StakingPolicy(ABC):
    """A deterministic bet-sizing policy. ``size`` is pure and reproducible."""

    policy_id: str = "abstract"
    version: int = 1
    schema_version: int = 1

    @abstractmethod
    def size(self, ctx: StakingContext) -> StakingDecision:
        """Return the recommended size for one bet."""
        raise NotImplementedError


class FractionalKellyByTier(StakingPolicy):
    """Tier-scaled fractional Kelly with a per-bet unit cap.

    The default configuration is **bit-identical** to the legacy
    ``recommend_stake``: A=0.50, B=0.25, C=0.10 of full Kelly, capped at 5 units;
    an unknown tier falls back to the B multiplier. ``recommend_stake`` delegates
    to an instance of this policy.
    """

    policy_id = "fractional_kelly_by_tier"
    version = 1

    def __init__(
        self,
        tier_multipliers: dict[str, float] | None = None,
        unit_cap: float = DEFAULT_UNIT_CAP,
    ) -> None:
        self._tier_multipliers = dict(tier_multipliers or DEFAULT_TIER_MULTIPLIERS)
        self._unit_cap = unit_cap

    def _scaled_kelly(self, ctx: StakingContext) -> float:
        raw = kelly_fraction(ctx.true_prob, ctx.odds)
        mult = self._tier_multipliers.get(ctx.confidence_tier.upper(), self._tier_multipliers["B"])
        return raw * mult

    def size(self, ctx: StakingContext) -> StakingDecision:
        scaled = self._scaled_kelly(ctx)
        units, stake_amount, capped_by = _units_and_stake(scaled, ctx.bankroll, self._unit_cap)
        return StakingDecision(
            units=units,
            kelly_fraction=round(scaled, 4),
            stake_amount=stake_amount,
            capped_by=capped_by,
            policy_id=self.policy_id,
            policy_version=self.version,
        )


class FlatKelly(StakingPolicy):
    """A single Kelly multiplier regardless of confidence tier."""

    policy_id = "flat_kelly"
    version = 1

    def __init__(self, multiplier: float = 0.25, unit_cap: float = DEFAULT_UNIT_CAP) -> None:
        self._multiplier = multiplier
        self._unit_cap = unit_cap

    def size(self, ctx: StakingContext) -> StakingDecision:
        scaled = kelly_fraction(ctx.true_prob, ctx.odds) * self._multiplier
        units, stake_amount, capped_by = _units_and_stake(scaled, ctx.bankroll, self._unit_cap)
        return StakingDecision(
            units=units,
            kelly_fraction=round(scaled, 4),
            stake_amount=stake_amount,
            capped_by=capped_by,
            policy_id=self.policy_id,
            policy_version=self.version,
        )


class CappedFractionalKelly(FractionalKellyByTier):
    """Fractional-Kelly-by-tier with a configurable unit cap and optional Kelly cap.

    ``max_kelly_fraction`` clamps the scaled Kelly fraction before the unit cap, so
    a high-edge bet can be bounded independently of the 5-unit ceiling.
    """

    policy_id = "capped_fractional_kelly"
    version = 1

    def __init__(
        self,
        tier_multipliers: dict[str, float] | None = None,
        unit_cap: float = DEFAULT_UNIT_CAP,
        max_kelly_fraction: float | None = None,
    ) -> None:
        super().__init__(tier_multipliers=tier_multipliers, unit_cap=unit_cap)
        self._max_kelly_fraction = max_kelly_fraction

    def size(self, ctx: StakingContext) -> StakingDecision:
        scaled = self._scaled_kelly(ctx)
        extra_caps: tuple[str, ...] = ()
        if self._max_kelly_fraction is not None and scaled > self._max_kelly_fraction:
            scaled = self._max_kelly_fraction
            extra_caps = ("max_kelly_fraction",)
        units, stake_amount, capped_by = _units_and_stake(scaled, ctx.bankroll, self._unit_cap)
        return StakingDecision(
            units=units,
            kelly_fraction=round(scaled, 4),
            stake_amount=stake_amount,
            capped_by=capped_by + extra_caps,
            policy_id=self.policy_id,
            policy_version=self.version,
        )
