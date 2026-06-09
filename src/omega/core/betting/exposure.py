"""Enforced exposure limits for portfolio-aware bet sizing.

Turns the parlay correlation *warnings* into enforced *constraints*: caps on how
much open dollar risk may sit on any one game / team / player / league / sport /
correlated group, plus a total open-risk cap. A future portfolio selector seeds
an :class:`ExposureLedger` from current open positions, then asks
:meth:`ExposurePolicy.admit` for each candidate.

Keys are the namespaced strings produced by
``omega.trace.portfolio_state.entity_keys_for`` (``sport:`` / ``league:`` /
``game:`` / ``selection:``) plus correlated-group keys from
``parlay.correlation_group_key`` (``corr:``) and any ``team:`` / ``player:`` keys
a caller adds. ``cap_for_key`` maps a key's prefix to its limit; an unmapped
prefix (e.g. ``selection:``) is uncapped per-key and bounded only by the total.

Bounded-autonomy note (AGENTS.md): this only ever *reduces* risk. An exposure
verdict never authorizes a stake larger than the one the staking policy already
computed — it ACCEPTs, DOWNSIZEs, or SKIPs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Float slack so a stake landing exactly on a cap is allowed (not a breach).
_EPS = 1e-9


@dataclass(frozen=True)
class ExposureLimits:
    """Open-risk caps as a fraction of bankroll. Versioned."""

    schema_version: int = 1
    max_total_open_pct: float = 0.25
    max_per_game_pct: float = 0.05
    max_per_team_pct: float = 0.08
    max_per_player_pct: float = 0.04
    max_per_league_pct: float = 0.15
    max_per_sport_pct: float = 0.20
    max_per_correlated_group_pct: float = 0.06

    def cap_for_key(self, key: str) -> float | None:
        """The fractional cap governing ``key`` (by prefix), or None if uncapped.

        ``selection:`` and any unknown prefix return None — those are bounded
        only by the per-bet stake and the total-open cap.
        """
        prefix = key.split(":", 1)[0]
        return {
            "sport": self.max_per_sport_pct,
            "league": self.max_per_league_pct,
            "game": self.max_per_game_pct,
            "team": self.max_per_team_pct,
            "player": self.max_per_player_pct,
            "corr": self.max_per_correlated_group_pct,
        }.get(prefix)


@dataclass
class ExposureLedger:
    """Mutable accumulator of open dollar exposure per entity key + total.

    Seed from a portfolio's current open exposure, then the selector calls
    :meth:`headroom` / :meth:`would_breach` to test a candidate and :meth:`add`
    once it commits. ``_total`` is the sum of distinct open *stakes* (not the sum
    over keys, which double-counts because one bet touches several keys).
    """

    limits: ExposureLimits = field(default_factory=ExposureLimits)
    _by_key: dict[str, float] = field(default_factory=dict)
    _total: float = 0.0

    @classmethod
    def seeded(
        cls,
        limits: ExposureLimits | None = None,
        *,
        exposure_by_entity: dict[str, float] | None = None,
        total_open: float = 0.0,
    ) -> ExposureLedger:
        """Build a ledger pre-loaded with existing open exposure.

        ``exposure_by_entity`` and ``total_open`` come straight from a
        ``PortfolioState`` (``exposure_by_entity`` and the summed open stakes),
        so this module needs no dependency on the persistence layer.
        """
        return cls(
            limits=limits or ExposureLimits(),
            _by_key=dict(exposure_by_entity or {}),
            _total=float(total_open),
        )

    def exposure_for(self, key: str) -> float:
        return self._by_key.get(key, 0.0)

    @property
    def total_open(self) -> float:
        return self._total

    def would_breach(
        self, entity_keys: tuple[str, ...], stake: float, bankroll: float
    ) -> tuple[bool, str | None]:
        """Whether adding ``stake`` on ``entity_keys`` breaches any cap.

        Returns ``(True, binding_key)`` naming the first cap that would be
        exceeded (``"total_open"`` for the portfolio cap), else ``(False, None)``.
        """
        if self._total + stake > self.limits.max_total_open_pct * bankroll + _EPS:
            return True, "total_open"
        for key in entity_keys:
            cap = self.limits.cap_for_key(key)
            if cap is None:
                continue
            if self._by_key.get(key, 0.0) + stake > cap * bankroll + _EPS:
                return True, key
        return False, None

    def headroom(self, entity_keys: tuple[str, ...], bankroll: float) -> float:
        """Max additional stake on ``entity_keys`` before any cap breaches (>= 0)."""
        hr = self.limits.max_total_open_pct * bankroll - self._total
        for key in entity_keys:
            cap = self.limits.cap_for_key(key)
            if cap is None:
                continue
            hr = min(hr, cap * bankroll - self._by_key.get(key, 0.0))
        return max(0.0, hr)

    def add(self, entity_keys: tuple[str, ...], stake: float) -> None:
        """Commit ``stake`` against ``entity_keys`` and the total."""
        for key in entity_keys:
            self._by_key[key] = self._by_key.get(key, 0.0) + stake
        self._total += stake


class ExposureAction(str, Enum):
    ACCEPT = "accept"
    DOWNSIZE = "downsize"
    SKIP = "skip"


@dataclass(frozen=True)
class ExposureVerdict:
    """The outcome of admitting a candidate. ``stake`` is never > the desired."""

    action: ExposureAction
    stake: float
    reason: str | None = None


class ExposurePolicy:
    """Admits candidate bets against an :class:`ExposureLedger`.

    Read-only on the ledger: the caller commits an accepted/downsized stake via
    ``ledger.add`` so it controls when exposure is recorded.
    """

    def __init__(self, limits: ExposureLimits | None = None, *, min_stake: float = 0.0) -> None:
        self.limits = limits or ExposureLimits()
        self.min_stake = min_stake

    def admit(
        self,
        *,
        entity_keys: tuple[str, ...],
        desired_stake: float,
        ledger: ExposureLedger,
        bankroll: float,
    ) -> ExposureVerdict:
        headroom = ledger.headroom(entity_keys, bankroll)
        if headroom <= self.min_stake:
            _, reason = ledger.would_breach(
                entity_keys, max(desired_stake, self.min_stake) + _EPS, bankroll
            )
            return ExposureVerdict(ExposureAction.SKIP, 0.0, reason or "exposure_cap")
        if headroom + _EPS < desired_stake:
            return ExposureVerdict(ExposureAction.DOWNSIZE, round(headroom, 2), "exposure_headroom")
        return ExposureVerdict(ExposureAction.ACCEPT, desired_stake, None)
