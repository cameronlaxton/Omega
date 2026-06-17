"""Deterministic, portfolio-aware multi-bet selection.

Generalizes the legacy single greedy ``_pick_best_bet`` (one max-EV BetSlip per
trace) into a *set* of sized bets for a slate, subject to per-entity exposure
caps and a slate risk budget. Pure and reproducible: no RNG, no dict-iteration
dependence — the only ordering is an explicit total sort, so shuffling the input
yields identical output.

Bounded-autonomy note (AGENTS.md): every number here is engine-computed.
Candidates carry engine-authored edge fields; sizing comes from the deterministic
``StakingPolicy``; exposure only caps/skips via ``ExposurePolicy``. The LLM never
authors a unit, stake, or EV.

This module is decoupled from persistence: candidates arrive with their entity
keys already built (by the caller, via ``portfolio_state.entity_keys_for``), so
``core/betting`` gains no dependency on ``omega.trace``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omega.core.betting.exposure import ExposureAction, ExposureLedger, ExposurePolicy
from omega.core.betting.staking_policy import (
    FractionalKellyByTier,
    StakingContext,
    StakingPolicy,
)

_EPS = 1e-9


@dataclass(frozen=True)
class BetCandidate:
    """One actionable edge as a selection/sizing candidate.

    ``calibrated_prob`` is what staking sizes against (matching the legacy
    ``recommend_stake(true_prob=best.calibrated_prob, ...)`` call). ``entity_keys``
    are the exposure keys for this bet (sport/league/game/selection + optional
    team/player/corr), built by the caller.
    """

    selection: str
    selection_descriptor: str
    market: str
    calibrated_prob: float
    odds: float
    edge_pct: float
    ev_pct: float
    confidence_tier: str
    entity_keys: tuple[str, ...] = ()
    league: str | None = None


@dataclass(frozen=True)
class SizedBet:
    candidate: BetCandidate
    units: float
    kelly_fraction: float
    stake_amount: float
    capped_by: tuple[str, ...] = ()
    policy_id: str = ""
    policy_version: int = 1


@dataclass(frozen=True)
class PortfolioSelection:
    bets: tuple[SizedBet, ...] = ()
    skipped: tuple[tuple[BetCandidate, str], ...] = ()


def _sort_key(c: BetCandidate) -> tuple:
    # EV desc, then edge desc, then prob desc, then a lexicographic tie-break so
    # the order is total and independent of input order (reproducible).
    return (-c.ev_pct, -c.edge_pct, -c.calibrated_prob, c.selection_descriptor, c.selection)


def select_portfolio(
    candidates: list[BetCandidate],
    *,
    bankroll: float,
    staking_policy: StakingPolicy | None = None,
    exposure_policy: ExposurePolicy | None = None,
    exposure_by_entity: dict[str, float] | None = None,
    total_open: float = 0.0,
    budget_pct: float = 0.10,
    max_bets: int | None = None,
    min_stake: float = 0.0,
) -> PortfolioSelection:
    """Select and size a portfolio of bets from ``candidates``.

    Greedy by EV under constraints: filter (tier in A/B and ev>0) -> sort -> seed
    an exposure ledger from existing open exposure -> admit each candidate against
    the per-entity caps and the remaining slate budget. Every dropped or downsized
    candidate is recorded with a reason.
    """
    staking_policy = staking_policy or FractionalKellyByTier()
    exposure_policy = exposure_policy or ExposurePolicy()
    ledger = ExposureLedger.seeded(
        exposure_policy.limits, exposure_by_entity=exposure_by_entity, total_open=total_open
    )

    actionable: list[BetCandidate] = []
    skipped: list[tuple[BetCandidate, str]] = []
    for c in candidates:
        if c.confidence_tier in ("A", "B") and c.ev_pct > 0:
            actionable.append(c)
        else:
            skipped.append((c, "tier_or_ev_filtered"))

    actionable.sort(key=_sort_key)

    remaining_budget = max(0.0, bankroll * budget_pct)
    bets: list[SizedBet] = []
    for c in actionable:
        if max_bets is not None and len(bets) >= max_bets:
            skipped.append((c, "max_bets_reached"))
            continue

        decision = staking_policy.size(
            StakingContext(
                true_prob=c.calibrated_prob,
                odds=c.odds,
                bankroll=bankroll,
                confidence_tier=c.confidence_tier,
                league=c.league,
                market=c.market,
                open_exposure=ledger.total_open,
                entity_keys=c.entity_keys,
            )
        )
        if decision.stake_amount <= 0:
            skipped.append((c, "zero_stake"))
            continue
        if remaining_budget <= min_stake:
            skipped.append((c, "budget_exhausted"))
            continue

        desired = min(decision.stake_amount, remaining_budget)
        verdict = exposure_policy.admit(
            entity_keys=c.entity_keys, desired_stake=desired, ledger=ledger, bankroll=bankroll
        )
        if verdict.action is ExposureAction.SKIP:
            skipped.append((c, verdict.reason or "exposure_cap"))
            continue
        stake = verdict.stake
        if stake <= min_stake:
            skipped.append((c, "below_min_stake"))
            continue

        # Reproduce the policy's units exactly when nothing reduced the stake;
        # recompute from the actual stake when budget/exposure downsized it.
        reduced = stake + _EPS < decision.stake_amount
        if reduced:
            units = round(stake / bankroll * 100.0, 2)
            extra = []
            if desired + _EPS < decision.stake_amount:
                extra.append("budget")
            if verdict.action is ExposureAction.DOWNSIZE:
                extra.append("exposure_headroom")
            capped_by = decision.capped_by + tuple(extra)
        else:
            units = decision.units
            capped_by = decision.capped_by

        ledger.add(c.entity_keys, stake)
        remaining_budget -= stake
        bets.append(
            SizedBet(
                candidate=c,
                units=units,
                kelly_fraction=decision.kelly_fraction,
                stake_amount=round(stake, 2),
                capped_by=capped_by,
                policy_id=decision.policy_id,
                policy_version=decision.policy_version,
            )
        )

    return PortfolioSelection(bets=tuple(bets), skipped=tuple(skipped))
