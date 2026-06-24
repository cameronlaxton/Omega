"""Phase 3 (Issue #22) — co-occurrence / correlation damping behind a flag.

Signals that share a ``damping_family`` and co-occur in the same plane bucket are
collapsed into one sign-preserving family factor (sequence step 4) instead of
stacking independently, then family-capped (step 5) and confidence-weighted by
the primary's confidence (step 6). With the flag off, behaviour is Phase 2.
"""

from __future__ import annotations

import warnings

import pytest

from omega.core.calibration.adjustment_policy import AdjustmentPolicy
from omega.core.contracts.evidence import EvidenceSignal
from omega.core.simulation.evidence_handlers import (
    compute_game_adjustment,
    compute_player_adjustment,
)


def _policy(
    *,
    damping: bool = False,
    confidence: bool = False,
    family_cap: float | None = None,
    plane_cap: float | None = None,
    weight: float = 0.5,
) -> AdjustmentPolicy:
    return AdjustmentPolicy(
        policy_id="p_damp",
        version=1,
        enable_correlation_damping=damping,
        enable_confidence_weighting=confidence,
        correlation_damping_weight=weight,
        family_cap=family_cap,
        plane_cap=plane_cap,
        coefficients={
            "recent_form": {"weight": 0.35, "cap": 0.18},
            "series_avg": {"weight": 0.30, "cap": 0.18},
            "def_matchup_weak": {"mean_mult": 1.06, "cap": 0.15},
        },
    )


def _sig(signal_type: str, value, *, confidence: float = 1.0, direction=None) -> EvidenceSignal:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(
            signal_type=signal_type,
            category="player_form" if signal_type in ("recent_form", "series_avg") else "matchup",
            plane="player" if signal_type in ("recent_form", "series_avg") else "game",
            value=value,
            source="agent_reasoning",
            confidence=confidence,
            window="last_5",
            stat_key="pts",
            direction=direction,
        )


def _player_adj(policy, evidence, mode="live"):
    return compute_player_adjustment(
        player_context={"pts_mean": 25.0},
        evidence=evidence,
        league="NBA",
        prop_type="pts",
        policy=policy,
        evidence_mode=mode,
    )


# recent_form [30,30,30] -> raw 1.07 ; series_avg 28 -> raw 1.036
_RECENT = lambda c=1.0: _sig("recent_form", [30.0, 30.0, 30.0], confidence=c)  # noqa: E731
_SERIES = lambda c=1.0: _sig("series_avg", 28.0, confidence=c)  # noqa: E731

_NAIVE_PRODUCT = 1.07 * 1.036  # 1.10852 — Phase 2 independent stacking
_DAMPED = 1.0 + 0.07 + 0.036 * 0.5  # 1.088 — primary + half the secondary delta


class TestDampingDisabledPreservesBehavior:
    def test_family_stacks_independently_when_off(self):
        adj = _player_adj(_policy(damping=False), [_RECENT(), _SERIES()])
        assert adj.mean_factor == pytest.approx(_NAIVE_PRODUCT)

    def test_both_records_applied_when_off(self):
        adj = _player_adj(_policy(damping=False), [_RECENT(), _SERIES()])
        assert all(r.applied for r in adj.records)
        assert all(r.family_role == "singleton" for r in adj.records)


class TestDampingEnabled:
    def test_family_collapses_to_one_damped_factor(self):
        adj = _player_adj(_policy(damping=True), [_RECENT(), _SERIES()])
        assert adj.mean_factor == pytest.approx(_DAMPED)
        # Damping must pull the combined effect below naive stacking.
        assert adj.mean_factor < _NAIVE_PRODUCT

    def test_primary_carries_factor_secondary_folded(self):
        adj = _player_adj(_policy(damping=True), [_RECENT(), _SERIES()])
        by_type = {r.signal_type: r for r in adj.records}
        primary, secondary = by_type["recent_form"], by_type["series_avg"]
        assert primary.family_role == "primary"
        assert primary.family_size == 2
        assert primary.applied is True
        assert primary.factor == pytest.approx(_DAMPED)
        assert secondary.family_role == "secondary"
        assert secondary.applied is False
        assert secondary.factor == pytest.approx(1.0)
        # The secondary still preserves its own pre-family value for scoring.
        assert secondary.per_signal_capped_factor == pytest.approx(1.036)

    def test_sign_preserved_for_suppressing_family(self):
        evidence = [
            _sig("recent_form", [20.0, 20.0, 20.0]),  # raw 0.93
            _sig("series_avg", 22.0),  # raw 0.964
        ]
        adj = _player_adj(_policy(damping=True), evidence)
        assert adj.mean_factor < 1.0  # never flips to a boost

    def test_lone_family_member_is_not_damped(self):
        adj = _player_adj(_policy(damping=True), [_RECENT()])
        assert adj.mean_factor == pytest.approx(1.07)
        assert adj.records[0].family_role == "singleton"
        assert adj.records[0].family_size == 1

    def test_probation_signal_cannot_be_reenabled_by_family_damping(self):
        policy = _policy(damping=True).model_copy(
            update={"signal_lifecycle": {"series_avg": "probation"}}
        )
        adj = _player_adj(policy, [_RECENT(), _SERIES()])
        by_type = {r.signal_type: r for r in adj.records}
        assert adj.mean_factor == pytest.approx(1.07)
        assert by_type["recent_form"].applied is True
        assert by_type["series_avg"].applied is False
        assert by_type["series_avg"].family_role == "singleton"


class TestCapsBind:
    def test_family_cap_binds(self):
        adj = _player_adj(_policy(damping=True, family_cap=0.05), [_RECENT(), _SERIES()])
        # family_damped 1.088 clamped to 1.05
        assert adj.mean_factor == pytest.approx(1.05)
        primary = next(r for r in adj.records if r.signal_type == "recent_form")
        assert primary.family_capped_factor == pytest.approx(1.05)

    def test_plane_cap_binds(self):
        # Plane cap clamps the aggregated product even with damping off.
        adj = _player_adj(_policy(damping=False, plane_cap=0.03), [_RECENT(), _SERIES()])
        assert adj.mean_factor == pytest.approx(1.03)

    def test_no_caps_when_none(self):
        adj = _player_adj(_policy(damping=True), [_RECENT(), _SERIES()])
        assert adj.mean_factor == pytest.approx(_DAMPED)


class TestDampingWithConfidence:
    def test_primary_confidence_scales_family_factor(self):
        adj = _player_adj(
            _policy(damping=True, confidence=True),
            [_RECENT(c=0.8), _SERIES(c=0.5)],
        )
        # family_capped 1.088 -> primary confidence 0.8 -> 1 + 0.8*0.088 = 1.0704
        assert adj.mean_factor == pytest.approx(1.0704)


class TestGamePlaneDamping:
    def test_same_direction_family_is_damped(self):
        evidence = [
            _sig("def_matchup_weak", True, direction="home"),
            _sig("def_matchup_weak", True, direction="home"),
        ]
        adj = compute_game_adjustment(
            evidence=evidence,
            league="NBA",
            policy=_policy(damping=True),
            evidence_mode="live",
        )
        # each raw 1.06 -> damped 1 + 0.06 + 0.06*0.5 = 1.09
        assert adj.home_factor == pytest.approx(1.09)
        assert adj.away_factor == pytest.approx(1.0)

    def test_opposite_directions_are_not_grouped(self):
        evidence = [
            _sig("def_matchup_weak", True, direction="home"),
            _sig("def_matchup_weak", True, direction="away"),
        ]
        adj = compute_game_adjustment(
            evidence=evidence,
            league="NBA",
            policy=_policy(damping=True),
            evidence_mode="live",
        )
        # different buckets -> each is a singleton on its own side, no damping
        assert adj.home_factor == pytest.approx(1.06)
        assert adj.away_factor == pytest.approx(1.06)
