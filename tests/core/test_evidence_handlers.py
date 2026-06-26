"""
Tests for Phase B — deterministic evidence handlers and service wiring.

Covers handler determinism + cap enforcement, shadow-vs-live mode, the
zero-behavior-change guarantee in shadow mode, and trace-identity behavior.
"""

from __future__ import annotations

import warnings

from omega.core.calibration.adjustment_policy import AdjustmentPolicyRegistry
from omega.core.contracts.evidence import EvidenceSignal
from omega.core.contracts.schemas import GameAnalysisRequest, PlayerPropRequest
from omega.core.contracts.service import analyze
from omega.core.simulation.evidence_handlers import (
    HANDLER_REGISTRY,
    _cap_factor,
    compute_game_adjustment,
    compute_player_adjustment,
    resolve_evidence_mode,
)

_POLICY = AdjustmentPolicyRegistry().get_production_policy()


def _player_signal(**overrides) -> EvidenceSignal:
    base = dict(
        signal_type="usage_spike",
        category="player_form",
        plane="player",
        value=0.2,
        source="agent_reasoning",
        confidence=0.7,
        window="matchup",
    )
    base.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(**base)


def _prop_request(evidence=None) -> PlayerPropRequest:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PlayerPropRequest(
            player_name="Test Player",
            league="NBA",
            prop_type="pts",
            line=20.0,
            odds_over=-110,
            odds_under=-110,
            player_context={"pts_mean": 25.0, "pts_std": 6.0},
            game_context={"is_playoff": False, "rest_days": 2},
            home_team="Home Team",
            away_team="Away Team",
            game_date="2026-05-22",
            n_iterations=2000,
            seed=12345,
            evidence=evidence or [],
        )


class TestCapAndDeterminism:
    def test_cap_clamps_high(self):
        assert _cap_factor(1.50, 0.20) == 1.20

    def test_cap_clamps_low(self):
        assert _cap_factor(0.50, 0.20) == 0.80

    def test_cap_zero_is_noop(self):
        assert _cap_factor(1.5, 0.0) == 1.0

    def test_handler_is_deterministic(self):
        sig = _player_signal()
        coeffs = _POLICY.coeffs_for("usage_spike")
        handler = HANDLER_REGISTRY["usage_spike"]
        first = handler(sig, coeffs, 25.0)
        second = handler(sig, coeffs, 25.0)
        assert first == second

    def test_every_registry_signal_has_a_handler(self):
        from omega.core.contracts.evidence import SIGNAL_REGISTRY

        for signal_type in SIGNAL_REGISTRY:
            assert signal_type in HANDLER_REGISTRY


class TestModeResolution:
    def test_env_override_live(self, monkeypatch):
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "live")
        assert resolve_evidence_mode(_POLICY) == "live"

    def test_env_override_legacy_shadow_maps_to_score_only(self, monkeypatch):
        # Legacy binary "shadow" normalizes to the graduated "score_only".
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "shadow")
        assert resolve_evidence_mode(_POLICY) == "score_only"

    def test_env_override_bounded_live(self, monkeypatch):
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "bounded_live")
        assert resolve_evidence_mode(_POLICY) == "bounded_live"

    def test_invalid_env_falls_through_to_policy(self, monkeypatch):
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "garbage")
        assert resolve_evidence_mode(_POLICY) == _POLICY.mode

    def test_policy_default_is_bounded_live(self, monkeypatch):
        # The production seed now ships at bounded_live (graduated-apply default):
        # evidence moves predictions under hard caps, scaled by reliability.
        monkeypatch.delenv("OMEGA_EVIDENCE_MODE", raising=False)
        assert resolve_evidence_mode(_POLICY) == "bounded_live"


class TestComputePlayerAdjustment:
    def test_shadow_mean_factor_is_identity(self):
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=[_player_signal()],
            league="NBA",
            prop_type="pts",
            policy=_POLICY,
            evidence_mode="shadow",
        )
        assert adj.mean_factor == 1.0  # shadow never applies
        # ...but the record still carries the computed counterfactual factor.
        assert adj.records[0].factor > 1.0
        assert adj.records[0].applied is False

    def test_live_applies_factor_damped_by_unfitted_prior(self):
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=[_player_signal(value=0.2)],
            league="NBA",
            prop_type="pts",
            policy=_POLICY,
            evidence_mode="live",
        )
        # usage_spike value 0.2 -> raw 1.2; it is unscored, so the 0.25 unfitted
        # prior damps it even in live mode: 1 + 0.25*(1.2-1) = 1.05 (cap 0.22 idle).
        assert abs(adj.mean_factor - 1.05) < 1e-9
        assert adj.records[0].applied is True

    def test_records_align_with_evidence_list(self):
        evidence = [_player_signal(), _player_signal(signal_type="recent_form")]
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=evidence,
            league="NBA",
            prop_type="pts",
            policy=_POLICY,
            evidence_mode="shadow",
        )
        assert len(adj.records) == len(evidence)

    def test_multi_sport_gating_skips_wrong_sport(self):
        # usage_spike does not apply to baseball.
        adj = compute_player_adjustment(
            player_context={"hits_mean": 1.2},
            evidence=[_player_signal(stat_key="hits")],
            league="MLB",
            prop_type="hits",
            policy=_POLICY,
            evidence_mode="live",
        )
        assert adj.records[0].target == "skip"
        assert adj.mean_factor == 1.0

    def test_game_plane_signal_skipped_on_prop(self):
        game_sig = _player_signal(
            signal_type="motivation_edge", category="situational", plane="game"
        )
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=[game_sig],
            league="NBA",
            prop_type="pts",
            policy=_POLICY,
            evidence_mode="live",
        )
        assert adj.records[0].target == "skip"


class TestComputeGameAdjustment:
    def test_directional_signal_targets_one_team(self):
        sig = EvidenceSignal(
            signal_type="motivation_edge",
            category="situational",
            plane="game",
            value=1.0,
            source="agent_reasoning",
            confidence=0.6,
            window="matchup",
            direction="home",
        )
        adj = compute_game_adjustment(
            evidence=[sig], league="NBA", policy=_POLICY, evidence_mode="live"
        )
        assert adj.home_factor != 1.0
        assert adj.away_factor == 1.0


class TestZeroBehaviorChangeShadow:
    """score_only mode must not change predictions vs. supplying no evidence.

    Under the graduated-apply default (bounded_live) evidence DOES move the math;
    these tests pin OMEGA_EVIDENCE_MODE=score_only to assert the non-applying
    rung stays a pure no-op (records the counterfactual, never touches the result).
    """

    def test_prop_result_identical_with_and_without_evidence(self, monkeypatch):
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "score_only")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = analyze(_prop_request(), session_id="s", bankroll=1000.0)
            withev = analyze(
                _prop_request(evidence=[_player_signal(value=0.3)]),
                session_id="s",
                bankroll=1000.0,
            )
        assert plain["result"]["over_prob"] == withev["result"]["over_prob"]
        assert plain["result"]["under_prob"] == withev["result"]["under_prob"]

    def test_envelope_carries_evidence_application(self, monkeypatch):
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "score_only")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = analyze(
                _prop_request(evidence=[_player_signal()]),
                session_id="s",
                bankroll=1000.0,
            )
        assert out["evidence_mode"] == "score_only"
        assert out["evidence_rollout_mode"] == "score_only"
        assert len(out["evidence_application"]) == 1
        app = out["evidence_application"][0]
        assert app["applied"] is False  # score_only never applies to math
        assert app["factor"] > 1.0  # counterfactual still recorded


class TestGraduatedApplyDefault:
    """The shipped default (bounded_live) applies evidence, scaled by reliability."""

    def _adjust(self, policy, *, mode="bounded_live", value=0.5):
        pol = policy.bounded_live_effective() if mode == "bounded_live" else policy
        return compute_player_adjustment(
            player_context={"pts_mean": 25.0, "pts_std": 6.0},
            evidence=[_player_signal(value=value)],
            league="NBA",
            prop_type="pts",
            policy=pol,
            evidence_mode=mode,
        )

    def test_default_mode_applies_unfitted_signal_as_a_sliver(self, monkeypatch):
        # No env override -> the production seed's bounded_live default applies.
        monkeypatch.delenv("OMEGA_EVIDENCE_MODE", raising=False)
        assert resolve_evidence_mode(_POLICY) == "bounded_live"
        adj = self._adjust(_POLICY)
        assert adj.records[0].applied is True
        assert adj.mean_factor != 1.0  # evidence now moves the math by default

    def test_unfitted_prior_damps_vs_full_trust(self):
        # Compare in live mode (loose per-signal cap) so the reliability scaling,
        # not the hard cap, drives the difference. value 0.2 -> raw 1.2.
        sliver = self._adjust(_POLICY, mode="live", value=0.2)
        full = self._adjust(
            _POLICY.model_copy(update={"unfitted_reliability_prior": 1.0}),
            mode="live",
            value=0.2,
        )
        assert abs(sliver.mean_factor - 1.0) < abs(full.mean_factor - 1.0)

    def test_zero_reliability_self_neutralizes_even_when_applying(self):
        coeffs = {st: dict(p) for st, p in _POLICY.coefficients.items()}
        coeffs["usage_spike"]["reliability_weight"] = 0.0
        pol0 = _POLICY.model_copy(update={"coefficients": coeffs})
        adj = self._adjust(pol0)
        assert adj.mean_factor == 1.0  # fit-proved noise is a no-op
        assert adj.records[0].applied is False


class TestLiveModeShiftsPrediction:
    def test_live_usage_spike_raises_over_prob(self, monkeypatch):
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "live")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shadow_like = analyze(_prop_request(), session_id="s", bankroll=1000.0)
            live = analyze(
                _prop_request(evidence=[_player_signal(value=0.3)]),
                session_id="s",
                bankroll=1000.0,
            )
        # Boosting the mean above a line of 20 must raise P(Over).
        assert live["result"]["over_prob"] > shadow_like["result"]["over_prob"]
        assert live["evidence_application"][0]["applied"] is True


class TestTraceIdentity:
    def test_same_request_same_hash_prefix(self, monkeypatch):
        monkeypatch.delenv("OMEGA_EVIDENCE_MODE", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = analyze(_prop_request(evidence=[_player_signal()]), session_id="s", bankroll=1000.0)
            b = analyze(_prop_request(evidence=[_player_signal()]), session_id="s", bankroll=1000.0)
        # trace_id = sandbox-<hash>-<nonce>; the hash prefix must be stable.
        assert a["trace_id"].rsplit("-", 1)[0] == b["trace_id"].rsplit("-", 1)[0]

    def test_different_evidence_changes_hash_prefix(self, monkeypatch):
        monkeypatch.delenv("OMEGA_EVIDENCE_MODE", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = analyze(_prop_request(), session_id="s", bankroll=1000.0)
            b = analyze(_prop_request(evidence=[_player_signal()]), session_id="s", bankroll=1000.0)
        assert a["trace_id"].rsplit("-", 1)[0] != b["trace_id"].rsplit("-", 1)[0]


class TestGamePlaneShadow:
    def test_game_shadow_does_not_change_win_prob(self, monkeypatch):
        # motivation_edge is not a curated full-trust signal, so under the
        # bounded_live default it would nudge the prediction by its 0.25 prior.
        # Pin score_only to assert the non-applying rung is a pure no-op.
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "score_only")
        sig = EvidenceSignal(
            signal_type="motivation_edge",
            category="situational",
            plane="game",
            value=1.0,
            source="agent_reasoning",
            confidence=0.6,
            window="matchup",
            direction="home",
        )
        ctx = {"off_rating": 115.0, "def_rating": 110.0, "pace": 100.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_kwargs = dict(
                home_team="H",
                away_team="A",
                league="NBA",
                home_context=dict(ctx),
                away_context=dict(ctx),
                game_context={"is_playoff": False, "rest_days": 2},
                seed=999,
                n_iterations=2000,
            )
            plain = analyze(GameAnalysisRequest(**base_kwargs), session_id="s", bankroll=1000.0)
            withev = analyze(
                GameAnalysisRequest(**base_kwargs, evidence=[sig]),
                session_id="s",
                bankroll=1000.0,
            )
        assert (
            plain["result"]["simulation"]["home_win_prob"]
            == withev["result"]["simulation"]["home_win_prob"]
        )
