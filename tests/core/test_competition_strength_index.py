"""Phase 5 (Issue #22) — structural competition_strength_index soccer path.

The index adjusts soccer team-context attack/concede rates BEFORE Bivariate
Poisson lambda derivation (never as a late home/away factor), preserving raw and
adjusted values plus the final lambdas in a trace/debug payload. Gated on
AdjustmentPolicy.enable_competition_strength_index; a no-op leaves the backend
bit-identical.
"""

from __future__ import annotations

import warnings

import pytest

from omega.core.calibration.adjustment_policy import AdjustmentPolicy
from omega.core.contracts.evidence import (
    SIGNAL_REGISTRY,
    EvidenceSignal,
    signal_applies,
)
from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import (
    _competition_strength_index,
    analyze,
)
from omega.core.simulation import engine as _engine  # noqa: F401  (registers backends)
from omega.core.simulation.backends import GameSimulationInput
from omega.core.simulation.evidence_handlers import HANDLER_REGISTRY
from omega.core.simulation.soccer_bivariate_poisson import (
    SoccerPoissonBackend,
    _resolve_competition_strength_index,
)

_SIG = "competition_strength_index"
_HOME_CTX = {"xg_for": 1.5, "xg_against": 1.1}
_AWAY_CTX = {"xg_for": 1.2, "xg_against": 1.3}


def _backend_request(csi, *, rho: float = -0.13) -> GameSimulationInput:
    return GameSimulationInput(
        home_team="Arsenal",
        away_team="Chelsea",
        league="EPL",
        n_iterations=4000,
        home_context=dict(_HOME_CTX),
        away_context=dict(_AWAY_CTX),
        seed=42,
        prior_payload={"rho": rho},
        competition_strength_index=csi,
    )


def _csi_evidence(value: float, direction=None) -> EvidenceSignal:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(
            signal_type=_SIG,
            category="situational",
            plane="game",
            value=value,
            source="agent_reasoning",
            confidence=0.6,
            window="matchup",
            direction=direction,
        )


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_signal_spec(self):
        spec = SIGNAL_REGISTRY[_SIG]
        assert spec.category == "situational"
        assert spec.plane == "game"
        assert spec.applies_to_sports == frozenset({"soccer"})
        assert spec.value_kind == "scalar"

    def test_has_audit_only_handler(self):
        # Must be in HANDLER_REGISTRY (engine invariant) but never apply a factor.
        assert _SIG in HANDLER_REGISTRY

    def test_applies_to_soccer_only(self):
        assert signal_applies(_SIG, "soccer") is True
        assert signal_applies(_SIG, "basketball") is False


# ---------------------------------------------------------------------------
# Backend application
# ---------------------------------------------------------------------------


class TestBackendApplication:
    def test_no_index_leaves_no_debug_payload(self):
        result = SoccerPoissonBackend().run(_backend_request(None))
        assert result["success"] is True
        assert "competition_strength_adjustment" not in result

    def test_neutral_index_is_a_noop(self):
        result = SoccerPoissonBackend().run(_backend_request({"home": 1.0, "away": 1.0}))
        assert "competition_strength_adjustment" not in result

    def test_applied_payload_carries_raw_and_adjusted(self):
        result = SoccerPoissonBackend().run(_backend_request({"home": 1.2, "away": 1.0}))
        csi = result["competition_strength_adjustment"]
        assert csi["index"] == {"home": 1.2, "away": 1.0}
        # home attack scaled up by 1.2, home concede divided by 1.2
        assert csi["raw"]["home_attack_rate"] == pytest.approx(1.5)
        assert csi["adjusted"]["home_attack_rate"] == pytest.approx(round(1.5 * 1.2, 4))
        assert csi["raw"]["home_concede_rate"] == pytest.approx(1.1)
        assert csi["adjusted"]["home_concede_rate"] == pytest.approx(round(1.1 / 1.2, 4))
        # away index 1.0 -> away rates unchanged
        assert csi["adjusted"]["away_attack_rate"] == pytest.approx(1.2)
        assert csi["raw"]["home_xg_for"] == 1.5  # raw context echoed
        assert isinstance(csi["home_lambda"], float)
        assert isinstance(csi["away_lambda"], float)

    def test_directional_lambda_effect(self):
        # Home strong vs away strong: home_lambda should move opposite to away_lambda.
        home_strong = SoccerPoissonBackend().run(_backend_request({"home": 1.3, "away": 1.0}))[
            "competition_strength_adjustment"
        ]
        away_strong = SoccerPoissonBackend().run(_backend_request({"home": 1.0, "away": 1.3}))[
            "competition_strength_adjustment"
        ]
        assert home_strong["home_lambda"] > away_strong["home_lambda"]
        assert home_strong["away_lambda"] < away_strong["away_lambda"]

    def test_index_shifts_win_probability(self):
        # Exact eval -> deterministic; a stronger home raises home win prob.
        base = SoccerPoissonBackend().run(_backend_request(None))
        stronger_home = SoccerPoissonBackend().run(_backend_request({"home": 1.3, "away": 0.9}))
        assert stronger_home["home_win_prob"] > base["home_win_prob"]


class TestResolveHelper:
    @pytest.mark.parametrize(
        "value",
        [
            None,
            {},
            {"home": 1.0, "away": 1.0},
            {"home": 0.0, "away": 1.2},
            {"home": "x"},
            # non-finite must be rejected — nan/inf slip past the <=0 / ==1 guards
            {"home": float("nan"), "away": 1.2},
            {"home": 1.2, "away": float("inf")},
        ],
    )
    def test_noop_cases_return_none(self, value):
        assert _resolve_competition_strength_index(value) is None

    def test_valid_index_returns_tuple(self):
        assert _resolve_competition_strength_index({"home": 1.2, "away": 0.9}) == (1.2, 0.9)


# ---------------------------------------------------------------------------
# Service extraction (gating + direction)
# ---------------------------------------------------------------------------


def _game_request(evidence, league="EPL") -> GameAnalysisRequest:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GameAnalysisRequest(
            home_team="Arsenal",
            away_team="Chelsea",
            league=league,
            home_context=dict(_HOME_CTX),
            away_context=dict(_AWAY_CTX),
            game_context={"is_playoff": False, "rest_days": 2},
            seed=42,
            n_iterations=2000,
            evidence=evidence,
        )


def _flag_on() -> AdjustmentPolicy:
    return AdjustmentPolicy(policy_id="csi-on", version=1, enable_competition_strength_index=True)


class TestServiceExtraction:
    def test_flag_off_returns_none(self):
        policy = AdjustmentPolicy(policy_id="off", version=1)
        req = _game_request([_csi_evidence(1.2, direction="home")])
        assert _competition_strength_index(req, policy) is None

    def test_non_soccer_returns_none(self):
        req = _game_request([_csi_evidence(1.2, direction="home")], league="NBA")
        assert _competition_strength_index(req, _flag_on()) is None

    def test_home_directed_sets_home_side(self):
        req = _game_request([_csi_evidence(1.2, direction="home")])
        assert _competition_strength_index(req, _flag_on()) == {"home": 1.2, "away": 1.0}

    def test_away_directed_sets_away_side(self):
        req = _game_request([_csi_evidence(1.15, direction="away")])
        assert _competition_strength_index(req, _flag_on()) == {"home": 1.0, "away": 1.15}

    def test_neutral_sets_both_sides(self):
        req = _game_request([_csi_evidence(1.1, direction=None)])
        assert _competition_strength_index(req, _flag_on()) == {"home": 1.1, "away": 1.1}

    def test_no_signal_returns_none(self):
        req = _game_request([])
        assert _competition_strength_index(req, _flag_on()) is None

    def test_player_plane_signal_is_ignored(self):
        # A misclassified player-plane CSI signal must not alter the game lambdas.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = EvidenceSignal(
                signal_type=_SIG,
                category="situational",
                plane="player",
                value=1.2,
                source="agent_reasoning",
                confidence=0.6,
                window="matchup",
                direction="home",
            )
        req = _game_request([sig])
        assert _competition_strength_index(req, _flag_on()) is None

    def test_neutral_value_returns_none(self):
        req = _game_request([_csi_evidence(1.0, direction="home")])
        assert _competition_strength_index(req, _flag_on()) is None


# ---------------------------------------------------------------------------
# End-to-end: analyze() threads the index into the trace
# ---------------------------------------------------------------------------


def test_analyze_threads_index_into_trace(monkeypatch):
    from omega.core.contracts import service as svc

    monkeypatch.setattr(svc, "_load_adjustment_policy", _flag_on)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        req = GameAnalysisRequest(
            home_team="Arsenal",
            away_team="Chelsea",
            league="EPL",
            home_context=dict(_HOME_CTX),
            away_context=dict(_AWAY_CTX),
            game_context={"is_playoff": False, "rest_days": 2},
            seed=42,
            n_iterations=3000,
            simulation_backend="soccer_bivariate_poisson_dc",
            prior_payload={"rho": -0.13},
            evidence=[_csi_evidence(1.25, direction="home")],
        )
    out = analyze(req, session_id="csi-sess", bankroll=1000.0)
    sim = out["result"]["simulation"]
    assert sim["competition_strength_adjustment"] is not None
    assert sim["competition_strength_adjustment"]["index"] == {"home": 1.25, "away": 1.0}
