"""FrozenArtifact governed substrate (plan 5.4 — closes the P8.3 follow-up).

Schema v2 artifacts carry the production run's ``simulation_backend`` and
``prior_payload`` so the backtest re-sim runs on the same substrate and
param-bound calibration profiles apply in backtest exactly as in production.
Recovery is fail-closed: a trace whose execution echoed governed provenance
that the request prior cannot reproduce is marked ``substrate_unresolved``
and the backtest keeps today's ungoverned default re-sim.
"""

from __future__ import annotations

import pytest

import omega.strategy.backtest.engine as bt_engine
from omega.strategy.artifacts import FrozenArtifact, trace_to_artifact
from omega.strategy.backtest.engine import BacktestEngine
from omega.strategy.models import StrategyEntry

_PROFILE_REF = {
    "backend_name": "soccer_bivariate_poisson_dc",
    "backend_component_version": "soccer_bvp_dc_v1",
    "param_profile_id": "soccer_bivariate_poisson_dc__EPL__v1__abc123def456",
}

_PRIOR = {
    "rho": -0.10,
    "rho_profile_id": _PROFILE_REF["param_profile_id"],
    "parameter_profile_ref": dict(_PROFILE_REF),
}


def _governed_trace(prior_payload=_PRIOR, echo=True):
    """A minimal persisted-trace dict for a governed soccer DC analysis."""
    trace = {
        "trace_id": "sandbox-test-governed-1",
        "matchup": "Arsenal @ Chelsea",
        "league": "EPL",
        "timestamp": "2026-05-01T18:00:00+00:00",
        "simulation_seed": 1234,
        "input_snapshot": {
            "simulation_backend": "soccer_bivariate_poisson_dc",
            "home_context": {"xg_for": 1.6, "xg_against": 1.1},
            "away_context": {"xg_for": 1.4, "xg_against": 1.2},
            "game_context": {},
        },
        "execution_result": {
            "simulation_backend": "soccer_bivariate_poisson_dc",
            "component_version": "soccer_bvp_dc_v1",
            "home_context": {"xg_for": 1.6, "xg_against": 1.1},
            "away_context": {"xg_for": 1.4, "xg_against": 1.2},
        },
        "odds_snapshot": {
            "moneyline_home": -120,
            "moneyline_away": +250,
            "spread_home": -0.5,
            "over_under": 2.5,
        },
    }
    if prior_payload is not None:
        trace["input_snapshot"]["prior_payload"] = dict(prior_payload)
    if echo:
        # Sim-result echo as the soccer DC backend persists it.
        trace["result"] = {
            "simulation": {
                "dc_rho": -0.10,
                "rho_profile_id": _PROFILE_REF["param_profile_id"],
                "parameter_profile_ref": dict(_PROFILE_REF),
                "simulation_backend": "soccer_bivariate_poisson_dc",
                "component_version": "soccer_bvp_dc_v1",
            }
        }
    return trace


class TestSubstrateRecovery:
    def test_governed_trace_recovers_backend_and_prior(self):
        art = trace_to_artifact(_governed_trace())
        assert art.schema_version == 2
        assert art.simulation_backend == "soccer_bivariate_poisson_dc"
        assert art.prior_payload is not None
        assert art.prior_payload["rho"] == pytest.approx(-0.10)
        assert art.substrate_unresolved is False

    def test_governed_echo_without_request_prior_is_unresolved(self):
        art = trace_to_artifact(_governed_trace(prior_payload=None))
        assert art.substrate_unresolved is True
        assert art.prior_payload is None

    def test_conflicting_profile_ids_are_unresolved(self):
        conflicting = dict(_PRIOR)
        conflicting["rho_profile_id"] = "some_other_profile"
        conflicting["parameter_profile_ref"] = {
            **_PROFILE_REF,
            "param_profile_id": "some_other_profile",
        }
        art = trace_to_artifact(_governed_trace(prior_payload=conflicting))
        assert art.substrate_unresolved is True
        assert art.prior_payload is None

    def test_ungoverned_trace_carries_raw_prior_without_flag(self):
        # No governed echo: a raw-knob prior (no profile ref) is carried as-is.
        trace = _governed_trace(prior_payload={"rho": -0.08}, echo=False)
        art = trace_to_artifact(trace)
        assert art.prior_payload == {"rho": -0.08}
        assert art.substrate_unresolved is False

    def test_v1_artifact_dict_round_trips_unchanged(self):
        v1 = {
            "artifact_id": "a" * 16,
            "schema_version": 1,
            "home_team": "A",
            "away_team": "B",
            "league": "NBA",
            "date": "2026-01-01",
        }
        art = FrozenArtifact(**v1)
        assert art.schema_version == 1
        assert art.simulation_backend is None
        assert art.prior_payload is None
        assert art.substrate_unresolved is False


def _strategy() -> StrategyEntry:
    return StrategyEntry(
        strategy_id="test-substrate",
        name="test-substrate",
        description="substrate parity test",
        edge_threshold=0.0,
        leagues=["EPL"],
    )


def _artifact(**overrides) -> FrozenArtifact:
    base = dict(
        artifact_id="b" * 16,
        source_trace_id="sandbox-test-governed-1",
        home_team="Chelsea",
        away_team="Arsenal",
        league="EPL",
        date="2026-05-01",
        home_context={"xg_for": 1.6, "xg_against": 1.1, "off_rating": 1.6, "def_rating": 1.1},
        away_context={"xg_for": 1.4, "xg_against": 1.2, "off_rating": 1.4, "def_rating": 1.2},
        game_context={},
        odds={
            "moneyline_home": -120,
            "moneyline_away": +250,
            "spread_home": -0.5,
            "over_under": 2.5,
        },
        simulation_seed=1234,
        outcome={"home_score": 2, "away_score": 1},
    )
    base.update(overrides)
    return FrozenArtifact(**base)


class TestBacktestGovernedResim:
    def _run_and_capture_substrates(self, artifact, monkeypatch):
        captured: list[dict] = []
        real = bt_engine.apply_calibration

        def spy(raw_prob, **kwargs):
            captured.append(kwargs.get("substrate_ref"))
            return real(raw_prob, **kwargs)

        monkeypatch.setattr(bt_engine, "apply_calibration", spy)
        result = BacktestEngine(n_iterations=400, seed=7).run(_strategy(), [artifact])
        return result, captured

    def test_governed_artifact_resim_reports_param_profile_id(self, monkeypatch):
        art = _artifact(
            simulation_backend="soccer_bivariate_poisson_dc",
            prior_payload=dict(_PRIOR),
        )
        result, substrates = self._run_and_capture_substrates(art, monkeypatch)
        assert substrates, "expected the re-sim to succeed and calibrate at least one market"
        for ref in substrates:
            assert ref["backend_name"] == "soccer_bivariate_poisson_dc"
            assert ref["param_profile_id"] == _PROFILE_REF["param_profile_id"]

    def test_unresolved_substrate_falls_back_to_ungoverned_default(self, monkeypatch):
        art = _artifact(
            simulation_backend="soccer_bivariate_poisson_dc",
            prior_payload=dict(_PRIOR),
            substrate_unresolved=True,
        )
        result, substrates = self._run_and_capture_substrates(art, monkeypatch)
        assert substrates
        for ref in substrates:
            # Default fast_score re-sim: no governed profile id.
            assert ref["param_profile_id"] is None

    def test_v1_artifact_backtest_is_unchanged(self, monkeypatch):
        # A v1-shaped artifact (no substrate fields) must produce bet rows
        # identical to before the schema change: same default backend, same
        # seed, same probabilities.
        art = _artifact()
        result, substrates = self._run_and_capture_substrates(art, monkeypatch)
        assert substrates
        for ref in substrates:
            assert ref["param_profile_id"] is None

    def test_unregistered_backend_name_falls_back(self, monkeypatch):
        art = _artifact(
            simulation_backend="not_a_registered_backend",
            prior_payload={"rho": -0.10},
        )
        result, substrates = self._run_and_capture_substrates(art, monkeypatch)
        # Fail-closed fallback: default backend, no prior threaded.
        assert substrates
        for ref in substrates:
            assert ref["param_profile_id"] is None
