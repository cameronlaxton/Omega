"""Issue #27 acceleration lane, first slice — replay rows, output cache, CRN.

HistoricalReplayRow is a derived view over FrozenArtifact (never a parallel
replay source): rows enumerate the priced market sides with decision-time
provenance, fail closed on missing provenance or post-outcome contamination,
and regroup losslessly into the artifact the one BacktestEngine consumes.
SimulationOutputCache reuses raw deterministic sim outputs only, keyed on every
substrate axis. crn_seed gives stable per-row common-random-number streams for
MC-only paths without displacing exact evaluation.
"""

from __future__ import annotations

import json

import pytest

from omega.strategy.artifacts import FrozenArtifact, compute_artifact_id
from omega.strategy.backtest.acceleration import (
    LookaheadContamination,
    MissingDecisionProvenance,
    SimulationOutputCache,
    artifact_from_replay_rows,
    crn_seed,
    replay_rows_from_artifact,
)
from omega.strategy.backtest.engine import BacktestEngine
from omega.strategy.models import StrategyEntry

_FULL_ODDS = {
    "moneyline_home": -180,
    "moneyline_away": 155,
    "spread_home": -4.5,
    "spread_home_price": -110,
    "spread_away_price": -110,
    "over_under": 224.5,
    "total_over_price": -110,
    "total_under_price": -105,
}


def _artifact(home: str = "Celtics", away: str = "Pacers", **over) -> FrozenArtifact:
    base = dict(
        source_trace_id="trace-1",
        home_team=home,
        away_team=away,
        league="NBA",
        date="2026-05-21",
        home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        away_context={"off_rating": 112.0, "def_rating": 112.0, "pace": 99.0},
        game_context={"is_playoff": True, "rest_days": 2},
        odds=dict(_FULL_ODDS),
        closing_odds={"moneyline_home": -195, "moneyline_away": 165},
        simulation_seed=1234,
        calibration_policy="static_v1",
        outcome={"home_score": 118, "away_score": 109},
    )
    base.update(over)
    return FrozenArtifact(
        artifact_id=compute_artifact_id(home, away, base["league"], base["date"]),
        **base,
    )


def _strategy() -> StrategyEntry:
    return StrategyEntry(
        strategy_id="accel",
        name="accel",
        leagues=["NBA"],
        edge_threshold=0.02,
        confidence_tiers=["A", "B", "C"],
    )


def _slate() -> list[FrozenArtifact]:
    return [
        _artifact("Celtics", "Pacers"),
        _artifact(
            "Lakers",
            "Magic",
            home_context={"off_rating": 115.0, "def_rating": 110.0, "pace": 101.0},
            away_context={"off_rating": 110.0, "def_rating": 113.0, "pace": 98.0},
            outcome={"home_score": 101, "away_score": 108},
            simulation_seed=77,
        ),
        _artifact(
            "Heat",
            "Hawks",
            home_context={"off_rating": 113.0, "def_rating": 111.0, "pace": 99.0},
            away_context={"off_rating": 114.0, "def_rating": 110.0, "pace": 100.0},
            outcome={"home_score": 104, "away_score": 112},
            simulation_seed=99,
        ),
    ]


def _result_signature(result) -> tuple:
    return (
        result.total_bets_placed,
        result.win_count,
        result.net_units,
        result.results_by_market,
    )


class TestHistoricalReplayRow:
    def test_rows_enumerate_priced_sides(self):
        rows = replay_rows_from_artifact(_artifact())
        by_key = {(r.market, r.side): r for r in rows}
        assert set(by_key) == {
            ("moneyline", "home"),
            ("moneyline", "away"),
            ("spread", "home"),
            ("spread", "away"),
            ("total", "over"),
            ("total", "under"),
        }
        assert by_key[("moneyline", "home")].offered_odds == -180
        assert by_key[("moneyline", "home")].line is None
        assert by_key[("spread", "home")].line == -4.5
        assert by_key[("spread", "away")].line == 4.5  # bet at the negated line
        assert by_key[("total", "under")].offered_odds == -105
        assert by_key[("total", "under")].line == 224.5
        # Closing odds map per side; sides without a captured close carry None.
        assert by_key[("moneyline", "home")].closing_odds == -195
        assert by_key[("spread", "home")].closing_odds is None

    def test_rows_preserve_provenance(self):
        prior = {"parameter_profile_ref": {"param_profile_id": "pp_v2"}, "rho": -0.01}
        artifact = _artifact(simulation_backend="fast_score", prior_payload=prior)
        row = replay_rows_from_artifact(artifact)[0]
        assert row.artifact_id == artifact.artifact_id
        assert row.source_trace_id == "trace-1"
        assert row.date == "2026-05-21"
        assert row.simulation_seed == 1234
        assert row.simulation_backend == "fast_score"
        assert row.prior_payload == prior
        assert row.calibration_policy == "static_v1"
        assert row.outcome == {"home_score": 118, "away_score": 109}

    def test_round_trip_is_lossless(self):
        artifact = _artifact(
            simulation_backend="fast_score",
            prior_payload={"rho": -0.01},
        )
        rows = replay_rows_from_artifact(artifact)
        rebuilt = artifact_from_replay_rows(rows)
        assert rebuilt.model_dump() == artifact.model_dump()

    def test_round_trip_backtest_behavior_unchanged(self):
        originals = _slate()
        rebuilt = [artifact_from_replay_rows(replay_rows_from_artifact(a)) for a in originals]
        strat = _strategy()
        a = BacktestEngine(n_iterations=1000, exact_eval=True).run(strat, originals)
        b = BacktestEngine(n_iterations=1000, exact_eval=True).run(strat, rebuilt)
        assert _result_signature(a) == _result_signature(b)

    def test_rows_from_multiple_artifacts_do_not_regroup(self):
        rows = replay_rows_from_artifact(_artifact("Celtics", "Pacers"))
        rows += replay_rows_from_artifact(_artifact("Lakers", "Magic"))
        with pytest.raises(ValueError, match="span multiple artifacts"):
            artifact_from_replay_rows(rows)

    def test_outcome_never_in_model_inputs(self):
        rows = replay_rows_from_artifact(_artifact())
        for row in rows:
            assert set(row.model_inputs) == {"home_context", "away_context", "game_context"}
            dumped = json.dumps(row.model_inputs)
            assert "home_score" not in dumped
            assert "outcome" not in dumped


class TestLookaheadGuards:
    def test_missing_odds_fails_closed(self):
        with pytest.raises(MissingDecisionProvenance, match="no decision-time odds"):
            replay_rows_from_artifact(_artifact(odds={}))

    def test_missing_contexts_fails_closed(self):
        with pytest.raises(MissingDecisionProvenance, match="no decision-time team contexts"):
            replay_rows_from_artifact(_artifact(home_context={}, away_context={}))

    def test_line_without_price_is_not_replayable(self):
        # A spread line with no prices (and nothing else priced) has no
        # replayable side — fail closed rather than fabricate a decision.
        with pytest.raises(MissingDecisionProvenance, match="prices no replayable"):
            replay_rows_from_artifact(_artifact(odds={"spread_home": -4.5}))

    def test_non_strict_mode_yields_empty_not_guessed(self):
        assert replay_rows_from_artifact(
            _artifact(odds={}), require_decision_provenance=False
        ) == []

    def test_contaminated_game_context_raises(self):
        artifact = _artifact(game_context={"is_playoff": True, "home_score": 118})
        with pytest.raises(LookaheadContamination, match="home_score"):
            replay_rows_from_artifact(artifact)

    def test_contamination_check_is_recursive(self):
        artifact = _artifact(home_context={"recent_form": {"result": "W-L-W"}})
        with pytest.raises(LookaheadContamination, match="result"):
            replay_rows_from_artifact(artifact)


class TestSimulationOutputCache:
    _BASE_KEY = dict(
        backend_name="fast_score",
        component_version="fs_v1",
        model_version="backtest_resim_v1",
        league="NBA",
        market="game_raw_sim",
        line={"spread_home": -4.5, "over_under": 224.5},
        context_hash="abc123",
        evidence_policy="replay_no_evidence",
        calibration_ref="static_v1",
    )

    def test_key_is_deterministic(self):
        assert SimulationOutputCache.make_key(**self._BASE_KEY) == SimulationOutputCache.make_key(
            **self._BASE_KEY
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("backend_name", "markov_state"),
            ("component_version", "fs_v2"),
            ("model_version", "backtest_resim_v2"),
            ("league", "NFL"),
            ("market", "prop_raw_sim"),
            ("line", {"spread_home": -3.5, "over_under": 224.5}),
            ("context_hash", "def456"),
            ("evidence_policy", "bounded_live"),
            ("calibration_ref", "iso_nba_v3"),
        ],
    )
    def test_key_changes_when_any_component_changes(self, field, value):
        changed = {**self._BASE_KEY, field: value}
        assert SimulationOutputCache.make_key(**changed) != SimulationOutputCache.make_key(
            **self._BASE_KEY
        )

    def test_hit_is_deterministic_and_mutation_safe(self):
        cache = SimulationOutputCache()
        key = SimulationOutputCache.make_key(**self._BASE_KEY)
        assert cache.get(key) is None
        cache.put(key, {"home_win_prob": 61.0}, audit={"source": "exact"})
        first = cache.get(key)
        second = cache.get(key)
        assert first == second == {"home_win_prob": 61.0}
        first["home_win_prob"] = 0.0  # a caller mutating its copy...
        assert cache.get(key) == {"home_win_prob": 61.0}  # ...never corrupts the cache
        assert cache.get_audit(key) == {"source": "exact"}
        assert cache.hits == 3
        assert cache.misses == 1

    def test_engine_cache_preserves_results_and_hits_on_reuse(self):
        slate = _slate()
        strat = _strategy()
        baseline = BacktestEngine(n_iterations=1000, exact_eval=True).run(strat, slate)

        cache = SimulationOutputCache()
        first = BacktestEngine(n_iterations=1000, exact_eval=True, output_cache=cache).run(
            strat, slate
        )
        assert cache.misses == len(slate)
        assert cache.hits == 0
        assert _result_signature(first) == _result_signature(baseline)

        second = BacktestEngine(n_iterations=1000, exact_eval=True, output_cache=cache).run(
            strat, slate
        )
        assert cache.hits == len(slate)  # every sim served from cache
        assert _result_signature(second) == _result_signature(baseline)

    def test_engine_cache_mc_path_is_deterministic(self):
        slate = _slate()
        strat = _strategy()
        cache = SimulationOutputCache()
        first = BacktestEngine(n_iterations=2000, exact_eval=False, output_cache=cache).run(
            strat, slate
        )
        second = BacktestEngine(n_iterations=2000, exact_eval=False, output_cache=cache).run(
            strat, slate
        )
        assert cache.hits == len(slate)
        assert _result_signature(first) == _result_signature(second)

    def test_engine_cache_not_shared_across_exact_and_mc(self):
        # exact flag is part of the context hash: an exact run never serves an
        # MC run (and vice versa) — no stale substrate reuse.
        slate = _slate()
        strat = _strategy()
        cache = SimulationOutputCache()
        BacktestEngine(n_iterations=1000, exact_eval=True, output_cache=cache).run(strat, slate)
        BacktestEngine(n_iterations=1000, exact_eval=False, output_cache=cache).run(strat, slate)
        assert cache.hits == 0
        assert cache.misses == 2 * len(slate)


class TestCommonRandomNumbers:
    def test_seed_is_stable(self):
        assert crn_seed("artifact-1") == crn_seed("artifact-1")
        assert crn_seed("artifact-1", "model_a") == crn_seed("artifact-1", "model_a")

    def test_seed_differs_by_row_and_model_version(self):
        assert crn_seed("artifact-1") != crn_seed("artifact-2")
        assert crn_seed("artifact-1", "model_a") != crn_seed("artifact-1", "model_b")
        assert crn_seed("artifact-1", "fast_score/fs_v1") != crn_seed(
            "artifact-1", "fast_score/fs_v2"
        )

    def test_paired_engines_share_streams(self):
        # Two engines under paired comparison pass the SAME salt: identical
        # random streams per row -> identical MC results run-to-run.
        slate = _slate()
        strat = _strategy()
        a = BacktestEngine(n_iterations=2000, exact_eval=False, crn_salt="pair-1").run(
            strat, slate
        )
        b = BacktestEngine(n_iterations=2000, exact_eval=False, crn_salt="pair-1").run(
            strat, slate
        )
        assert _result_signature(a) == _result_signature(b)

    def test_exact_eval_is_not_displaced_by_crn(self):
        # Exact evaluation has no sampling: enabling CRN changes nothing.
        slate = _slate()
        strat = _strategy()
        plain = BacktestEngine(n_iterations=1000, exact_eval=True).run(strat, slate)
        crn = BacktestEngine(n_iterations=1000, exact_eval=True, crn_salt="pair-1").run(
            strat, slate
        )
        assert _result_signature(plain) == _result_signature(crn)
