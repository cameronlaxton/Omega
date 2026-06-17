"""
Tests for retrospective evidence-signal scoring (Phase C).

Covers realized-outcome resolution, per-trace scoring, the aggregation math,
and the signal_performance store round-trip.
"""

from __future__ import annotations

import tempfile

from omega.strategy.signal_performance import (
    ScoredSignal,
    SignalPerformanceRow,
    accumulate_signal_performance,
    realized_game_direction,
    realized_prop_direction,
    score_trace_signals,
)
from omega.trace.store import TraceStore


def _tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _ev_row(**overrides) -> dict:
    base = {
        "signal_type": "recent_form",
        "source": "boxscore_derived",
        "obs_window": "last_5",
        "league": "NBA",
        "confidence": 0.8,
        "direction": "over",
        "stat_key": "pts",
    }
    base.update(overrides)
    return base


class TestRealizedDirection:
    def test_prop_over(self):
        assert realized_prop_direction([{"stat_value": 30.0, "line": 27.5}]) == "over"

    def test_prop_under(self):
        assert realized_prop_direction([{"stat_value": 13.0, "line": 27.5}]) == "under"

    def test_prop_push_is_none(self):
        assert realized_prop_direction([{"stat_value": 27.5, "line": 27.5}]) is None

    def test_prop_empty_is_none(self):
        assert realized_prop_direction([]) is None
        assert realized_prop_direction(None) is None

    def test_game_home(self):
        assert realized_game_direction({"result": "home_win"}) == "home"

    def test_game_away(self):
        assert realized_game_direction({"result": "away_win"}) == "away"

    def test_game_draw(self):
        # 3-way draw now resolves to a scorable 'draw' direction (Gap 2).
        assert realized_game_direction({"result": "draw"}) == "draw"

    def test_game_missing_is_none(self):
        assert realized_game_direction(None) is None
        assert realized_game_direction({}) is None


class TestScoreTraceSignals:
    def test_prop_signal_scored_correct(self):
        trace = {"_prop_outcomes": [{"stat_value": 30.0, "line": 27.5}]}
        scored = score_trace_signals(trace, [_ev_row(direction="over")])
        assert len(scored) == 1
        assert scored[0].direction_correct is True

    def test_prop_signal_scored_incorrect(self):
        trace = {"_prop_outcomes": [{"stat_value": 13.0, "line": 27.5}]}
        scored = score_trace_signals(trace, [_ev_row(direction="over")])
        assert scored[0].direction_correct is False

    def test_neutral_signal_skipped(self):
        trace = {"_prop_outcomes": [{"stat_value": 30.0, "line": 27.5}]}
        scored = score_trace_signals(trace, [_ev_row(direction="neutral")])
        assert scored == []

    def test_none_direction_skipped(self):
        trace = {"_prop_outcomes": [{"stat_value": 30.0, "line": 27.5}]}
        scored = score_trace_signals(trace, [_ev_row(direction=None)])
        assert scored == []

    def test_push_outcome_skips_signal(self):
        trace = {"_prop_outcomes": [{"stat_value": 27.5, "line": 27.5}]}
        scored = score_trace_signals(trace, [_ev_row(direction="over")])
        assert scored == []

    def test_game_signal_scored(self):
        trace = {"_outcome": {"result": "home_win"}}
        row = _ev_row(direction="home", signal_type="motivation_edge", stat_key=None)
        scored = score_trace_signals(trace, [row])
        assert scored[0].direction_correct is True

    def test_draw_signal_scored_correct(self):
        # A draw-direction game signal scores against a drawn outcome (Gap 2).
        trace = {"_outcome": {"result": "draw"}}
        row = _ev_row(direction="draw", signal_type="parity_edge", stat_key=None, league="EPL")
        scored = score_trace_signals(trace, [row])
        assert len(scored) == 1
        assert scored[0].direction_correct is True

    def test_draw_signal_scored_incorrect(self):
        trace = {"_outcome": {"result": "home_win"}}
        row = _ev_row(direction="draw", signal_type="parity_edge", stat_key=None, league="EPL")
        scored = score_trace_signals(trace, [row])
        assert len(scored) == 1
        assert scored[0].direction_correct is False


class TestAccumulate:
    def test_groups_by_key(self):
        scored = [
            ScoredSignal("recent_form", "boxscore", "last_5", "NBA", 0.8, True),
            ScoredSignal("recent_form", "boxscore", "last_5", "NBA", 0.8, True),
            ScoredSignal("usage_spike", "agent", "matchup", "NBA", 0.6, False),
        ]
        rows = accumulate_signal_performance(scored)
        assert len(rows) == 2

    def test_aggregate_math(self):
        scored = [
            ScoredSignal("recent_form", "boxscore", "last_5", "NBA", 0.8, True),
            ScoredSignal("recent_form", "boxscore", "last_5", "NBA", 0.8, True),
            ScoredSignal("recent_form", "boxscore", "last_5", "NBA", 0.6, False),
        ]
        row = accumulate_signal_performance(scored)[0]
        assert row.sample_size == 3
        assert row.direction_correct == 2
        assert abs(row.direction_accuracy - 2 / 3) < 1e-9
        assert abs(row.mean_confidence - (0.8 + 0.8 + 0.6) / 3) < 1e-9
        # gap = mean_conf - accuracy
        assert abs(row.calibration_gap - (0.7333333 - 0.6666667)) < 1e-4
        # brier = ((0.8-1)^2 + (0.8-1)^2 + (0.6-0)^2) / 3
        assert abs(row.brier - (0.04 + 0.04 + 0.36) / 3) < 1e-9

    def test_deterministic_ordering(self):
        scored = [
            ScoredSignal("zzz", "s", "season", "NBA", 0.5, True),
            ScoredSignal("aaa", "s", "season", "NBA", 0.5, True),
        ]
        rows = accumulate_signal_performance(scored)
        assert [r.signal_type for r in rows] == ["aaa", "zzz"]


class TestSignalPerformanceStore:
    def _row(self, signal_type: str = "recent_form") -> SignalPerformanceRow:
        return SignalPerformanceRow(
            signal_type=signal_type,
            source="boxscore_derived",
            obs_window="last_5",
            league="NBA",
            sample_size=40,
            direction_correct=29,
            direction_accuracy=0.725,
            mean_confidence=0.80,
            realized_hit_rate=0.725,
            calibration_gap=0.075,
            brier=0.18,
        )

    def test_upsert_and_read_back(self):
        store = TraceStore(db_path=_tmp_db())
        n = store.upsert_signal_performance([self._row()], dataset_hash="h1")
        assert n == 1
        rows = store.get_signal_performance(league="NBA")
        assert len(rows) == 1
        assert rows[0]["signal_type"] == "recent_form"
        assert rows[0]["sample_size"] == 40
        assert abs(rows[0]["direction_accuracy"] - 0.725) < 1e-9
        store.close()

    def test_reupsert_same_hash_is_idempotent(self):
        store = TraceStore(db_path=_tmp_db())
        store.upsert_signal_performance([self._row()], dataset_hash="h1")
        store.upsert_signal_performance([self._row()], dataset_hash="h1")
        rows = store.get_signal_performance(league="NBA")
        assert len(rows) == 1  # replaced, not duplicated
        store.close()

    def test_get_returns_empty_when_unscored(self):
        store = TraceStore(db_path=_tmp_db())
        assert store.get_signal_performance(league="NBA") == []
        store.close()

    def test_get_filters_by_league(self):
        store = TraceStore(db_path=_tmp_db())
        store.upsert_signal_performance(
            [self._row(), SignalPerformanceRow(
                "recent_form", "s", "last_5", "NHL", 10, 5, 0.5, 0.5, 0.5, 0.0, 0.25
            )],
            dataset_hash="h1",
        )
        nba = store.get_signal_performance(league="NBA")
        assert all(r["league"] == "NBA" for r in nba)
        store.close()
