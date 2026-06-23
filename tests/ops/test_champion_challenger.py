from __future__ import annotations

from omega.ops.run_champion_challenger import _load_graded_traces, _trace_to_game


def test_load_graded_traces_filters_null_league_without_fallback(monkeypatch):
    class FakeStore:
        def query_traces(self, has_outcome: bool, limit: int):
            assert has_outcome is True
            assert limit == 200
            return [
                {"trace_id": "null-league", "league": None},
                {"trace_id": "nba-game", "league": "NBA"},
                {"trace_id": "mlb-game", "league": "MLB"},
            ]

    import omega.trace.store as store_mod

    monkeypatch.setattr(store_mod, "TraceStore", FakeStore)

    traces = _load_graded_traces("NBA")

    assert [t["trace_id"] for t in traces] == ["nba-game"]


def test_trace_to_game_skips_unusable_rows_individually():
    assert _trace_to_game({"matchup": "bad", "league": None}) is None
