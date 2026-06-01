"""Tests for `scripts/fetch_outcomes_props.py --dry-run` explainability.

Pins the "0 processed" diagnostics: effective DB path, UTC window, and the
per-league ungraded-trace count â€” so a window/DB mismatch is visible rather
than looking identical to "nothing to do".
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fetch_outcomes_props  # type: ignore  # noqa: E402

from omega.trace.store import TraceStore  # noqa: E402


def _prop_trace(trace_id: str, game_date: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": "r",
        "timestamp": f"{game_date}T18:00:00Z",
        "prompt": "prop",
        "league": "NBA",
        "matchup": "Player pts 25.5",
        "execution_mode": "sandbox_prop",
        "simulation_seed": 1,
        "aggregate_quality": 0.8,
        "predictions": {"over_prob": 0.5, "under_prob": 0.5},
        "recommendations": [],
        "odds_snapshot": {},
        "downgrades": [],
        "kind": "prop",
        "result": {"status": "success", "recommendation": "over", "over_prob": 0.5, "under_prob": 0.5},
        "input_snapshot": {
            "player_name": "Test Player",
            "prop_type": "points",
            "line": 25.5,
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "game_date": game_date,
        },
        "trace_quality": {"calibration_eligible": True},
    }


def test_dry_run_reports_db_window_and_ungraded_count(tmp_path, monkeypatch, caplog):
    db = tmp_path / "t.db"
    store = TraceStore(db_path=str(db))
    store.persist(_prop_trace("sandbox-prop-1", "2026-05-20"))
    store.close()

    # Scoreboard returns nothing â†’ the prop is unmatched (a skip reason), but the
    # pre-scan must still report it as 1 ungraded prop in the window.
    def _empty_scoreboard(_league: str, _d) -> list:
        return []

    with caplog.at_level(logging.INFO):
        rc = fetch_outcomes_props.main(
            [
                "--since",
                "2026-05-20",
                "--until",
                "2026-05-20",
                "--league",
                "NBA",
                "--db",
                str(db),
                "--dry-run",
            ],
            scoreboard_fetcher=_empty_scoreboard,
        )

    assert rc == 0
    text = "\n".join(r.message for r in caplog.records)
    assert "Outcome window (UTC): 2026-05-20 .. 2026-05-20" in text
    assert "ungraded traces in window" in text
    assert str(db) in text  # effective DB path surfaced

