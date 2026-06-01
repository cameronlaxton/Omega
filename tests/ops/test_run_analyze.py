from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_analyze  # type: ignore  # noqa: E402


def _game_request() -> dict:
    return {
        "home_team": "Boston Celtics",
        "away_team": "Indiana Pacers",
        "league": "NBA",
        "n_iterations": 100,
        "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
        "game_context": {"is_playoff": False, "rest_days": 2},
        "odds": {"moneyline_home": -150, "moneyline_away": 130},
    }


def test_run_analyze_injects_deterministic_seed_and_writes_trace(tmp_path):
    request_path = tmp_path / "request.json"
    request = _game_request()
    request_path.write_text(json.dumps(request), encoding="utf-8")

    first = run_analyze.run(
        kind="game",
        request_json=request_path,
        session_id="sess-test",
        bankroll=1000.0,
        trace_out=tmp_path / "traces",
    )
    second = run_analyze.run(
        kind="game",
        request_json=request_path,
        session_id="sess-test",
        bankroll=1000.0,
    )

    assert first["kind"] == "game"
    assert first["session_id"] == "sess-test"
    assert first["input_snapshot"]["seed"] == second["input_snapshot"]["seed"]
    assert Path(first["_trace_out_path"]).exists()
    written = json.loads(Path(first["_trace_out_path"]).read_text(encoding="utf-8"))
    assert written["trace_id"] == first["trace_id"]
    assert written["result"]["status"] == "success"


def test_run_analyze_explicit_seed_wins(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_game_request()), encoding="utf-8")

    trace = run_analyze.run(
        kind="game",
        request_json=request_path,
        session_id="sess-test",
        bankroll=1000.0,
        seed=12345,
    )

    assert trace["input_snapshot"]["seed"] == 12345


def test_run_analyze_accepts_utf8_bom_request_json(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_game_request()), encoding="utf-8-sig")

    trace = run_analyze.run(
        kind="game",
        request_json=request_path,
        session_id="sess-test",
        bankroll=1000.0,
    )

    assert trace["kind"] == "game"
    assert trace["result"]["status"] == "success"

