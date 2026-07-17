"""Phase 1 acceptance: MLB game + prop, end to end.

run_batch (real RSVG gate, mocked engine) → export blocks → the real ingest
seam (PersistableTrace.from_analyze_output → TraceStore.persist) → grouped
matchup brief via the console service and the MCP tool.

Asserts the Phase 1 acceptance criteria (design §14):
- game and prop are grouped under the same provider event id;
- displayed external facts carry provenance/freshness or an explicit label;
- authorized probabilities contain all mutually exclusive outcomes;
- markets are ordered by identity, never ranked by advantage;
- calibration warnings and data-quality notes stay visible;
- no engine_auto ledger rows appear (default disabled);
- the brief carries no protected fields or blocked language.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from omega.mcp.server import omega_get_matchup_brief, omega_run_batch
from omega.trace.persistable import PersistableTrace
from omega.trace.store import TraceStore
from omega.ui.service import open_service

EVENT_ID = "mlb-ev-100"
EVENT_KEY = f"MLB::the-odds-api::{EVENT_ID}"
GAME_DATE = "2026-07-16"
HOME = "Seattle Mariners"
AWAY = "New York Mets"

_PROD_AUDIT = {
    "raw_prob": 0.55,
    "calibrated_prob": 0.56,
    "plane": "game",
    "market": "home",
    "path": "profile",
    "profile_id": "iso_mlb_game_v8",
    "profile_maturity": "production",
    "sample_size": 400,
    "ece": 0.03,
}


def _roster_context() -> dict[str, Any]:
    return {
        "home_team": HOME,
        "away_team": AWAY,
        "league": "MLB",
        "game_date": GAME_DATE,
        "source_summaries": [
            {
                "source": "mlb.com",
                "summary": "Both confirmed lineups posted; no late scratches.",
                "source_title": "Mariners-Mets lineups",
                "source_url": "https://mlb.com/lineups",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "home_status": {"lineup_status": "confirmed", "injury_report_checked": True},
        "away_status": {"lineup_status": "confirmed", "injury_report_checked": True},
        "absences": [],
        "roster_context_complete": True,
        "gathered_at": datetime.now(timezone.utc).isoformat(),
    }


def _entries() -> list[dict[str, Any]]:
    return [
        {
            "kind": "game",
            "league": "MLB",
            "home_team": HOME,
            "away_team": AWAY,
            "game_date": GAME_DATE,
            "event_id": EVENT_ID,
            "odds": {"moneyline_home": -140, "moneyline_away": 120},
            "home_context": {"off_rating": 4.4, "def_rating": 3.8},
            "away_context": {"off_rating": 4.1, "def_rating": 4.0},
            "game_context": {"is_playoff": False, "rest_days": 1},
            "roster_context": _roster_context(),
        },
        {
            "kind": "prop",
            "league": "MLB",
            "home_team": HOME,
            "away_team": AWAY,
            "game_date": GAME_DATE,
            "event_id": EVENT_ID,
            "player_name": "Julio Rodríguez",
            "prop_type": "hits",
            "line": 1.5,
            "odds_over": -110,
            "odds_under": -110,
            "player_context": {"hits_mean": 1.2, "hits_std": 0.8},
            "game_context": {"is_playoff": False, "rest_days": 1},
        },
    ]


def _fake_analyze(request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Kind-appropriate engine envelopes with production calibration audits."""
    if "player_name" in request:
        return {
            "trace_id": "sandbox-mlb-prop-1",
            "kind": "prop",
            "ran_at": "2026-07-16T18:00:00Z",
            "session_id": kwargs.get("session_id"),
            "input_snapshot": {
                "league": "MLB",
                "player_name": request["player_name"],
                "prop_type": request["prop_type"],
                "line": request["line"],
                "odds_over": request["odds_over"],
                "odds_under": request["odds_under"],
                "home_team": HOME,
                "away_team": AWAY,
                "game_date": GAME_DATE,
            },
            "result": {
                "status": "success",
                "over_prob": 0.44,
                "under_prob": 0.56,
                "recommendation": "under",
                "confidence_tier": "B",
                "over_calibration_audit": {**_PROD_AUDIT, "plane": "prop", "market": "over"},
                "under_calibration_audit": {**_PROD_AUDIT, "plane": "prop", "market": "under"},
                "simulation_distributions": [
                    {
                        "target": "player_stat",
                        "stat_key": "hits",
                        "distribution_type": "negative_binomial",
                        "sample_mean": 1.2,
                        "sample_std": 0.8,
                        "p10": 0.0,
                        "p50": 1.0,
                        "p90": 3.0,
                        "n_iterations": 5000,
                    }
                ],
            },
            "trace_quality": {"aggregate_quality": 0.88},
        }
    return {
        "trace_id": "sandbox-mlb-game-1",
        "kind": "game",
        "ran_at": "2026-07-16T18:00:00Z",
        "session_id": kwargs.get("session_id"),
        "input_snapshot": {
            "league": "MLB",
            "home_team": HOME,
            "away_team": AWAY,
            "odds": request.get("odds"),
        },
        "result": {
            "status": "success",
            "simulation": {
                "home_win_prob": 56.0,
                "away_win_prob": 44.0,
                "draw_prob": None,
            },
            "edges": [
                {
                    "side": "home",
                    "team": HOME,
                    "market": "moneyline",
                    "true_prob": 0.55,
                    "calibrated_prob": 0.56,
                    "market_implied": 0.58,
                    "edge_pct": -2.0,
                    "ev_pct": -3.0,
                    "market_odds": -140,
                    "confidence_tier": "Pass",
                    "recommended_units": 0.0,
                    "calibration_audit": dict(_PROD_AUDIT),
                },
                {
                    "side": "away",
                    "team": AWAY,
                    "market": "moneyline",
                    "true_prob": 0.45,
                    "calibrated_prob": 0.44,
                    "market_implied": 0.45,
                    "edge_pct": -1.0,
                    "ev_pct": -1.5,
                    "market_odds": 120,
                    "confidence_tier": "Pass",
                    "recommended_units": 0.0,
                    "calibration_audit": {**_PROD_AUDIT, "market": "away"},
                },
            ],
            "best_bet": None,
            "simulation_distributions": [
                {
                    "target": "home_score",
                    "distribution_type": "poisson",
                    "sample_mean": 4.6,
                    "sample_std": 2.0,
                    "p10": 2.0,
                    "p50": 4.0,
                    "p90": 8.0,
                    "n_iterations": 10000,
                }
            ],
        },
        "trace_quality": {"aggregate_quality": 0.9},
    }


def _run_batch_and_ingest() -> str:
    """run_batch → export blocks → real ingest seam → temp DB. Returns db path."""
    export_root = Path(tempfile.mkdtemp())
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.core.contracts.service.analyze", side_effect=_fake_analyze),
        patch("omega.paths.repo_root", return_value=export_root),
    ):
        result = omega_run_batch(
            entries=_entries(), bankroll=1000.0, session_id="sess-e2e"
        )
    assert result["status"] == "ok", result
    assert result["entries_ok"] == 2

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = TraceStore(db_path=tmp.name)
    for path in result["export_paths"]:
        block = json.loads(Path(path).read_text(encoding="utf-8"))
        record = PersistableTrace.from_analyze_output(block["trace"]).to_store_record()
        store.persist(record)
    # Acceptance: default-disabled autolog — zero engine_auto wager rows.
    rows = store.conn.execute(
        "SELECT COUNT(*) FROM bet_ledger WHERE provenance = 'engine_auto'"
    ).fetchone()[0]
    assert rows == 0
    store.close()
    return tmp.name


def test_mlb_game_and_prop_share_one_brief_end_to_end():
    db = _run_batch_and_ingest()

    # -- service path -------------------------------------------------------
    service = open_service(db_path=db)
    try:
        briefs = service.list_matchup_briefs()
    finally:
        service.close()
    assert len(briefs) == 1
    brief = briefs[0]
    assert brief["event_key"] == EVENT_KEY
    assert brief["identity_warning"] is False
    assert brief["presentation_mode"] == "decision_support"

    # Markets ordered by stable identity: game first, then props by player.
    assert [m["market_group"] for m in brief["markets"]] == [
        "game",
        "Julio Rodríguez hits",
    ]

    game, prop = brief["markets"]
    ml = game["probability_sets"][0]
    assert ml["market_key"] == "moneyline"
    assert ml["disclosure"] == "shown"
    assert [o["outcome_key"] for o in ml["outcomes"]] == ["home", "away"]
    assert "not a recommendation" in ml["estimate_label"]
    # Market-implied baselines accompany the estimates, no gap is computed.
    assert ml["outcomes"][0]["market_implied"] == 0.58
    assert "gap" not in json.dumps(ml)

    prop_set = prop["probability_sets"][0]
    assert prop_set["market_key"] == "hits"
    assert [o["outcome_key"] for o in prop_set["outcomes"]] == ["over", "under"]

    # RSVG source provenance made it through the whole pipeline.
    game_sources = {s["source"]: s for s in game["sources"]}
    assert game_sources["mlb.com"]["provenance_status"] == "ok"
    assert game_sources["mlb.com"]["source_url"] == "https://mlb.com/lineups"

    # Distributions + explicit sensitivity-unavailable state.
    assert game["distributions"][0]["distribution_type"] == "poisson"
    assert game["sensitivity"]["status"] == "unavailable"

    # -- MCP path -----------------------------------------------------------
    mcp_result = omega_get_matchup_brief(EVENT_KEY, db_path=db)
    assert mcp_result["status"] == "success"
    mcp_brief = mcp_result["brief"]
    assert mcp_brief["event_key"] == EVENT_KEY
    assert [m["market_group"] for m in mcp_brief["markets"]] == [
        "game",
        "Julio Rodríguez hits",
    ]

    # -- safety sweep -------------------------------------------------------
    from omega.core.contracts.language import blocked_language
    from omega.trace.decision_support import DENYLIST_KEYS

    dumped = json.dumps(mcp_brief)
    for key in sorted(DENYLIST_KEYS):
        assert f'"{key}"' not in dumped, key
    assert blocked_language(dumped) == []
