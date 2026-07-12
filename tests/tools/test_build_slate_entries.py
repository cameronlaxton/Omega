import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from pathlib import Path
from unittest.mock import patch

from tools.build_slate_entries import main

def test_build_slate_entries_valid(tmp_path):
    slate = {
        "league": "MLB",
        "game_date": "2026-07-04",
        "games": [
            {
                "home_team": "Team A",
                "away_team": "Team B",
                "home_context": {"power": 1},
                "away_context": {"power": 2},
                "game_context": {"is_playoff": False, "rest_days": 1},
            }
        ]
    }
    slate_file = tmp_path / "slate.json"
    slate_file.write_text(json.dumps(slate))
    
    out_file = tmp_path / "out.json"
    args = ["--slate", str(slate_file), "--output", str(out_file)]
    
    assert main(args) == 0
    
    entries = json.loads(out_file.read_text())
    assert len(entries) == 1
    assert entries[0]["kind"] == "game"
    assert entries[0]["home_team"] == "Team A"
    
def test_build_slate_entries_missing_fields(tmp_path):
    slate = {
        "league": "MLB",
        "game_date": "2026-07-04",
        "games": [
            {
                "home_team": "Team A",
                # away_team is missing
            }
        ]
    }
    slate_file = tmp_path / "slate.json"
    slate_file.write_text(json.dumps(slate))

    out_file = tmp_path / "out.json"
    args = ["--slate", str(slate_file), "--output", str(out_file)]

    assert main(args) != 0


def _roster_context(home, away, summary):
    return {
        "home_team": home,
        "away_team": away,
        "league": "MLB",
        "game_date": "2026-07-04",
        "source_summaries": [{"source": "mlb.com", "summary": summary}],
        "home_status": {"lineup_status": "confirmed", "injury_report_checked": True},
        "away_status": {"lineup_status": "confirmed", "injury_report_checked": True},
        "absences": [],
        "roster_context_complete": True,
        "gathered_at": "2026-07-04T18:00:00Z",
    }


def _game_item(home, away, summary):
    return {
        "home_team": home,
        "away_team": away,
        "home_context": {"power": 1},
        "away_context": {"power": 2},
        "game_context": {"is_playoff": False, "rest_days": 1},
        "roster_context": _roster_context(home, away, summary),
    }


def test_build_fails_on_cross_matchup_duplicate_summaries(tmp_path, capsys):
    """Identical source_summaries text for two DIFFERENT matchups is boilerplate
    RSVG will downgrade — the build must fail closed so the operator researches
    each matchup instead of shipping unfixable entries."""
    slate = {
        "league": "MLB",
        "game_date": "2026-07-04",
        "games": [
            _game_item("Team A", "Team B", "Same boilerplate."),
            _game_item("Team C", "Team D", "Same boilerplate."),
        ],
    }
    slate_file = tmp_path / "slate.json"
    slate_file.write_text(json.dumps(slate))

    out_file = tmp_path / "out.json"
    assert main(["--slate", str(slate_file), "--output", str(out_file)]) == 1
    assert "reused verbatim" in capsys.readouterr().out


def test_build_allows_same_matchup_summary_inheritance(tmp_path):
    """A prop entry inheriting its own game's roster_context (same matchup) is
    legitimate reuse and must not fail the build."""
    slate = {
        "league": "MLB",
        "game_date": "2026-07-04",
        "games": [_game_item("Team A", "Team B", "A/B specific lineup notes.")],
        "props": [
            {
                "player_name": "Player One",
                "prop_type": "hits",
                "home_team": "Team A",
                "away_team": "Team B",
                "player_context": {"hits_mean": 1.1, "hits_std": 0.7},
                "game_context": {"is_playoff": False, "rest_days": 1},
            }
        ],
    }
    slate_file = tmp_path / "slate.json"
    slate_file.write_text(json.dumps(slate))

    out_file = tmp_path / "out.json"
    assert main(["--slate", str(slate_file), "--output", str(out_file)]) == 0
    entries = json.loads(out_file.read_text())
    assert len(entries) == 2
    # The prop inherited the game's roster_context for the same matchup.
    assert entries[1]["kind"] == "prop"
    assert entries[1]["roster_context"]["source_summaries"][0]["summary"] == (
        "A/B specific lineup notes."
    )


def test_build_fails_on_distinct_matchup_summaries_ok(tmp_path):
    """Distinct per-matchup summaries build cleanly (control case)."""
    slate = {
        "league": "MLB",
        "game_date": "2026-07-04",
        "games": [
            _game_item("Team A", "Team B", "A/B specific lineup notes."),
            _game_item("Team C", "Team D", "C/D specific lineup notes."),
        ],
    }
    slate_file = tmp_path / "slate.json"
    slate_file.write_text(json.dumps(slate))

    out_file = tmp_path / "out.json"
    assert main(["--slate", str(slate_file), "--output", str(out_file)]) == 0
    assert len(json.loads(out_file.read_text())) == 2
