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
