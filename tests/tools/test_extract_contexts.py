import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.extract_contexts import main, _stamp_ages

def test_stamp_ages():
    from datetime import datetime, timezone, timedelta
    now = datetime(2026, 7, 4, tzinfo=timezone.utc)
    provenance = {
        "recent": {"trace_timestamp": (now - timedelta(days=5)).isoformat()},
        "stale": {"trace_timestamp": (now - timedelta(days=40)).isoformat()},
    }
    stale = _stamp_ages(provenance, now, 30.0)
    assert stale == ["stale"]
    assert provenance["recent"]["age_days"] == 5.0
    assert provenance["stale"]["age_days"] == 40.0

@patch("tools.extract_contexts.TraceStore")
def test_extract_contexts_main(mock_store, tmp_path):
    mock_instance = mock_store.return_value
    mock_instance.db_path = "fake.db"
    
    mock_instance.conn.execute.return_value = [
        ("trace_1", json.dumps({
            "timestamp": "2026-07-04T12:00:00Z",
            "kind": "game",
            "result": {"status": "success"},
            "input_snapshot": {
                "home_team": "Team A",
                "away_team": "Team B",
                "home_context": {"power": 1, "weather_wind_mph": 10},
                "away_context": {"power": 2}
            }
        }))
    ]
    
    out_dir = tmp_path / "packs"
    args = ["--league", "MLB", "--kind", "team", "--output-dir", str(out_dir)]
    
    assert main(args) == 0
    
    pack_file = out_dir / "mlb_team_contexts.json"
    assert pack_file.exists()
    pack = json.loads(pack_file.read_text())
    assert "Team A" in pack["contexts"]
    assert "weather_wind_mph" not in pack["contexts"]["Team A"]
    assert pack["contexts"]["Team A"]["power"] == 1
