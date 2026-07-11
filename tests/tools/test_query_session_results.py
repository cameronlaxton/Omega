import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from unittest.mock import patch

from tools.query_session_results import main

@patch("tools.query_session_results.TraceStore")
def test_query_session_results(mock_store, tmp_path):
    mock_instance = mock_store.return_value
    mock_instance.conn.execute.return_value = [
        ("trace_1", "MLB", json.dumps({
            "kind": "game",
            "input_snapshot": {"home_team": "A", "away_team": "B"},
            "result": {"status": "success", "best_bet": {"selection": "A ML", "edge": 0.05}}
        }))
    ]
    
    json_out = tmp_path / "out.json"
    args = ["--session-id", "sess-123", "--json-out", str(json_out)]
    
    assert main(args) == 0
    
    out_data = json.loads(json_out.read_text())
    assert len(out_data) == 1
    assert out_data[0]["trace_id"] == "trace_1"
