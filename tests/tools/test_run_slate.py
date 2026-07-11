import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.run_slate import main

@patch("tools.run_slate.omega_run_batch")
@patch("tools.run_slate._audit")
def test_run_slate(mock_audit, mock_run_batch, tmp_path):
    entries = [{"kind": "game", "league": "MLB"}]
    entries_file = tmp_path / "entries.json"
    entries_file.write_text(json.dumps(entries))
    
    args = ["--entries", str(entries_file), "--session-id", "sess-123", "--allow-missing-sidecar"]
    
    mock_run_batch.return_value = {
        "status": "ok",
        "entries_total": 1,
        "entries_ok": 1,
        "entries_skipped": 0,
        "entries_error": 0,
        "results": [{"status": "ok", "identifier": "A @ B", "trace_id": "trace_1"}],
        "trace_ids": ["trace_1"]
    }
    
    assert main(args) == 0
    mock_run_batch.assert_called_once()
    
    result_file = tmp_path / "entries.result.json"
    assert result_file.exists()
    assert json.loads(result_file.read_text())["status"] == "ok"
