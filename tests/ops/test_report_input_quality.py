import pytest
from unittest.mock import patch, MagicMock

from omega.ops.report_input_quality import main


def test_report_input_quality(monkeypatch):
    monkeypatch.setattr("sys.argv", ["report_input_quality.py", "--league", "NFL"])

    mock_store = MagicMock()
    mock_store.return_value.get_recent_traces.return_value = [
        {"context_labels": ["thursday"], "market": {"game": {}}, "home_context": {"x": 1}, "away_context": {"y": 1}},
        {"market": {"game": {}}, "home_context": {"x": 1}, "away_context": {"y": 1}}, # missing context_labels
        {"context_labels": ["sunday"]}, # missing market, home, away
    ]

    with patch("omega.ops.report_input_quality.TraceStore", mock_store):
        assert main() == 0
