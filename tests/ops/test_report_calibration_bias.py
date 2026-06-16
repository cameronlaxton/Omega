from unittest.mock import MagicMock, patch

from omega.ops.report_calibration_bias import main


def test_report_calibration_bias(monkeypatch):
    monkeypatch.setattr("sys.argv", ["report_calibration_bias.py", "--league", "NBA", "--min-samples", "1"])

    mock_store = MagicMock()
    mock_store.return_value.get_recent_traces.return_value = [
        {"context_labels": ["playoff"], "market": {"game": {"home_prob": 0.6, "away_prob": 0.4}}, "outcome": {"home_win": 1}},
        {"context_labels": ["playoff"], "market": {"game": {"home_prob": 0.8, "away_prob": 0.2}}, "outcome": {"home_win": 1}},
        {"context_labels": ["back_to_back"], "market": {"game": {"home_prob": 0.5, "away_prob": 0.5}}, "outcome": {"home_win": 0}},
    ]

    with patch("omega.ops.report_calibration_bias.TraceStore", mock_store):
        # We also need to mock fitter.extract_pairs because the dummy traces aren't fully formed
        with patch(
            "omega.core.calibration.fitter.CalibrationFitter.extract_pairs",
            return_value=([0.6, 0.8, 0.5], [1, 1, 0]),
        ):
            assert main() == 0
