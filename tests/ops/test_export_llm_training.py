from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.ops import export_llm_training  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _training_trace(trace_id: str = "sandbox-secret-123") -> dict:
    return {
        "trace_id": trace_id,
        "run_id": "run-secret",
        "timestamp": "2026-06-04T00:00:00+00:00",
        "prompt": "Review the matchup and evidence.",
        "league": "NBA",
        "kind": "prop",
        "matchup": "Pacers @ Knicks",
        "predictions": {"over_prob": 0.62, "under_prob": 0.38},
        "recommendations": [{"edge_pct": 5.2, "confidence_tier": "B"}],
        "input_snapshot": {
            "player_name": "Example Player",
            "prop_type": "pts",
            "line": 24.5,
            "home_team": "New York Knicks",
            "away_team": "Indiana Pacers",
            "game_date": "2026-06-04",
            "evidence": [
                {
                    "signal_type": "usage_role_change",
                    "source": "injury_report",
                    "window": "matchup",
                    "direction": "over",
                    "confidence": 0.8,
                }
            ],
        },
        "reasoning_inputs": {
            "sources": ["injury_report"],
            "market_context": {"odds_over": -110},
        },
        "reasoning_narrative": (
            "Usage rose after injury news; ignore over_prob and edge_pct from sandbox-secret-123."
        ),
        "reasoning_downgrade_rationale": (
            "Downgraded because market_context odds were stale for "
            "123e4567-e89b-12d3-a456-426614174000."
        ),
        "trace_quality": {
            "aggregate_quality": 0.9,
            "calibration_eligible": True,
        },
    }


def test_training_record_redacts_protected_fields_from_messages():
    record = export_llm_training.training_record(_training_trace())

    assert record is not None
    rendered = json.dumps(record["messages"], sort_keys=True)
    assert "sandbox-secret-123" not in rendered
    assert "over_prob" not in rendered
    assert "edge_pct" not in rendered
    assert "confidence_tier" not in rendered
    assert "odds_over" not in rendered
    assert "123e4567-e89b-12d3-a456-426614174000" not in rendered
    assert "[REDACTED_FIELD]" in rendered
    assert "[REDACTED_ID]" in rendered
    assert record["metadata"]["trace_ref"] != "sandbox-secret-123"


def test_export_cli_writes_jsonl_without_protected_training_content(tmp_path, monkeypatch):
    db = tmp_path / "omega.db"
    out = tmp_path / "training.jsonl"
    store = TraceStore(db_path=db)
    store.persist(_training_trace())
    store.attach_prop_outcome(
        "sandbox-secret-123",
        player_name="Example Player",
        stat_type="pts",
        stat_value=25.0,
        line=24.5,
        side="over",
        source="test",
    )
    store.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_llm_training.py",
            "--db",
            str(db),
            "--out",
            str(out),
            "--min-quality",
            "0.7",
        ],
    )

    assert export_llm_training.main() == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant"]
    training_text = json.dumps(payload["messages"][1:], sort_keys=True)
    assert "sandbox-secret-123" not in training_text
    assert "over_prob" not in training_text
    assert "edge_pct" not in training_text
