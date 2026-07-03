"""
Tests for omega-ingest-traces — sandbox export ingestion.

Covers:
- Export-block (shape A): wrapped trace + bet_record
- Raw analyze() output (shape B): top-level trace_id/kind, no wrapper
- Bad JSON / missing trace_id → moved to failed/ with sibling .error.txt
- Idempotent re-run on processed/ files (no duplicate row)
- new bet records require selection_descriptor; legacy exports can infer it
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.ops import ingest_traces  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_analyze_out(trace_id: str = "sandbox-abc123", kind: str = "prop") -> dict[str, Any]:
    """A minimal but realistic core analyze() return value."""
    if kind == "prop":
        return {
            "trace_id": trace_id,
            "model_version": "omega-core-phase6h",
            "ran_at": "2026-05-14T19:23:11Z",
            "kind": "prop",
            "input_snapshot": {
                "player_name": "Jayson Tatum",
                "league": "NBA",
                "prop_type": "pts",
                "line": 27.5,
                "odds_over": -115,
                "odds_under": -105,
                "home_team": "Miami Heat",
                "away_team": "Boston Celtics",
                "game_date": "2026-05-14",
                "seed": 42,
            },
            "result": {
                "player_name": "Jayson Tatum",
                "league": "NBA",
                "prop_type": "pts",
                "line": 27.5,
                "status": "success",
                "over_prob": 0.56,
                "under_prob": 0.44,
                "edge_over": 4.2,
                "edge_under": -3.1,
                "recommendation": "over",
                "confidence_tier": "B",
            },
            "quality_gate": {
                "applied": True,
                "aggregate_quality": 0.82,
                "data_completeness": {},
                "downgrades": [],
            },
        }
    # game
    return {
        "trace_id": trace_id,
        "model_version": "omega-core-phase6h",
        "ran_at": "2026-05-14T19:23:11Z",
        "kind": "game",
        "input_snapshot": {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "league": "NBA",
            "odds": {"moneyline_home": -150, "moneyline_away": 130},
            "seed": 42,
        },
        "result": {
            "matchup": "Celtics @ Lakers",
            "league": "NBA",
            "status": "success",
            "simulation": {"home_win_prob": 58.0, "away_win_prob": 42.0},
            "edges": [],
            "best_bet": None,
        },
        "quality_gate": {
            "applied": True,
            "aggregate_quality": 0.85,
            "data_completeness": {},
            "downgrades": [],
        },
    }


def _make_export_block(trace_id: str = "sandbox-abc123", with_bet: bool = True) -> dict[str, Any]:
    block = {
        "trace": _make_analyze_out(trace_id=trace_id, kind="prop"),
        "bet_record": None,
    }
    if with_bet:
        block["bet_record"] = {
            "book": "DraftKings",
            "market": "player_prop:pts",
            "selection": "Tatum Over 27.5 pts",
            "selection_descriptor": "Tatum_over_27.5_pts",
            "line_taken": 27.5,
            "odds_taken": -115,
            "stake_units": 1.0,
            "decision_timestamp": "2026-05-14T19:25:00Z",
        }
    return block


def _make_legacy_export_block(trace_id: str = "sandbox-legacy") -> dict[str, Any]:
    block = _make_export_block(trace_id=trace_id, with_bet=True)
    block["bet_record"].pop("selection_descriptor")
    block["clv_capture_instructions"] = {
        "selection_descriptor": "Tatum_over_27.5_pts",
    }
    return block


@pytest.fixture()
def workspace(tmp_path: Path):
    """Provide an isolated inbox + db pair."""
    inbox = tmp_path / "inbox" / "traces"
    inbox.mkdir(parents=True)
    db_path = tmp_path / "test_traces.db"
    yield inbox, db_path


def _write_file(inbox: Path, name: str, payload: dict[str, Any]) -> Path:
    p = inbox / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _write_sidecar(
    sidecar_dir: Path,
    session_id: str,
    *,
    qa_failed: bool,
    qa_failed_trace_ids: list[str] | None = None,
    gate_ts: str = "2026-05-28T18:30:00Z",
) -> Path:
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "session_id": session_id,
        "opened_at": "2026-05-28T18:00:00Z",
        "closed_at": "2026-05-28T19:00:00Z",
        "model_version": "omega-core-phase6h",
        "purpose": "ingest gate test",
        "bankroll": 1000.0,
        "bankroll_confirmed": True,
        "exec_stats": {},
        "agent_notes": "",
        "audit_events": [],
    }
    if qa_failed:
        payload["audit_events"].append(
            {
                "ts": gate_ts,
                "event_type": "quality_gate",
                "step": "qa_failed_quarantine_0528",
                "status": "fail",
                "notes": "QA-failed quarantine",
                "trace_ids": qa_failed_trace_ids or [],
            }
        )
    path = sidecar_dir / f"{session_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


_ELIGIBLE_TQ = {
    "calibration_eligible": True,
    "context_source": "provided",
    "identity_status": "complete",
    "evidence_status": "present",
    "downgrades": [],
    "calibration_exclusion_reasons": [],
}


# ---------------------------------------------------------------------------
# Shape A: export block
# ---------------------------------------------------------------------------


class TestExportBlock:
    def test_round_trip_with_bet(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "sandbox-abc123.json", _make_export_block("sandbox-abc123"))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest_traces.py",
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
            ],
        )
        rc = ingest_traces.main()
        assert rc == 0

        store = TraceStore(db_path=str(db_path))
        retrieved = store.get_trace("sandbox-abc123")
        assert retrieved is not None
        assert retrieved["trace_id"] == "sandbox-abc123"
        assert retrieved["league"] == "NBA"
        assert retrieved["kind"] == "prop"

        bets = store.get_bet_records("sandbox-abc123")
        assert len(bets) == 1
        assert bets[0]["book"] == "DraftKings"
        assert bets[0]["selection_descriptor"] == "Tatum_over_27.5_pts"
        assert bets[0]["odds_taken"] == -115.0
        assert bets[0]["status"] == "pending"
        store.close()

        # File moved to processed/
        assert not (inbox / "sandbox-abc123.json").exists()
        assert (inbox / "processed" / "sandbox-abc123.json").exists()

    def test_round_trip_no_bet(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "sandbox-xyz.json", _make_export_block("sandbox-xyz", with_bet=False))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest_traces.py",
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
            ],
        )
        assert ingest_traces.main() == 0

        store = TraceStore(db_path=str(db_path))
        assert store.get_trace("sandbox-xyz") is not None
        assert store.get_bet_records("sandbox-xyz") == []
        store.close()

    def test_legacy_selection_descriptor_sibling_still_ingests(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "legacy.json", _make_legacy_export_block("sandbox-legacy"))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest_traces.py",
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
            ],
        )
        assert ingest_traces.main() == 0

        store = TraceStore(db_path=str(db_path))
        bets = store.get_bet_records("sandbox-legacy")
        assert len(bets) == 1
        assert bets[0]["selection_descriptor"] == "Tatum_over_27.5_pts"
        store.close()

    def test_new_bet_record_requires_selection_descriptor(self, workspace):
        inbox, db_path = workspace
        payload = _make_export_block("sandbox-missing-descriptor", with_bet=True)
        payload["bet_record"].pop("selection_descriptor")
        path = _write_file(inbox, "missing_descriptor.json", payload)

        store = TraceStore(db_path=str(db_path))
        with pytest.raises(ValueError, match="selection_descriptor"):
            ingest_traces.ingest_file(path, store)
        store.close()


# ---------------------------------------------------------------------------
# Shape B: raw analyze() output
# ---------------------------------------------------------------------------


class TestRawAnalyzeOutput:
    def test_raw_output_is_accepted(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "raw.json", _make_analyze_out("sandbox-raw1", kind="game"))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest_traces.py",
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
            ],
        )
        assert ingest_traces.main() == 0

        store = TraceStore(db_path=str(db_path))
        trace = store.get_trace("sandbox-raw1")
        assert trace is not None
        assert trace["league"] == "NBA"
        assert trace["matchup"] == "Celtics @ Lakers"
        store.close()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


class TestFailedFiles:
    def test_malformed_json_moves_to_failed(self, workspace, monkeypatch):
        inbox, db_path = workspace
        bad = inbox / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest_traces.py",
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
            ],
        )
        assert ingest_traces.main() == 0  # main returns 0 even with file-level failures

        assert not bad.exists()
        assert (inbox / "failed" / "bad.json").exists()
        assert (inbox / "failed" / "bad.json.error.txt").exists()

    def test_missing_trace_id_moves_to_failed(self, workspace, monkeypatch):
        inbox, db_path = workspace
        bad_payload = _make_export_block("sandbox-broken")
        bad_payload["trace"].pop("trace_id")
        _write_file(inbox, "no_id.json", bad_payload)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest_traces.py",
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
            ],
        )
        assert ingest_traces.main() == 0

        assert (inbox / "failed" / "no_id.json").exists()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotent:
    def test_double_ingest_same_trace(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "first.json", _make_export_block("sandbox-dup"))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest_traces.py",
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
            ],
        )
        assert ingest_traces.main() == 0

        # Drop the same trace_id again as a new inbox file
        _write_file(inbox, "second.json", _make_export_block("sandbox-dup"))
        assert ingest_traces.main() == 0

        store = TraceStore(db_path=str(db_path))
        # Only one row for this trace_id
        rows = store.conn.execute(
            "SELECT COUNT(*) AS n FROM traces WHERE trace_id = ?",
            ("sandbox-dup",),
        ).fetchone()
        assert rows["n"] == 1
        # Only one bet (UNIQUE on trace_id+market+selection_descriptor)
        bets = store.get_bet_records("sandbox-dup")
        assert len(bets) == 1
        store.close()


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_does_not_persist_or_move(self, workspace, monkeypatch):
        inbox, db_path = workspace
        path = _write_file(inbox, "dry.json", _make_export_block("sandbox-dry"))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest_traces.py",
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
                "--dry-run",
            ],
        )
        assert ingest_traces.main() == 0

        assert path.exists()  # not moved
        store = TraceStore(db_path=str(db_path))
        assert store.get_trace("sandbox-dry") is None  # not persisted
        store.close()


# ---------------------------------------------------------------------------
# QA-failed sidecar quarantine
# ---------------------------------------------------------------------------


class TestTraceScopedQaIngest:
    """Trace-scoped QA: valid artifacts are always persisted; QA fails persist
    audit-only (calibration-ineligible) and never block unrelated traces."""

    def _eligible_payload(self, trace_id: str, session_id: str) -> dict[str, Any]:
        payload = _make_export_block(trace_id, with_bet=False)
        payload["trace"]["session_id"] = session_id
        payload["trace"]["trace_quality"] = dict(_ELIGIBLE_TQ)
        return payload

    def test_unrelated_failed_quality_gate_does_not_block_later_valid_trace(
        self, workspace, tmp_path
    ):
        inbox, db_path = workspace
        session_id = "sess-mixed"
        sidecar_dir = tmp_path / "sessions"
        # Gate failed for some OTHER trace, not the one we ingest.
        _write_sidecar(
            sidecar_dir, session_id, qa_failed=True, qa_failed_trace_ids=["some-other-trace"]
        )

        payload = self._eligible_payload("sandbox-valid", session_id)
        path = _write_file(inbox, "valid.json", payload)

        store = TraceStore(db_path=str(db_path))
        trace_id, _ = ingest_traces.ingest_file(path, store, sidecar_dir=sidecar_dir)
        trace = store.get_trace(trace_id)
        assert trace is not None
        assert trace["trace_quality"]["calibration_eligible"] is True
        # Still calibration-eligible: unrelated failure did not condemn it.
        ids = {t["trace_id"] for t in store.query_traces(calibration_eligible_only=True)}
        assert trace_id in ids
        verdict = store.get_qa_verdict(trace_id)
        assert verdict["verdict"] == "pass"
        assert verdict["scope"] == "unrelated_session_failure"
        store.close()

    def test_trace_specific_qa_failed_trace_persists_as_audit_only(self, workspace, tmp_path):
        inbox, db_path = workspace
        session_id = "sess-qa"
        sidecar_dir = tmp_path / "sessions"
        _write_sidecar(
            sidecar_dir, session_id, qa_failed=True, qa_failed_trace_ids=["sandbox-qa-trace"]
        )

        payload = self._eligible_payload("sandbox-qa-trace", session_id)
        path = _write_file(inbox, "qa.json", payload)

        store = TraceStore(db_path=str(db_path))
        trace_id, _ = ingest_traces.ingest_file(path, store, sidecar_dir=sidecar_dir)
        # Ledger-preserved (audit-visible)...
        trace = store.get_trace(trace_id)
        assert trace is not None
        tq = trace["trace_quality"]
        assert tq["calibration_eligible"] is False
        assert "qa_failed" in tq["calibration_exclusion_reasons"]
        # ...with a recorded trace-scoped verdict.
        verdict = store.get_qa_verdict(trace_id)
        assert verdict["verdict"] == "fail"
        assert verdict["scope"] == "trace_id"
        store.close()

    def test_omitted_sidecar_dir_resolves_at_call_time(self, workspace, tmp_path, monkeypatch):
        inbox, db_path = workspace
        session_id = "sess-lazy"
        sidecar_dir = tmp_path / "lazy" / "sessions"
        _write_sidecar(
            sidecar_dir, session_id, qa_failed=True, qa_failed_trace_ids=["sandbox-lazy"]
        )
        monkeypatch.setattr(ingest_traces, "session_inbox_dir", lambda: sidecar_dir)

        payload = self._eligible_payload("sandbox-lazy", session_id)
        path = _write_file(inbox, "lazy.json", payload)

        store = TraceStore(db_path=str(db_path))
        trace_id, _ = ingest_traces.ingest_file(path, store)
        trace = store.get_trace(trace_id)
        assert trace is not None
        assert trace["trace_quality"]["calibration_eligible"] is False
        verdict = store.get_qa_verdict(trace_id)
        assert verdict["verdict"] == "fail"
        store.close()

    def test_explicit_none_sidecar_dir_still_disables_qa(self, workspace, tmp_path, monkeypatch):
        inbox, db_path = workspace
        session_id = "sess-none"
        sidecar_dir = tmp_path / "lazy" / "sessions"
        _write_sidecar(
            sidecar_dir, session_id, qa_failed=True, qa_failed_trace_ids=["sandbox-none"]
        )
        monkeypatch.setattr(ingest_traces, "session_inbox_dir", lambda: sidecar_dir)

        payload = self._eligible_payload("sandbox-none", session_id)
        path = _write_file(inbox, "none.json", payload)

        store = TraceStore(db_path=str(db_path))
        trace_id, _ = ingest_traces.ingest_file(path, store, sidecar_dir=None)
        trace = store.get_trace(trace_id)
        assert trace is not None
        assert trace["trace_quality"]["calibration_eligible"] is True
        assert store.get_qa_verdict(trace_id) is None
        store.close()

    def test_qa_failed_trace_is_not_calibration_eligible(self, workspace, tmp_path):
        inbox, db_path = workspace
        session_id = "sess-qa2"
        sidecar_dir = tmp_path / "sessions"
        _write_sidecar(sidecar_dir, session_id, qa_failed=True, qa_failed_trace_ids=["sandbox-qa2"])

        payload = self._eligible_payload("sandbox-qa2", session_id)
        path = _write_file(inbox, "qa2.json", payload)

        store = TraceStore(db_path=str(db_path))
        trace_id, _ = ingest_traces.ingest_file(path, store, sidecar_dir=sidecar_dir)
        assert store.query_traces(calibration_eligible_only=True) == []
        store.attach_prop_outcome(trace_id, "Jayson Tatum", "pts", 31.0, 27.5, "over")
        assert store.get_graded_traces() == []
        store.close()

    def test_force_ingest_qa_failed_does_not_mark_calibration_eligible(self, workspace, tmp_path):
        inbox, db_path = workspace
        session_id = "sess-qa3"
        sidecar_dir = tmp_path / "sessions"
        _write_sidecar(sidecar_dir, session_id, qa_failed=True, qa_failed_trace_ids=["sandbox-qa3"])

        payload = self._eligible_payload("sandbox-qa3", session_id)
        path = _write_file(inbox, "qa3.json", payload)

        store = TraceStore(db_path=str(db_path))
        # The deprecated force flag must not confer eligibility.
        trace_id, _ = ingest_traces.ingest_file(
            path, store, sidecar_dir=sidecar_dir, force_ingest_qa_failed=True
        )
        trace = store.get_trace(trace_id)
        assert trace is not None
        assert trace["trace_quality"]["calibration_eligible"] is False
        assert store.query_traces(calibration_eligible_only=True) == []
        store.close()

    def test_session_fallback_does_not_block_ledger_ingest(self, workspace, tmp_path):
        inbox, db_path = workspace
        session_id = "sess-fallback"
        sidecar_dir = tmp_path / "sessions"
        # Unstructured failed gate (no trace_ids): conservative session fallback.
        _write_sidecar(sidecar_dir, session_id, qa_failed=True)

        payload = self._eligible_payload("sandbox-fallback", session_id)
        path = _write_file(inbox, "fallback.json", payload)

        store = TraceStore(db_path=str(db_path))
        # Does NOT raise; the trace is preserved in the ledger.
        trace_id, _ = ingest_traces.ingest_file(path, store, sidecar_dir=sidecar_dir)
        trace = store.get_trace(trace_id)
        assert trace is not None
        assert trace["trace_quality"]["calibration_eligible"] is False
        verdict = store.get_qa_verdict(trace_id)
        assert verdict["verdict"] == "fail"
        assert verdict["scope"] == "session_fallback"
        store.close()


# ---------------------------------------------------------------------------
# Prop matchup derivation: with vs. without game identity
# ---------------------------------------------------------------------------


class TestPropMatchupDerivation:
    """Phase 6h: prop traces should denormalize matchup to the game pair when
    home_team/away_team are present on input_snapshot. This makes prop traces
    discoverable by the existing time-window + league query path used by
    omega-fetch-outcomes-props.
    """

    def test_prop_with_game_identity_denormalizes_to_game_pair(self):
        analyze_out = {
            "trace_id": "sandbox-prop-g1",
            "kind": "prop",
            "ran_at": "2026-05-14T19:00:00Z",
            "input_snapshot": {
                "player_name": "Jayson Tatum",
                "league": "NBA",
                "prop_type": "pts",
                "line": 27.5,
                "home_team": "Miami Heat",
                "away_team": "Boston Celtics",
                "game_date": "2026-05-14",
            },
            "result": {"status": "success"},
        }
        adapted = ingest_traces._adapt_sandbox_trace(analyze_out)
        assert adapted["matchup"] == "Boston Celtics @ Miami Heat"
        # Prop descriptor still recoverable from input_snapshot inside full_trace
        assert adapted["input_snapshot"]["player_name"] == "Jayson Tatum"
        assert adapted["input_snapshot"]["prop_type"] == "pts"

    def test_prop_without_game_identity_falls_back_to_descriptor(self):
        """Legacy prop traces without home/away keep the old descriptor form."""
        analyze_out = {
            "trace_id": "sandbox-prop-legacy",
            "kind": "prop",
            "ran_at": "2026-05-14T19:00:00Z",
            "input_snapshot": {
                "player_name": "Jayson Tatum",
                "league": "NBA",
                "prop_type": "pts",
                "line": 27.5,
            },
            "result": {},
        }
        adapted = ingest_traces._adapt_sandbox_trace(analyze_out)
        assert adapted["matchup"] == "Jayson Tatum pts 27.5"

    def test_game_kind_unchanged(self):
        """Game traces continue to denormalize matchup as before."""
        analyze_out = {
            "trace_id": "sandbox-game-1",
            "kind": "game",
            "ran_at": "2026-05-14T19:00:00Z",
            "input_snapshot": {
                "home_team": "Lakers",
                "away_team": "Celtics",
                "league": "NBA",
            },
            "result": {},
        }
        adapted = ingest_traces._adapt_sandbox_trace(analyze_out)
        assert adapted["matchup"] == "Celtics @ Lakers"


# ---------------------------------------------------------------------------
# Wrapped-payload sibling merge: reasoning/quality fields attached at the
# export-block top level are promoted into the inner trace before adaptation.
# ---------------------------------------------------------------------------


class TestTopLevelCompatMerge:
    def _wrapped_with_siblings(self) -> dict[str, Any]:
        block = _make_export_block("sandbox-sib-1", with_bet=False)
        block["reasoning_inputs"] = {
            "sources": ["espn", "rotowire"],
            "fields_gathered": ["pace", "rest"],
            "missing_fields": [],
        }
        block["reasoning_narrative"] = "Pace edge favors over."
        block["reasoning_downgrade_rationale"] = ""
        block["trace_quality"] = {
            "aggregate_quality": 0.88,
            "calibration_eligible": True,
            "identity_status": "complete",
            "context_source": "provided",
            "evidence_status": "present",
        }
        return block

    def test_sibling_fields_land_on_persistable_trace(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "sib.json", self._wrapped_with_siblings())

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest_traces.py",
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
            ],
        )
        assert ingest_traces.main() == 0

        store = TraceStore(db_path=str(db_path))
        trace = store.get_trace("sandbox-sib-1")
        assert trace is not None
        assert trace["reasoning_narrative"] == "Pace edge favors over."
        assert trace["reasoning_inputs"]["sources"] == ["espn", "rotowire"]
        assert trace["trace_quality"]["aggregate_quality"] == 0.88
        assert trace["trace_quality"]["calibration_eligible"] is True
        store.close()

    def test_inner_trace_value_wins_over_sibling(self):
        block = self._wrapped_with_siblings()
        # Conflicting inner value: trace already carries reasoning_narrative.
        block["trace"]["reasoning_narrative"] = "INNER WINS"
        block["reasoning_narrative"] = "sibling should lose"

        ingest_traces._merge_top_level_compat_fields(block)

        assert block["trace"]["reasoning_narrative"] == "INNER WINS"

    def test_export_schema_version_stays_wrapper_metadata(self):
        block = self._wrapped_with_siblings()
        block["export_schema_version"] = 2

        ingest_traces._merge_top_level_compat_fields(block)

        assert "export_schema_version" not in block["trace"]

    def test_raw_pattern_b_payloads_are_unaffected(self):
        raw = _make_analyze_out("sandbox-rawb-1", kind="prop")
        raw["reasoning_narrative"] = "already-nested"
        # Pattern B has no outer wrapper, so the merge is a no-op.
        payload = {"trace": raw, "bet_record": None}
        ingest_traces._merge_top_level_compat_fields(payload)
        assert payload["trace"]["reasoning_narrative"] == "already-nested"


class TestPrePersistExportGate:
    """The pre-persist export gate (omega/trace/export_validator.py).

    Default is lenient: its error set mirrors what ingest already rejected, so
    the default gate is zero-behavior-change. --strict (strict=True) adds fresh-
    export quality checks (e.g. NBA game_context). Structurally unsafe wrapper
    shapes fail before persist. The fix for a rejected export is to re-wrap/re-
    export — never to re-run analyze().
    """

    def test_lenient_default_preserves_current_behavior(self, workspace):
        inbox, db_path = workspace
        # A normal export with no game_context ingests fine under the default gate.
        path = _write_file(
            inbox, "lenient.json", _make_export_block("sandbox-lenient-1", with_bet=False)
        )
        store = TraceStore(db_path=str(db_path))
        trace_id, _ = ingest_traces.ingest_file(path, store, sidecar_dir=None)
        assert trace_id == "sandbox-lenient-1"
        assert store.get_trace("sandbox-lenient-1") is not None
        store.close()

    def test_strict_rejects_fresh_export_missing_game_context_before_persist(self, workspace):
        inbox, db_path = workspace
        # The prop export omits game_context (is_playoff/rest_days); strict mode
        # hard-fails NBA traces that lack it.
        path = _write_file(
            inbox, "strict.json", _make_export_block("sandbox-strict-1", with_bet=False)
        )
        store = TraceStore(db_path=str(db_path))
        with pytest.raises(ValueError, match="pre-persist validation"):
            ingest_traces.ingest_file(path, store, sidecar_dir=None, strict=True)
        # The gate fires before store.persist(): nothing was written.
        assert store.get_trace("sandbox-strict-1") is None
        store.close()

    def test_unsafe_wrapper_shape_fails_before_persist(self, workspace):
        inbox, db_path = workspace
        # Neither shape A ({"trace": ...}) nor shape B (top-level trace_id+kind).
        path = _write_file(inbox, "bad.json", {"not_a_trace": True})
        store = TraceStore(db_path=str(db_path))
        with pytest.raises(ValueError):
            ingest_traces.ingest_file(path, store, sidecar_dir=None)
        # No row created.
        n = store.conn.execute("SELECT COUNT(*) AS n FROM traces").fetchone()["n"]
        assert n == 0
        store.close()

    def test_gate_error_says_rewrap_not_rerun_analyze(self, workspace):
        inbox, db_path = workspace
        path = _write_file(
            inbox, "strict2.json", _make_export_block("sandbox-strict-2", with_bet=False)
        )
        store = TraceStore(db_path=str(db_path))
        with pytest.raises(ValueError) as exc:
            ingest_traces.ingest_file(path, store, sidecar_dir=None, strict=True)
        msg = str(exc.value).lower()
        assert "re-wrap" in msg or "re-drop" in msg
        assert "do not re-run analyze" in msg
        store.close()


# ---------------------------------------------------------------------------
# LLM interface contract: reasoning surface survives export -> ingest -> store
# ---------------------------------------------------------------------------


class TestReasoningSurfaceRoundTrip:
    def test_reasoning_fields_survive_ingest_into_full_trace(self, workspace, monkeypatch):
        """reasoning_presentation, downgrade rationale, evidence, and the RSVG
        gate verdict must ride the export block through ingest into the stored
        full_trace unchanged — typed and trace-persisted, not prompt-only."""
        inbox, db_path = workspace
        block = _make_export_block("sandbox-reason-rt", with_bet=False)
        trace = block["trace"]
        trace["session_id"] = "sess-reason-rt"
        trace["input_snapshot"]["evidence"] = [
            {
                "signal_type": "usage_role_change",
                "category": "situational",
                "plane": "game",
                "value": "key_absence:out",
                "source": "injury_report",
                "confidence": 0.9,
                "window": "matchup",
                "direction": "away",
                "note": "RSVG key absence: Star A (out, Miami Heat)",
            }
        ]
        trace["reasoning_presentation"] = {
            "thesis": "RSVG-verified matchup.",
            "market_read": None,
            "why": "Lineups confirmed; one key absence verified.",
            "risks": "Late scratch risk.",
            "verdict": "RESEARCH_CANDIDATE — no formal actionable output.",
        }
        trace["reasoning_downgrade_rationale"] = "RSVG research_candidate: context stale"
        trace["reasoning_inputs"] = {
            "sources": ["espn.com"],
            "fields_gathered": ["evidence"],
            "missing_fields": [],
        }
        trace["trace_quality"] = {
            "aggregate_quality": 0.7,
            "rsvg": {"gate": "rsvg", "status": "research_candidate", "schema_version": 1},
        }

        _write_file(inbox, "sandbox-reason-rt.json", block)
        monkeypatch.setattr(
            sys, "argv", ["ingest_traces.py", "--inbox", str(inbox), "--db", str(db_path)]
        )
        assert ingest_traces.main() == 0

        store = TraceStore(db_path=str(db_path))
        stored = store.get_trace("sandbox-reason-rt")
        assert stored is not None
        assert stored["reasoning_presentation"]["verdict"].startswith("RESEARCH_CANDIDATE")
        assert stored["reasoning_presentation"]["thesis"] == "RSVG-verified matchup."
        assert stored["reasoning_downgrade_rationale"] == "RSVG research_candidate: context stale"
        assert stored["reasoning_inputs"]["sources"] == ["espn.com"]
        assert stored["trace_quality"]["rsvg"]["status"] == "research_candidate"
        assert stored["input_snapshot"]["evidence"][0]["signal_type"] == "usage_role_change"
        store.close()
