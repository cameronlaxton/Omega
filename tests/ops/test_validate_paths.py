"""Regression tests pinning the validator default inbox paths.

The session-sidecar / trace inboxes are canonically under ``var/`` (see
docs/phase6/ARTIFACT_AUTHORITY.md). A default that drops the ``var/`` segment
makes ``omega-validate-session-sidecars`` / ``omega-validate-all`` validate a
stale/empty directory and report a false green while the live sidecars go
unchecked. These tests fail if that regresses.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from omega.ops import validate_all, validate_session_sidecars  # noqa: E402
from omega.trace.session_sidecar import bootstrap_payload, create_sidecar  # noqa: E402


def test_validate_session_sidecars_default_is_var_inbox():
    default = validate_session_sidecars._DEFAULT_SESSIONS_INBOX
    assert default.parts[-3:] == ("var", "inbox", "sessions"), default


def test_validate_all_defaults_are_var_inbox():
    assert validate_all._DEFAULT_SESSIONS_INBOX.parts[-3:] == ("var", "inbox", "sessions")
    assert validate_all._DEFAULT_TRACES.parts[-3:] == ("var", "inbox", "traces")


def test_validate_all_default_matches_subvalidator_default():
    # The orchestrator and the sub-validator must agree on where sidecars live.
    assert validate_all._DEFAULT_SESSIONS_INBOX == validate_session_sidecars._DEFAULT_SESSIONS_INBOX


def test_validate_directory_counts_valid_and_invalid(tmp_path):
    # Behavioral smoke: a valid sidecar counts valid; a malformed one counts invalid.
    good = tmp_path / "sess-20260607-good.json"
    create_sidecar(
        good,
        bootstrap_payload("sess-20260607-good", model_version="m", purpose="p", bankroll=100.0),
    )
    (tmp_path / "sess-20260607-bad.json").write_text("{not valid json", encoding="utf-8")

    valid, invalid, warned = validate_session_sidecars.validate_directory(tmp_path)
    assert (valid, invalid) == (1, 1)
    assert warned == 0  # the good sidecar has no mirror/close inconsistency


def test_validate_directory_warns_on_mirror_count_mismatch(tmp_path):
    # A structurally valid sidecar whose JSONL mirror has more lines than the JSON
    # summary's audit_events (the F2/F3 drift signature) is WARNed, not counted
    # invalid, and never fails the run.
    from omega.trace.session_sidecar import append_audit_events

    good = tmp_path / "sess-20260607-drift.json"
    create_sidecar(
        good,
        bootstrap_payload("sess-20260607-drift", model_version="m", purpose="p", bankroll=100.0),
    )
    append_audit_events(
        good,
        [
            {
                "ts": "2026-06-07T00:00:00Z",
                "event_type": "note",
                "step": "x",
                "status": "ok",
                "trace_ids": [],
            }
        ],
    )
    # Simulate drift: append an extra line to the mirror only.
    mirror = good.with_suffix(".events.jsonl")
    with mirror.open("a", encoding="utf-8") as fh:
        fh.write('{"ts": "2026-06-07T00:00:01Z", "event_type": "note", "step": "y", "status": "ok"}\n')

    valid, invalid, warned = validate_session_sidecars.validate_directory(tmp_path)
    assert (valid, invalid, warned) == (1, 0, 1)


def test_validate_all_warns_for_root_inbox_even_when_var_paths_are_clean(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "repo"
    (root / "inbox").mkdir(parents=True)
    sessions = root / "var" / "inbox" / "sessions"
    traces = root / "var" / "inbox" / "traces"
    sessions.mkdir(parents=True)
    traces.mkdir(parents=True)

    monkeypatch.setattr(validate_all, "repo_root", lambda: root)
    monkeypatch.setattr(
        validate_all,
        "_run_module",
        lambda name, module, *args: validate_all.StepResult(name, "PASS", "stub"),
    )

    code = validate_all.main(
        ["--skip-tests", "--sessions-inbox", str(sessions), "--traces", str(traces)]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "WARN  legacy-runtime-dirs" in out
    assert str(root / "inbox") in out
    assert "SUCCESS: all executed checks passed." in out


def test_validate_all_warns_for_root_reports_even_when_var_paths_are_clean(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "repo"
    (root / "reports").mkdir(parents=True)
    sessions = root / "var" / "inbox" / "sessions"
    traces = root / "var" / "inbox" / "traces"
    sessions.mkdir(parents=True)
    traces.mkdir(parents=True)

    monkeypatch.setattr(validate_all, "repo_root", lambda: root)
    monkeypatch.setattr(
        validate_all,
        "_run_module",
        lambda name, module, *args: validate_all.StepResult(name, "PASS", "stub"),
    )

    code = validate_all.main(
        ["--skip-tests", "--sessions-inbox", str(sessions), "--traces", str(traces)]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "WARN  legacy-runtime-dirs" in out
    assert str(root / "reports") in out
    assert "SUCCESS: all executed checks passed." in out
