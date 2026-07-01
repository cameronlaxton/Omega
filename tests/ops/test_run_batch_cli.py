"""omega-run-batch: thin CLI over the single omega_run_batch implementation.

These tests cover the CLI's own plumbing (entries loading, exit-code mapping)
without re-running the engine — the batch execution itself is the MCP tool's
tested code path, delegated to here.
"""

from __future__ import annotations

import json

import pytest

from omega.ops import run_batch


def _write(tmp_path, obj):
    p = tmp_path / "entries.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_load_entries_accepts_top_level_list(tmp_path):
    p = _write(tmp_path, [{"kind": "game"}, {"kind": "prop"}])
    assert run_batch._load_entries(p) == [{"kind": "game"}, {"kind": "prop"}]


def test_load_entries_accepts_entries_object(tmp_path):
    p = _write(tmp_path, {"entries": [{"kind": "game"}], "bankroll": 500})
    assert run_batch._load_entries(p) == [{"kind": "game"}]


def test_load_entries_rejects_bad_shapes(tmp_path):
    with pytest.raises(ValueError):
        run_batch._load_entries(_write(tmp_path, {"no_entries": 1}))
    with pytest.raises(ValueError):
        run_batch._load_entries(_write(tmp_path, [1, 2, 3]))


def test_main_returns_1_on_unparseable_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(run_batch, "_check_preflight_sentinel", lambda: None)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    rc = main_with_stub(monkeypatch, tmp_path, entries_path=bad, fake=None)
    assert rc == 1


def test_main_returns_2_when_sentinel_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_batch, "_check_preflight_sentinel", lambda: "truncated!")
    p = _write(tmp_path, [{"kind": "game"}])
    rc = run_batch.main(
        ["--entries-json", str(p), "--session-id", "sess-x", "--bankroll", "1000"]
    )
    assert rc == 2


@pytest.mark.parametrize(
    "envelope,expected_rc",
    [
        ({"status": "ok"}, 0),
        ({"status": "partial"}, 0),
        ({"status": "error"}, 1),
        ({"status": "error", "error_code": "formal_output_blocked"}, 2),
    ],
)
def test_main_maps_envelope_status_to_exit_code(
    tmp_path, monkeypatch, envelope, expected_rc
):
    monkeypatch.setattr(run_batch, "_check_preflight_sentinel", lambda: None)
    p = _write(tmp_path, [{"kind": "game"}])
    rc = main_with_stub(monkeypatch, tmp_path, entries_path=p, fake=lambda *a, **kw: envelope)
    assert rc == expected_rc


def main_with_stub(monkeypatch, tmp_path, *, entries_path, fake):
    """Run main() with omega_run_batch stubbed so the engine is never invoked."""
    import types

    stub = types.ModuleType("omega.mcp.server")
    stub.omega_run_batch = fake if fake is not None else (lambda *a, **kw: {"status": "ok"})
    monkeypatch.setitem(__import__("sys").modules, "omega.mcp.server", stub)
    return run_batch.main(
        ["--entries-json", str(entries_path), "--session-id", "sess-x", "--bankroll", "1000"]
    )
