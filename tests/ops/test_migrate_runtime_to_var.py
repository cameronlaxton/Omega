"""Behavioral tests for the one-time root->var runtime migration helper."""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import omega.ops.migrate_runtime_to_var as M  # noqa: E402


def _seed(root: Path) -> Path:
    """Create legacy root inbox/ + reports/ with a few files. Returns runtime root."""
    (root / "inbox" / "traces" / "processed").mkdir(parents=True)
    (root / "inbox" / "sessions").mkdir(parents=True)
    (root / "inbox" / "closing_lines").mkdir(parents=True)  # empty -> just pruned
    (root / "reports" / "run_audits").mkdir(parents=True)
    (root / "inbox" / "traces" / "processed" / "a.json").write_text("A", encoding="utf-8")
    (root / "inbox" / "sessions" / "s1.json").write_text("S1", encoding="utf-8")
    (root / "reports" / "latest.md").write_text("ROOT", encoding="utf-8")
    (root / "reports" / "run_audits" / "x.audit.md").write_text("AUD", encoding="utf-8")
    return root / "var"


def _patch(monkeypatch, root: Path, runtime: Path) -> None:
    monkeypatch.setattr(M, "repo_root", lambda: root)
    monkeypatch.setattr(M, "runtime_root", lambda: runtime)


def test_dry_run_moves_nothing(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    runtime = _seed(root)
    _patch(monkeypatch, root, runtime)

    assert M.main([]) == 0
    assert (root / "inbox" / "sessions" / "s1.json").exists()
    assert not (runtime / "inbox" / "sessions" / "s1.json").exists()


def test_apply_moves_and_deletes_root(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    runtime = _seed(root)
    _patch(monkeypatch, root, runtime)

    assert M.main(["--apply"]) == 0
    assert (runtime / "inbox" / "traces" / "processed" / "a.json").read_text(encoding="utf-8") == "A"
    assert (runtime / "inbox" / "sessions" / "s1.json").read_text(encoding="utf-8") == "S1"
    assert (runtime / "reports" / "latest.md").read_text(encoding="utf-8") == "ROOT"
    assert (runtime / "reports" / "run_audits" / "x.audit.md").read_text(encoding="utf-8") == "AUD"
    assert not (root / "inbox").exists()
    assert not (root / "reports").exists()
    # idempotent
    assert M.main(["--apply"]) == 0


def test_collision_keeps_newer_destination(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    runtime = _seed(root)
    (runtime / "reports").mkdir(parents=True, exist_ok=True)
    time.sleep(0.02)
    (runtime / "reports" / "latest.md").write_text("CANONICAL", encoding="utf-8")
    _patch(monkeypatch, root, runtime)

    assert M.main(["--apply"]) == 0
    # The newer canonical copy is kept; the stale root copy is dropped.
    assert (runtime / "reports" / "latest.md").read_text(encoding="utf-8") == "CANONICAL"
    assert not (root / "reports").exists()


def test_apply_removes_readonly_dirs(tmp_path, monkeypatch):
    # Reproduces the Windows WinError-5 case: a read-only dir attribute blocks
    # rmdir. On POSIX this attribute does not block rmdir, so the test still
    # passes there (it just exercises the chmod path harmlessly).
    root = tmp_path / "repo"
    runtime = _seed(root)
    os.chmod(root / "inbox" / "traces" / "processed", stat.S_IREAD)
    _patch(monkeypatch, root, runtime)

    assert M.main(["--apply"]) == 0
    assert not (root / "inbox").exists()


def test_refuses_when_runtime_equals_repo_root(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _seed(root)
    _patch(monkeypatch, root, root)  # unsafe: runtime == repo root

    assert M.main(["--apply"]) == 1
    assert (root / "inbox").exists()  # nothing migrated
