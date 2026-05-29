from __future__ import annotations

import errno
import importlib.metadata
import subprocess
from pathlib import Path

from scripts import cowork_preflight


def test_python_below_310_fails():
    failures = cowork_preflight.check_python((3, 9, 18))

    assert len(failures) == 1
    assert "Python 3.10+ is required" in failures[0]


def test_missing_distribution_points_to_editable_mcp_install(monkeypatch):
    def fake_version(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(cowork_preflight.importlib.metadata, "version", fake_version)

    failures = cowork_preflight.check_distribution()

    assert failures == [
        "Omega is not installed in this interpreter. Run: python -m pip install -e .[mcp]"
    ]


def test_direct_only_skips_mcp_import(monkeypatch):
    imported: list[str] = []

    monkeypatch.setattr(cowork_preflight, "check_python", lambda: [])
    monkeypatch.setattr(cowork_preflight, "check_distribution", lambda: [])
    monkeypatch.setattr(cowork_preflight, "check_omega_import_binding", lambda _repo: [])
    monkeypatch.setattr(cowork_preflight, "repair_source_integrity", lambda _repo: [])
    monkeypatch.setattr(cowork_preflight, "verify_against_git", lambda _repo: [])

    def fake_check_import(module: str, install_hint: str) -> list[str]:
        imported.append(module)
        return []

    monkeypatch.setattr(cowork_preflight, "check_import", fake_check_import)

    failures = cowork_preflight.run_checks(require_mcp=False)

    assert failures == []
    assert "mcp.server.fastmcp" not in imported
    assert "omega.core.contracts.service" in imported


def test_omega_import_binding_fails_when_import_resolves_outside_repo(monkeypatch, tmp_path):
    from types import SimpleNamespace

    wrong = tmp_path / "other" / "omega" / "__init__.py"
    wrong.parent.mkdir(parents=True)
    wrong.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        cowork_preflight.importlib,
        "import_module",
        lambda _module: SimpleNamespace(__file__=str(wrong)),
    )

    failures = cowork_preflight.check_omega_import_binding(tmp_path / "repo")

    assert len(failures) == 1
    assert "Omega import binding mismatch" in failures[0]
    assert str(wrong.resolve()) in failures[0]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    service = repo / "omega" / "core" / "contracts" / "service.py"
    service.parent.mkdir(parents=True)
    service.write_text("VALUE = 1\n", encoding="utf-8")
    other = repo / "other.py"
    other.write_text("VALUE = 1\n", encoding="utf-8")
    test_file = repo / "tests" / "core" / "test_thing.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    return repo


def test_repair_from_git_restores_syntax_corrupt_tracked_file(tmp_path):
    repo = _init_repo(tmp_path)
    service = repo / "omega" / "core" / "contracts" / "service.py"
    service.write_text("def broken(:\n", encoding="utf-8")

    failures = cowork_preflight.verify_against_git(repo, repair=True)

    assert failures == []
    assert service.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_repair_from_git_syncs_and_reparses(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)
    service = repo / "omega" / "core" / "contracts" / "service.py"
    service.write_text("def broken(:\n", encoding="utf-8")
    synced = []
    parsed = []

    monkeypatch.setattr(cowork_preflight.os, "sync", lambda: synced.append(True), raising=False)

    def fake_parse(path: Path):
        parsed.append(path.name)
        return None

    monkeypatch.setattr(cowork_preflight, "_ast_parse_file", fake_parse)

    failures = cowork_preflight.verify_against_git(repo, repair=True)

    assert failures == []
    assert synced == [True]
    assert "service.py" in parsed


def test_repair_from_git_blocks_dirty_non_target_file(tmp_path):
    repo = _init_repo(tmp_path)
    service = repo / "omega" / "core" / "contracts" / "service.py"
    other = repo / "other.py"
    service.write_text("def broken(:\n", encoding="utf-8")
    other.write_text("VALUE = 2\n", encoding="utf-8")

    failures = cowork_preflight.verify_against_git(repo, repair=True)

    assert len(failures) == 1
    assert "Refusing to --repair-from-git" in failures[0]
    assert "other.py" in failures[0]
    assert service.read_text(encoding="utf-8") == "def broken(:\n"


def test_force_repair_restores_all_divergent_tracked_python(tmp_path):
    repo = _init_repo(tmp_path)
    service = repo / "omega" / "core" / "contracts" / "service.py"
    other = repo / "other.py"
    service.write_text("def broken(:\n", encoding="utf-8")
    other.write_text("VALUE = 2\n", encoding="utf-8")

    failures = cowork_preflight.verify_against_git(repo, repair=True, force_repair=True)

    assert failures == []
    assert service.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert other.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_repair_from_git_proceeds_when_only_tests_are_dirty(tmp_path):
    """BUG-PREFLIGHT-3: a dirty test file must not block repair of a corrupt
    core source file. Core tier is restored; the test edit is left alone."""
    repo = _init_repo(tmp_path)
    service = repo / "omega" / "core" / "contracts" / "service.py"
    test_file = repo / "tests" / "core" / "test_thing.py"
    service.write_text("def broken(:\n", encoding="utf-8")
    test_file.write_text("VALUE = 2\n", encoding="utf-8")

    failures = cowork_preflight.verify_against_git(repo, repair=True)

    assert failures == []
    assert service.read_text(encoding="utf-8") == "VALUE = 1\n"
    # The dirty test edit is preserved — we deliberately did not touch it.
    assert test_file.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_non_repair_only_test_divergence_is_warning_not_failure(tmp_path, capsys):
    """When the core tier is clean and only tests/ diverge, verify_against_git
    returns no failures but logs a warning line on stdout."""
    repo = _init_repo(tmp_path)
    test_file = repo / "tests" / "core" / "test_thing.py"
    test_file.write_text("VALUE = 2\n", encoding="utf-8")

    failures = cowork_preflight.verify_against_git(repo)

    assert failures == []
    captured = capsys.readouterr().out
    assert "tests/core/test_thing.py" in captured
    assert "not blocking" in captured


def test_non_repair_non_critical_core_divergence_is_warning_not_failure(tmp_path, capsys):
    """Non-critical core files with intentional edits emit a warning but do
    not cause verify_against_git to return failures in the non-repair path."""
    repo = _init_repo(tmp_path)
    other = repo / "other.py"
    other.write_text("VALUE = 2\n", encoding="utf-8")  # intentional edit

    failures = cowork_preflight.verify_against_git(repo)

    assert failures == []
    captured = capsys.readouterr().out
    assert "other.py" in captured
    assert "[warning]" in captured


def test_dirty_core_still_blocks_repair_when_not_a_target(tmp_path):
    """The original block-on-dirty-non-target behavior is preserved for the
    core tier (other.py is at repo root, treated as core)."""
    repo = _init_repo(tmp_path)
    service = repo / "omega" / "core" / "contracts" / "service.py"
    other = repo / "other.py"
    service.write_text("def broken(:\n", encoding="utf-8")
    other.write_text("VALUE = 2\n", encoding="utf-8")

    failures = cowork_preflight.verify_against_git(repo, repair=True)

    assert len(failures) == 1
    assert "Refusing to --repair-from-git" in failures[0]
    assert "other.py" in failures[0]
    # Service.py left in its corrupt state because the guard blocked.
    assert service.read_text(encoding="utf-8") == "def broken(:\n"


def test_split_tiers_separates_tests_from_core():
    core, test = cowork_preflight._split_tiers(
        [
            "omega/core/contracts/service.py",
            "tests/core/test_engine.py",
            "scripts/run_analyze.py",
            "tests/scripts/test_cowork_preflight.py",
            "OMEGA_COWORK.md",
        ]
    )
    assert core == [
        "omega/core/contracts/service.py",
        "scripts/run_analyze.py",
        "OMEGA_COWORK.md",
    ]
    assert test == [
        "tests/core/test_engine.py",
        "tests/scripts/test_cowork_preflight.py",
    ]


def test_clean_stale_bytecode_summarizes_unlink_failures(monkeypatch, tmp_path):
    pyc = tmp_path / "module.pyc"
    pyc.write_bytes(b"bytecode")

    def fake_unlink(_path: Path) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    errors = cowork_preflight.clean_stale_bytecode(tmp_path)

    assert errors == ["Skipped 1 stale .pyc file(s) (EPERM); likely host-locked."]


def test_formal_output_gate_requires_clean_git_parity(monkeypatch, tmp_path):
    monkeypatch.setattr(cowork_preflight, "run_checks", lambda **_kwargs: [])
    monkeypatch.setattr(cowork_preflight, "_tracked_python_files", lambda _repo: (["a.py"], []))
    monkeypatch.setattr(cowork_preflight, "_diverged_tracked_files", lambda _repo, _files: ["a.py"])

    failures = cowork_preflight.run_formal_output_gate(repo_root=tmp_path, require_mcp=False)

    assert len(failures) == 1
    assert "clean source parity" in failures[0]
    assert "a.py" in failures[0]


def test_formal_output_gate_runs_smoke_when_clean(monkeypatch, tmp_path):
    smoke_called = []

    monkeypatch.setattr(cowork_preflight, "run_checks", lambda **_kwargs: [])
    monkeypatch.setattr(cowork_preflight, "_tracked_python_files", lambda _repo: ([], []))
    monkeypatch.setattr(cowork_preflight, "_diverged_tracked_files", lambda _repo, _files: [])
    monkeypatch.setattr(
        cowork_preflight,
        "check_formal_output_smoke",
        lambda: smoke_called.append(True) or [],
    )

    failures = cowork_preflight.run_formal_output_gate(repo_root=tmp_path, require_mcp=False)

    assert failures == []
    assert smoke_called == [True]


# ---------------------------------------------------------------------------
# Sentinel check tests (Deliverable 1)
# ---------------------------------------------------------------------------

def test_sentinel_present_in_script():
    """The EOF sentinel must be the final non-empty content in cowork_preflight.py."""
    script = Path(__file__).parent.parent.parent / "scripts" / "cowork_preflight.py"
    text = script.read_text(encoding="utf-8")
    assert cowork_preflight._PREFLIGHT_SENTINEL in text, (
        "cowork_preflight.py is missing its EOF sentinel. "
        "The file may have been truncated."
    )


def test_sentinel_missing_is_detectable(tmp_path):
    """A truncated preflight script (missing sentinel) is detectable via raw text check."""
    truncated = tmp_path / "cowork_preflight.py"
    truncated.write_text("def main():\n    pass\n", encoding="utf-8")
    text = truncated.read_text(encoding="utf-8")
    assert cowork_preflight._PREFLIGHT_SENTINEL not in text


# ---------------------------------------------------------------------------
# Empty stdout guard: run_checks always emits stdout before returning
# ---------------------------------------------------------------------------

def test_run_checks_is_not_silent_on_success(monkeypatch, tmp_path, capsys):
    """run_checks should not return silently with exit 0 and empty stdout on success;
    the caller banner (cowork_preflight_ready) is emitted by main(), not run_checks itself,
    but we verify run_checks at minimum emits no suppressed errors."""
    monkeypatch.setattr(cowork_preflight, "check_python", lambda: [])
    monkeypatch.setattr(cowork_preflight, "check_distribution", lambda: [])
    monkeypatch.setattr(cowork_preflight, "check_omega_import_binding", lambda _repo: [])
    monkeypatch.setattr(cowork_preflight, "check_import", lambda _m, _h: [])
    monkeypatch.setattr(cowork_preflight, "repair_source_integrity", lambda _r: [])
    monkeypatch.setattr(cowork_preflight, "verify_against_git", lambda _r: [])
    monkeypatch.setattr(cowork_preflight, "check_git_health", lambda _r: [])
    monkeypatch.setattr(cowork_preflight, "clean_stale_bytecode", lambda _r: [])
    monkeypatch.setattr(cowork_preflight, "_ast_parse_file", lambda _p: None)

    failures = cowork_preflight.run_checks(require_mcp=False, repo_root=tmp_path)

    assert failures == []


# ---------------------------------------------------------------------------
# Git health checks (Deliverable 1)
# ---------------------------------------------------------------------------

def test_git_health_relocates_index_lock_and_continues(monkeypatch, tmp_path, capsys):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    index_lock = git_dir / "index.lock"
    index_lock.write_text("lock\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    def fail_unlink(self):
        raise AssertionError(f"Path.unlink should not be used for {self}")

    monkeypatch.setattr(cowork_preflight.subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    failures = cowork_preflight.check_git_health(tmp_path)

    assert failures == []
    assert not index_lock.exists()
    assert (git_dir / "index.lock.bak").read_text(encoding="utf-8") == "lock\n"
    assert "Moved stale .git/index.lock" in capsys.readouterr().out


def test_git_health_fails_when_index_lock_rename_fails(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    index_lock = git_dir / "index.lock"
    index_lock.write_text("lock\n", encoding="utf-8")

    def fake_rename(_src: str, _dst: str) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(cowork_preflight.os, "rename", fake_rename)

    failures = cowork_preflight.check_git_health(tmp_path)

    assert len(failures) == 1
    assert "GitEnvironmentCorrupt" in failures[0]
    assert "FUSE-safe rename" in failures[0]
    assert index_lock.exists()


def test_git_health_uses_timestamped_index_lock_backup_when_bak_exists(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    index_lock = git_dir / "index.lock"
    index_lock.write_text("lock\n", encoding="utf-8")
    existing_bak = git_dir / "index.lock.bak"
    existing_bak.write_text("old\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cowork_preflight.subprocess, "run", fake_run)

    failures = cowork_preflight.check_git_health(tmp_path)

    assert failures == []
    assert existing_bak.read_text(encoding="utf-8") == "old\n"
    stamped = sorted(git_dir.glob("index.lock.bak.*"))
    assert len(stamped) == 1
    assert stamped[0].read_text(encoding="utf-8") == "lock\n"


def test_git_health_fails_when_git_status_errors(monkeypatch, tmp_path):
    import subprocess as sp

    def fake_run(cmd, **kwargs):
        return sp.CompletedProcess(cmd, returncode=128, stdout="", stderr="not a git repo")

    monkeypatch.setattr(cowork_preflight.subprocess, "run", fake_run)

    failures = cowork_preflight.check_git_health(tmp_path)

    assert len(failures) == 1
    assert "GitEnvironmentCorrupt" in failures[0]
    assert "exit 128" in failures[0]


def test_git_health_passes_in_clean_git_repo(tmp_path):
    """A valid git repo with no index.lock should pass health check."""
    import subprocess as sp
    sp.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    sp.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True, capture_output=True)
    sp.run(["git", "config", "user.name", "T"], cwd=tmp_path, check=True, capture_output=True)

    failures = cowork_preflight.check_git_health(tmp_path)

    assert failures == []


# ---------------------------------------------------------------------------
# Repair taint lockfile (Deliverable 2)
# ---------------------------------------------------------------------------

def test_repair_writes_taint_lockfile(tmp_path):
    """A successful --repair-from-git run must write the repair taint lockfile."""
    repo = _init_repo(tmp_path)
    service = repo / "omega" / "core" / "contracts" / "service.py"
    service.write_text("def broken(:\n", encoding="utf-8")

    cowork_preflight.verify_against_git(repo, repair=True)

    taint_path = cowork_preflight._get_repair_taint_path(repo)
    assert taint_path.exists(), "Taint lockfile should exist after successful repair"


def test_formal_output_gate_blocked_while_taint_present(monkeypatch, tmp_path):
    """run_formal_output_gate must block (return a failure) if the repair taint
    lockfile exists, even when all source checks would otherwise pass."""
    # Patch away source and smoke checks so only the taint decides the outcome.
    monkeypatch.setattr(cowork_preflight, "run_checks", lambda **_kwargs: [])
    monkeypatch.setattr(cowork_preflight, "_tracked_python_files", lambda _repo: ([], []))
    monkeypatch.setattr(cowork_preflight, "_diverged_tracked_files", lambda _repo, _files: [])
    monkeypatch.setattr(cowork_preflight, "check_formal_output_smoke", lambda: [])

    # Write a taint file manually.
    taint_path = cowork_preflight._get_repair_taint_path(tmp_path)
    taint_path.parent.mkdir(parents=True, exist_ok=True)
    taint_path.write_text("repair_tainted\n", encoding="utf-8")

    failures = cowork_preflight.run_formal_output_gate(repo_root=tmp_path, require_mcp=False)

    assert len(failures) == 1
    assert "TAINT_CLEARED" in failures[0]
    # Taint file should be removed after the gate processes it.
    assert not taint_path.exists(), "Taint file should be cleared after gate processes it"


def test_formal_output_gate_clean_after_taint_cleared(monkeypatch, tmp_path):
    """After taint is cleared (prior run), the gate passes cleanly."""
    smoke_called = []
    monkeypatch.setattr(cowork_preflight, "run_checks", lambda **_kwargs: [])
    monkeypatch.setattr(cowork_preflight, "_tracked_python_files", lambda _repo: ([], []))
    monkeypatch.setattr(cowork_preflight, "_diverged_tracked_files", lambda _repo, _files: [])
    monkeypatch.setattr(
        cowork_preflight, "check_formal_output_smoke", lambda: smoke_called.append(True) or []
    )

    # No taint file — should pass cleanly.
    failures = cowork_preflight.run_formal_output_gate(repo_root=tmp_path, require_mcp=False)

    assert failures == []
    assert smoke_called == [True]
