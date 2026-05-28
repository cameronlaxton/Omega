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
