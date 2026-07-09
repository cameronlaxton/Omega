from __future__ import annotations

from omega.ops import validate_docs


def test_backtick_reference_to_existing_file_passes(tmp_path, monkeypatch):
    (tmp_path / "REAL.md").write_text("content", encoding="utf-8")
    doc = tmp_path / "citing.md"
    doc.write_text("See `REAL.md` for details.", encoding="utf-8")

    monkeypatch.setattr(validate_docs, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_docs, "_BACKTICK_SCAN_DOCS", [doc])

    assert validate_docs.check_backtick_root_doc_references() is True


def test_backtick_reference_to_missing_file_fails(tmp_path, monkeypatch, capsys):
    doc = tmp_path / "citing.md"
    doc.write_text("See `GHOST_DOC.md` for details.", encoding="utf-8")

    monkeypatch.setattr(validate_docs, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_docs, "_BACKTICK_SCAN_DOCS", [doc])

    assert validate_docs.check_backtick_root_doc_references() is False
    assert "GHOST_DOC.md" in capsys.readouterr().out


def test_backtick_reference_to_archived_file_still_passes(tmp_path, monkeypatch):
    """A doc may legitimately cite a retired file that only exists under
    archive/ (e.g. AGENTS.md citing OMEGA_RUN_RECIPE.md as retired). That is
    correct, not drift, and must not fail the check."""
    archive_dir = tmp_path / "archive" / "historical"
    archive_dir.mkdir(parents=True)
    (archive_dir / "RETIRED.md").write_text("old content", encoding="utf-8")
    doc = tmp_path / "citing.md"
    doc.write_text("Retired to archive/historical/: `RETIRED.md`.", encoding="utf-8")

    monkeypatch.setattr(validate_docs, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_docs, "_BACKTICK_SCAN_DOCS", [doc])

    assert validate_docs.check_backtick_root_doc_references() is True
