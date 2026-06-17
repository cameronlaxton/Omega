"""Regression tests pinning that ops CLIs default to the canonical ``var/``
runtime root, not the legacy repo-root ``inbox/`` / ``reports/`` dirs.

The runtime helpers live in ``omega.paths`` (``session_inbox_dir`` etc.); a
default that drops the ``var/`` segment makes a CLI read/write a stale or
legacy location (see docs/phase6/ARTIFACT_AUTHORITY.md). These tests fail if a
default regresses to a root path or stops using the helper. The source-contract
style mirrors ``tests/ops/test_sync_to_mount_contract.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import omega.paths as paths  # noqa: E402

_OPS = _SRC / "omega" / "ops"

# CLI module -> omega.paths helpers it must import and use for its defaults.
_CLI_HELPERS = {
    "report_calibration.py": ("session_inbox_dir", "latest_report_path"),
    "render_session_audits.py": ("session_inbox_dir", "run_audits_dir"),
    "ingest_traces.py": ("session_inbox_dir", "trace_inbox_dir"),
    "ingest_closing_lines.py": ("closing_lines_inbox_dir",),
    "mark_session_qa_failed.py": ("session_inbox_dir",),
    "sanitize_trace_exports.py": ("trace_inbox_dir",),
}

# Root-anchored runtime defaults that must never come back.
_ROOT_DEFAULT_PATTERNS = (
    '_REPO_ROOT / "inbox"',
    '_REPO_ROOT / "reports"',
    'Path("inbox")',
    'Path("reports")',
)


@pytest.mark.parametrize("fname,helpers", list(_CLI_HELPERS.items()))
def test_cli_defaults_use_var_helpers_not_root(fname, helpers):
    text = (_OPS / fname).read_text(encoding="utf-8")
    for pat in _ROOT_DEFAULT_PATTERNS:
        assert pat not in text, f"{fname} still defaults to a root runtime path: {pat!r}"
    for helper in helpers:
        assert helper in text, f"{fname} should use omega.paths.{helper} for its default"


def test_path_helpers_resolve_under_var():
    assert paths.session_inbox_dir().parts[-3:] == ("var", "inbox", "sessions")
    assert paths.trace_inbox_dir().parts[-3:] == ("var", "inbox", "traces")
    assert paths.closing_lines_inbox_dir().parts[-3:] == ("var", "inbox", "closing_lines")
    assert paths.latest_report_path().parts[-3:] == ("var", "reports", "latest.md")
    assert paths.run_audits_dir().parts[-3:] == ("var", "reports", "run_audits")
