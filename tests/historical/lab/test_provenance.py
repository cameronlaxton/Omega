"""Git provenance helpers: shape on the real repo + fail-closed on unreadable git."""

from __future__ import annotations

import re

from omega.historical.lab import provenance


def test_git_commit_shape_on_repo():
    sha = provenance.git_commit()
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)


def test_working_tree_dirty_returns_bool():
    assert isinstance(provenance.working_tree_dirty(), bool)


def test_capture_has_three_keys():
    cap = provenance.capture()
    assert set(cap) == {"code_version", "git_commit", "working_tree_dirty"}
    assert isinstance(cap["working_tree_dirty"], bool)


def test_fail_closed_when_git_unreadable(monkeypatch):
    # Simulate a non-repo / corrupt git: rc != 0 for every call.
    monkeypatch.setattr(provenance, "_git", lambda root, args: (1, ""))
    assert provenance.git_commit() == "unknown"
    # Unreadable git state must be treated as dirty so auto-promote refuses.
    assert provenance.working_tree_dirty() is True
