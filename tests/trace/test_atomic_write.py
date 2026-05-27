from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from omega.trace._atomic import atomic_write_text


def test_writes_content_to_path(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.json"
    atomic_write_text(target, '{"a": 1}\n')

    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_no_tmp_left_on_happy_path(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    atomic_write_text(target, "{}")

    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_overwrite_preserves_atomicity(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    target.write_text("original", encoding="utf-8")

    atomic_write_text(target, "updated")

    assert target.read_text(encoding="utf-8") == "updated"


def test_failure_during_write_leaves_original_intact(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    target.write_text("original-bytes", encoding="utf-8")

    with mock.patch("os.fsync", side_effect=OSError("simulated disk fail")):
        with pytest.raises(OSError, match="simulated disk fail"):
            atomic_write_text(target, "new-bytes-that-should-not-land")

    assert target.read_text(encoding="utf-8") == "original-bytes"
    assert list(tmp_path.glob("*.tmp")) == []


def test_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "file.txt"
    atomic_write_text(target, "hi")

    assert target.read_text(encoding="utf-8") == "hi"
