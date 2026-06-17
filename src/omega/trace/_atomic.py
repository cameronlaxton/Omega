"""Atomic text-file write helper shared by sidecar and audit renderer.

Writes go to ``<path>.tmp`` first, fsync, then ``os.replace`` over the
target. Readers never observe a partial file: either the previous bytes
are intact or the new bytes are fully visible.

This module is intentionally tiny — adding cross-process locking, retries,
or POSIX-only flags belongs elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Behavior on failure: if writing the temp file or fsync raises, the
    original ``path`` is untouched and the temp file is removed before the
    exception propagates. On success, no ``.tmp`` artifact is left behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
