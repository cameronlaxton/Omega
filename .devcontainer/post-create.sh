#!/usr/bin/env bash
set -euo pipefail

SOURCE=/opt/omega-source
WORKSPACE=/workspaces/Omega

if [ ! -f "$WORKSPACE/pyproject.toml" ]; then
  mkdir -p "$WORKSPACE"
  rsync -a --delete \
    --exclude ".git/index.lock" \
    "$SOURCE"/ "$WORKSPACE"/
fi

cd "$WORKSPACE"
git config --global --add safe.directory "$WORKSPACE"

python -m pip install --upgrade pip
python -m pip install -e '.[dev,mcp]'

omega-cowork-preflight --direct-only
python - <<'PY'
import sqlite3
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    db_path = Path(tmp) / "wal_probe.db"
    conn = sqlite3.connect(db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower()
        if mode != "wal":
            raise SystemExit(f"SQLite WAL probe returned {mode!r}")
    finally:
        conn.close()
PY
