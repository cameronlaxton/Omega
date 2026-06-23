#!/usr/bin/env python
"""
recover_db.py - recover lost NBA traces into the active var/omega_traces.db.

Background / findings
---------------------
The legacy root-level ``omega_traces.db`` (pre-"Moved sports data" refactor) is
GONE. Only two orphans survive at the repo root:

    omega_traces.db-shm   (32 KiB, stale shared-memory index)
    omega_traces.db-wal   (0 bytes, already checkpointed/empty)

A WAL/SHM pair cannot reconstruct a database without its main file, and the WAL
is empty anyway. So there is nothing to ``ATTACH DATABASE`` and merge -- the
schema-diff/legacy-copy path the runbook anticipated is moot.

However, the historical NBA traces were committed to git as JSON export blocks
under ``inbox/traces/`` (and ``.../processed/``) and were dropped from the
working tree during the refactor. Git is therefore the authoritative recovery
source. This script:

    1. Reads the set of trace_ids already present in the active DB (READ-ONLY).
    2. Walks all git history for added ``inbox/traces/**/*.json`` blobs.
    3. Keeps only those whose ``trace.input_snapshot.league == "NBA"``.
    4. Dedupes by trace_id (prefers the ``processed/`` copy when both exist).
    5. Skips any trace_id already in the DB (defence-in-depth; ingest is also
       INSERT OR IGNORE, so this is belt-and-suspenders).
    6. Stages the survivors as ``inbox/traces/<trace_id>.json`` for the native
       ``omega-ingest-traces`` pipeline.

Why staging + native ingest instead of direct INSERTs
------------------------------------------------------
The active DB carries the current Phase 6/7 schema (session_id, evidence_signals
explosion, bet_ledger, trace_qa_verdicts, ...). The native ingest path
(``omega.ops.ingest_traces``) maps a legacy export block onto that schema
*correctly*, runs the pre-persist export validator, and uses INSERT OR IGNORE on
trace_id. Hand-rolling INSERTs here is exactly how schema corruption happens, so
the actual DB write is delegated to that audited code. This script never opens
the DB for writing and never overwrites the DB file.

Usage
-----
    python tools/recover_db.py --explain   # stage + dry-run validation report
    python tools/recover_db.py             # stage NBA traces, do not ingest
    python tools/recover_db.py --ingest    # stage + run native ingest for real
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "var" / "omega_traces.db"
INBOX = REPO_ROOT / "var" / "inbox" / "traces"

_TRACE_JSON_RE = re.compile(r"^(?:var/)?inbox/traces/.*\.json$")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def existing_trace_ids() -> set[str]:
    """Read trace_ids from the active DB without taking a write lock."""
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        return {row[0] for row in con.execute("SELECT trace_id FROM traces")}
    finally:
        con.close()


def historical_trace_paths() -> list[str]:
    """Every inbox/traces JSON path ever ADDED in git history (deduped)."""
    out = _git("log", "--all", "--diff-filter=A", "--pretty=format:", "--name-only")
    paths = {line.strip() for line in out.splitlines() if _TRACE_JSON_RE.match(line.strip())}
    return sorted(paths)


def _add_commit_for(path: str) -> str | None:
    out = _git("log", "--all", "--diff-filter=A", "-1", "--pretty=format:%H", "--", path)
    return out.strip() or None


def _blob_at(commit: str, path: str) -> dict[str, Any] | None:
    try:
        raw = _git("show", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _league_of(payload: dict[str, Any]) -> str | None:
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else payload
    snap = (trace or {}).get("input_snapshot") or {}
    return snap.get("league") or trace.get("league")


def _trace_id_of(payload: dict[str, Any]) -> str | None:
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else payload
    return (trace or {}).get("trace_id")


def collect_nba(existing: set[str]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Return ({trace_id: (path, raw_json)}, skipped_existing_ids)."""
    chosen: dict[str, tuple[str, str]] = {}
    chosen_is_processed: dict[str, bool] = {}
    skipped: list[str] = []

    for path in historical_trace_paths():
        commit = _add_commit_for(path)
        if not commit:
            continue
        try:
            raw = _git("show", f"{commit}:{path}")
        except subprocess.CalledProcessError:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if _league_of(payload) != "NBA":
            continue
        tid = _trace_id_of(payload)
        if not tid:
            continue
        if tid in existing:
            if tid not in skipped:
                skipped.append(tid)
            continue
        is_processed = "/processed/" in path
        # Prefer the processed/ copy (post-validation, canonical) when duplicated.
        if tid in chosen and chosen_is_processed.get(tid) and not is_processed:
            continue
        chosen[tid] = (path, raw)
        chosen_is_processed[tid] = is_processed

    return chosen, skipped


def stage(chosen: dict[str, tuple[str, str]]) -> list[Path]:
    INBOX.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for tid, (_path, raw) in sorted(chosen.items()):
        dst = INBOX / f"{tid}.json"
        dst.write_text(raw, encoding="utf-8")
        written.append(dst)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest", action="store_true", help="Run native ingest after staging")
    ap.add_argument(
        "--explain", action="store_true", help="Stage, then run ingest --explain (no DB writes)"
    )
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"FATAL: active DB not found at {DB_PATH}", file=sys.stderr)
        return 1

    existing = existing_trace_ids()
    print(f"Active DB: {DB_PATH}")
    print(f"Existing traces in DB: {len(existing)}")

    chosen, skipped = collect_nba(existing)
    print(f"NBA traces found in git history (new, not already in DB): {len(chosen)}")
    if skipped:
        print(f"NBA trace_ids already present in DB (skipped): {len(skipped)}")

    written = stage(chosen)
    print(f"Staged {len(written)} file(s) into {INBOX}")
    for p in written:
        print(f"  staged {p.name}")

    if args.ingest or args.explain:
        from omega.ops import ingest_traces

        argv = ["--inbox", str(INBOX), "--db", str(DB_PATH)]
        if args.explain:
            argv.append("--explain")
        print(f"\nRunning native ingest: omega-ingest-traces {' '.join(argv)}\n")
        sys.argv = ["omega-ingest-traces", *argv]
        return ingest_traces.main()

    print("\nStaging complete. Run with --explain to validate, or --ingest to load.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "src"))
    raise SystemExit(main())
