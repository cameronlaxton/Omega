"""
omega.trace.store — SQLite-backed trace persistence and retrieval.

TraceStore is the single persistence interface for ExecutionTrace artifacts.
It handles:
- Persist: write a trace dict to SQLite (idempotent on trace_id)
- Query: retrieve traces by league, time range, outcome status
- Attach outcome: link an actual result to a persisted trace
- Graded traces: return traces with attached outcomes (for calibration)

Thread safety: uses one connection per TraceStore instance with WAL mode.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omega.paths import default_trace_db_path
from omega.trace.bet_record import BetRecord
from omega.trace.bet_settlement import compute_pnl, extract_recommended_bet
from omega.trace.db import resolve_backend
from omega.trace.ledger_bet import (
    DEFAULT_BANKROLL,
    BetProvenance,
    LedgerBet,
    LedgerStatus,
)
from omega.trace.market_snapshot import (
    EarlyMarketSnapshot,
    MarketMovement,
    MarketSnapshot,
)
from omega.trace.prop_outcome import derive_prop_outcome_result, normalize_prop_side
from omega.trace.schema import (
    CURRENT_VERSION,
    SCHEMA_V1,
    SCHEMA_V2,
    SCHEMA_V3,
    SCHEMA_V5,
    SCHEMA_V6,
    SCHEMA_V9,
    SCHEMA_V10,
    SCHEMA_V11,
    SCHEMA_V12,
    SCHEMA_V13,
    SCHEMA_V16,
    SCHEMA_V17,
    SCHEMA_V18,
    SCHEMA_V19,
    apply_v4_migration,
    apply_v7_migration,
    apply_v8_migration,
    apply_v15_migration,
    apply_v20_migration,
)
from omega.trace.parameter_profiles import extract_parameter_profile_ref

if TYPE_CHECKING:
    from omega.trace.session_sidecar import TraceQaVerdict

UTC = timezone.utc

logger = logging.getLogger("omega.trace.store")

_DEFAULT_DB_NAME = "var/omega_traces.db"

# Path-resolution sources reported back to callers and tests so the redirect
# decision is observable without parsing log lines.
_PATH_SOURCE_REQUESTED = "requested"
_PATH_SOURCE_ENV_OVERRIDE = "env_override"
_PATH_SOURCE_AUTO_REDIRECT = "auto_redirect_network_fs"
_PATH_SOURCE_DEFAULT = "default"

_NETWORK_FS_TYPES = frozenset(
    {"fuse", "fuse.exfat", "fuse.ntfs", "fuseblk", "cifs", "smb", "smb2", "smb3", "nfs", "nfs4"}
)


def _local_runtime_db_path() -> Path:
    """Per-user runtime DB path that lives on the local SSD, not a mount.

    Windows: %LOCALAPPDATA%\\omega\\runtime\\var/omega_traces.db
    POSIX:   ~/.omega/runtime/omega_traces.db
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("USERPROFILE") or "."
        return Path(base) / "omega" / "runtime" / _DEFAULT_DB_NAME
    return Path.home() / ".omega" / "runtime" / _DEFAULT_DB_NAME


def _is_network_filesystem(path: str | os.PathLike[str]) -> bool:
    """Best-effort detection of a CIFS/SMB/FUSE/NFS mount under ``path``.

    Pure prefix matching is not enough: WSL2 9p, mapped drives whose provider
    is local virtualization, and bind-mounts can all look like network paths
    superficially. This helper combines:
      - UNC / ``\\?\\UNC\\`` prefix on Windows.
      - ``ctypes.windll.kernel32.GetDriveTypeW`` returning DRIVE_REMOTE (4) for
        the resolved drive letter (Windows only).
      - ``/proc/self/mountinfo`` scan on POSIX for FUSE/CIFS/NFS-backed mounts
        whose mount-point is a prefix of the resolved path.

    Returns False on any inspection error — Layer 2's contract is soft-warn,
    not hard-fail, so a false negative here just means the auto-redirect does
    not trigger and the existing WAL→DELETE fallback in ``conn`` is the next
    line of defense.
    """
    try:
        target = Path(path).expanduser()
    except (TypeError, ValueError):
        return False

    # Resolve far enough to follow symlinks, but only to the closest existing
    # ancestor — the DB file itself may not exist yet on first use.
    probe = target
    for _ in range(8):
        if probe.exists() or probe.parent == probe:
            break
        probe = probe.parent
    try:
        resolved = probe.resolve()
    except (OSError, RuntimeError):
        resolved = probe

    resolved_str = str(resolved)

    if os.name == "nt":
        # UNC / extended UNC prefix.
        if resolved_str.startswith("\\\\?\\UNC\\") or resolved_str.startswith("\\\\"):
            return True
        # Drive-letter remote check.
        if len(resolved_str) >= 2 and resolved_str[1] == ":":
            drive_root = resolved_str[:3] if len(resolved_str) >= 3 else resolved_str[:2] + "\\"
            try:
                import ctypes

                DRIVE_REMOTE = 4
                drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive_root)  # type: ignore[attr-defined]
                if drive_type == DRIVE_REMOTE:
                    return True
            except (AttributeError, OSError, ImportError):
                pass
        return False

    # POSIX: scan /proc/self/mountinfo for the deepest mount point that is a
    # prefix of the resolved path; check its fs type.
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.exists():
        return False
    try:
        entries = mountinfo.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False

    best_mount = ""
    best_fs = ""
    for line in entries:
        # mountinfo format: <id> <parent> <maj:min> <root> <mount-point> <opts> ... - <fs-type> <source> <super-opts>
        parts = line.split(" - ")
        if len(parts) != 2:
            continue
        left = parts[0].split()
        right = parts[1].split()
        if len(left) < 5 or len(right) < 1:
            continue
        mount_point = left[4]
        fs_type = right[0]
        if mount_point == "/" or resolved_str.startswith(mount_point.rstrip("/") + "/") or resolved_str == mount_point:
            if len(mount_point) > len(best_mount):
                best_mount = mount_point
                best_fs = fs_type

    if not best_fs:
        return False
    fs_lower = best_fs.lower()
    if fs_lower in _NETWORK_FS_TYPES:
        return True
    # FUSE shows up as "fuse.<flavor>" for many backends.
    if fs_lower.startswith("fuse"):
        return True
    return False


def _resolve_db_path(requested: str | None) -> tuple[str, str]:
    """Return (effective_path, source) for the TraceStore SQLite location.

    Resolution order (BUG-FUSE-2):
      1. ``OMEGA_TRACE_DB`` env override (absolute path).
      2. The caller's ``requested`` path, unchanged, unless it resolves onto a
         FUSE/SMB/CIFS/NFS mount — in which case we redirect to a local
         per-user runtime DB and log a single WARNING. **No** ``atexit`` sync
         hook is registered; archival back to the mount is owned by
         ``tools/windows/sync_to_mount.ps1`` (see plan §Layer 3).
      3. ``omega.paths.trace_db_path()``. This defaults to the repo
         ``var/omega_traces.db`` and is affected by ``OMEGA_RUNTIME_DIR``.
    """
    env_override = os.environ.get("OMEGA_TRACE_DB")
    if env_override:
        return env_override, _PATH_SOURCE_ENV_OVERRIDE

    if requested is None:
        default_path = str(default_trace_db_path())
        if _is_network_filesystem(default_path):
            redirect = _local_runtime_db_path()
            redirect.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "TraceStore: default DB path %s is on a network/FUSE mount; "
                "redirecting writes to %s. See OMEGA_COWORK.md §2c.",
                default_path,
                redirect,
            )
            return str(redirect), _PATH_SOURCE_AUTO_REDIRECT
        return default_path, _PATH_SOURCE_DEFAULT

    if _is_network_filesystem(requested):
        redirect = _local_runtime_db_path()
        redirect.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "TraceStore: requested DB path %s is on a network/FUSE mount; "
            "redirecting writes to %s. See OMEGA_COWORK.md §2c.",
            requested,
            redirect,
        )
        return str(redirect), _PATH_SOURCE_AUTO_REDIRECT

    return requested, _PATH_SOURCE_REQUESTED


def _repo_default_db_path() -> str:
    """The repository default DB path (the source for FUSE redirects)."""
    return str(default_trace_db_path())


def _integrity_ok(path: str) -> bool:
    """Run ``PRAGMA integrity_check`` read-only. False on any open error.

    Never instantiates TraceStore — safe to call when the redirect guard would
    raise (db_status guardrail).
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        uri = p.resolve().as_uri() + "?mode=ro&immutable=0"
        conn = sqlite3.connect(uri, uri=True)
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return bool(row) and str(row[0]).lower() == "ok"


def _raw_trace_count(path: str) -> int | None:
    """Trace count via a raw read-only connection. None if unreadable/absent."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        uri = p.resolve().as_uri() + "?mode=ro&immutable=0"
        conn = sqlite3.connect(uri, uri=True)
        try:
            row = conn.execute("SELECT COUNT(*) FROM traces").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return int(row[0]) if row else None


def _raw_latest_session_ids(path: str, limit: int = 5) -> list[str]:
    """Most recent session_ids via a raw read-only connection. [] if unreadable."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        uri = p.resolve().as_uri() + "?mode=ro&immutable=0"
        conn = sqlite3.connect(uri, uri=True)
        try:
            rows = conn.execute(
                "SELECT session_id, MAX(timestamp) AS last_ts FROM traces "
                "WHERE session_id IS NOT NULL GROUP BY session_id "
                "ORDER BY last_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [str(r[0]) for r in rows]


def _empty_history_mode_enabled() -> bool:
    return os.environ.get("OMEGA_ALLOW_EMPTY_DB") == "1"


def _apply_redirect_guard(runtime_path: str, source_path: str) -> bool:
    """FUSE-redirect safety guard. Returns True iff EMPTY_HISTORY_MODE is active.

    Honors the revision constraints: never copies/merges here (seeding is the
    explicit ``db_status --seed`` action); never overwrites an existing runtime
    DB; fails LOUD rather than presenting believable-empty history.
    """
    runtime = Path(runtime_path)
    if runtime.exists():
        return False  # use existing runtime DB; never overwrite
    if _empty_history_mode_enabled():
        logger.warning(
            "EMPTY_HISTORY_MODE=true: OMEGA_ALLOW_EMPTY_DB=1 set; creating EMPTY "
            "runtime DB at %s (trace_count=0). Empty history is intentional, not a "
            "failed ingest.",
            runtime,
        )
        return True
    src = Path(source_path)
    if not src.exists():
        raise RuntimeError(
            f"TraceStore: source DB {src} is missing while redirecting off a "
            "network/FUSE mount. Refusing to create believable-empty history "
            "(the mount may be detached). Pass --db, set OMEGA_TRACE_DB, run "
            "`omega-db-status --seed`, or set OMEGA_ALLOW_EMPTY_DB=1 "
            "to start empty on purpose."
        )
    if not _integrity_ok(str(src)):
        raise RuntimeError(
            f"TraceStore: source DB {src} fails integrity_check. Refusing to "
            "redirect to a fresh empty runtime DB. Repair/replace the source, or "
            "set OMEGA_ALLOW_EMPTY_DB=1 to start empty on purpose."
        )
    n = _raw_trace_count(str(src)) or 0
    if n > 0:
        raise RuntimeError(
            f"TraceStore: runtime DB {runtime} is absent and source {src} holds "
            f"{n} traces. Refusing to start with empty history. Run "
            "`omega-db-status --seed` to populate the runtime DB, or "
            "set OMEGA_ALLOW_EMPTY_DB=1 to start empty on purpose."
        )
    # Source is valid but genuinely empty → a fresh runtime DB is correct.
    logger.warning(
        "EMPTY_HISTORY_MODE=true: source DB %s is valid but empty; creating an "
        "empty runtime DB at %s.",
        src,
        runtime,
    )
    return True


def db_status(requested: str | None = None) -> dict[str, Any]:
    """Read-only inspection of DB path/state. Never opens a TraceStore.

    Reports the repo/default path, would-be runtime path, existence, integrity,
    trace counts, divergence, and a recommended action — even when the normal
    redirect guard would raise (db_status guardrail).
    """
    env_override = os.environ.get("OMEGA_TRACE_DB")
    default_path = _repo_default_db_path()
    effective_path, source = _resolve_db_path(requested)
    runtime_path = str(_local_runtime_db_path())

    default_count = _raw_trace_count(default_path)
    effective_count = _raw_trace_count(effective_path)

    divergence: dict[str, Any] | None = None
    if (
        source == _PATH_SOURCE_AUTO_REDIRECT
        and Path(effective_path).exists()
        and Path(default_path).exists()
        and default_count != effective_count
    ):
        divergence = {
            "source_path": default_path,
            "source_trace_count": default_count,
            "runtime_path": effective_path,
            "runtime_trace_count": effective_count,
            "note": "runtime and source diverge; no auto-merge — sync or seed deliberately",
        }

    empty_history_mode = _empty_history_mode_enabled()
    recommended = "ok"
    if source == _PATH_SOURCE_AUTO_REDIRECT and not Path(effective_path).exists():
        if not Path(default_path).exists():
            recommended = "source DB missing — verify the mount is attached before running"
        elif not _integrity_ok(default_path):
            recommended = "source DB malformed — repair/replace before running"
        elif (default_count or 0) > 0 and not empty_history_mode:
            recommended = "run `omega-db-status --seed` to populate the runtime DB"
        elif empty_history_mode:
            recommended = "EMPTY_HISTORY_MODE active — empty history is intentional"
    elif divergence is not None:
        recommended = "review divergence; seed/sync deliberately (no auto-merge)"
    elif effective_count is None and Path(effective_path).exists():
        recommended = "effective DB unreadable — check integrity"

    return {
        "requested": requested,
        "env_override": env_override,
        "repo_default_path": default_path,
        "effective_path": effective_path,
        "source": source,
        "would_be_runtime_path": runtime_path,
        "default_exists": Path(default_path).exists(),
        "effective_exists": Path(effective_path).exists(),
        "runtime_exists": Path(runtime_path).exists(),
        "default_integrity_ok": _integrity_ok(default_path) if Path(default_path).exists() else None,
        "effective_integrity_ok": _integrity_ok(effective_path)
        if Path(effective_path).exists()
        else None,
        "default_trace_count": default_count,
        "effective_trace_count": effective_count,
        "latest_session_ids": _raw_latest_session_ids(
            effective_path if Path(effective_path).exists() else default_path
        ),
        "schema_version": None,
        "divergence": divergence,
        "empty_history_mode": empty_history_mode,
        "recommended_action": recommended,
    }


def seed_runtime_db(source: str, runtime: str) -> dict[str, Any]:
    """Copy a valid non-empty source DB into an ABSENT runtime path (WAL-safe).

    The ONLY mutating DB helper. Preconditions (all enforced, raise otherwise):
    runtime must not exist (never overwrite), source must exist, pass
    integrity_check, and hold at least one trace. No merging.
    """
    src, rt = Path(source), Path(runtime)
    if rt.exists():
        raise RuntimeError(f"refusing to overwrite existing runtime DB at {rt}")
    if not src.exists():
        raise RuntimeError(f"source DB {src} does not exist")
    if not _integrity_ok(str(src)):
        raise RuntimeError(f"source DB {src} fails integrity_check; not seeding")
    n = _raw_trace_count(str(src)) or 0
    if n == 0:
        raise RuntimeError(f"source DB {src} is empty; nothing to seed")
    rt.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(rt))
    try:
        with dst_conn:
            src_conn.backup(dst_conn)
    finally:
        src_conn.close()
        dst_conn.close()
    logger.warning("Seeded runtime DB %s from %s (%d traces).", rt, src, n)
    return {"source": str(src), "runtime": str(rt), "trace_count": n}


def log_effective_db(store: TraceStore, log: logging.Logger) -> None:
    """One-line startup log of the effective DB so the agent never queries the
    wrong (e.g. silently redirected, empty) database unknowingly."""
    try:
        count = store.count()
    except sqlite3.Error:
        count = -1
    log.info(
        "TraceStore DB: path=%s source=%s trace_count=%d EMPTY_HISTORY_MODE=%s",
        store.db_path,
        store.db_path_source,
        count,
        str(bool(store.empty_history_mode)).lower(),
    )


class TraceStore:
    """SQLite-backed trace persistence."""

    def __init__(
        self,
        db_path: str | None = None,
        journal_mode: str | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        if journal_mode is not None and journal_mode.upper() not in {"WAL", "DELETE", "AUTO"}:
            raise ValueError("journal_mode must be 'WAL', 'DELETE', 'AUTO', or None")
        self._repo: Any | None = None
        # Explicit, scoped suppression of the bet_ledger dual-write. Replay sets
        # this via autolog_suppressed() so historical traces never auto-log an
        # engine_auto bet into the (backtest) ledger; staking entries are written
        # explicitly with provenance=historical_replay instead. Falls back to the
        # OMEGA_BET_LEDGER_AUTOLOG env var when unset (legacy behavior preserved).
        self._autolog_suppressed = False
        backend = resolve_backend(db_path)
        if backend.backend == "postgres":
            if journal_mode is not None:
                raise ValueError("journal_mode is sqlite-only when DATABASE_URL routes to Postgres")
            from omega.trace.repository import PostgresRepository

            self._db_path = backend.target or ""
            self._db_path_source = "database_url"
            self._read_only = bool(read_only)
            self._empty_history_mode = False
            self._conn: sqlite3.Connection | None = None
            self._journal_mode: str | None = None
            self._requested_journal_mode = "AUTO"
            self._repo = PostgresRepository(self._db_path, read_only=self._read_only)
            return

        effective_path, source = _resolve_db_path(backend.target)
        self._db_path = effective_path
        self._db_path_source = source
        self._read_only = bool(read_only)
        self._empty_history_mode = False
        self._conn: sqlite3.Connection | None = None
        self._journal_mode: str | None = None
        self._requested_journal_mode = (journal_mode or "AUTO").upper()
        # FUSE-redirect safety guard: a redirected write path must never silently
        # become believable-empty history. Read-only opens skip the guard (they
        # only inspect, never create). See _apply_redirect_guard.
        if source == _PATH_SOURCE_AUTO_REDIRECT and not self._read_only:
            origin = db_path if db_path is not None else _repo_default_db_path()
            self._empty_history_mode = _apply_redirect_guard(effective_path, origin)
        if not self._read_only:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    @property
    def empty_history_mode(self) -> bool:
        """True when this store was opened as an intentional empty runtime DB
        (OMEGA_ALLOW_EMPTY_DB=1 or a valid-but-empty source on FUSE redirect)."""
        return self._empty_history_mode

    @property
    def read_only(self) -> bool:
        """True when this store was opened read-only.

        Backend-neutral: both the SQLite and Postgres paths set ``_read_only`` at
        construction. A read-only store opens its SQLite connection with the
        ``mode=ro`` URI and ``PRAGMA query_only=ON``; callers (e.g. the read-only
        operator console) assert on this to enforce the no-mutation boundary at
        the DB layer.
        """
        return self._read_only

    @property
    def db_path(self) -> str:
        """The DB path the store will actually read/write."""
        return self._db_path

    @property
    def db_path_source(self) -> str:
        """How ``db_path`` was chosen: requested / env_override / auto_redirect_network_fs / default."""
        return self._db_path_source

    @contextmanager
    def autolog_suppressed(self):
        """Disable the bet_ledger dual-write for the duration of the block.

        Deterministic and scoped — unlike toggling OMEGA_BET_LEDGER_AUTOLOG, this
        cannot leak across processes or outlive the ``with`` block. Used by
        historical replay so persisted replay traces never auto-log an
        ``engine_auto`` ledger row. Re-entrant: a prior suppression is restored
        on exit rather than force-cleared.
        """
        prev = self._autolog_suppressed
        self._autolog_suppressed = True
        try:
            yield self
        finally:
            self._autolog_suppressed = prev

    @property
    def conn(self) -> sqlite3.Connection:
        if self._repo is not None:
            raise RuntimeError(
                "Raw sqlite3 connection is unavailable for Postgres TraceStore; "
                "use public TraceStore APIs."
            )
        if self._conn is None:
            if self._read_only:
                # Build a read-only URI that survives Windows paths with spaces
                # and ``?`` characters. ``as_uri()`` handles the escaping.
                uri = Path(self._db_path).resolve().as_uri() + "?mode=ro&immutable=0"
                self._conn = sqlite3.connect(uri, uri=True)
                # Read-only opens MUST NOT touch journal mode or schema — both
                # would attempt writes.
                self._conn.execute("PRAGMA query_only=ON")
                self._conn.execute("PRAGMA busy_timeout=5000")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.execute("PRAGMA temp_store=MEMORY")
                self._conn.row_factory = sqlite3.Row
                return self._conn

            self._conn = sqlite3.connect(self._db_path)
            requested = self._requested_journal_mode
            desired = "DELETE" if requested == "AUTO" and os.environ.get("COWORK_SANDBOX") else requested
            if desired == "AUTO":
                desired = "WAL"
            try:
                row = self._conn.execute(f"PRAGMA journal_mode={desired}").fetchone()
            except sqlite3.OperationalError as exc:
                if desired != "WAL":
                    raise
                logger.warning(
                    "SQLite WAL mode unsupported on this mount. Falling back to DELETE mode. "
                    "Concurrency degraded: ensure trace writes and calibration run sequentially. "
                    "(%s)",
                    exc,
                )
                row = self._conn.execute("PRAGMA journal_mode=DELETE").fetchone()
            self._journal_mode = str(row[0]).lower() if row else None
            # PRAGMA hardening (BUG-FUSE-2): give writers room to wait when WAL
            # is contended, and keep temp tables off whatever the mount is.
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist and record schema version stamps.

        Migrations are forward-additive: each version's DDL uses CREATE TABLE IF
        NOT EXISTS so a fresh DB and an old-version DB converge to CURRENT_VERSION
        without an explicit migration step.
        """
        # Capture the pre-migration version so the V2/V7 bet_records steps are
        # not re-applied (which would resurrect the table) once V14 has dropped
        # it. Must be read before any _record_version() call below.
        prior_version = self._existing_schema_version()

        # Fast path: an up-to-date DB skips the whole forward-additive replay.
        # Every open used to re-run all ~17 idempotent executescripts plus the
        # bet_records consolidation probe — pure no-op DDL that turned hot
        # paths (e.g. per-request prior injection, batch loops) into serialized
        # schema churn. schema_versions stamping CURRENT_VERSION guarantees the
        # DDL already ran; manual table drops are out-of-contract and surface
        # via the repo-state validators, not via per-open self-healing.
        if prior_version >= CURRENT_VERSION:
            return

        # V1: traces, outcomes, schema_versions
        self.conn.executescript(SCHEMA_V1)
        self._record_version(1, "Initial schema: traces, outcomes, schema_versions")

        # V2: bet_records (legacy CLV substrate; removed at V14). Only (re)create
        # on DBs that have not yet been consolidated, so the V14 drop sticks.
        if prior_version < 14:
            self.conn.executescript(SCHEMA_V2)
        self._record_version(2, "Phase 6d: bet_records table for CLV tracking")

        # V3: closing_lines (market close snapshots for CLV)
        self.conn.executescript(SCHEMA_V3)
        self._record_version(3, "Phase 6e: closing_lines table for CLV computation")

        # V4: traces.session_id (groups traces by Claude Project chat session)
        apply_v4_migration(self.conn)
        self._record_version(4, "Phase 6f: traces.session_id column")

        # V5: market_snapshots (provider observations for line movement)
        self.conn.executescript(SCHEMA_V5)
        self._record_version(5, "Phase 6g: market_snapshots table for line movement")

        # V6: prop_outcomes (player-prop grading; separate from outcomes)
        self.conn.executescript(SCHEMA_V6)
        self._record_version(6, "Phase 6h: prop_outcomes table for player-prop grading")

        # V7: bet_records.session_id + BUG-3 cleanup. Gated like V2 — the
        # ALTER targets bet_records, which no longer exists post-V14.
        if prior_version < 14:
            apply_v7_migration(self.conn)
        self._record_version(
            7,
            "Phase 6i: bet_records.session_id column + BUG-3 prop-trace outcome cleanup",
        )

        # V8: one game outcome per trace
        apply_v8_migration(self.conn)
        self._record_version(
            8,
            "Phase 6j: enforce one game outcome row per trace",
        )

        # V9: evidence_signals + signal_performance (structured reasoning loop)
        self.conn.executescript(SCHEMA_V9)
        self._record_version(
            9,
            "Phase 6i: evidence_signals + signal_performance tables",
        )

        # V10: queryable simulation distribution summaries for continuous grading
        self.conn.executescript(SCHEMA_V10)
        self._record_version(
            10,
            "Phase 6k: simulation_distributions + dynamic outcome view",
        )

        # V11: early_market_snapshots (low-liquidity early lines; isolated from CLV)
        self.conn.executescript(SCHEMA_V11)
        self._record_version(
            11,
            "Phase 7: early_market_snapshots table (segregated from closing_lines)",
        )

        # V12: trace_qa_verdicts (trace-scoped QA audit; eligibility stays in JSON)
        self.conn.executescript(SCHEMA_V12)
        self._record_version(
            12,
            "Trace-scoped QA: trace_qa_verdicts audit table",
        )

        # V13: bet_ledger (dollar/PnL bet log; the single bet table from V14 on)
        self.conn.executescript(SCHEMA_V13)
        self._record_version(
            13,
            "Bet logging: bet_ledger table + dashboard view (dollar-denominated PnL)",
        )

        # V14: consolidate bet_records into bet_ledger, then drop bet_records.
        self._consolidate_legacy_bet_records()
        self._record_version(
            14,
            "Consolidate bet_records into bet_ledger (provenance=user_confirmed); drop bet_records",
        )

        # V15: bet_ledger sizing-audit columns (staking policy / exposure / corr).
        apply_v15_migration(self.conn)
        self._record_version(
            15,
            "Sizing audit: bet_ledger staking_policy/exposure_limits/sizing_reasons/correlation_group columns",
        )

        # V16: soccer dynamic priors (Dixon-Coles rho profiles + team xG).
        self.conn.executescript(SCHEMA_V16)
        self._record_version(
            16,
            "Phase 7 M2: priors_dixon_coles + priors_xg tables (soccer dynamic priors)",
        )

        # V17: tennis dynamic priors (SPW/RPW rolling rates + pressure deltas).
        self.conn.executescript(SCHEMA_V17)
        self._record_version(
            17,
            "Phase 7 M3: priors_tennis + priors_tennis_pressure tables (tennis dynamic priors)",
        )

        # V18: NFL dispersion priors (NB k with hierarchical-shrinkage provenance).
        self.conn.executescript(SCHEMA_V18)
        self._record_version(
            18,
            "Phase 7 M4: priors_nfl_dispersion table (NFL NB dispersion k + shrinkage source)",
        )

        # V19: governed backend parameter-profile table (Phase 8). One production
        # profile per (backend_name, competition_bucket); promoted fail-closed
        # through the shared gate engine. Generalizes the priors_dixon_coles
        # rho-profile pattern to a backend's full structural-parameter set.
        self.conn.executescript(SCHEMA_V19)
        self._record_version(
            19,
            "Phase 8: parameter_profiles table (governed backend structural-parameter profiles)",
        )

        # V20: trace-level parameter-profile provenance column. ALTER ADD COLUMN,
        # so apply via the probe-first helper. Surfaces the governed param set that
        # priced a trace into a queryable column, so a probability is attributable
        # to its exact parameters and a replay/lab can pin from the ref.
        apply_v20_migration(self.conn)
        self._record_version(
            20,
            "Phase 8: traces.parameter_profile_ref column (trace-level parameter provenance)",
        )

    def _existing_schema_version(self) -> int:
        """Highest applied schema version, or 0 if the DB is brand new."""
        try:
            row = self.conn.execute("SELECT MAX(version) FROM schema_versions").fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row and row[0] is not None else 0

    def _consolidate_legacy_bet_records(self) -> int:
        """Migrate any rows from the legacy bet_records table into bet_ledger
        (as provenance='user_confirmed'), then drop bet_records.

        Idempotent: a no-op once bet_records is gone. Dollar stake is derived
        from the units-based stake (1 unit = 1% of bankroll; default $1000), and
        settled rows get payout/PnL recomputed from odds + status.
        """
        tbl = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bet_records'"
        ).fetchone()
        if tbl is None:
            return 0

        rows = self.conn.execute(
            """SELECT b.bet_id, b.trace_id, b.book, b.market, b.selection,
                      b.selection_descriptor, b.line_taken, b.odds_taken,
                      b.stake_units, b.decision_timestamp, b.status, b.session_id,
                      t.league, t.matchup
               FROM bet_records b JOIN traces t ON t.trace_id = b.trace_id"""
        ).fetchall()

        migrated = 0
        for r in rows:
            status = LedgerStatus(r["status"]) if r["status"] in {s.value for s in LedgerStatus} else LedgerStatus.PENDING
            bankroll = DEFAULT_BANKROLL
            stake = round((r["stake_units"] or 0.0) * (bankroll / 100.0), 2) or 25.0
            odds = float(r["odds_taken"]) if r["odds_taken"] is not None else 0.0
            payout, net = compute_pnl(status, odds, stake) if odds else (None, None)
            sport = None
            if r["league"]:
                try:
                    from omega.core.config.leagues import get_league_config

                    sport = get_league_config(str(r["league"])).get("sport")
                except Exception:  # noqa: BLE001 - sport is a nicety, never fatal
                    sport = None
            ts = r["decision_timestamp"] or ""
            self.conn.execute(
                """INSERT OR IGNORE INTO bet_ledger
                   (ledger_id, trace_id, bet_date, league, sport, matchup, market,
                    bookmaker, selection, selection_descriptor, line, odds,
                    stake_amount, payout_amount, net_pnl, bankroll_at_open, status,
                    provenance, decision_timestamp, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r["bet_id"],
                    r["trace_id"],
                    ts[:10] if len(ts) >= 10 else None,
                    r["league"],
                    sport,
                    r["matchup"],
                    r["market"],
                    r["book"],
                    r["selection"],
                    r["selection_descriptor"],
                    r["line_taken"],
                    odds,
                    stake,
                    payout,
                    net,
                    bankroll,
                    status.value,
                    BetProvenance.USER_CONFIRMED.value,
                    r["decision_timestamp"],
                    r["session_id"] if "session_id" in r.keys() else None,
                ),
            )
            migrated += 1

        self.conn.execute("DROP TABLE IF EXISTS bet_records")
        self.conn.commit()
        if migrated:
            logger.info(
                "V14: migrated %d bet_records row(s) into bet_ledger (user_confirmed) "
                "and dropped bet_records.",
                migrated,
            )
        return migrated

    def _record_version(self, version: int, description: str) -> None:
        """Idempotently stamp a schema version into schema_versions."""
        existing = self.conn.execute(
            "SELECT version FROM schema_versions WHERE version = ?",
            (version,),
        ).fetchone()
        if not existing:
            self.conn.execute(
                "INSERT INTO schema_versions (version, description) VALUES (?, ?)",
                (version, description),
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def persist(self, trace: dict[str, Any] | Any) -> str:
        """Write a trace to SQLite. Idempotent on trace_id.

        Args:
            trace: PersistableTrace or serialized trace dict containing trace_id, run_id, timestamp.

        Returns:
            trace_id of the persisted record.

        Raises:
            ValueError: if required fields are missing.
        """
        if self._repo is not None:
            return self._repo.persist(trace)
        if hasattr(trace, "to_store_record"):
            trace = trace.to_store_record()
        elif hasattr(trace, "model_dump"):
            trace = trace.model_dump(mode="json")

        trace_id = str(trace.get("trace_id", ""))
        run_id = str(trace.get("run_id", ""))
        timestamp = str(trace.get("timestamp", ""))

        if not trace_id or not run_id or not timestamp:
            raise ValueError(
                f"Trace missing required fields: trace_id={trace_id!r}, "
                f"run_id={run_id!r}, timestamp={timestamp!r}"
            )

        full_trace = json.dumps(trace, default=str)

        session_id = trace.get("session_id")
        if session_id is not None:
            session_id = str(session_id) or None

        # Trace-level parameter-profile provenance (V20): the governed param set
        # that priced this trace, surfaced from the full_trace blob into a queryable
        # column so a probability is attributable and a replay/lab can pin from it.
        param_ref = extract_parameter_profile_ref(trace)
        param_ref_json = json.dumps(param_ref, default=str) if param_ref else None

        cur = self.conn.execute(
            """INSERT OR IGNORE INTO traces
               (trace_id, run_id, timestamp, prompt, league, matchup,
                execution_mode, simulation_seed, aggregate_quality,
                predictions, recommendations, odds_snapshot, downgrades,
                full_trace, schema_version, session_id, parameter_profile_ref)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                run_id,
                timestamp,
                trace.get("prompt", ""),
                trace.get("league"),
                trace.get("matchup"),
                trace.get("execution_mode"),
                trace.get("simulation_seed"),
                trace.get("aggregate_quality", 0.0),
                json.dumps(trace.get("predictions"), default=str)
                if trace.get("predictions")
                else None,
                json.dumps(trace.get("recommendations"), default=str)
                if trace.get("recommendations")
                else None,
                json.dumps(trace.get("odds_snapshot"), default=str)
                if trace.get("odds_snapshot")
                else None,
                json.dumps(trace.get("downgrades", []), default=str),
                full_trace,
                CURRENT_VERSION,
                session_id,
                param_ref_json,
            ),
        )
        # Explode evidence into queryable rows only on a genuine first insert.
        # persist() is idempotent on trace_id; guarding on rowcount keeps the
        # evidence_signals table free of duplicates when a trace is re-persisted.
        if cur.rowcount and cur.rowcount > 0:
            self._write_evidence_signals(trace_id, trace)
            self._write_simulation_distributions(trace_id, trace)
            self._maybe_autolog_ledger_bet(trace_id, trace)
        self.conn.commit()
        return trace_id

    def _maybe_autolog_ledger_bet(self, trace_id: str, trace: dict[str, Any]) -> None:
        """Gated dual-write: log the recommended selection into bet_ledger.

        Runs only on a genuine first insert (guarded by the caller's rowcount
        check, like the other explode helpers). Gated behind
        OMEGA_BET_LEDGER_AUTOLOG (default on) so the deliberate calibration/bet
        decouple stays explicit and reversible. Never raises into persist(): a
        malformed recommendation must not cost us the trace write.
        """
        # Explicit scoped suppression wins over the env default (used by replay).
        if self._autolog_suppressed:
            return
        if os.environ.get("OMEGA_BET_LEDGER_AUTOLOG", "1") not in ("1", "true", "True"):
            return
        try:
            result = extract_recommended_bet(trace, provenance=BetProvenance.ENGINE_AUTO)
            if result.bet is not None:
                # Defer to persist()'s single commit so the trace + its children +
                # the autologged ledger bet land in ONE transaction.
                self.record_ledger_bet(result.bet, commit=False)
        except Exception as exc:  # noqa: BLE001 - never let autolog break persist
            logger.warning("bet_ledger autolog skipped for %s: %s", trace_id, exc)

    def _write_simulation_distributions(self, trace_id: str, trace: dict[str, Any]) -> int:
        """Explode deterministic distribution summaries into V10 query rows."""
        rows_in = trace.get("simulation_distributions")
        if not isinstance(rows_in, list) or not rows_in:
            result = trace.get("result") or {}
            rows_in = result.get("simulation_distributions") or []
        if not isinstance(rows_in, list) or not rows_in:
            return 0

        rows: list[tuple[Any, ...]] = []
        for item in rows_in:
            if not isinstance(item, dict):
                continue
            dist_type = item.get("distribution_type")
            target = item.get("target")
            if not dist_type or not target:
                continue
            params = item.get("distribution_params") or {}
            rows.append(
                (
                    trace_id,
                    trace.get("kind"),
                    trace.get("league"),
                    target,
                    item.get("market"),
                    item.get("stat_key"),
                    dist_type,
                    json.dumps(params, default=str, sort_keys=True),
                    int(item.get("params_schema_version") or 1),
                    item.get("sample_mean"),
                    item.get("sample_std"),
                    item.get("p10"),
                    item.get("p50"),
                    item.get("p90"),
                    item.get("n_iterations"),
                    item.get("seed", trace.get("simulation_seed")),
                    item.get("context_hash"),
                    item.get("component_version") or trace.get("model_version"),
                )
            )
        if rows:
            self.conn.executemany(
                """INSERT INTO simulation_distributions
                   (trace_id, kind, league, target, market, stat_key,
                    distribution_type, distribution_params, params_schema_version,
                    sample_mean, sample_std, p10, p50, p90, n_iterations, seed,
                    context_hash, component_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def get_simulation_distributions(self, trace_id: str) -> list[dict[str, Any]]:
        """Return V10 simulation distribution rows attached to one trace."""
        if self._repo is not None:
            return self._repo.get_simulation_distributions(trace_id)
        rows = self.conn.execute(
            """SELECT distribution_id, trace_id, kind, league, target, market,
                      stat_key, distribution_type, distribution_params,
                      params_schema_version, sample_mean, sample_std, p10, p50,
                      p90, n_iterations, seed, context_hash, component_version,
                      created_at
               FROM simulation_distributions
               WHERE trace_id = ?
               ORDER BY distribution_id""",
            (trace_id,),
        ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            try:
                data["distribution_params"] = json.loads(data["distribution_params"])
            except (TypeError, json.JSONDecodeError):
                data["distribution_params"] = {}
            result.append(data)
        return result

    def _prop_outcomes_by_traces(self, trace_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Batch-load prop outcomes for many traces (one query per ~900-id chunk).

        Output per trace_id is byte-identical to :meth:`get_prop_outcomes`
        (same columns, same within-trace ``attached_at`` order) — this is the
        N+1 elimination for ``query_traces``, not a behavior change.
        """
        out: dict[str, list[dict[str, Any]]] = {}
        for start in range(0, len(trace_ids), 900):
            chunk = trace_ids[start : start + 900]
            ph = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"""SELECT prop_outcome_id, trace_id, player_name, stat_type,
                           stat_value, line, side, result, source, attached_at
                    FROM prop_outcomes WHERE trace_id IN ({ph})
                    ORDER BY trace_id, attached_at""",
                chunk,
            ).fetchall()
            for row in rows:
                out.setdefault(row["trace_id"], []).append(dict(row))
        return out

    def _distributions_by_traces(self, trace_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Batch-load V10 distribution rows for many traces (chunked IN query).

        Per-trace output matches :meth:`get_simulation_distributions` exactly
        (same columns, within-trace ``distribution_id`` order, JSON-decoded
        ``distribution_params``).
        """
        out: dict[str, list[dict[str, Any]]] = {}
        for start in range(0, len(trace_ids), 900):
            chunk = trace_ids[start : start + 900]
            ph = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"""SELECT distribution_id, trace_id, kind, league, target, market,
                           stat_key, distribution_type, distribution_params,
                           params_schema_version, sample_mean, sample_std, p10, p50,
                           p90, n_iterations, seed, context_hash, component_version,
                           created_at
                    FROM simulation_distributions WHERE trace_id IN ({ph})
                    ORDER BY trace_id, distribution_id""",
                chunk,
            ).fetchall()
            for row in rows:
                data = dict(row)
                try:
                    data["distribution_params"] = json.loads(data["distribution_params"])
                except (TypeError, json.JSONDecodeError):
                    data["distribution_params"] = {}
                out.setdefault(row["trace_id"], []).append(data)
        return out

    def _write_evidence_signals(self, trace_id: str, trace: dict[str, Any]) -> int:
        """Explode input_snapshot.evidence into queryable evidence_signals rows.

        Source of truth stays the full_trace JSON blob; this table only exists
        so retrospective scoring can JOIN signals to outcomes. Phase B writes a
        per-signal `evidence_application` list (aligned by index) describing
        whether/how the engine applied each signal; when absent (Phase-A traces
        or no engine apply) every signal is recorded as unapplied.

        Returns the number of evidence rows written.
        """
        input_snap = trace.get("input_snapshot") or {}
        evidence = input_snap.get("evidence") or []
        if not isinstance(evidence, list) or not evidence:
            return 0

        league = trace.get("league")
        application = trace.get("evidence_application")
        if not isinstance(application, list):
            application = []
        trace_evidence_mode = trace.get("evidence_mode")

        rows: list[tuple[Any, ...]] = []
        for idx, sig in enumerate(evidence):
            if not isinstance(sig, dict):
                continue
            app = (
                application[idx]
                if idx < len(application) and isinstance(application[idx], dict)
                else {}
            )
            rows.append(
                (
                    trace_id,
                    sig.get("signal_type"),
                    sig.get("category"),
                    sig.get("plane"),
                    sig.get("source"),
                    sig.get("confidence"),
                    sig.get("window"),
                    sig.get("direction"),
                    sig.get("stat_key"),
                    league,
                    json.dumps(sig.get("value"), default=str),
                    1 if app.get("applied") else 0,
                    app.get("factor"),
                    app.get("policy_version"),
                    app.get("evidence_mode") or trace_evidence_mode,
                )
            )
        if rows:
            self.conn.executemany(
                """INSERT INTO evidence_signals
                   (trace_id, signal_type, category, plane, source, confidence,
                    obs_window, direction, stat_key, league, value_json,
                    applied, applied_factor, policy_version, evidence_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def get_evidence_signals(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all evidence signal rows attached to a trace (may be empty)."""
        if self._repo is not None:
            return self._repo.get_evidence_signals(trace_id)
        rows = self.conn.execute(
            """SELECT id, trace_id, signal_type, category, plane, source,
                      confidence, obs_window, direction, stat_key, league,
                      value_json, applied, applied_factor, policy_version,
                      evidence_mode, created_at
               FROM evidence_signals WHERE trace_id = ? ORDER BY id""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Trace-scoped QA verdicts (audit aid; eligibility stays in JSON)
    # ------------------------------------------------------------------

    def write_qa_verdict(
        self,
        trace_id: str,
        verdict: TraceQaVerdict,
        *,
        session_id: str | None = None,
        ran_at: str | None = None,
    ) -> None:
        """Upsert the trace-scoped QA verdict audit row (idempotent on trace_id).

        This is an audit/query aid only: the canonical calibration-eligibility
        flag lives in trace_quality.calibration_eligible inside the full_trace
        blob and is reconciled separately. The trace row must already exist (FK).
        """
        if self._repo is not None:
            return self._repo.write_qa_verdict(
                trace_id, verdict, session_id=session_id, ran_at=ran_at
            )
        self.conn.execute(
            """INSERT INTO trace_qa_verdicts
                   (trace_id, session_id, verdict, scope, gate_name, reason,
                    event_id, matched_trace_id, ran_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(trace_id) DO UPDATE SET
                   session_id=excluded.session_id,
                   verdict=excluded.verdict,
                   scope=excluded.scope,
                   gate_name=excluded.gate_name,
                   reason=excluded.reason,
                   event_id=excluded.event_id,
                   matched_trace_id=excluded.matched_trace_id,
                   ran_at=excluded.ran_at""",
            (
                trace_id,
                session_id,
                verdict.verdict,
                verdict.scope,
                verdict.gate_name,
                verdict.reason,
                verdict.event_id,
                verdict.matched_trace_id,
                ran_at,
            ),
        )
        self.conn.commit()

    def get_qa_verdict(self, trace_id: str) -> dict[str, Any] | None:
        """Return the trace-scoped QA verdict audit row, or None."""
        if self._repo is not None:
            return self._repo.get_qa_verdict(trace_id)
        row = self.conn.execute(
            "SELECT * FROM trace_qa_verdicts WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Signal performance (Phase 6i — retrospective evidence scoring)
    # ------------------------------------------------------------------

    def upsert_signal_performance(
        self, rows: list[Any], dataset_hash: str
    ) -> int:
        """Write retrospective signal-performance aggregates for one scoring run.

        ``rows`` are SignalPerformanceRow-shaped objects (see
        omega/strategy/signal_performance.py). ``dataset_hash`` identifies the
        scored dataset; all rows from one run share it and a single ``scored_at``
        timestamp so the report can read the latest run cleanly. Idempotent on
        (signal_type, source, obs_window, league, dataset_hash): re-running the
        same dataset replaces its rows rather than duplicating them.

        Returns the number of rows written.
        """
        if self._repo is not None:
            return self._repo.upsert_signal_performance(rows, dataset_hash)
        scored_at = datetime.now(UTC).isoformat()
        payload = [
            (
                r.signal_type,
                r.source,
                r.obs_window,
                r.league,
                int(r.sample_size),
                int(r.direction_correct),
                float(r.direction_accuracy),
                float(r.mean_confidence),
                float(r.realized_hit_rate),
                float(r.calibration_gap),
                float(r.brier),
                dataset_hash,
                scored_at,
            )
            for r in rows
        ]
        if payload:
            self.conn.executemany(
                """INSERT OR REPLACE INTO signal_performance
                   (signal_type, source, obs_window, league, sample_size,
                    direction_correct, direction_accuracy, mean_confidence,
                    realized_hit_rate, calibration_gap, brier, dataset_hash,
                    scored_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                payload,
            )
            self.conn.commit()
        return len(payload)

    def get_signal_performance(
        self, league: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Return the most recent scoring run's signal-performance rows.

        "Most recent" is the latest ``scored_at`` (optionally scoped to a
        league). Older runs stay in the table for history but are not returned.
        """
        if self._repo is not None:
            return self._repo.get_signal_performance(league=league, limit=limit)
        latest = self.conn.execute(
            "SELECT scored_at FROM signal_performance "
            + ("WHERE league = ? " if league else "")
            + "ORDER BY scored_at DESC LIMIT 1",
            (league,) if league else (),
        ).fetchone()
        if latest is None:
            return []

        clauses = ["scored_at = ?"]
        params: list[Any] = [latest["scored_at"]]
        if league:
            clauses.append("league = ?")
            params.append(league)
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT signal_type, source, obs_window, league, sample_size,
                       direction_correct, direction_accuracy, mean_confidence,
                       realized_hit_rate, calibration_gap, brier, dataset_hash,
                       scored_at
                FROM signal_performance
                WHERE {" AND ".join(clauses)}
                ORDER BY sample_size DESC, signal_type
                LIMIT ?""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Outcome attachment
    # ------------------------------------------------------------------

    def attach_outcome(
        self,
        trace_id: str,
        home_score: int,
        away_score: int,
        source: str = "manual",
        result_override: str | None = None,
    ) -> str:
        """Attach an actual outcome to a persisted trace.

        Args:
            trace_id: Must reference an existing trace.
            home_score: Final home team score.
            away_score: Final away team score.
            source: How the outcome was obtained ("manual", "api", "backtest").
            result_override: When set ("home_win"|"away_win"|"draw"), use this as
                the graded 3-way result instead of deriving it from the scores.
                Used for single-leg knockouts decided after regulation, where the
                1X2 market settles on the 90' result (a draw) even though the
                stored scores may reflect the extra-time/shootout winner.

        Returns:
            outcome_id of the created record.

        Raises:
            ValueError: if trace_id does not exist.
        """
        if self._repo is not None:
            return self._repo.attach_outcome(
                trace_id=trace_id,
                home_score=home_score,
                away_score=away_score,
                source=source,
                result_override=result_override,
            )
        # Verify trace exists
        row = self.conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No trace found with trace_id={trace_id!r}")

        existing = self.conn.execute(
            "SELECT outcome_id FROM outcomes WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if existing:
            raise ValueError(
                f"Outcome already attached for trace_id={trace_id!r}; "
                "delete the existing outcome explicitly before re-grading"
            )

        # Determine result (an explicit override wins — see result_override doc).
        if result_override is not None:
            result = result_override
        elif home_score > away_score:
            result = "home_win"
        elif away_score > home_score:
            result = "away_win"
        else:
            result = "draw"

        outcome_id = uuid.uuid4().hex[:12]
        try:
            self.conn.execute(
                """INSERT INTO outcomes (outcome_id, trace_id, home_score, away_score, result, source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (outcome_id, trace_id, home_score, away_score, result, source),
            )
        except sqlite3.IntegrityError as exc:
            # Lost a concurrent attach race (the UNIQUE(trace_id) index fired after
            # our existence check passed). Surface the same friendly error as the
            # check-then-insert path instead of an unhandled IntegrityError.
            raise ValueError(
                f"Outcome already attached for trace_id={trace_id!r}; "
                "delete the existing outcome explicitly before re-grading"
            ) from exc
        self.conn.commit()
        return outcome_id

    def get_outcome(self, trace_id: str) -> dict[str, Any] | None:
        """Return the game outcome attached to a trace, or None."""
        if self._repo is not None:
            return self._repo.get_outcome(trace_id)
        row = self.conn.execute(
            """SELECT outcome_id, trace_id, home_score, away_score, result,
                      attached_at, source
               FROM outcomes WHERE trace_id = ? ORDER BY attached_at LIMIT 1""",
            (trace_id,),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Prop outcome attachment (Phase 6h — player-prop grading)
    # ------------------------------------------------------------------

    def attach_prop_outcome(
        self,
        trace_id: str,
        player_name: str,
        stat_type: str,
        stat_value: float,
        line: float,
        side: str,
        source: str = "manual",
        void: bool = False,
    ) -> str:
        """Attach a graded player-prop outcome to a persisted trace.

        Idempotent on (trace_id, player_name, stat_type): re-attaching returns the
        existing row's id, mirroring closing_lines semantics.

        When ``void=True`` (DNP / no-action: player absent from the box score),
        the win/loss/push math is skipped and ``result`` is recorded as ``void``
        so settlement returns VOID (stake returned). ``stat_value`` is ignored.

        Args:
            trace_id: Must reference an existing trace.
            player_name: Canonical player name (matches input_snapshot.player_name).
            stat_type: Stat graded (e.g. "points", "rebounds", "hits", "strikeouts").
            stat_value: Actual stat the player produced.
            line: Prop line at decision time.
            side: "over" or "under" — the selection being graded.
            source: How the outcome was obtained (e.g. "manual",
                "api:espn_boxscore", "manual:espn_boxscore_YYYYMMDD").

        Returns:
            prop_outcome_id of the row (existing or newly inserted).

        Raises:
            ValueError: if trace_id does not exist or side is not over/under.
        """
        if self._repo is not None:
            return self._repo.attach_prop_outcome(
                trace_id=trace_id,
                player_name=player_name,
                stat_type=stat_type,
                stat_value=stat_value,
                line=line,
                side=side,
                source=source,
                void=void,
            )
        side_norm = normalize_prop_side(side)

        row = self.conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No trace found with trace_id={trace_id!r}")

        existing = self.conn.execute(
            """SELECT prop_outcome_id FROM prop_outcomes
               WHERE trace_id = ? AND player_name = ? AND stat_type = ?""",
            (trace_id, player_name, stat_type),
        ).fetchone()
        if existing:
            return existing["prop_outcome_id"]

        result, side_norm = derive_prop_outcome_result(
            stat_value=stat_value,
            line=line,
            side=side_norm,
            void=void,
        )

        prop_outcome_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            """INSERT INTO prop_outcomes
               (prop_outcome_id, trace_id, player_name, stat_type,
                stat_value, line, side, result, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prop_outcome_id,
                trace_id,
                player_name,
                stat_type,
                float(stat_value),
                float(line),
                side_norm,
                result,
                source,
            ),
        )
        self.conn.commit()
        return prop_outcome_id

    def get_prop_outcomes(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all prop outcomes attached to a trace (may be empty)."""
        if self._repo is not None:
            return self._repo.get_prop_outcomes(trace_id)
        rows = self.conn.execute(
            """SELECT prop_outcome_id, trace_id, player_name, stat_type,
                      stat_value, line, side, result, source, attached_at
               FROM prop_outcomes WHERE trace_id = ? ORDER BY attached_at""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # User-confirmed wagers (CLV substrate) — now stored in bet_ledger as
    # provenance='user_confirmed'. The legacy bet_records table was dropped at
    # schema V14; these methods preserve the historical API on top of the
    # unified bet_ledger so wager-tracking callers (ingest, audit) are unchanged.
    # ------------------------------------------------------------------

    def record_bet(self, bet: BetRecord) -> str:
        """Persist a user-confirmed wager into bet_ledger. Idempotent on
        (trace_id, market, selection_descriptor).

        Accepts the legacy units-based BetRecord (the LLM export-block DTO) and
        converts it to a dollar LedgerBet tagged provenance='user_confirmed'.
        Returns the ledger_id (== bet.bet_id).

        Raises:
            ValueError: if the referenced trace does not exist.
        """
        if self._repo is not None:
            return self._repo.record_bet(bet)
        stake = round((bet.stake_units or 0.0) * (DEFAULT_BANKROLL / 100.0), 2) or 25.0
        ts = bet.decision_timestamp or ""
        ledger = LedgerBet(
            ledger_id=bet.bet_id,
            trace_id=bet.trace_id,
            bet_date=ts[:10] if len(ts) >= 10 else None,
            league=None,
            sport=None,
            matchup="",
            market=bet.market,
            bookmaker=bet.book,
            selection=bet.selection,
            selection_descriptor=bet.selection_descriptor,
            line=bet.line_taken,
            odds=bet.odds_taken,
            stake_amount=stake,
            bankroll_at_open=DEFAULT_BANKROLL,
            status=LedgerStatus(bet.status.value)
            if bet.status.value in {s.value for s in LedgerStatus}
            else LedgerStatus.PENDING,
            provenance=BetProvenance.USER_CONFIRMED,
            decision_timestamp=bet.decision_timestamp,
        )
        return self.record_ledger_bet(ledger)

    def get_bet_records(self, trace_id: str) -> list[dict[str, Any]]:
        """Return user-confirmed wagers for a trace in the legacy bet_records
        shape (bet_id/book/odds_taken/line_taken/stake_units), read from
        bet_ledger where provenance='user_confirmed'. May be empty."""
        if self._repo is not None:
            return self._repo.get_bet_records(trace_id)
        rows = self.conn.execute(
            """SELECT ledger_id, trace_id, bookmaker, market, selection,
                      selection_descriptor, line, odds, stake_amount,
                      bankroll_at_open, decision_timestamp, status, created_at,
                      session_id
               FROM bet_ledger
               WHERE trace_id = ? AND provenance = 'user_confirmed'
               ORDER BY created_at""",
            (trace_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            bankroll = r["bankroll_at_open"] or DEFAULT_BANKROLL
            units = round(r["stake_amount"] / (bankroll / 100.0), 4) if bankroll else None
            out.append(
                {
                    "bet_id": r["ledger_id"],
                    "trace_id": r["trace_id"],
                    "book": r["bookmaker"],
                    "market": r["market"],
                    "selection": r["selection"],
                    "selection_descriptor": r["selection_descriptor"],
                    "line_taken": r["line"],
                    "odds_taken": r["odds"],
                    "stake_units": units,
                    "decision_timestamp": r["decision_timestamp"],
                    "status": r["status"],
                    "recorded_at": r["created_at"],
                    "session_id": r["session_id"],
                }
            )
        return out

    def query_ungraded_prop_bet_traces(
        self,
        league: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return traces linked to pending player-prop bets (bet_ledger) that
        have no prop_outcome yet.

        Defense-in-depth for BUG-2 (docs/session_bugs_20260519.md): when the
        agent minted a separate bet-confirmation trace, the bet's trace_id and
        the analysis trace_id end up disjoint, so prop_outcomes attached to the
        analysis trace never reach the bet. This method surfaces the bet's
        trace_id directly so the grading pipeline can attach under it.

        Filters mirror query_traces() for league/time so window semantics are
        consistent. Returns full trace dicts; callers are expected to call
        the same _prop_fields()/grading path as analysis-trace candidates.
        """
        if self._repo is not None:
            return self._repo.query_ungraded_prop_bet_traces(
                league=league, start=start, end=end, limit=limit
            )
        clauses = [
            "b.status = 'pending'",
            "b.market LIKE 'player_prop:%'",
            "b.provenance = 'user_confirmed'",
            "NOT EXISTS (SELECT 1 FROM prop_outcomes p WHERE p.trace_id = b.trace_id)",
        ]
        params: list[Any] = []
        if league:
            clauses.append("t.league = ?")
            params.append(league)
        if start:
            clauses.append("t.timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("t.timestamp <= ?")
            params.append(end)
        params.append(limit)

        sql = f"""
            SELECT DISTINCT t.trace_id, t.full_trace
            FROM bet_ledger b
            JOIN traces t ON t.trace_id = b.trace_id
            WHERE {" AND ".join(clauses)}
            ORDER BY t.timestamp DESC
            LIMIT ?
        """
        rows = self.conn.execute(sql, params).fetchall()
        return [json.loads(row["full_trace"]) for row in rows]

    def update_bet_status(self, bet_id: str, status: str) -> None:
        """Mark a bet won/lost/void/push after outcome resolves (bet_ledger)."""
        if self._repo is not None:
            return self._repo.update_bet_status(bet_id, status)
        self.conn.execute(
            "UPDATE bet_ledger SET status = ? WHERE ledger_id = ?",
            (status, bet_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Bet ledger (V13 — dollar-denominated PnL log; the single bet table)
    # ------------------------------------------------------------------

    _LEDGER_COLUMNS = (
        "ledger_id, trace_id, bet_date, league, sport, matchup, market, bookmaker, "
        "selection, selection_descriptor, line, odds, stake_amount, payout_amount, "
        "net_pnl, bankroll_at_open, status, provenance, decision_timestamp, "
        "graded_at, session_id, created_at, staking_policy_id, staking_policy_version, "
        "exposure_limits_version, sizing_reasons, correlation_group"
    )

    def record_ledger_bet(self, bet: LedgerBet, *, commit: bool = True) -> str:
        """Persist a ledger bet. Idempotent on (trace_id, market, selection_descriptor).

        session_id is sourced from the linked trace so it stays consistent with
        traces.session_id without the caller plumbing it through.

        ``commit`` lets the autolog path inside :meth:`persist` defer to persist's
        single commit, so a trace and its dual-written ledger bet land in ONE
        transaction instead of two (no mid-persist commit).

        Provenance upgrade: the unique key omits provenance, so the engine's
        gated autolog (engine_auto) and a later user confirmation of the same
        selection would otherwise collide. A `user_confirmed` write upgrades an
        existing non-user_confirmed row in place — promoting provenance and
        overwriting the wager fields with the user's actual price/book/stake —
        so the CLV/closing-line pipeline (which reads provenance='user_confirmed')
        sees the real wager. The upgrade is guarded to PENDING rows so an already
        graded row's settled PnL is never silently invalidated. All other
        conflicts (engine_auto/backfill onto an existing row, or a duplicate
        user_confirmed) are no-ops, preserving the prior INSERT-OR-IGNORE
        behavior.

        Returns the ledger_id of the stored row (the pre-existing row's id on an
        upgrade/no-op, not necessarily ``bet.ledger_id``).

        Raises:
            ValueError: if the referenced trace does not exist.
        """
        if self._repo is not None:
            return self._repo.record_ledger_bet(bet)
        row = self.conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id = ?", (bet.trace_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No trace found with trace_id={bet.trace_id!r}")

        self.conn.execute(
            """INSERT INTO bet_ledger
               (ledger_id, trace_id, bet_date, league, sport, matchup, market,
                bookmaker, selection, selection_descriptor, line, odds,
                stake_amount, payout_amount, net_pnl, bankroll_at_open, status,
                provenance, decision_timestamp, graded_at,
                staking_policy_id, staking_policy_version, exposure_limits_version,
                sizing_reasons, correlation_group, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?,
                       (SELECT session_id FROM traces WHERE trace_id = ?))
               ON CONFLICT(trace_id, market, selection_descriptor) DO UPDATE SET
                   provenance         = excluded.provenance,
                   bookmaker          = excluded.bookmaker,
                   selection          = excluded.selection,
                   line               = excluded.line,
                   odds               = excluded.odds,
                   stake_amount       = excluded.stake_amount,
                   bankroll_at_open   = excluded.bankroll_at_open,
                   bet_date           = excluded.bet_date,
                   decision_timestamp = excluded.decision_timestamp,
                   staking_policy_id = excluded.staking_policy_id,
                   staking_policy_version = excluded.staking_policy_version,
                   exposure_limits_version = excluded.exposure_limits_version,
                   sizing_reasons = excluded.sizing_reasons,
                   correlation_group = excluded.correlation_group
               WHERE excluded.provenance = 'user_confirmed'
                 AND bet_ledger.provenance != 'user_confirmed'
                 AND bet_ledger.status = 'pending'""",
            (
                bet.ledger_id,
                bet.trace_id,
                bet.bet_date,
                bet.league,
                bet.sport,
                bet.matchup,
                bet.market,
                bet.bookmaker,
                bet.selection,
                bet.selection_descriptor,
                bet.line,
                bet.odds,
                bet.stake_amount,
                bet.payout_amount,
                bet.net_pnl,
                bet.bankroll_at_open,
                bet.status.value,
                bet.provenance.value,
                bet.decision_timestamp,
                bet.graded_at,
                bet.staking_policy_id,
                bet.staking_policy_version,
                bet.exposure_limits_version,
                json.dumps(bet.sizing_reasons) if bet.sizing_reasons is not None else None,
                bet.correlation_group,
                bet.trace_id,
            ),
        )
        if commit:
            self.conn.commit()
        stored = self.conn.execute(
            "SELECT ledger_id FROM bet_ledger "
            "WHERE trace_id = ? AND market = ? AND selection_descriptor = ?",
            (bet.trace_id, bet.market, bet.selection_descriptor),
        ).fetchone()
        return stored["ledger_id"] if stored else bet.ledger_id

    def grade_ledger_bet(
        self,
        ledger_id: str,
        status: LedgerStatus | str,
        payout_amount: float | None,
        net_pnl: float | None,
    ) -> None:
        """Settle a ledger bet: write status + money + graded_at."""
        if self._repo is not None:
            return self._repo.grade_ledger_bet(
                ledger_id, status, payout_amount, net_pnl
            )
        status_val = status.value if isinstance(status, LedgerStatus) else str(status)
        self.conn.execute(
            """UPDATE bet_ledger
               SET status = ?, payout_amount = ?, net_pnl = ?,
                   graded_at = ?
               WHERE ledger_id = ?""",
            (
                status_val,
                payout_amount,
                net_pnl,
                datetime.now(UTC).isoformat(),
                ledger_id,
            ),
        )
        self.conn.commit()

    def get_ledger_bets(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all ledger bets attached to a trace (may be empty)."""
        if self._repo is not None:
            return self._repo.get_ledger_bets(trace_id)
        rows = self.conn.execute(
            f"SELECT {self._LEDGER_COLUMNS} FROM bet_ledger "
            "WHERE trace_id = ? ORDER BY created_at",
            (trace_id,),
        ).fetchall()
        return [self._decode_ledger_row(row) for row in rows]

    def get_ledger_bet(self, ledger_id: str) -> dict[str, Any] | None:
        """Return a single ledger bet by ledger_id, or None.

        Backend-neutral read helper (the read-only operator console needs a
        by-id lookup for the bet-detail page; the by-trace_id readers above do
        not cover it). Pure SELECT — safe on a read-only store.
        """
        if self._repo is not None:
            return self._repo.get_ledger_bet(ledger_id)
        row = self.conn.execute(
            f"SELECT {self._LEDGER_COLUMNS} FROM bet_ledger WHERE ledger_id = ?",
            (ledger_id,),
        ).fetchone()
        return self._decode_ledger_row(row) if row else None

    def query_ledger(
        self,
        league: str | None = None,
        sport: str | None = None,
        status: str | None = None,
        provenance: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Return ledger bets matching filters, newest first (for dashboard/export)."""
        if self._repo is not None:
            return self._repo.query_ledger(
                league=league,
                sport=sport,
                status=status,
                provenance=provenance,
                start=start,
                end=end,
                limit=limit,
            )
        clauses: list[str] = []
        params: list[Any] = []
        if league:
            clauses.append("league = ?")
            params.append(league)
        if sport:
            clauses.append("sport = ?")
            params.append(sport)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if provenance:
            clauses.append("provenance = ?")
            params.append(provenance)
        if start:
            clauses.append("decision_timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("decision_timestamp <= ?")
            params.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT {self._LEDGER_COLUMNS} FROM bet_ledger {where} "
            "ORDER BY decision_timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._decode_ledger_row(row) for row in rows]

    @staticmethod
    def _decode_ledger_row(row: Any) -> dict[str, Any]:
        data = dict(row)
        if data.get("sizing_reasons") is not None:
            data["sizing_reasons"] = json.loads(data["sizing_reasons"])
        return data

    # ------------------------------------------------------------------
    # Closing lines (Phase 6e — CLV resolution)
    # ------------------------------------------------------------------

    def attach_closing_line(
        self,
        trace_id: str,
        market: str,
        selection_descriptor: str,
        closing_odds: float,
        closing_line: float | None,
        closing_timestamp: str,
        source: str,
    ) -> str:
        """Attach a market-close snapshot to a trace + selection.

        Idempotent on (trace_id, market, selection_descriptor): re-running with a
        new source/timestamp leaves the first attached close in place. To force
        an overwrite, delete the row explicitly. This protects against a
        misconfigured cron clobbering a verified close.

        Args:
            trace_id: Must reference an existing trace.
            market: e.g. "moneyline", "spread", "total", "player_prop:pts".
            selection_descriptor: Canonical snake_case form (matches BetRecord).
            closing_odds: American odds at close.
            closing_line: Point/total at close; None for moneyline.
            closing_timestamp: ISO 8601 of the close snapshot.
            source: e.g. "the-odds-api:draftkings".

        Returns:
            closing_id of the row (existing or newly inserted).
        """
        if self._repo is not None:
            return self._repo.attach_closing_line(
                trace_id=trace_id,
                market=market,
                selection_descriptor=selection_descriptor,
                closing_odds=closing_odds,
                closing_line=closing_line,
                closing_timestamp=closing_timestamp,
                source=source,
            )
        row = self.conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No trace found with trace_id={trace_id!r}")

        existing = self.conn.execute(
            """SELECT closing_id FROM closing_lines
               WHERE trace_id = ? AND market = ? AND selection_descriptor = ?""",
            (trace_id, market, selection_descriptor),
        ).fetchone()
        if existing:
            return existing["closing_id"]

        closing_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            """INSERT INTO closing_lines
               (closing_id, trace_id, market, selection_descriptor,
                closing_line, closing_odds, closing_timestamp, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                closing_id,
                trace_id,
                market,
                selection_descriptor,
                closing_line,
                closing_odds,
                closing_timestamp,
                source,
            ),
        )
        self.conn.commit()
        return closing_id

    def get_closing_lines(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all closing-line snapshots attached to a trace."""
        if self._repo is not None:
            return self._repo.get_closing_lines(trace_id)
        rows = self.conn.execute(
            """SELECT closing_id, trace_id, market, selection_descriptor,
                      closing_line, closing_odds, closing_timestamp, source, captured_at
               FROM closing_lines WHERE trace_id = ? ORDER BY captured_at""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Early market snapshots (Phase 7 - low-liquidity early lines)
    # ------------------------------------------------------------------
    #
    # These intentionally do NOT touch closing_lines. CLV reads only
    # closing_lines, so early captures can never distort it. See red-team
    # finding 4 in docs/phase7/MULTI_SPORT_EXPANSION.md.

    def record_early_market_snapshot(self, snapshot: EarlyMarketSnapshot) -> str:
        """Persist one early low-liquidity line capture idempotently.

        Idempotent on (trace_id, league, market, selection_descriptor,
        captured_at). Writes ONLY to early_market_snapshots — never to
        closing_lines — so the CLV metric stays grounded in true closes.

        Returns the early_id of the row (existing or newly inserted).
        """
        if self._repo is not None:
            return self._repo.record_early_market_snapshot(snapshot)
        early_id = snapshot.stable_id()
        self.conn.execute(
            """INSERT OR IGNORE INTO early_market_snapshots
               (early_id, trace_id, league, market, selection_descriptor,
                early_line, early_odds, liquidity_profile, captured_at, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                early_id,
                snapshot.trace_id,
                snapshot.league.upper(),
                snapshot.market,
                snapshot.selection_descriptor,
                snapshot.early_line,
                snapshot.early_odds,
                snapshot.liquidity_profile,
                snapshot.captured_at,
                snapshot.source,
            ),
        )
        self.conn.commit()
        return early_id

    def get_early_market_snapshots(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all early-market snapshots captured for a trace."""
        if self._repo is not None:
            return self._repo.get_early_market_snapshots(trace_id)
        rows = self.conn.execute(
            """SELECT early_id, trace_id, league, market, selection_descriptor,
                      early_line, early_odds, liquidity_profile, captured_at,
                      source, recorded_at
               FROM early_market_snapshots WHERE trace_id = ? ORDER BY captured_at""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Market snapshots (Phase 6g - line movement substrate)
    # ------------------------------------------------------------------

    def record_market_snapshot(self, snapshot: MarketSnapshot) -> str:
        """Persist one provider market observation idempotently."""
        if self._repo is not None:
            return self._repo.record_market_snapshot(snapshot)
        snapshot_id = snapshot.stable_id()
        self.conn.execute(
            """INSERT OR IGNORE INTO market_snapshots
               (snapshot_id, league, provider, provider_event_id, home_team,
                away_team, commence_time, bookmaker, market, selection, player,
                point, price, snapshot_timestamp, provider_last_update, source,
                schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot_id,
                snapshot.league.upper(),
                snapshot.provider,
                snapshot.provider_event_id,
                snapshot.home_team,
                snapshot.away_team,
                snapshot.commence_time,
                snapshot.bookmaker,
                snapshot.market,
                snapshot.selection,
                snapshot.player,
                snapshot.point,
                snapshot.price,
                snapshot.snapshot_timestamp,
                snapshot.provider_last_update,
                snapshot.source,
                snapshot.schema_version,
            ),
        )
        self.conn.commit()
        return snapshot_id

    def get_market_snapshots(
        self,
        provider_event_id: str,
        market: str | None = None,
        bookmaker: str | None = None,
        selection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return provider market observations for movement analysis."""
        if self._repo is not None:
            return self._repo.get_market_snapshots(
                provider_event_id=provider_event_id,
                market=market,
                bookmaker=bookmaker,
                selection=selection,
            )
        clauses = ["provider_event_id = ?"]
        params: list[Any] = [provider_event_id]
        if market:
            clauses.append("market = ?")
            params.append(market)
        if bookmaker:
            clauses.append("bookmaker = ?")
            params.append(bookmaker)
        if selection:
            clauses.append("selection = ?")
            params.append(selection)
        rows = self.conn.execute(
            f"""SELECT snapshot_id, league, provider, provider_event_id, home_team,
                       away_team, commence_time, bookmaker, market, selection,
                       player, point, price, snapshot_timestamp,
                       provider_last_update, source, schema_version, captured_at
                FROM market_snapshots
                WHERE {" AND ".join(clauses)}
                ORDER BY snapshot_timestamp""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def compute_market_movement(
        self,
        provider_event_id: str,
        market: str,
        selection: str,
        bookmaker: str,
    ) -> dict[str, Any] | None:
        """Compute simple first-to-last line movement for an exact market row."""
        if self._repo is not None:
            return self._repo.compute_market_movement(
                provider_event_id=provider_event_id,
                market=market,
                selection=selection,
                bookmaker=bookmaker,
            )
        rows = self.get_market_snapshots(
            provider_event_id=provider_event_id,
            market=market,
            bookmaker=bookmaker,
            selection=selection,
        )
        if len(rows) < 2:
            return None
        first = rows[0]
        last = rows[-1]
        point_delta = None
        if first["point"] is not None and last["point"] is not None:
            point_delta = float(last["point"]) - float(first["point"])
        movement = MarketMovement(
            market=market,
            selection=selection,
            bookmaker=bookmaker,
            first_timestamp=first["snapshot_timestamp"],
            last_timestamp=last["snapshot_timestamp"],
            first_point=first["point"],
            last_point=last["point"],
            first_price=float(first["price"]),
            last_price=float(last["price"]),
            point_delta=point_delta,
            price_delta=float(last["price"]) - float(first["price"]),
        )
        return movement.model_dump()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Retrieve the full trace dict by ID."""
        if self._repo is not None:
            return self._repo.get_trace(trace_id)
        row = self.conn.execute(
            "SELECT full_trace FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["full_trace"])

    def query_traces(
        self,
        league: str | None = None,
        start: str | None = None,
        end: str | None = None,
        has_outcome: bool | None = None,
        execution_mode: str | None = None,
        limit: int = 100,
        calibration_eligible_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Query traces with optional filters.

        Args:
            league: Filter by league (e.g. "NBA").
            start: ISO timestamp lower bound.
            end: ISO timestamp upper bound.
            has_outcome: True = only graded, False = only ungraded, None = all.
            execution_mode: Filter by mode (e.g. "native_sim").
            limit: Max results (default 100).

        Returns:
            List of trace dicts.
        """
        if self._repo is not None:
            return self._repo.query_traces(
                league=league,
                start=start,
                end=end,
                has_outcome=has_outcome,
                execution_mode=execution_mode,
                limit=limit,
                calibration_eligible_only=calibration_eligible_only,
            )
        clauses: list[str] = []
        params: list[Any] = []

        if league:
            clauses.append("t.league = ?")
            params.append(league)
        if start:
            clauses.append("t.timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("t.timestamp <= ?")
            params.append(end)
        if execution_mode:
            clauses.append("t.execution_mode = ?")
            params.append(execution_mode)

        # "Outcome" for query purposes means either a game outcome OR one or
        # more prop outcomes attached to the trace. EXISTS subqueries avoid
        # the row-duplication that would come from LEFT JOINing prop_outcomes
        # (which is 1:N per trace).
        any_outcome_sql = (
            "(EXISTS (SELECT 1 FROM outcomes WHERE outcomes.trace_id = t.trace_id) "
            "OR EXISTS (SELECT 1 FROM prop_outcomes WHERE prop_outcomes.trace_id = t.trace_id))"
        )
        if has_outcome is True:
            clauses.append(any_outcome_sql)
        elif has_outcome is False:
            clauses.append(f"NOT {any_outcome_sql}")

        if calibration_eligible_only:
            # Single source of truth: omega.trace.eligibility computes the
            # canonical calibration_eligible flag (service.py writes it; ingest
            # reconciles QA fails into it). That flag already subsumes
            # result.status=success, context_source='provided', and
            # identity_status='complete', so we gate on the flag plus the one
            # independent structural prerequisite (a probability to fit).
            # Default-deny: legacy rows without the flag set are excluded.
            clauses.append("t.predictions IS NOT NULL")
            clauses.append(
                "json_extract(t.full_trace, '$.trace_quality.calibration_eligible') = 1"
            )

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        # LEFT JOIN outcomes stays for the convenient _outcome attach (1:1 in
        # practice). Prop outcomes are 1:N so we fetch them per-trace below.
        sql = f"""
            SELECT t.trace_id, t.full_trace,
                   o.outcome_id, o.home_score, o.away_score, o.result
            FROM traces t
            LEFT JOIN outcomes o ON t.trace_id = o.trace_id
            {where}
            ORDER BY t.timestamp DESC
            LIMIT ?
        """
        rows = self.conn.execute(sql, params).fetchall()

        # Batch-load the 1:N children once (2 queries) instead of per-row (2N).
        trace_ids = [row["trace_id"] for row in rows]
        prop_by_trace = self._prop_outcomes_by_traces(trace_ids)
        dist_by_trace = self._distributions_by_traces(trace_ids)

        results = []
        for row in rows:
            trace = json.loads(row["full_trace"])
            if row["outcome_id"]:
                trace["_outcome"] = {
                    "outcome_id": row["outcome_id"],
                    "home_score": row["home_score"],
                    "away_score": row["away_score"],
                    "result": row["result"],
                }
            prop_rows = prop_by_trace.get(row["trace_id"])
            if prop_rows:
                trace["_prop_outcomes"] = prop_rows
            distribution_rows = dist_by_trace.get(row["trace_id"])
            if distribution_rows:
                trace["_simulation_distributions"] = distribution_rows
            results.append(trace)
        return results

    def get_graded_traces(
        self, league: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Return calibration-eligible traces with attached outcomes.

        Only returns traces where the engine ran and produced model predictions
        (predictions IS NOT NULL). Excludes manual:no_engine_run, parlay, and
        pre-6h legacy traces that have outcomes attached but no probability to fit.
        Each returned dict has a '_outcome' key with the attached outcome.
        """
        if self._repo is not None:
            return self._repo.get_graded_traces(league=league, limit=limit)
        return self.query_traces(
            league=league,
            has_outcome=True,
            calibration_eligible_only=True,
            limit=limit,
        )

    def query_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Return every trace persisted under one session_id, with attached outcomes.

        Used by the session audit renderer. Numeric/quant fields shown in the
        audit must come from these rows, not from sidecar prose.
        """
        if self._repo is not None:
            return self._repo.query_by_session(session_id)
        rows = self.conn.execute(
            """SELECT t.trace_id, t.timestamp, t.league, t.matchup,
                      t.execution_mode, t.aggregate_quality, t.full_trace,
                      o.outcome_id, o.home_score, o.away_score, o.result
               FROM traces t
               LEFT JOIN outcomes o ON o.trace_id = t.trace_id
               WHERE t.session_id = ?
               ORDER BY t.timestamp""",
            (session_id,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            trace = json.loads(row["full_trace"])
            trace["_row"] = {
                "trace_id": row["trace_id"],
                "timestamp": row["timestamp"],
                "kind": trace.get("kind"),
                "league": row["league"],
                "matchup": row["matchup"],
                "execution_mode": row["execution_mode"],
                "aggregate_quality": row["aggregate_quality"],
            }
            if row["outcome_id"]:
                trace["_outcome"] = {
                    "outcome_id": row["outcome_id"],
                    "home_score": row["home_score"],
                    "away_score": row["away_score"],
                    "result": row["result"],
                }
            prop_rows = self.get_prop_outcomes(row["trace_id"])
            if prop_rows:
                trace["_prop_outcomes"] = prop_rows
            bet_rows = self.get_bet_records(row["trace_id"])
            if bet_rows:
                trace["_bet_records"] = bet_rows
            results.append(trace)
        return results

    def get_session_summary(
        self, league: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Aggregate trace counts grouped by session_id. NULL session_ids excluded.

        Used by report_calibration.py to surface per-session metrics.
        """
        if self._repo is not None:
            return self._repo.get_session_summary(league=league, limit=limit)
        clauses = ["t.session_id IS NOT NULL"]
        params: list[Any] = []
        if league:
            clauses.append("t.league = ?")
            params.append(league)
        where = " WHERE " + " AND ".join(clauses)
        params.append(limit)

        rows = self.conn.execute(
            f"""
            SELECT t.session_id,
                   COUNT(*) AS trace_count,
                   SUM(
                       CASE WHEN
                           EXISTS (
                               SELECT 1 FROM outcomes o
                               WHERE o.trace_id = t.trace_id
                           )
                           OR EXISTS (
                               SELECT 1 FROM prop_outcomes p
                               WHERE p.trace_id = t.trace_id
                           )
                       THEN 1 ELSE 0 END
                   ) AS graded_count,
                   MIN(t.timestamp) AS first_ts,
                   MAX(t.timestamp) AS last_ts
            FROM traces t
            {where}
            GROUP BY t.session_id
            ORDER BY last_ts DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def schema_version(self) -> int:
        """Return the current schema version."""
        if self._repo is not None:
            return self._repo.schema_version()
        row = self.conn.execute("SELECT MAX(version) as v FROM schema_versions").fetchone()
        return row["v"] if row and row["v"] else 0

    def count(self) -> int:
        """Return total number of persisted traces."""
        if self._repo is not None:
            return self._repo.count()
        row = self.conn.execute("SELECT COUNT(*) as n FROM traces").fetchone()
        return row["n"]

    def close(self) -> None:
        """Close the database connection."""
        if self._repo is not None:
            return self._repo.close()
        if self._conn:
            self._conn.close()
            self._conn = None
