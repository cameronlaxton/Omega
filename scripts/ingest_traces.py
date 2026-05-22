"""
scripts/ingest_traces.py — drain `inbox/traces/*.json` into `omega_traces.db`.

Workflow:
    1. Scan `inbox/traces/*.json` (non-recursive, processed/ and failed/ are skipped).
    2. For each file: parse → adapt analyze() output to TraceStore shape → persist trace
       → persist bet_record if present → move file to processed/.
    3. On parse or persistence error: move file to failed/ with a sibling `.error.txt`.

Idempotent: re-running over the same processed/ directory is a no-op
(`TraceStore.persist()` uses INSERT OR IGNORE on trace_id, and bet_records uses
INSERT OR IGNORE on the (trace_id, market, selection_descriptor) UNIQUE).

Usage:
    python scripts/ingest_traces.py
    python scripts/ingest_traces.py --inbox <path> --db <path>
    python scripts/ingest_traces.py --dry-run

Exit codes:
    0 — all files processed (some may have failed; check failed/)
    1 — fatal error before scanning (bad args, inbox missing)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

# Allow running as a script from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.trace.bet_record import BetRecord  # noqa: E402
from omega.trace.persistable import PersistableTrace  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("ingest_traces")


# ---------------------------------------------------------------------------
# Adapters: engine analyze() output → TraceStore.persist() shape
# ---------------------------------------------------------------------------


def _adapt_sandbox_trace(analyze_out: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible wrapper around the explicit persistable trace contract."""
    return PersistableTrace.from_analyze_output(analyze_out).to_store_record()


# ---------------------------------------------------------------------------
# File-level ingest
# ---------------------------------------------------------------------------


def _load_payload(path: Path) -> dict[str, Any]:
    """Parse the JSON file and return the export-block dict.

    Accepts two shapes:
      A) The Phase 6h export block: {"trace": {...}, "bet_record": ...}
      B) The raw analyze() output: {"trace_id": "sandbox-...", "kind": "...", ...}
    Shape B is wrapped into A for uniform downstream handling.
    """
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON must be an object, got {type(payload).__name__}")

    if "trace" in payload and isinstance(payload["trace"], dict):
        return payload  # shape A
    if "trace_id" in payload and "kind" in payload:
        return {"trace": payload, "bet_record": None}  # shape B
    raise ValueError(
        "JSON must contain either 'trace' (export block) or top-level 'trace_id'+'kind'"
    )


# Drift thresholds for the BUG-5 consistency check. Anything beyond these
# bounds between analysis-trace state and the user-confirmed bet is worth a
# warning but not worth failing ingest — line moves and odds drift are
# expected.
_LINE_DRIFT_WARN = 1.0
_ODDS_DRIFT_WARN_AMERICAN = 25


def _validate_bet_with_prop_identity(
    trace_id: str, kind: str, input_snap: dict[str, Any], bet_block: dict[str, Any]
) -> None:
    """BUG-4 defense: when a bet_record is attached to a prop trace, the trace
    MUST carry home_team/away_team/game_date so fetch_outcomes_props.py can
    resolve the box score.

    Enforces the OMEGA_COWORK.md §6 single-trace policy at the ingest seam.
    """
    if kind != "prop":
        return
    missing = [f for f in ("home_team", "away_team", "game_date") if not input_snap.get(f)]
    if missing:
        raise ValueError(
            f"prop trace {trace_id} carries a bet_record but is missing "
            f"input_snapshot fields {missing}. Per OMEGA_COWORK.md §6 the "
            "bet must attach to the original analysis trace (single-trace "
            "policy); do not mint a stripped-down confirmation trace."
        )


def _warn_drift(trace_id: str, input_snap: dict[str, Any], bet_block: dict[str, Any]) -> None:
    """BUG-5 defense: log a warning when bet_record.line_taken or odds_taken
    diverge meaningfully from the analysis trace's snapshot. Warnings only —
    line/odds shopping is legitimate; we just want the audit trail."""
    line_taken = bet_block.get("line_taken")
    analysis_line = input_snap.get("line")
    if line_taken is not None and analysis_line is not None:
        try:
            delta = abs(float(line_taken) - float(analysis_line))
        except (TypeError, ValueError):
            delta = 0.0
        if delta > _LINE_DRIFT_WARN:
            logger.warning(
                "line drift trace=%s analysis_line=%s bet_line=%s delta=%.2f",
                trace_id,
                analysis_line,
                line_taken,
                delta,
            )

    odds_taken = bet_block.get("odds_taken")
    # The bet's selection_descriptor encodes the side (over/under). Compare
    # against the matching snapshot odds when we can resolve it; otherwise
    # compare to the closer of odds_over/odds_under.
    if odds_taken is not None:
        side_hint = str(bet_block.get("selection_descriptor", "")).lower()
        if "under" in side_hint:
            snap_odds = input_snap.get("odds_under")
        elif "over" in side_hint:
            snap_odds = input_snap.get("odds_over")
        else:
            snap_odds = None
        if snap_odds is not None:
            try:
                odds_delta = abs(float(odds_taken) - float(snap_odds))
            except (TypeError, ValueError):
                odds_delta = 0.0
            if odds_delta > _ODDS_DRIFT_WARN_AMERICAN:
                logger.warning(
                    "odds drift trace=%s analysis_odds=%s bet_odds=%s delta=%.0f",
                    trace_id,
                    snap_odds,
                    odds_taken,
                    odds_delta,
                )


def ingest_file(path: Path, store: TraceStore, dry_run: bool = False) -> tuple[str, str | None]:
    """Ingest one file. Returns (trace_id, bet_id or None). Raises on error."""
    payload = _load_payload(path)
    analyze_out = payload["trace"]

    # session_id may live on the trace object (preferred) or at the export-block
    # top level (fallback). Trace-level value wins if both are present.
    if not analyze_out.get("session_id") and payload.get("session_id"):
        analyze_out = {**analyze_out, "session_id": payload["session_id"]}

    adapted = _adapt_sandbox_trace(analyze_out)
    if not adapted["trace_id"]:
        raise ValueError("trace.trace_id is missing or empty")
    if not adapted["timestamp"]:
        raise ValueError("trace.ran_at is missing or empty")

    # Pre-persist validation: a bet_record on a prop trace must come with
    # full game identity (BUG-4 defense). Warnings about line/odds drift
    # (BUG-5) are emitted before we hand off to the store.
    bet_block = payload.get("bet_record")
    if isinstance(bet_block, dict):
        _validate_bet_with_prop_identity(
            adapted["trace_id"],
            adapted.get("kind", ""),
            adapted.get("input_snapshot") or {},
            bet_block,
        )
        _warn_drift(adapted["trace_id"], adapted.get("input_snapshot") or {}, bet_block)

    # P5: reject manual traces with no engine run and no model predictions.
    # These cannot contribute calibration pairs and inflate the graded-count metric.
    # Exception: sandbox_parlay traces are intentionally engine-less — they are
    # still ingested for bet-record purposes but noted here for clarity.
    downgrades = adapted.get("downgrades") or []
    if (
        "manual:no_engine_run" in downgrades
        and adapted.get("predictions") is None
        and adapted.get("execution_mode") != "sandbox_parlay"
    ):
        raise ValueError(
            f"Trace {adapted['trace_id']} has 'manual:no_engine_run' downgrade and no "
            "model predictions. Manual traces without predictions cannot contribute "
            "calibration pairs. Use analyze() to produce engine-run traces, or attach "
            "this trace to a bet_record manually via backfill_outcomes_manual.py."
        )

    if dry_run:
        return (adapted["trace_id"], None)

    trace_id = store.persist(adapted)

    bet_id: str | None = None
    if isinstance(bet_block, dict):
        # Phase 6h writes selection_descriptor directly on bet_record. Legacy
        # processed exports may still carry it on the retired sibling block.
        if "selection_descriptor" not in bet_block:
            clv = payload.get("clv_capture_instructions") or {}
            if clv.get("selection_descriptor"):
                bet_block = {**bet_block, "selection_descriptor": clv["selection_descriptor"]}
            else:
                raise ValueError("bet_record.selection_descriptor is required")
        bet = BetRecord.from_export_block(
            trace_id=trace_id,
            bet_id=uuid.uuid4().hex[:12],
            block=bet_block,
        )
        bet_id = store.record_bet(bet)

    return (trace_id, bet_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _move_to(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    # On collision, append a uuid suffix so we don't lose data
    if dst.exists():
        dst = dst_dir / f"{src.stem}.{uuid.uuid4().hex[:8]}{src.suffix}"
    shutil.move(str(src), str(dst))
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest sandbox trace exports into omega_traces.db"
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        default=_REPO_ROOT / "inbox" / "traces",
        help="Directory containing *.json trace exports",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite path (default: repo-root omega_traces.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and adapt but skip writes; do not move files",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    inbox: Path = args.inbox
    if not inbox.exists():
        logger.error("Inbox directory does not exist: %s", inbox)
        return 1

    processed_dir = inbox / "processed"
    failed_dir = inbox / "failed"

    files = sorted(p for p in inbox.glob("*.json") if p.is_file())
    if not files:
        logger.info("No new trace files in %s", inbox)
        return 0

    store = TraceStore(db_path=args.db)
    ok = 0
    failed = 0

    for path in files:
        try:
            trace_id, bet_id = ingest_file(path, store, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 — we want to capture every failure
            failed += 1
            logger.warning("FAILED %s: %s", path.name, exc)
            if not args.dry_run:
                moved = _move_to(path, failed_dir)
                error_path = moved.with_suffix(moved.suffix + ".error.txt")
                error_path.write_text(
                    f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
            continue

        ok += 1
        suffix = f" bet={bet_id}" if bet_id else ""
        logger.info("OK %s -> %s%s", path.name, trace_id, suffix)
        if not args.dry_run:
            _move_to(path, processed_dir)

    logger.info("Done. %d ingested, %d failed.", ok, failed)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
