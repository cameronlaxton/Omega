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
from typing import Any, Dict, Optional, Tuple

# Allow running as a script from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.trace.bet_record import BetRecord  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


logger = logging.getLogger("ingest_traces")


# ---------------------------------------------------------------------------
# Adapters: engine analyze() output → TraceStore.persist() shape
# ---------------------------------------------------------------------------

def _adapt_sandbox_trace(analyze_out: Dict[str, Any]) -> Dict[str, Any]:
    """Map an `analyze()` return value to the TraceStore persist schema.

    TraceStore.persist requires: trace_id, run_id, timestamp. It uses other
    denormalized fields (league, matchup, execution_mode, ...) for querying but
    falls back gracefully if absent. The `full_trace` blob preserves the raw
    analyze output for downstream reconstruction.
    """
    trace_id = analyze_out.get("trace_id", "")
    ran_at = analyze_out.get("ran_at") or analyze_out.get("analyzed_at") or ""
    kind = analyze_out.get("kind", "unknown")
    input_snap = analyze_out.get("input_snapshot") or {}
    result = analyze_out.get("result") or {}
    gate = analyze_out.get("quality_gate") or {}

    league = input_snap.get("league") or result.get("league") or ""
    matchup = _derive_matchup(kind, input_snap, result)
    seed = input_snap.get("seed")
    aggregate_quality = gate.get("aggregate_quality")
    downgrades = gate.get("downgrades") or []

    # Predictions/recommendations vary by kind; we pull what's available
    predictions = result.get("simulation") if kind == "game" else {
        k: result.get(k) for k in ("over_prob", "under_prob") if result.get(k) is not None
    } or None
    recommendations = result.get("edges") or result.get("best_bet") or None
    odds_snapshot = input_snap.get("odds") or _prop_odds_snapshot(input_snap) or None

    return {
        "trace_id": trace_id,
        # run_id: one analyze() call = one run in sandbox path
        "run_id": analyze_out.get("run_id") or trace_id,
        "timestamp": ran_at,
        "prompt": _derive_prompt(kind, input_snap, league, matchup),
        "league": league,
        "matchup": matchup,
        "execution_mode": f"sandbox_{kind}",
        "simulation_seed": seed,
        "aggregate_quality": aggregate_quality,
        "predictions": predictions,
        "recommendations": recommendations,
        "odds_snapshot": odds_snapshot,
        "downgrades": downgrades,
        "session_id": analyze_out.get("session_id"),
        # Preserve the raw analyze output for downstream consumers
        "model_version": analyze_out.get("model_version"),
        "kind": kind,
        "input_snapshot": input_snap,
        "result": result,
        "quality_gate": gate,
    }


def _derive_matchup(kind: str, input_snap: Dict[str, Any], result: Dict[str, Any]) -> str:
    if kind == "game":
        home = input_snap.get("home_team") or ""
        away = input_snap.get("away_team") or ""
        if home and away:
            return f"{away} @ {home}"
    if kind == "prop":
        player = input_snap.get("player_name") or ""
        prop = input_snap.get("prop_type") or ""
        line = input_snap.get("line")
        if player and prop:
            line_str = f" {line}" if line is not None else ""
            return f"{player} {prop}{line_str}"
    return result.get("matchup", "") or ""


def _derive_prompt(kind: str, input_snap: Dict[str, Any], league: str, matchup: str) -> str:
    base = f"{league} {kind}: {matchup}".strip()
    return base or json.dumps(input_snap, default=str)[:200]


def _prop_odds_snapshot(input_snap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    over = input_snap.get("odds_over")
    under = input_snap.get("odds_under")
    if over is None and under is None:
        return None
    return {"odds_over": over, "odds_under": under}


# ---------------------------------------------------------------------------
# File-level ingest
# ---------------------------------------------------------------------------

def _load_payload(path: Path) -> Dict[str, Any]:
    """Parse the JSON file and return the export-block dict.

    Accepts two shapes:
      A) The system-prompt §10 export block: {"trace": {...}, "bet_record": ..., "clv_capture_instructions": ...}
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
        return {"trace": payload, "bet_record": None, "clv_capture_instructions": None}  # shape B
    raise ValueError("JSON must contain either 'trace' (export block) or top-level 'trace_id'+'kind'")


def ingest_file(path: Path, store: TraceStore, dry_run: bool = False) -> Tuple[str, Optional[str]]:
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

    if dry_run:
        return (adapted["trace_id"], None)

    trace_id = store.persist(adapted)

    bet_id: Optional[str] = None
    bet_block = payload.get("bet_record")
    if isinstance(bet_block, dict):
        # selection_descriptor may live on the sibling clv_capture_instructions
        clv = payload.get("clv_capture_instructions") or {}
        if "selection_descriptor" not in bet_block and clv.get("selection_descriptor"):
            bet_block = {**bet_block, "selection_descriptor": clv["selection_descriptor"]}
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
    parser = argparse.ArgumentParser(description="Ingest sandbox trace exports into omega_traces.db")
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

    files = sorted(
        p for p in inbox.glob("*.json")
        if p.is_file()
    )
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
