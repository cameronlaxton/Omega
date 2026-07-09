"""
omega-ingest-traces â€” drain `var/inbox/traces/*.json` into `var/omega_traces.db`.

Workflow:
    1. Scan `var/inbox/traces/*.json` (non-recursive; processed/ and failed/ skipped unless --include-processed is passed).
    2. For each file: parse â†’ adapt analyze() output to TraceStore shape â†’ persist trace
       â†’ persist bet_record if present â†’ move file to processed/.
    3. On parse or persistence error: move file to failed/ with a sibling `.error.txt`.

Idempotent: re-running over the same processed/ directory is a no-op
(`TraceStore.persist()` uses INSERT OR IGNORE on trace_id, and bet_records uses
INSERT OR IGNORE on the (trace_id, market, selection_descriptor) UNIQUE).

Pre-persist export gate:
    Each file is validated by `omega/trace/export_validator.py` BEFORE any write.
    A wrong wrapper or incomplete export is rejected (routed to failed/) â€” the
    fix is to re-wrap/re-export, never to re-run analyze(). The default (lenient)
    error set mirrors what ingest already rejects (zero behavior change). Use
    --strict for fresh exports to also require export-quality fields, or
    --no-validate to bypass the gate entirely (rollback escape hatch).

Usage:
    omega-ingest-traces
    omega-ingest-traces --inbox <path> --db <path>
    omega-ingest-traces --dry-run
    omega-ingest-traces --explain            # no-write validation report
    omega-ingest-traces --strict             # fresh-export discipline
    omega-ingest-traces --no-validate        # bypass gate (rollback)
    omega-ingest-traces --include-processed  # recover traces stuck in processed/

Exit codes:
    0 â€” all files processed (some may have failed; check failed/)
    1 â€” fatal error before scanning (bad args, inbox missing)
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
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.paths import session_inbox_dir, trace_inbox_dir  # noqa: E402
from omega.trace.bet_record import BetRecord  # noqa: E402
from omega.trace.eligibility import REASON_QA_FAILED  # noqa: E402
from omega.trace.export_validator import validate_export_block  # noqa: E402
from omega.trace.persistable import PersistableTrace  # noqa: E402
from omega.trace.session_sidecar import (  # noqa: E402
    load_sidecar_safe,
    quality_gate_verdict_for_trace,
)
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("ingest_traces")


class _ArgSentinel:
    pass


_ARG_SENTINEL = _ArgSentinel()


# ---------------------------------------------------------------------------
# Adapters: engine analyze() output â†’ TraceStore.persist() shape
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
# warning but not worth failing ingest â€” line moves and odds drift are
# expected.
_LINE_DRIFT_WARN = 1.0
_ODDS_DRIFT_WARN_AMERICAN = 25


def _validate_bet_with_prop_identity(
    trace_id: str, kind: str, input_snap: dict[str, Any], bet_block: dict[str, Any]
) -> None:
    """BUG-4 defense: when a bet_record is attached to a prop trace, the trace
    MUST carry home_team/away_team/game_date so fetch_outcomes_props.py can
    resolve the box score.

    Enforces the OMEGA_RUNTIME.md Â§6a single-trace policy at the ingest seam.
    """
    if kind != "prop":
        return
    missing = [f for f in ("home_team", "away_team", "game_date") if not input_snap.get(f)]
    if missing:
        raise ValueError(
            f"prop trace {trace_id} carries a bet_record but is missing "
            f"input_snapshot fields {missing}. Per OMEGA_RUNTIME.md Â§6a the "
            "bet must attach to the original analysis trace (single-trace "
            "policy); do not mint a stripped-down confirmation trace."
        )


def _warn_drift(trace_id: str, input_snap: dict[str, Any], bet_block: dict[str, Any]) -> None:
    """BUG-5 defense: log a warning when bet_record.line_taken or odds_taken
    diverge meaningfully from the analysis trace's snapshot. Warnings only â€”
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


# Keys at the export-block root that are structural (not trace-level fields).
# These are never promoted into the inner trace by _merge_top_level_compat_fields.
_BLOCK_STRUCTURAL_KEYS = frozenset(
    {"trace", "bet_record", "clv_capture_instructions", "export_schema_version"}
)

_PROP_IDENTITY_FIELDS = ("player_name", "home_team", "away_team", "game_date", "line")
_REASONING_INPUTS_REQUIRED_KEYS = ("sources", "fields_gathered", "missing_fields")


def _merge_top_level_compat_fields(block: dict[str, Any]) -> None:
    """Promote sibling fields from an export block's root into the inner trace.

    Export blocks may carry reasoning/quality fields at the block root rather than
    nested inside 'trace'. This merges them into block["trace"] in-place so that
    PersistableTrace.from_analyze_output sees them. Inner values win: a key already
    present (and non-None) in the trace is never overwritten.

    Structural block keys ('trace', 'bet_record', 'clv_capture_instructions') are
    never promoted. Pattern-B payloads (no siblings beyond structural keys) are
    effectively a no-op.
    """
    trace = block.get("trace")
    if not isinstance(trace, dict):
        return
    for key, value in block.items():
        if key in _BLOCK_STRUCTURAL_KEYS:
            continue
        if key not in trace or trace[key] is None:
            trace[key] = value


def _warn_prop_identity(adapted: dict) -> None:
    """Warn when a prop trace is missing identity fields in input_snapshot.

    Fires for all prop traces â€” not just those with bet_records. Traces without
    full identity cannot support outcome attachment or calibration slice fitting.
    """
    if adapted.get("kind") != "prop":
        return
    snap = adapted.get("input_snapshot") or {}
    missing = [f for f in _PROP_IDENTITY_FIELDS if not snap.get(f)]
    if missing:
        logger.warning(
            "prop trace %s missing identity fields %s in input_snapshot â€” "
            "outcome attachment and calibration slice fitting unavailable",
            adapted.get("trace_id", "?"),
            missing,
        )


def _warn_empty_evidence(adapted: dict) -> None:
    """Warn when a trace carries no structured evidence signals.

    Empty evidence means evidence_signals rows will not be written and
    retrospective signal scoring will be unavailable for this trace.
    """
    snap = adapted.get("input_snapshot") or {}
    if not snap.get("evidence"):
        logger.warning(
            "trace %s has no structured evidence signals â€” "
            "retrospective evidence scoring unavailable for this trace",
            adapted.get("trace_id", "?"),
        )


def _warn_reasoning_inputs(analyze_out: dict) -> None:
    """Warn when reasoning_inputs is present but missing its expected minimal keys."""
    ri = analyze_out.get("reasoning_inputs")
    if ri is None or not isinstance(ri, dict):
        return
    for key in _REASONING_INPUTS_REQUIRED_KEYS:
        if key not in ri:
            logger.warning(
                "trace %s reasoning_inputs missing expected key '%s'",
                analyze_out.get("trace_id", "?"),
                key,
            )


def _load_session_sidecar(sidecar_dir: Path | None, session_id: str | None):
    """Return the parsed session sidecar, or None if absent/unreadable.

    None is deliberately ambiguous between "no sidecar" and "unreadable"; both
    mean the QA history is UNKNOWN. The caller warns only when a file existed
    but failed to parse, so configuring no sidecar dir stays quiet.
    """
    if sidecar_dir is None or not session_id:
        return None
    path = sidecar_dir / f"{session_id}.json"
    if not path.exists():
        return None
    return load_sidecar_safe(path)


def _append_once(values: list[Any], value: str) -> list[Any]:
    return values if value in values else [*values, value]


def _mark_calibration_ineligible(adapted: dict[str, Any], reason: str) -> None:
    """Mark an audit-only trace permanently fitter-invisible.

    Reconciles a failed trace-scoped QA verdict into the canonical
    trace_quality.calibration_eligible flag (the single source of truth read by
    the calibration query path). The trace is still persisted to the ledger.
    """
    trace_quality = dict(adapted.get("trace_quality") or {})
    reasons = list(trace_quality.get("calibration_exclusion_reasons") or [])
    downgrades = list(trace_quality.get("downgrades") or adapted.get("downgrades") or [])

    trace_quality["calibration_eligible"] = False
    trace_quality["passed"] = False
    trace_quality["calibration_exclusion_reasons"] = _append_once(reasons, reason)
    trace_quality["downgrades"] = _append_once(downgrades, reason)

    adapted["trace_quality"] = trace_quality
    adapted["downgrades"] = _append_once(list(adapted.get("downgrades") or []), reason)


def ingest_file(
    path: Path,
    store: TraceStore,
    dry_run: bool = False,
    *,
    sidecar_dir: Path | None | _ArgSentinel = _ARG_SENTINEL,
    force_ingest_qa_failed: bool = False,
    validate: bool = True,
    strict: bool = False,
) -> tuple[str, str | None]:
    """Ingest one file. Returns (trace_id, bet_id or None). Raises on error.

    QA handling is trace-scoped (see omega/trace/session_sidecar.py
    quality_gate_verdict_for_trace). A valid trace artifact is ALWAYS persisted
    to the ledger; only a malformed/invalid artifact is rejected to failed/. A
    failed trace-scoped QA verdict persists the trace as audit-only
    (calibration-ineligible) and records a trace_qa_verdicts row â€” it never
    blocks ingest, and an unrelated session failure never condemns this trace.
    ``force_ingest_qa_failed`` is retained for backward compatibility but is now
    a no-op: audit-only persistence is the default, and no flag can confer
    calibration eligibility on a QA-failed trace.

    The pre-persist export gate (`validate`/`strict`) runs the canonical export
    validator (`omega/trace/export_validator.py`) *before* any write. The default
    is `strict=False` (lenient): its error set mirrors what ingest already
    rejects, so the default gate is zero-behavior-change â€” it just routes the same
    rejections through one validator and catches a wrong wrapper shape loudly,
    with the fix being re-wrap/re-export, never re-running analyze(). `strict=True`
    (CLI `--strict`) additionally hard-fails fresh exports on missing export-
    quality fields (session_id, result.status, predictions, identity, NBA
    game_context). `validate=False` (`--no-validate`) is the rollback escape
    hatch â€” it skips the gate, but the inline integrity checks below (BUG-4 prop
    identity, manual-no-predictions) still apply.
    """
    if isinstance(sidecar_dir, _ArgSentinel):
        sidecar_dir = session_inbox_dir()

    payload = _load_payload(path)
    _merge_top_level_compat_fields(payload)
    analyze_out = payload["trace"]

    # session_id may live on the trace object (preferred) or at the export-block
    # top level (fallback). Trace-level value wins if both are present.
    # _merge_top_level_compat_fields already handles this for blocks that carry
    # session_id as a sibling; the guard below is kept for safety.
    if not analyze_out.get("session_id") and payload.get("session_id"):
        analyze_out = {**analyze_out, "session_id": payload["session_id"]}

    # Pre-persist export-shape/quality gate. A wrong wrapper or incomplete export
    # fails HERE, before any write â€” the fix is to re-wrap/re-export, never to
    # re-run analyze(). This is the single seam where the canonical validator
    # (omega/trace/export_validator.py) is enforced on the write path.
    if validate:
        report = validate_export_block(payload, strict=strict)
        if not report.ok:
            reasons = "; ".join(f"{i.code}: {i.message}" for i in report.errors)
            raise ValueError(
                f"trace export failed pre-persist validation "
                f"({'strict' if strict else 'lenient'}): {reasons}. "
                "Fix the export wrapper and re-drop the file; do NOT re-run analyze()."
            )

    adapted = _adapt_sandbox_trace(analyze_out)
    if not adapted["trace_id"]:
        raise ValueError("trace.trace_id is missing or empty")
    if not adapted["timestamp"]:
        raise ValueError("trace.ran_at is missing or empty")

    # Trace quality warnings â€” emitted for all traces regardless of bet_record.
    _warn_prop_identity(adapted)
    _warn_empty_evidence(adapted)
    _warn_reasoning_inputs(analyze_out)

    session_id = adapted.get("session_id")
    sid = str(session_id) if session_id else None
    sidecar_path = (sidecar_dir / f"{sid}.json") if (sidecar_dir is not None and sid) else None
    sidecar = _load_session_sidecar(sidecar_dir, sid)
    if sidecar is None and sidecar_path is not None and sidecar_path.exists():
        # File present but failed to parse: QA history is UNKNOWN, not clean.
        logger.warning(
            "Session %s sidecar unreadable; ingesting %s without QA confirmation. "
            "Quarantine the sidecar via `validate_session_sidecars.py --quarantine`.",
            session_id,
            adapted["trace_id"],
        )

    # Trace-scoped QA verdict. A failed verdict marks the trace audit-only
    # (calibration-ineligible) but never blocks ledger ingest; an unrelated
    # session failure leaves this trace eligible.
    qa_verdict = quality_gate_verdict_for_trace(
        sidecar, adapted["trace_id"], adapted.get("timestamp")
    )
    if qa_verdict.verdict == "fail":
        _mark_calibration_ineligible(adapted, REASON_QA_FAILED)
        logger.info(
            "trace %s QA verdict=fail scope=%s; persisting audit-only (calibration-ineligible): %s",
            adapted["trace_id"],
            qa_verdict.scope,
            qa_verdict.reason,
        )

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

    # P5: reject manual traces with no engine run, no model predictions, and no
    # bet_record. These cannot contribute calibration pairs and carry no
    # bet-record audit value either, so they'd only inflate the graded-count
    # metric. Exception: any trace carrying a real bet_record (sandbox_parlay
    # included) is intentionally engine-less but still ingested for
    # bet-record/ledger purposes -- the calibration-ineligibility is already
    # enforced separately via predictions being None.
    downgrades = adapted.get("downgrades") or []
    if (
        "manual:no_engine_run" in downgrades
        and adapted.get("predictions") is None
        and adapted.get("execution_mode") != "sandbox_parlay"
        and not isinstance(bet_block, dict)
    ):
        raise ValueError(
            f"Trace {adapted['trace_id']} has 'manual:no_engine_run' downgrade, no "
            "model predictions, and no bet_record. Manual traces without predictions "
            "cannot contribute calibration pairs and carry no bet-record audit value. "
            "Use analyze() to produce an engine-run trace, or add a bet_record so this "
            "can still be ingested for ledger/audit purposes."
        )

    if dry_run:
        return (adapted["trace_id"], None)

    trace_id = store.persist(adapted)

    # Record the trace-scoped QA verdict as an audit row (the canonical
    # eligibility flag already lives in trace_quality). Skip the trivial
    # no_sidecar case to avoid noise for sessions without a sidecar.
    if qa_verdict.scope != "no_sidecar":
        store.write_qa_verdict(
            trace_id,
            qa_verdict,
            session_id=sid,
            ran_at=adapted.get("timestamp"),
        )

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
        description="Ingest sandbox trace exports into var/omega_traces.db"
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        default=None,
        help="Directory containing *.json trace exports (default: var/inbox/traces)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite path (default: var/omega_traces.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and adapt but skip writes; do not move files",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Dry-run + per-file validation report (VALID/REJECT with reasons). "
            "Use this to fix a wrong export wrapper instead of re-running analyze()."
        ),
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help="Directory containing <session_id>.json sidecars for QA gates (default: var/inbox/sessions)",
    )
    parser.add_argument(
        "--allow-audit-only-qa-failed",
        "--force-ingest-qa-failed",
        dest="force_ingest_qa_failed",
        action="store_true",
        help=(
            "DEPRECATED / no-op. QA-failed traces are now persisted audit-only "
            "by default (trace-scoped) and remain calibration-ineligible "
            "regardless of this flag; no flag can confer calibration eligibility. "
            "Kept for backward compatibility."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Hard-fail fresh exports on missing export-quality fields (session_id, "
            "result.status, predictions, identity, NBA game_context) before persist. "
            "Default (lenient) mirrors the historical ingest hard-fail set."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "ROLLBACK ESCAPE HATCH: skip the pre-persist export gate entirely. "
            "Core integrity checks (BUG-4 prop identity, manual-no-predictions) "
            "still apply. Use only to bypass a regression in the gate."
        ),
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help=(
            "Also scan <inbox>/processed/*.json for recovery. Since ingest is "
            "INSERT OR IGNORE, already-ingested traces are silently skipped. "
            "Use when traces were moved to processed/ before being persisted "
            "(e.g. after a stash-pop conflict or partial session export)."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    inbox: Path = args.inbox if args.inbox is not None else trace_inbox_dir()
    sidecar_dir: Path = args.sidecar_dir if args.sidecar_dir is not None else session_inbox_dir()
    if not inbox.exists():
        logger.error("Inbox directory does not exist: %s", inbox)
        return 1

    processed_dir = inbox / "processed"
    failed_dir = inbox / "failed"

    files = sorted(p for p in inbox.glob("*.json") if p.is_file())
    if args.include_processed and processed_dir.exists():
        recovered = sorted(p for p in processed_dir.glob("*.json") if p.is_file())
        if recovered:
            logger.info(
                "--include-processed: found %d file(s) in processed/ to recover.", len(recovered)
            )
            files = recovered + files
    if not files:
        logger.info("No new trace files in %s", inbox)
        return 0

    store = TraceStore(db_path=args.db)
    log_effective_db(store, logger)
    if args.no_validate:
        logger.warning(
            "PRE-PERSIST EXPORT GATE DISABLED (--no-validate): traces are being "
            "ingested without export-shape/quality validation. Inline integrity "
            "checks still apply."
        )
    elif args.strict:
        logger.info("Pre-persist export gate: STRICT mode (fresh exports).")
    ok = 0
    failed = 0

    # --explain is a no-write reporting mode: validate each file and print the
    # exact reason it would be rejected, so a malformed *wrapper* on otherwise
    # valid analyze() output gets re-wrapped rather than re-analyzed.
    explain = args.explain
    dry_run = args.dry_run or explain

    for path in files:
        if explain:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                failed += 1
                logger.warning("REJECT %s: unreadable JSON: %s", path.name, exc)
                continue
            report = validate_export_block(payload, strict=False)
            if report.ok:
                ok += 1
                warns = ",".join(i.code for i in report.warnings)
                logger.info(
                    "VALID %s -> %s [%s]%s",
                    path.name,
                    report.trace_id,
                    report.kind,
                    f" warn={warns}" if warns else "",
                )
            else:
                failed += 1
                reasons = "; ".join(f"{i.code}: {i.message}" for i in report.errors)
                logger.warning("REJECT %s: %s", path.name, reasons)
            continue

        try:
            trace_id, bet_id = ingest_file(
                path,
                store,
                dry_run=dry_run,
                sidecar_dir=sidecar_dir,
                force_ingest_qa_failed=args.force_ingest_qa_failed,
                validate=not args.no_validate,
                strict=args.strict,
            )
        except Exception as exc:  # noqa: BLE001 â€” we want to capture every failure
            failed += 1
            logger.warning("FAILED %s: %s", path.name, exc)
            if not dry_run:
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
        if not dry_run:
            _move_to(path, processed_dir)

    if explain:
        logger.info("Explain (no writes): %d valid, %d rejected.", ok, failed)
    else:
        logger.info("Done. %d ingested, %d failed.", ok, failed)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
