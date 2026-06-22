"""
omega.trace.export_validator — pre-ingest validation of trace export files.

Purpose: tell the agent whether an exported trace will ingest cleanly **before**
it runs ingest — and, when the raw ``analyze()`` output is valid but the export
wrapper is wrong, make it obvious that the fix is to re-wrap/re-export, NOT to
re-run ``analyze()``.

Two modes:
- ``strict=True`` (default for new formal exports): the extra export-quality
  checks (session_id, result.status, prediction fields, identity, NBA
  game_context) are ERRORS.
- ``strict=False`` (lenient; for legacy/backfill inspection): those extra checks
  are downgraded to WARNINGS. The set of *errors* in lenient mode mirrors what
  ``omega-ingest-traces`` actually rejects, so lenient ⊆ ingest reality.

This module is the single home of export-shape detection (``split_export_block``);
``ingest_traces.py`` keeps its own historical ValueError messages but the shape
rules are intentionally identical and covered by tests asserting parity.

It NEVER tightens ingestion: ingest's acceptance gate is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from omega.trace.bet_record import BetRecord
from omega.trace.persistable import PersistableTrace

Level = Literal["error", "warn"]

_NBA_GAME_CONTEXT_FIELDS = ("is_playoff", "rest_days")
_PROP_IDENTITY_FIELDS = ("player_name", "prop_type", "line", "home_team", "away_team", "game_date")
_GAME_IDENTITY_FIELDS = ("home_team", "away_team")


@dataclass
class ValidationIssue:
    level: Level
    code: str
    message: str


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    trace_id: str | None = None
    kind: str | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warn"]

    def summary(self) -> str:
        if self.ok and not self.issues:
            return f"VALID {self.trace_id or '?'} [{self.kind or '?'}]"
        head = "VALID" if self.ok else "REJECT"
        parts = [f"{head} {self.trace_id or '?'} [{self.kind or '?'}]"]
        for i in self.issues:
            parts.append(f"  [{i.level}] {i.code}: {i.message}")
        return "\n".join(parts)


def split_export_block(
    payload: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Return (trace, bet_record, clv_instructions, error).

    Mirrors ``ingest_traces._load_payload`` shape detection without raising:
      A) ``{"trace": {...}, "bet_record": ..., "clv_capture_instructions": ...}``
      B) raw ``analyze()`` output with top-level ``trace_id`` + ``kind``.
    On an unsupported shape, returns ``error`` set and the rest None.
    """
    if not isinstance(payload, dict):
        return None, None, None, f"top-level JSON must be an object, got {type(payload).__name__}"
    if "trace" in payload and isinstance(payload["trace"], dict):
        return (
            payload["trace"],
            payload.get("bet_record"),
            payload.get("clv_capture_instructions"),
            None,
        )
    if "trace_id" in payload and "kind" in payload:
        return payload, None, None, None
    return (
        None,
        None,
        None,
        "missing outer 'trace' block and no top-level 'trace_id'+'kind' "
        "(unsupported wrapper shape)",
    )


def _has_predictions(adapted: dict[str, Any]) -> bool:
    preds = adapted.get("predictions")
    if isinstance(preds, dict):
        return any(v is not None for v in preds.values())
    if isinstance(preds, list):
        return len(preds) > 0
    return preds is not None


def _identity_missing(kind: str | None, snap: dict[str, Any]) -> list[str]:
    fields = _PROP_IDENTITY_FIELDS if kind == "prop" else _GAME_IDENTITY_FIELDS
    return [f for f in fields if snap.get(f) in (None, "")]


def validate_export_block(
    payload: Any,
    *,
    strict: bool = True,
    league_hint: str | None = None,
) -> ValidationReport:
    """Validate one export-block dict. See module docstring for strict/lenient."""
    issues: list[ValidationIssue] = []
    trace, bet, clv, shape_err = split_export_block(payload)
    if shape_err or trace is None:
        return ValidationReport(
            ok=False,
            issues=[ValidationIssue("error", "shape", shape_err or "no trace")],
        )

    # session_id fallback (block top-level), mirroring ingest.
    session_id = trace.get("session_id")
    if not session_id and isinstance(payload, dict):
        session_id = payload.get("session_id")

    try:
        adapted = PersistableTrace.from_analyze_output(trace).to_store_record()
    except Exception as exc:  # noqa: BLE001
        return ValidationReport(
            ok=False,
            issues=[
                ValidationIssue(
                    "error",
                    "adapt_failed",
                    f"PersistableTrace.from_analyze_output failed: {exc}",
                )
            ],
            trace_id=trace.get("trace_id"),
            kind=trace.get("kind"),
        )

    trace_id = adapted.get("trace_id")
    kind = adapted.get("kind")
    snap = adapted.get("input_snapshot") or {}
    result = adapted.get("result") or trace.get("result") or {}

    # ── Errors that mirror ingest's hard-fails (apply in both modes) ──────────
    if not trace_id:
        issues.append(ValidationIssue("error", "trace_id", "trace.trace_id is missing or empty"))
    if not adapted.get("timestamp"):
        issues.append(
            ValidationIssue("error", "ran_at", "trace.ran_at/timestamp is missing or empty")
        )

    if kind == "prop" and isinstance(bet, dict):
        miss = [f for f in ("home_team", "away_team", "game_date") if not snap.get(f)]
        if miss:
            issues.append(
                ValidationIssue(
                    "error",
                    "prop_bet_identity",
                    f"prop trace carries a bet_record but is missing identity {miss} "
                    "(OMEGA_COWORK.md sec.6 single-trace policy)",
                )
            )

    downgrades = adapted.get("downgrades") or []
    if (
        "manual:no_engine_run" in downgrades
        and adapted.get("predictions") is None
        and adapted.get("execution_mode") != "sandbox_parlay"
    ):
        issues.append(
            ValidationIssue(
                "error",
                "manual_no_predictions",
                "manual:no_engine_run with no model predictions cannot contribute calibration pairs",
            )
        )

    if isinstance(bet, dict):
        b = dict(bet)
        if (
            "selection_descriptor" not in b
            and isinstance(clv, dict)
            and clv.get("selection_descriptor")
        ):
            b["selection_descriptor"] = clv["selection_descriptor"]
        try:
            BetRecord.from_export_block(trace_id=trace_id or "x", bet_id="x", block=b)
        except Exception as exc:  # noqa: BLE001
            issues.append(ValidationIssue("error", "bet_record", f"malformed bet_record: {exc}"))

    # ── Strict export-quality checks (error in strict, warn in lenient) ───────
    lvl: Level = "error" if strict else "warn"
    if not session_id:
        issues.append(ValidationIssue(lvl, "session_id", "missing session_id"))
    if not result.get("status"):
        issues.append(ValidationIssue(lvl, "result_status", "missing result.status"))
    if not _has_predictions(adapted):
        issues.append(ValidationIssue(lvl, "predictions", "missing model prediction fields"))
    ident_missing = _identity_missing(kind, snap)
    if ident_missing:
        issues.append(
            ValidationIssue(
                lvl, "identity", f"missing {kind or 'game'} identity fields {ident_missing}"
            )
        )
    league = str(snap.get("league") or trace.get("league") or league_hint or "").upper()
    if league == "NBA":
        gc = snap.get("game_context") or result.get("game_context") or {}
        gc_missing = [f for f in _NBA_GAME_CONTEXT_FIELDS if f not in gc]
        if gc_missing:
            issues.append(
                ValidationIssue(
                    lvl, "nba_game_context", f"NBA trace missing game_context {gc_missing}"
                )
            )

    ok = not any(i.level == "error" for i in issues)
    return ValidationReport(ok=ok, issues=issues, trace_id=trace_id, kind=kind)
