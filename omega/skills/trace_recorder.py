"""
omega.skills.trace_recorder — persist ExecutionTrace to SQLite.

Invariants:
- trace_id, run_id, and timestamp must be present
- The current trace schema version (omega.trace.schema.CURRENT_VERSION) is
  injected before write — never a hard-coded integer
- Persistence: omega.trace.store.TraceStore (SQLite)
- Failure mode: loud — a SQLite write failure surfaces as a finding on the
  SkillObservation. No silent JSONL fallback (Phase 6h: favor visible
  failure over a fallback path that has no consumer).
"""

from __future__ import annotations

import logging
from typing import Any

from omega.skills import register
from omega.skills.base import SkillBase, SkillObservation
from omega.trace.schema import CURRENT_VERSION

logger = logging.getLogger("omega.skills.trace_recorder")

_REQUIRED_FIELDS = {"trace_id", "run_id", "timestamp"}


@register("trace-recorder")
class TraceRecorder(SkillBase):
    name = "trace-recorder"
    stage = "composition"

    def _run(self, *, trace: dict[str, Any], **_: Any) -> SkillObservation:  # type: ignore[override]
        findings: list[str] = []

        # Validate required fields
        missing = _REQUIRED_FIELDS - set(trace.keys())
        if missing:
            findings.extend(f"missing_field:{f}" for f in sorted(missing))
            return SkillObservation(
                skill=self.name,
                stage=self.stage,
                ok=False,
                findings=findings,
            )

        # Inject schema version (matches the table's current migration version
        # so the recorded blob never lies about its on-disk shape)
        record = dict(trace)
        record["schema_version"] = CURRENT_VERSION

        # Write
        write_error = _write_trace(record)
        if write_error:
            findings.append(f"write_failed:{write_error}")
            return SkillObservation(
                skill=self.name,
                stage=self.stage,
                ok=False,
                findings=findings,
            )

        return SkillObservation(skill=self.name, stage=self.stage, ok=True)


def _write_trace(record: dict[str, Any]) -> str:
    """Persist trace to SQLite. Returns empty string on success, error message
    on failure."""
    try:
        from omega.trace.store import TraceStore

        store = TraceStore()
        store.persist(record)
        store.close()
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("trace-recorder SQLite persist failed: %s", exc)
        return str(exc)
