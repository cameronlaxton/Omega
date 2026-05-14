"""
omega.skills.trace_recorder — persist ExecutionTrace to SQLite + JSONL fallback.

Invariants:
- trace_id, run_id, and timestamp must be present
- schema_version:1 is injected before write
- Primary persistence: omega.trace.store.TraceStore (SQLite)
- Fallback: JSONL at omega/skills/logs/traces.jsonl
- Findings are diagnostic only; write failures do not propagate
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from omega.skills import register
from omega.skills.base import SkillBase, SkillObservation

logger = logging.getLogger("omega.skills.trace_recorder")

_SCHEMA_VERSION = 1
_REQUIRED_FIELDS = {"trace_id", "run_id", "timestamp"}


@register("trace-recorder")
class TraceRecorder(SkillBase):
    name = "trace-recorder"
    stage = "composition"

    def _run(self, *, trace: Dict[str, Any], **_: Any) -> SkillObservation:
        findings: list[str] = []

        # Validate required fields
        missing = _REQUIRED_FIELDS - set(trace.keys())
        if missing:
            findings.extend(f"missing_field:{f}" for f in sorted(missing))
            return SkillObservation(
                skill=self.name, stage=self.stage, ok=False, findings=findings,
            )

        # Inject schema version
        record = dict(trace)
        record["schema_version"] = _SCHEMA_VERSION

        # Write
        write_error = _write_trace(record)
        if write_error:
            findings.append(f"write_failed:{write_error}")
            return SkillObservation(
                skill=self.name, stage=self.stage, ok=False, findings=findings,
            )

        return SkillObservation(skill=self.name, stage=self.stage, ok=True)


def _resolve_log_path() -> Path:
    """Resolve JSONL path from config or fall back to package default."""
    try:
        from omega.skills import config as skill_config
        cfg = skill_config()
        log_path = cfg.get("log_path", "omega/skills/logs/")
    except Exception:
        log_path = "omega/skills/logs/"

    # Resolve relative to repo root (two levels up from this file)
    base = Path(__file__).parent.parent.parent
    resolved = base / log_path
    return resolved / "traces.jsonl"


def _write_trace(record: Dict[str, Any]) -> str:
    """Persist trace to SQLite, falling back to JSONL on failure.

    Returns empty string on success, error message on failure.
    """
    # Primary: SQLite via TraceStore
    try:
        from omega.trace.store import TraceStore
        store = TraceStore()
        store.persist(record)
        store.close()
        return ""
    except Exception as exc:
        logger.debug("SQLite persist failed, falling back to JSONL: %s", exc)

    # Fallback: JSONL
    try:
        path = _resolve_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return ""
    except Exception as exc:
        logger.warning("trace-recorder write error (both paths failed): %s", exc)
        return str(exc)
