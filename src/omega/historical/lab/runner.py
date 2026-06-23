"""In-process command runner + JSONL audit ledger for a lab run.

The orchestrator chains existing ``main(argv) -> int`` CLIs and library calls.
Rather than shell out (which loses structured exit handling and is slow on the
Cowork FUSE mount), it invokes them in-process and records each step as a
reused :class:`~omega.trace.session_sidecar.AuditEvent` (``event_type="command"``)
appended to ``command_log.jsonl``. Appending per-step means a crash still leaves
a partial, inspectable trail.

Exit-code policy: ``0`` → ``ok``; any other *allowed* code → ``warn`` (e.g.
historical-live parity exit 2 = INCONCLUSIVE is allowed and non-fatal); a
disallowed code on a required step raises :class:`LabStepError`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omega.trace.session_sidecar import AuditEvent

UTC = timezone.utc


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class LabStepError(RuntimeError):
    """A required lab step returned a disallowed exit code."""

    def __init__(self, step: str, exit_code: int, allowed: Sequence[int]) -> None:
        self.step = step
        self.exit_code = exit_code
        self.allowed = tuple(allowed)
        super().__init__(f"lab step {step!r} failed: exit {exit_code} (allowed {self.allowed})")


class LabCommandRunner:
    """Run steps in-process, logging each as an AuditEvent to a JSONL ledger."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[AuditEvent] = []

    def _emit(self, event: AuditEvent) -> AuditEvent:
        self.events.append(event)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.model_dump(mode="json"), separators=(",", ":")))
            fh.write("\n")
        return event

    def record(
        self,
        step: str,
        status: str,
        *,
        notes: str | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        trace_ids: Iterable[str] | None = None,
    ) -> AuditEvent:
        """Log a non-CLI step (grid, seal, evidence, promote-decision)."""
        return self._emit(
            AuditEvent(
                ts=_now_iso(),
                event_type="command",
                step=step,
                status=status,
                notes=notes,
                inputs=inputs,
                outputs=outputs,
                trace_ids=list(trace_ids or []),
            )
        )

    def run_cli(
        self,
        step: str,
        main_fn: Callable[[list[str]], int],
        argv: Sequence[str],
        *,
        required: bool = True,
        ok_exits: Sequence[int] = (0,),
    ) -> int:
        """Invoke an existing ``main(argv) -> int`` CLI in-process and log the result."""
        argv = list(argv)
        started = time.monotonic()
        rc = int(main_fn(argv))
        elapsed = round(time.monotonic() - started, 3)

        allowed = rc in ok_exits
        if rc == 0:
            status = "ok"
        elif allowed:
            status = "warn"
        else:
            status = "fail"

        self.record(
            step,
            status,
            inputs={"argv": argv},
            outputs={"exit_code": rc, "elapsed_s": elapsed, "allowed": allowed},
        )
        if required and not allowed:
            raise LabStepError(step, rc, ok_exits)
        return rc
