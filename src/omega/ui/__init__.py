"""omega.ui — read-only operator console (Phase 8, Milestone A).

This package holds the backend-neutral read service, GET-only JSON API, and
Pydantic response schemas for ``omega-console`` (entry point:
``omega.ops.console_server``). Everything here is strictly read-only: it opens
``TraceStore(read_only=True)`` and reads validated session sidecars from the
filesystem. No module in this package may import or invoke a state-mutating
operation (settle, attach outcome, persist, promote, ingest, quarantine, write
sidecar, or any DB write). See ``tests/ui/test_console_no_mutation_imports.py``.
"""

from __future__ import annotations
