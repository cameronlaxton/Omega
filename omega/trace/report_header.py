"""Derived-artifact front-matter header for generated Omega reports.

Every generated markdown report (`reports/latest.md`,
`reports/run_audits/<sid>.audit.md`, …) is *derived* from the ledger and
sidecars — never a source of truth. This module emits a uniform YAML front-matter
block at the top of those files so a human or script can immediately see the file
is derived, which DB it was generated from, and how many traces existed at
generation time.

See `docs/phase6/ARTIFACT_AUTHORITY.md` for the source-of-truth rules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # avoid a runtime import cycle (store imports nothing from here)
    from omega.trace.store import TraceStore

UTC = timezone.utc


def render_derived_header(
    *,
    source_db_path: str,
    db_path_source: str,
    trace_count_at_generation: int,
    source_artifacts: Sequence[str],
    generated_at: str | None = None,
    canonical: bool = False,
) -> str:
    """Return a YAML front-matter block marking a report as a derived artifact.

    The block is the standard markdown front-matter form (delimited by ``---``)
    so it parses as metadata and renders unobtrusively. It must be the very first
    thing in the file, before any heading.
    """
    ts = generated_at or datetime.now(UTC).isoformat()
    lines = [
        "---",
        f"canonical: {str(canonical).lower()}",
        f"generated_at: {ts!r}",
        f"source_db_path: {source_db_path!r}",
        f"db_path_source: {db_path_source!r}",
        f"trace_count_at_generation: {int(trace_count_at_generation)}",
        "source_artifacts:",
    ]
    for artifact in source_artifacts:
        lines.append(f"  - {artifact}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def header_for_store(
    store: "TraceStore",
    source_artifacts: Sequence[str],
    *,
    generated_at: str | None = None,
) -> str:
    """Convenience wrapper: build a derived header from a live TraceStore.

    Reads the effective DB path, how it was resolved, and the trace count at
    generation time straight off the store so reports always name the DB they
    actually read.
    """
    return render_derived_header(
        source_db_path=store.db_path,
        db_path_source=store.db_path_source,
        trace_count_at_generation=store.count(),
        source_artifacts=source_artifacts,
        generated_at=generated_at,
    )
