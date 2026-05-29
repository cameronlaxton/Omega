"""Deterministic session audit renderer.

Inputs: a ``SessionSidecar`` JSON file + the ``omega_traces.db`` ledger.
Output: ``reports/run_audits/<session_id>.audit.md`` written atomically.

Hard rule: any numeric/quant value shown in the rendered markdown
(probabilities, edge%, EV%, Kelly, units, confidence tiers, fair/no-vig
prices, model probabilities, trace IDs) is sourced from the DB row, not
from sidecar prose. The sidecar contributes only QA narrative:
``audit_events``, ``agent_notes``, ``exec_stats``.

This module must never read ``RUN_TRACE.jsonl`` or ``RUN_AUDIT.md`` —
those legacy artifacts are retired.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omega.trace._atomic import atomic_write_text
from omega.trace.report_header import header_for_store
from omega.trace.session_sidecar import (
    SessionSidecar,
    bootstrap_payload,
    load_sidecar_safe,
)
from omega.trace.store import TraceStore

logger = logging.getLogger("omega.trace.audit_renderer")


def render_session_audit(
    session_id: str,
    *,
    db_path: str | Path | None,
    sidecar_dir: Path,
    out_dir: Path,
) -> Path:
    """Render one session's audit markdown. Returns the written path.

    If the sidecar does not exist — or exists but is malformed/unreadable — a
    minimal one is synthesized in-memory (NOT written back to disk) so the audit
    still shows the DB cross-section and is logged as ``degraded``. A malformed
    sidecar never crashes the render.
    """
    sidecar_path = Path(sidecar_dir) / f"{session_id}.json"
    loaded = load_sidecar_safe(sidecar_path) if sidecar_path.exists() else None
    if loaded is not None:
        sidecar = loaded
        degraded = False
    else:
        sidecar = SessionSidecar.model_validate(
            {
                **bootstrap_payload(
                    session_id,
                    model_version="unknown",
                    purpose=f"auto-bootstrapped audit for {session_id}",
                    bankroll=1.0,
                ),
                "audit_events": [],
            }
        )
        degraded = True
        logger.warning(
            "audit_renderer: no readable sidecar at %s; rendering DB-only (degraded)",
            sidecar_path,
        )

    store = TraceStore(db_path=str(db_path) if db_path else None, read_only=False)
    try:
        rows = store.query_by_session(session_id)
        # Derived-artifact front-matter, built from the live store so the audit
        # always names the DB it was rendered from (ARTIFACT_AUTHORITY.md).
        header = header_for_store(
            store, ["omega_traces.db", f"inbox/sessions/{session_id}.json"]
        )
    finally:
        store.close()

    markdown = header + _render_markdown(sidecar, rows, degraded=degraded)

    out_path = Path(out_dir) / f"{session_id}.audit.md"
    atomic_write_text(out_path, markdown)
    return out_path


def _render_markdown(
    sidecar: SessionSidecar,
    rows: list[dict[str, Any]],
    *,
    degraded: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# Session Audit — {sidecar.session_id}")
    lines.append("")
    if degraded:
        lines.append("> **Status: degraded** — no sidecar file. DB cross-section only.")
        lines.append("")

    # ---------------- Session header (sidecar) ----------------
    lines.append("## Session")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| session_id | `{sidecar.session_id}` |")
    lines.append(f"| opened_at | {sidecar.opened_at} |")
    lines.append(f"| closed_at | {sidecar.closed_at or '—'} |")
    lines.append(f"| model_version | {sidecar.model_version} |")
    lines.append(f"| bankroll | {sidecar.bankroll} (confirmed={sidecar.bankroll_confirmed}) |")
    lines.append(f"| purpose | {sidecar.purpose} |")
    lines.append("")

    # ---------------- Engine exec_stats (sidecar) ----------------
    if sidecar.exec_stats:
        lines.append("## exec_stats")
        lines.append("")
        lines.append("| Key | Value |")
        lines.append("|---|---|")
        for key in sorted(sidecar.exec_stats):
            lines.append(f"| {key} | {sidecar.exec_stats[key]} |")
        lines.append("")

    # ---------------- Traces (DB — source of truth for quant) ----------------
    lines.append(f"## Traces ({len(rows)})")
    lines.append("")
    if not rows:
        lines.append("_No traces in `omega_traces.db` for this session._")
        lines.append("")
    else:
        lines.append(
            "| trace_id | kind | league | matchup | quality | outcome |"
        )
        lines.append("|---|---|---|---|---|---|")
        for row in rows:
            meta = row.get("_row") or {}
            outcome = row.get("_outcome")
            prop_outcomes = row.get("_prop_outcomes") or []
            outcome_str = "—"
            if outcome:
                outcome_str = (
                    f"{outcome.get('result')} "
                    f"({outcome.get('home_score')}-{outcome.get('away_score')})"
                )
            elif prop_outcomes:
                outcome_str = ", ".join(
                    f"{p.get('player_name')} {p.get('stat_type')}={p.get('stat_value')} "
                    f"({p.get('result')})"
                    for p in prop_outcomes[:3]
                )
                if len(prop_outcomes) > 3:
                    outcome_str += f", +{len(prop_outcomes) - 3} more"
            quality = meta.get("aggregate_quality")
            quality_str = "—" if quality is None else f"{quality:.2f}"
            lines.append(
                f"| `{meta.get('trace_id', '?')}` "
                f"| {meta.get('kind', '?')} "
                f"| {meta.get('league') or '—'} "
                f"| {meta.get('matchup') or '—'} "
                f"| {quality_str} "
                f"| {outcome_str} |"
            )
        lines.append("")

    # ---------------- Bets (DB) ----------------
    bet_rows: list[dict[str, Any]] = []
    for row in rows:
        for bet in row.get("_bet_records") or []:
            bet_rows.append(bet)
    if bet_rows:
        lines.append(f"## Bets ({len(bet_rows)})")
        lines.append("")
        lines.append("| trace_id | book | market | selection | line | odds | units | status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for bet in bet_rows:
            lines.append(
                f"| `{bet.get('trace_id')}` "
                f"| {bet.get('book') or '—'} "
                f"| {bet.get('market') or '—'} "
                f"| {bet.get('selection') or '—'} "
                f"| {bet.get('line_taken') if bet.get('line_taken') is not None else '—'} "
                f"| {bet.get('odds_taken') if bet.get('odds_taken') is not None else '—'} "
                f"| {bet.get('stake_units') if bet.get('stake_units') is not None else '—'} "
                f"| {bet.get('status') or '—'} |"
            )
        lines.append("")

    # ---------------- audit_events (sidecar) ----------------
    lines.append(f"## audit_events ({len(sidecar.audit_events)})")
    lines.append("")
    if not sidecar.audit_events:
        lines.append("_No audit events recorded._")
        lines.append("")
    else:
        lines.append("| ts | event_type | step | status | notes | trace_ids |")
        lines.append("|---|---|---|---|---|---|")
        for ev in sidecar.audit_events:
            note_cell = (ev.notes or "").replace("|", "\\|").replace("\n", " ")
            if len(note_cell) > 120:
                note_cell = note_cell[:117] + "..."
            trace_cell = (
                ", ".join(f"`{tid}`" for tid in ev.trace_ids) if ev.trace_ids else "—"
            )
            lines.append(
                f"| {ev.ts} | {ev.event_type} | {ev.step} | {ev.status} "
                f"| {note_cell or '—'} | {trace_cell} |"
            )
        lines.append("")

        # Surface assumptions and bugs separately so reviewers don't miss them
        # in the row-per-event table.
        assumptions: list[tuple[str, str]] = []
        bugs: list[tuple[str, str]] = []
        for ev in sidecar.audit_events:
            for a in ev.assumptions:
                assumptions.append((ev.step, a))
            for b in ev.bugs:
                bugs.append((ev.step, b))
        if assumptions:
            lines.append("### Assumptions")
            lines.append("")
            for step, a in assumptions:
                lines.append(f"- **{step}**: {a}")
            lines.append("")
        if bugs:
            lines.append("### Bugs flagged")
            lines.append("")
            for step, b in bugs:
                lines.append(f"- **{step}**: {b}")
            lines.append("")

    # ---------------- agent_notes (sidecar) ----------------
    if sidecar.agent_notes:
        lines.append("## agent_notes")
        lines.append("")
        lines.append(sidecar.agent_notes.rstrip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
