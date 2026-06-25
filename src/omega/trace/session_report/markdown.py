"""Pure Markdown renderers for Omega session reports."""

from __future__ import annotations

from collections.abc import Iterable

from omega.trace.session_report.models import (
    AuditRow,
    ContextBullet,
    IntakeReportData,
    TraceReportCard,
)


def _clean(value: object) -> str:
    if value is None or value == "":
        return "not captured"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _yaml_str(value: object) -> str:
    if value is None:
        return "null"
    return repr(str(value))


def _frontmatter(data: IntakeReportData) -> str:
    lines = [
        "---",
        "canonical: false",
        f"report_kind: {data.report_kind!r}",
        f"generated_at: {data.generated_at!r}",
        f"source_db_path: {data.source_db_path!r}",
        f"source_db_fingerprint: {data.source_db_fingerprint!r}",
        f"source_session_id: {_yaml_str(data.source_session_id)}",
        f"context_mode: {data.context_mode!r}",
        f"context_bundle_id: {_yaml_str(data.context_bundle_id)}",
        f"trace_count: {data.trace_count}",
        f"ledger_count: {data.ledger_count}",
        "---",
        "",
    ]
    return "\n".join(lines)


def _rows(title: str, rows: Iterable[tuple[str, int]]) -> list[str]:
    out = [f"### {title}", "", "| Item | Count |", "|---|---:|"]
    empty = True
    for label, count in rows:
        empty = False
        out.append(f"| {_clean(label)} | {count} |")
    if empty:
        out.append("| none | 0 |")
    out.append("")
    return out


def _context_section(title: str, bullets: list[ContextBullet], bucket: str) -> list[str]:
    selected = [b for b in bullets if b.bucket == bucket]
    lines = [f"**{title}**"]
    if not selected:
        lines.append("- [missing] not captured")
        return lines
    for bullet in selected:
        source = bullet.source_type
        suffix = ""
        if bullet.source_title:
            suffix = f" ({bullet.source_title})"
        lines.append(f"- [{source}] {_clean(bullet.text)}{suffix}")
    return lines


def _engine_line(card: TraceReportCard) -> str:
    ev = card.engine_view
    bits = [
        f"model probability: {_clean(ev.model_probability)}",
        f"edge: {_clean(ev.edge)}",
        f"units: {_clean(ev.units)}",
        f"tier: {_clean(ev.tier)}",
        f"calibration: {_clean(ev.calibration_status)}",
    ]
    return "; ".join(bits)


def _honesty_lines(card: TraceReportCard) -> list[str]:
    """Truth-in-labeling block: how trustworthy is this recommendation?

    Always rendered (both output modes) — these are honesty signals, not
    protected edge/EV numbers.
    """
    ev = card.engine_view
    return [
        "**Honesty**",
        f"- confidence tier: {_clean(ev.tier)}"
        + (
            f" (capped: {_clean(ev.confidence_cap_reason)})"
            if ev.confidence_cap_reason
            else ""
        ),
        f"- trace quality: {_clean(ev.aggregate_quality)}/100 ({_clean(ev.quality_band)})",
        f"- evidence: mode={_clean(ev.evidence_mode)}, status={_clean(ev.evidence_status)}, "
        f"signals={_clean(ev.evidence_signal_count)}, applied_factor={_clean(ev.evidence_applied_factor)}",
        f"- calibration: path={_clean(ev.calibration_path)}, profile={_clean(ev.profile_id)}, "
        f"status={_clean(ev.profile_status)}, maturity={_clean(ev.profile_maturity)}",
        f"- profile metrics: n={_clean(ev.profile_sample_size)}, ECE={_clean(ev.profile_ece)}, "
        f"Brier={_clean(ev.profile_brier)}",
        f"- static_identity fallback used: {_clean(ev.static_identity_used)}",
        "",
    ]


def _analyst_lines(card: TraceReportCard) -> list[str]:
    """Analyst-note narrative block (prose only — rendered in both output modes).

    Thesis falls back to the legacy ``reasoning_narrative`` so traces filed before
    ``reasoning_presentation`` existed still read as analyst notes. These are qualitative
    sections; no protected edge/EV/units numbers appear here.
    """
    a = card.analyst
    thesis = a.thesis or card.reasoning_narrative
    return [
        "**Analyst Notes**",
        f"- Thesis: {_clean(thesis)}",
        f"- Market read: {_clean(a.market_read)}",
        f"- Why Omega likes it: {_clean(a.why)}",
        f"- Risks: {_clean(a.risks)}",
        f"- Verdict: {_clean(a.verdict)}",
        "",
    ]


def _render_card(card: TraceReportCard) -> list[str]:
    ledger = card.ledger_view
    lines = [
        f"### {_clean(card.matchup)}",
        "",
        "**Decision**",
        f"- selection: {_clean(card.selection)}",
        f"- market: {_clean(card.market)}",
        f"- book: {_clean(card.book)}",
        f"- stake/status: {_clean(card.stake_status)}",
        f"- trace id: `{card.trace_id}`",
        "",
        "**Engine View**",
        f"- {_engine_line(card)}",
        "",
        *_honesty_lines(card),
        *_analyst_lines(card),
        *_context_section("Support", card.context, "support"),
        "",
        *_context_section("Concern", card.context, "concern"),
        "",
        *_context_section("Missing Data", card.context, "missing"),
        "",
        "**Ledger**",
        f"- status: {_clean(ledger.status)}",
        f"- provenance: {_clean(ledger.provenance)}",
        f"- close attached: {str(ledger.close_attached).lower()}",
        f"- CLV available: {str(ledger.clv_available).lower()}",
        f"- outcome attached: {str(ledger.outcome_attached).lower()}",
        "",
        "**Quality**",
        f"- trace quality: {_clean(card.trace_quality_status)}",
        f"- sidecar: {_clean(card.sidecar_status)}",
        f"- evidence: {_clean(card.evidence_status)}",
        f"- calibration eligible: {_clean(card.calibration_eligible)}",
        f"- evidence-learning eligible: {_clean(card.evidence_learning_eligible)}",
        "",
    ]
    return lines


_AUDIT_HEADERS = [
    "trace_id",
    "league",
    "matchup",
    "market",
    "selection",
    "line",
    "odds",
    "bookmaker",
    "odds_at",
    "output_mode",
    "tier",
    "calib_elig",
    "quality",
    "evidence",
    "prior_status",
    "fallback",
    "ledger",
]


def _audit_row_cells(row: AuditRow) -> list[str]:
    warnings_tag = f"⚠ {len(row.resolver_warnings)}" if row.resolver_warnings else "—"
    return [
        f"`{row.trace_id}`",
        _clean(row.league),
        _clean(row.matchup),
        _clean(row.market_type),
        _clean(row.selection),
        _clean(row.line),
        _clean(row.odds),
        _clean(row.bookmaker),
        _clean(row.odds_resolved_at),
        _clean(row.output_mode),
        _clean(row.confidence_tier),
        _clean(row.calibration_eligible),
        _clean(row.aggregate_quality),
        str(row.evidence_count),
        _clean(row.prior_coverage_status),
        warnings_tag if row.fallback_usage else "—",
        _clean(row.ledger_status),
    ]


def _render_audit_table(rows: list[AuditRow]) -> list[str]:
    """Render the bet-level trust audit table as Markdown."""
    header = "| " + " | ".join(_AUDIT_HEADERS) + " |"
    sep = "|" + "|".join("---" for _ in _AUDIT_HEADERS) + "|"
    lines = ["## Bet-Level Trust Audit", "", header, sep]
    if not rows:
        placeholder = "| " + " | ".join("not captured" for _ in _AUDIT_HEADERS) + " |"
        lines.append(placeholder)
    else:
        for row in rows:
            cells = _audit_row_cells(row)
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "_Legend: `odds_at` = timestamp of the odds snapshot used; "
            "`prior_status` = data_provenance event from prior injection; "
            "`fallback` = ⚠ N means N resolver warnings present._",
            "",
        ]
    )
    return lines


def _zero_evidence_block(data: IntakeReportData) -> list[str]:
    """A prominent blocker section when too many traces reason blind."""
    if not data.zero_evidence_blocked:
        return []
    shown = data.zero_evidence_trace_ids[:20]
    more = (
        f" (+{len(data.zero_evidence_trace_ids) - len(shown)} more)"
        if len(data.zero_evidence_trace_ids) > len(shown)
        else ""
    )
    return [
        "## ⛔ BLOCKER: zero-evidence / empty-context",
        "",
        f"**{data.zero_evidence_count} of {data.trace_count} traces** have no structured "
        "evidence AND no provided context. These cannot calibrate, cannot learn, and must "
        "not produce actionable output. **This run summary is failed.**",
        "",
        f"Offending traces: {', '.join(f'`{t}`' for t in shown)}{more}",
        "",
    ]


def render_intake_markdown(data: IntakeReportData) -> str:
    """Render an intake report without DB access or recomputation."""
    lines: list[str] = [_frontmatter(data)]
    title = "Daily Intake Overview + Trace Ledger"
    if data.source_session_id:
        title += f" - {data.source_session_id}"
    lines.extend([f"# {title}", ""])
    lines.extend(_zero_evidence_block(data))
    lines.extend(
        [
            "## Snapshot",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| traces | {data.trace_count} |",
            f"| ledger rows | {data.ledger_count} |",
            f"| context mode | {data.context_mode} |",
            f"| sidecar | {data.sidecar_status} |",
            "",
        ]
    )
    lines.extend(_rows("Trace Coverage", ((r.label, r.count) for r in data.coverage)))
    lines.extend(_rows("Ledger Linkage", ((r.label, r.count) for r in data.ledger_linkage)))
    lines.extend(_rows("Provenance Split", ((r.label, r.count) for r in data.provenance_split)))

    lines.extend(["## Persisted Recommendation Cards", ""])
    if not data.cards:
        lines.extend(["_No traces matched this report scope._", ""])
    else:
        for card in data.cards:
            lines.extend(_render_card(card))

    lines.extend(["## Full Trace Ledger", ""])
    lines.extend(["| Trace | League | Market | Status | Ledger |", "|---|---|---|---|---|"])
    for card in data.cards:
        lines.append(
            f"| `{card.trace_id}` | {_clean(card.league)} | {_clean(card.market)} | "
            f"{_clean(card.trace_quality_status)} | {_clean(card.ledger_view.provenance or card.ledger_view.status)} |"
        )
    if not data.cards:
        lines.append("| none | not captured | not captured | not captured | not captured |")
    lines.append("")

    lines.extend(_render_audit_table(data.audit_rows))
    lines.extend(["## Appendix", ""])
    if data.unmatched_ledger_rows:
        lines.append("### Unmatched Ledger Rows")
        lines.append("")
        for trace_id in data.unmatched_ledger_rows:
            lines.append(f"- `{trace_id}`")
        lines.append("")
    if data.ignored_context_entries:
        lines.append("### Ignored Context Entries")
        lines.append("")
        for entry in data.ignored_context_entries:
            lines.append(f"- `{entry.entry_id}`: {_clean(entry.reason)}")
        lines.append("")
    if not data.unmatched_ledger_rows and not data.ignored_context_entries:
        lines.append("_No appendix issues._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
