"""Render a :class:`BacktestReport` to JSON and a human-readable text summary.

The renderer keeps **probability accuracy and betting ROI in separate blocks**,
reports the soccer draw plane separately when present, and surfaces the
leakage / identity / missing-odds health rates so data problems are visible.
"""

from __future__ import annotations

import json
from collections import Counter

from omega.historical.contracts import (
    BacktestReport,
    BettingBlock,
    MetricBlock,
    ReplayTraceManifest,
)


def to_json(report: BacktestReport, *, indent: int = 2) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=indent)


def _fmt(v: float | None, places: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{places}f}"


def _metric_line(name: str, m: MetricBlock) -> str:
    return (
        f"  {name:<22} n={m.n:<5} "
        f"brier {_fmt(m.raw_brier)} -> {_fmt(m.calibrated_brier)}  "
        f"ece {_fmt(m.raw_ece)} -> {_fmt(m.calibrated_ece)}  "
        f"logloss {_fmt(m.raw_log_loss)} -> {_fmt(m.calibrated_log_loss)}"
    )


def _betting_lines(b: BettingBlock) -> list[str]:
    return [
        f"  bets={b.n_bets}  roi={_fmt(b.roi)}  net_pnl={_fmt(b.net_pnl, 2)}  "
        f"hit_rate={_fmt(b.hit_rate)}  profit_factor={_fmt(b.profit_factor)}  "
        f"max_drawdown={_fmt(b.max_drawdown, 2)}  avg_clv={_fmt(b.avg_clv)}"
    ]


def to_text(report: BacktestReport) -> str:
    lines: list[str] = []
    lines.append(f"Historical Backtest — {report.league}")
    lines.append(f"manifest_id={report.manifest_id}  replay_id={report.replay_id}")
    lines.append(
        f"mode={report.walk_forward_config.mode}  folds={len(report.folds)}  "
        f"code_version={report.code_version}"
    )
    lines.append("")

    lines.append("== Probability accuracy (raw -> calibrated) ==")
    if not report.aggregate_metrics_by_market:
        lines.append("  (no calibration-eligible test pairs)")
    for market in sorted(report.aggregate_metrics_by_market):
        lines.append(_metric_line(market, report.aggregate_metrics_by_market[market]))
    # Call out the soccer draw plane explicitly if present.
    if "draw" in report.aggregate_metrics_by_market:
        lines.append("  (draw calibration reported separately above)")
    lines.append("")

    lines.append("== Betting performance ==")
    if report.aggregate_betting and report.aggregate_betting.n_bets:
        lines.extend(_betting_lines(report.aggregate_betting))
    else:
        lines.append("  (no graded bets — probability-only backtest)")
    lines.append("")

    h = report.aggregate_health
    lines.append("== Health / visibility ==")
    lines.append(
        f"  missing_odds_rate={_fmt(h.missing_odds_rate)}  "
        f"leakage_skip_count={h.leakage_skip_count}  "
        f"identity_failure_count={h.identity_failure_count}"
    )
    lines.append(
        f"  fallback_profile_rate={_fmt(h.fallback_profile_rate)}  "
        f"default_context_rate={_fmt(h.default_context_rate)}  "
        f"stale_context_rate={_fmt(h.stale_context_rate)}"
    )
    lines.append("")

    lines.append("== Folds ==")
    for f in report.folds:
        game = f.metrics_by_market.get("game")
        brier = (
            f"brier {_fmt(game.raw_brier)} -> {_fmt(game.calibrated_brier)}"
            if game
            else "no game plane"
        )
        lines.append(
            f"  [{f.fold_index}] test {f.test_start[:10]}..{f.test_end[:10]} "
            f"train={f.n_train} test={f.n_test} profiles={len(f.frozen_profiles)} {brier}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Replay run audit (calibration backfill)
# ---------------------------------------------------------------------------


def build_replay_summary(
    manifest: ReplayTraceManifest,
    *,
    eligible_count: int,
    eligible_denominator: int | None = None,
    league: str | None = None,
) -> dict:
    """Summarize a replay run for the RUN_AUDIT + machine-readable sidecar.

    ``eligible_count`` and ``eligible_denominator`` are supplied by the caller
    via DB queries so DB-wide appended seasons do not get divided by this run's
    manifest-local persisted count.
    """
    records = manifest.records
    n_events = len(records)
    persisted = [r for r in records if r.trace_id is not None]
    n_persisted = len(persisted)
    n_skipped = n_events - n_persisted

    context_source_distribution = dict(Counter(r.context_source for r in persisted))
    leakage_skipped = sum(1 for r in records if r.leakage_status == "skipped")
    identity_missing = sum(1 for r in records if r.identity_status == "missing")
    missing_odds = sum(1 for r in persisted if r.missing_odds)
    stale = sum(1 for r in persisted if r.is_stale)

    def _rate(n: int, denominator: int | None = None) -> float:
        den = n_persisted if denominator is None else denominator
        return round(n / den, 4) if den else 0.0

    eligible_denominator = n_persisted if eligible_denominator is None else eligible_denominator

    return {
        "schema_version": 1,
        "replay_id": manifest.replay_id,
        "dataset_manifest_id": manifest.dataset_manifest_id,
        "league": league or manifest.league,
        "code_version": manifest.code_version,
        "config_hash": manifest.config_hash,
        "created_at": manifest.created_at,
        "n_events": n_events,
        "n_persisted": n_persisted,
        "n_skipped": n_skipped,
        "eligible_count": eligible_count,
        "calibration_eligible_rate": _rate(eligible_count, eligible_denominator),
        "calibration_eligible_rate_denominator": eligible_denominator,
        "context_source_distribution": context_source_distribution,
        "missing_odds_rate": _rate(missing_odds),
        "stale_context_rate": _rate(stale),
        "leakage_skipped": leakage_skipped,
        "identity_missing": identity_missing,
    }


def render_run_audit(summary: dict) -> str:
    """Render a human-readable RUN_AUDIT.md from a :func:`build_replay_summary` dict."""
    csd = summary.get("context_source_distribution") or {}
    csd_str = ", ".join(f"{k}={v}" for k, v in sorted(csd.items())) or "(none)"
    # Derived-artifact front-matter. A replay RUN_AUDIT is a generated, non-canonical
    # artifact (PROJECT_STATE.md source-of-truth rules), so it carries the same
    # ``canonical: false`` marker the other generated reports get from
    # trace/report_header.py — here store-less, built from the summary's provenance.
    front_matter = [
        "---",
        "canonical: false",
        "artifact: replay_run_audit",
        f"replay_id: {summary.get('replay_id')!r}",
        f"dataset_manifest_id: {summary.get('dataset_manifest_id')!r}",
        f"league: {summary.get('league')!r}",
        f"code_version: {summary.get('code_version')!r}",
        f"config_hash: {summary.get('config_hash')!r}",
        f"created_at: {summary.get('created_at')!r}",
        "---",
        "",
    ]
    lines = front_matter + [
        f"# Replay Run Audit — {summary.get('replay_id')}",
        "",
        f"- League: {summary.get('league')}",
        f"- Dataset manifest: {summary.get('dataset_manifest_id')}",
        f"- Code version: {summary.get('code_version')}",
        f"- Config hash: {summary.get('config_hash')}",
        f"- Created: {summary.get('created_at')}",
        "",
        "## Counts",
        f"- Events: {summary.get('n_events')}",
        f"- Persisted traces: {summary.get('n_persisted')}",
        f"- Skipped (leakage/identity): {summary.get('n_skipped')}",
        (
            f"- Calibration-eligible graded traces: {summary.get('eligible_count')} "
            f"(rate {summary.get('calibration_eligible_rate')})"
        ),
        "",
        "## Quality / leakage visibility",
        f"- context_source distribution: {csd_str}",
        f"- missing_odds_rate: {summary.get('missing_odds_rate')}",
        f"- stale_context_rate: {summary.get('stale_context_rate')}",
        f"- leakage_skipped: {summary.get('leakage_skipped')}",
        f"- identity_missing: {summary.get('identity_missing')}",
        "",
        (
            "_Synthetic historical-replay traces. Not promotable until the "
            "backtest-parity and historical-vs-live parity gates pass "
            "(see the calibration backfill runbook)._"
        ),
    ]
    return "\n".join(lines)
