"""
omega-report-calibration â€” emit a markdown health & feedback report.

Read by the LLM at session start (Â§13 of system_prompt.txt) to surface
miscalibration trends, line-drift outliers, execution health, and pending
candidate gate status. Writes a single markdown file the operator uploads
to the Claude Project as `var/reports/latest.md`.

Sources of truth:
- `var/omega_traces.db` â€” traces, outcomes, bet_records, closing_lines, session_id
- `var/inbox/sessions/*.json` â€” session sidecars with exec_stats
- `CalibrationRegistry` â€” active production profile + candidate gate status

Determinism: this script makes no judgments. It tallies what is on disk. The
LLM is responsible for distinguishing signal from noise per Â§13.

Usage:
    omega-report-calibration
    omega-report-calibration --window-days 30 \\
        --out var/reports/2026-05-15-nba.md
    omega-report-calibration --sessions-inbox var/inbox/sessions

Exit codes:
    0 â€” report written
    1 â€” fatal error (DB missing, --out parent dir cannot be created)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.calibration.fitter import CalibrationFitter  # noqa: E402
from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus  # noqa: E402
from omega.core.calibration.registry import CalibrationRegistry  # noqa: E402
from omega.core.simulation.evidence_to_modifier import MAPPED_SIGNAL_TYPES  # noqa: E402
from omega.ops.output_modes import (  # noqa: E402
    OutputMode,
    classify_market_output_mode,
)
from omega.paths import latest_report_path, session_inbox_dir  # noqa: E402
from omega.strategy.distribution_metrics import (  # noqa: E402
    METRIC_VERSION as DISTRIBUTION_METRIC_VERSION,
)
from omega.strategy.distribution_metrics import crps_from_distribution_row  # noqa: E402
from omega.trace._atomic import atomic_write_text  # noqa: E402
from omega.trace.clv import compute_clv  # noqa: E402
from omega.trace.db import require_sqlite_backend  # noqa: E402
from omega.trace.portfolio import summarize_ledger  # noqa: E402
from omega.trace.report_header import header_for_store  # noqa: E402
from omega.trace.session_sidecar import load_sidecar_safe  # noqa: E402
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("report_calibration")


_CALIBRATION_ELIGIBLE_SQL = """
    t.predictions IS NOT NULL
    AND json_extract(t.full_trace, '$.result.status') = 'success'
    AND json_extract(t.full_trace, '$.trace_quality.calibration_eligible') = 1
    AND json_extract(t.full_trace, '$.trace_quality.context_source') = 'provided'
    AND json_extract(t.full_trace, '$.trace_quality.identity_status') = 'complete'
"""


def _load_session_sidecars(inbox: Path) -> list[dict[str, Any]]:
    """Read every var/inbox/sessions/*.json sidecar through the sidecar contract."""
    if not inbox.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(inbox.glob("*.json")):
        # Warn-only classify; read-only report never moves/marks files.
        sidecar = load_sidecar_safe(path)
        if sidecar is not None:
            out.append(sidecar.to_report_dict())
    return out


def _window_cutoff(window_days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=window_days)).isoformat()


def _league_filter(alias: str | None, league: str | None) -> tuple[str, tuple[str, ...]]:
    if not league:
        return "", ()
    prefix = f"{alias}." if alias else ""
    return f" AND {prefix}league = ?", (league,)


def _section_counts(store: TraceStore, league: str | None, cutoff: str) -> dict[str, int]:
    """Trace / bet / close counts within the window.

    "Graded" is the union of game outcomes and prop outcomes â€” a trace counts
    once whether it has a game-score result, one or more prop results, or both.
    """
    conn = store.conn
    league_filter, league_params = _league_filter(None, league)
    t_league_filter, t_league_params = _league_filter("t", league)
    n_traces = conn.execute(
        f"SELECT COUNT(*) FROM traces WHERE timestamp >= ?{league_filter}",
        (cutoff, *league_params),
    ).fetchone()[0]
    n_graded_game = conn.execute(
        """SELECT COUNT(DISTINCT t.trace_id) FROM traces t
           JOIN outcomes o ON t.trace_id = o.trace_id
           WHERE t.timestamp >= ?"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchone()[0]
    n_graded_prop = conn.execute(
        """SELECT COUNT(DISTINCT t.trace_id) FROM traces t
           JOIN prop_outcomes p ON t.trace_id = p.trace_id
           WHERE t.timestamp >= ?"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchone()[0]
    n_graded = conn.execute(
        """SELECT COUNT(DISTINCT t.trace_id) FROM traces t
           WHERE t.timestamp >= ?
             AND (EXISTS (SELECT 1 FROM outcomes o WHERE o.trace_id = t.trace_id)
               OR EXISTS (SELECT 1 FROM prop_outcomes p WHERE p.trace_id = t.trace_id))"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchone()[0]
    n_with_bet = conn.execute(
        """SELECT COUNT(DISTINCT t.trace_id) FROM traces t
           JOIN bet_ledger b ON t.trace_id = b.trace_id
           WHERE b.provenance = 'user_confirmed'
             AND t.timestamp >= ?"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchone()[0]
    n_with_close = conn.execute(
        """SELECT COUNT(DISTINCT t.trace_id) FROM traces t
           JOIN closing_lines c ON t.trace_id = c.trace_id
           WHERE t.timestamp >= ?"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchone()[0]
    n_with_predictions = conn.execute(
        f"""SELECT COUNT(*) FROM traces t
            WHERE timestamp >= ?
              AND {_CALIBRATION_ELIGIBLE_SQL}"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchone()[0]
    # Per-market coverage feeds per-market output-mode authorization. `kind`
    # lives in the full_trace JSON (no dedicated column), matching the rest of
    # the report's market scoping. The classify gate only checks > 0, so a coarse
    # kind scope is sufficient.
    n_with_predictions_game = conn.execute(
        f"""SELECT COUNT(*) FROM traces t
            WHERE timestamp >= ?
              AND {_CALIBRATION_ELIGIBLE_SQL}
              AND json_extract(t.full_trace, '$.kind') = 'game'"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchone()[0]
    n_with_predictions_prop = conn.execute(
        f"""SELECT COUNT(*) FROM traces t
            WHERE timestamp >= ?
              AND {_CALIBRATION_ELIGIBLE_SQL}
              AND json_extract(t.full_trace, '$.kind') = 'prop'"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchone()[0]
    n_graded_calibration = conn.execute(
        f"""SELECT COUNT(DISTINCT t.trace_id) FROM traces t
           WHERE t.timestamp >= ?
             AND {_CALIBRATION_ELIGIBLE_SQL}
             AND (EXISTS (SELECT 1 FROM outcomes o WHERE o.trace_id = t.trace_id)
               OR EXISTS (SELECT 1 FROM prop_outcomes p WHERE p.trace_id = t.trace_id))"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchone()[0]
    return {
        "traces": n_traces,
        "graded": n_graded,
        "graded_game": n_graded_game,
        "graded_prop": n_graded_prop,
        "with_bet": n_with_bet,
        "with_close": n_with_close,
        "with_predictions": n_with_predictions,
        "with_predictions_game": n_with_predictions_game,
        "with_predictions_prop": n_with_predictions_prop,
        "graded_calibration": n_graded_calibration,
    }


def _section_realized_metrics(
    store: TraceStore, league: str | None, cutoff: str
) -> dict[str, float] | None:
    """Brier / ECE / log_loss on graded traces in the window. Returns None if too few."""
    rows = store.query_traces(
        league=league,
        start=cutoff,
        has_outcome=True,
        calibration_eligible_only=True,
        limit=100_000,
    )
    if len(rows) < 10:
        return None
    fitter = CalibrationFitter()
    predictions, outcomes = fitter.extract_pairs(rows)
    if not predictions:
        return None

    # Score the predictions AS-IS (already calibrated by production profile at write time).
    n = len(predictions)
    brier = sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n
    from math import log

    eps = 1e-15
    log_loss = (
        -sum(
            o * log(max(eps, min(1 - eps, p))) + (1 - o) * log(max(eps, min(1 - eps, 1 - p)))
            for p, o in zip(predictions, outcomes)
        )
        / n
    )

    # ECE
    n_bins = 10
    bin_counts = [0] * n_bins
    bin_pred_sum = [0.0] * n_bins
    bin_out_sum = [0.0] * n_bins
    for p, o in zip(predictions, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bin_counts[idx] += 1
        bin_pred_sum[idx] += p
        bin_out_sum[idx] += o
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (
                abs(bin_pred_sum[i] / bin_counts[i] - bin_out_sum[i] / bin_counts[i])
                * bin_counts[i]
                / n
            )

    return {
        "n": float(n),
        "brier": brier,
        "ece": ece,
        "log_loss": log_loss,
    }


def _section_prop_realized_metrics(
    store: TraceStore, league: str | None, cutoff: str
) -> dict[str, float] | None:
    """Brier / ECE / log_loss on graded prop traces in the window.

    Separate from the game plane: prop calibration is a different forecasting
    problem (over/under stat lines, not home/away win). Suppressed when fewer
    than 10 prop (prediction, outcome) pairs are available.
    """
    rows = store.query_traces(
        league=league,
        start=cutoff,
        has_outcome=True,
        calibration_eligible_only=True,
        limit=100_000,
    )
    if not rows:
        return None
    predictions, outcomes = CalibrationFitter.extract_prop_pairs(rows)
    if len(predictions) < 10:
        return None

    n = len(predictions)
    brier = sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n
    from math import log

    eps = 1e-15
    log_loss = (
        -sum(
            o * log(max(eps, min(1 - eps, p))) + (1 - o) * log(max(eps, min(1 - eps, 1 - p)))
            for p, o in zip(predictions, outcomes)
        )
        / n
    )

    n_bins = 10
    bin_counts = [0] * n_bins
    bin_pred_sum = [0.0] * n_bins
    bin_out_sum = [0.0] * n_bins
    for p, o in zip(predictions, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bin_counts[idx] += 1
        bin_pred_sum[idx] += p
        bin_out_sum[idx] += o
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (
                abs(bin_pred_sum[i] / bin_counts[i] - bin_out_sum[i] / bin_counts[i])
                * bin_counts[i]
                / n
            )

    return {
        "n": float(n),
        "brier": brier,
        "ece": ece,
        "log_loss": log_loss,
    }


def _section_distribution_crps(
    store: TraceStore, league: str | None, cutoff: str
) -> dict[str, Any] | None:
    """Dynamic CRPS over V10 distribution rows joined to realized outcomes.

    Metrics are recomputed from ``v_distribution_outcomes`` at report time so a
    future CRPS refinement changes the report code, not historical ledger rows.
    """
    league_filter, league_params = _league_filter("d", league)
    rows = store.conn.execute(
        f"""SELECT d.*
           FROM v_distribution_outcomes d
           JOIN traces t ON t.trace_id = d.trace_id
           WHERE t.timestamp >= ?
             AND {_CALIBRATION_ELIGIBLE_SQL}
             AND d.stat_value IS NOT NULL"""
        + league_filter,
        (cutoff, *league_params),
    ).fetchall()
    values: list[float] = []
    by_stat: dict[str, list[float]] = {}
    skipped = 0
    for row in rows:
        data = dict(row)
        try:
            metric = crps_from_distribution_row(data, observed=float(data["stat_value"]))
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        value = float(metric["value"])
        values.append(value)
        stat_key = str(data.get("stat_key") or data.get("target") or "unknown")
        by_stat.setdefault(stat_key, []).append(value)

    if not values:
        return None

    return {
        "metric_version": DISTRIBUTION_METRIC_VERSION,
        "n": len(values),
        "mean_crps": sum(values) / len(values),
        "by_stat": {
            stat: {"n": len(stat_values), "mean_crps": sum(stat_values) / len(stat_values)}
            for stat, stat_values in sorted(by_stat.items())
        },
        "skipped": skipped,
    }


def _section_clv(store: TraceStore, league: str | None, cutoff: str) -> dict[str, Any]:
    """Mean CLV cents and beat-close rate across bets with attached closes in the window."""
    conn = store.conn
    t_league_filter, t_league_params = _league_filter("t", league)
    rows = conn.execute(
        """SELECT b.market, b.selection, b.line AS line_taken, b.odds AS odds_taken,
                  c.closing_line, c.closing_odds
           FROM bet_ledger b
           JOIN closing_lines c ON b.trace_id = c.trace_id
              AND b.market = c.market
              AND b.selection_descriptor = c.selection_descriptor
           JOIN traces t ON b.trace_id = t.trace_id
           WHERE b.provenance IN ('user_confirmed', 'engine_auto')
             AND t.timestamp >= ?"""
        + t_league_filter,
        (cutoff, *t_league_params),
    ).fetchall()

    if not rows:
        return {"n": 0, "mean_clv_cents": None, "beat_close_pct": None}

    clv_cents: list[float] = []
    beat_count = 0
    for row in rows:
        try:
            r = compute_clv(
                odds_taken=float(row["odds_taken"]),
                closing_odds=float(row["closing_odds"]),
                line_taken=float(row["line_taken"]) if row["line_taken"] is not None else None,
                closing_line=float(row["closing_line"])
                if row["closing_line"] is not None
                else None,
                side=str(row["selection"]).split()[0].lower() if row["selection"] else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLV computation failed for trace %s: %s", row.get("trace_id", "?"), exc)
            continue
        clv_cents.append(r.clv_cents)
        if r.beat_close:
            beat_count += 1

    if not clv_cents:
        return {"n": 0, "mean_clv_cents": None, "beat_close_pct": None}

    return {
        "n": len(clv_cents),
        "mean_clv_cents": sum(clv_cents) / len(clv_cents),
        "beat_close_pct": 100.0 * beat_count / len(clv_cents),
    }


def _section_portfolio(store: TraceStore, cutoff: str) -> dict[str, Any]:
    rows = store.query_ledger(start=cutoff, limit=100_000)
    return summarize_ledger(rows)


def _session_leagues(store: TraceStore, session_id: str) -> str:
    rows = store.conn.execute(
        """SELECT DISTINCT league FROM traces
           WHERE session_id = ? AND league IS NOT NULL
           ORDER BY league""",
        (session_id,),
    ).fetchall()
    leagues = [str(row["league"]) for row in rows if row["league"]]
    return ", ".join(leagues) if leagues else "?"


def _section_sessions(
    store: TraceStore, sidecars: list[dict[str, Any]], league: str | None, limit: int = 10
) -> list[dict[str, Any]]:
    """Join trace-level session summary with sidecar exec_stats."""
    summaries = store.get_session_summary(league=league, limit=limit)
    sidecar_index = {s.get("session_id"): s for s in sidecars if s.get("session_id")}
    out: list[dict[str, Any]] = []
    for s in summaries:
        sid = s["session_id"]
        sidecar = sidecar_index.get(sid, {})
        out.append(
            {
                "session_id": sid,
                "trace_count": s["trace_count"],
                "graded_count": s["graded_count"],
                "leagues": _session_leagues(store, sid),
                "first_ts": s["first_ts"],
                "last_ts": s["last_ts"],
                "model_version": sidecar.get("model_version"),
                "exec_stats": sidecar.get("exec_stats", {}),
                "pipeline_status": sidecar.get("pipeline_status", {}),
                "next_required_action": sidecar.get("next_required_action"),
                "effective_db_path": sidecar.get("effective_db_path"),
                "runtime_db_status": sidecar.get("runtime_db_status"),
                "agent_notes": (sidecar.get("agent_notes") or "")[:200],
            }
        )
    return out


def _section_signal_performance(store: TraceStore, league: str | None) -> list[dict[str, Any]]:
    """Most recent retrospective evidence-signal scoring run for the league."""
    return store.get_signal_performance(league=league, limit=200)


def _signal_verdict(sample_size: int, accuracy: float, calibration_gap: float) -> str:
    """Classify one signal-performance row for the agent.

    A directional signal is a coin flip at 0.50; >=0.55 accuracy on >=30 samples
    is treated as genuinely predictive. Below that it is noise. Thin samples are
    unproven. A large positive calibration gap flags overconfidence.
    """
    if sample_size < 30:
        return "insufficient_n"
    verdict = "predictive" if accuracy >= 0.55 else "noise"
    if calibration_gap > 0.10:
        verdict += " / overconfident"
    return verdict


def _signal_guidance(signal_perf: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket evidence signals into agent-facing bootstrap guidance.

    This is deliberately derived from ``signal_performance``. It does not create
    a second source-performance model; the detailed rows below remain the audit
    substrate, and this section is only a compact prompt-time readout.
    """
    buckets: dict[str, list[dict[str, Any]]] = {
        "trusted": [],
        "warnings": [],
        "insufficient": [],
    }
    for row in signal_perf:
        sample_size = int(row.get("sample_size") or 0)
        signal_type = str(row.get("signal_type") or "unknown")
        if signal_type not in MAPPED_SIGNAL_TYPES:
            signal_type = "unknown"
        accuracy = min(0.65, max(0.35, float(row.get("direction_accuracy") or 0.0)))
        calibration_gap = min(0.15, max(-0.15, float(row.get("calibration_gap") or 0.0)))
        brier = min(0.15, max(-0.15, float(row.get("brier") or 0.0)))
        normalized = {
            "signal_type": signal_type,
            "source": row.get("source") or "unknown",
            "obs_window": row.get("obs_window") or "unknown",
            "league": row.get("league") or "unknown",
            "sample_size": sample_size,
            "direction_accuracy": accuracy,
            "calibration_gap": calibration_gap,
            "brier": brier,
        }
        if sample_size < 30:
            buckets["insufficient"].append(normalized)
        elif accuracy >= 0.55 and calibration_gap <= 0.10:
            buckets["trusted"].append(normalized)
        else:
            buckets["warnings"].append(normalized)

    buckets["trusted"].sort(
        key=lambda r: (-r["direction_accuracy"], r["brier"], -r["sample_size"])
    )
    buckets["warnings"].sort(
        key=lambda r: (r["direction_accuracy"], -r["calibration_gap"], -r["sample_size"])
    )
    buckets["insufficient"].sort(key=lambda r: (-r["sample_size"], r["signal_type"]))
    return buckets


def _section_pending_candidates(registry: CalibrationRegistry, league: str | None) -> list[dict[str, Any]]:
    candidates = registry.list_profiles(league=league, status=ProfileStatus.CANDIDATE.value)
    out = []
    for p in candidates:
        out.append(
            {
                "profile_id": p.profile_id,
                "league": p.league,
                "market": p.market or "game",
                "method": p.method,
                "sample_size": p.sample_size,
                "metrics": p.metrics,
                "created_at": p.created_at,
            }
        )
    return out


def _section_production_profiles(registry: CalibrationRegistry, league: str | None) -> list[dict[str, Any]]:
    profiles = registry.list_profiles(league=league, status=ProfileStatus.PRODUCTION.value)
    out = []
    for p in sorted(
        profiles,
        key=lambda item: (item.league, item.market or "game", item.context_slice or ""),
    ):
        out.append(
            {
                "profile_id": p.profile_id,
                "league": p.league,
                "market": p.market or "game",
                "context_slice": p.context_slice,
                "method": p.method,
                "sample_size": p.sample_size,
                "metrics": p.metrics,
                "promoted_at": p.promoted_at,
            }
        )
    return out


_OUTPUT_MODE_MARKETS = ("game", "prop")


def _resolve_output_modes(
    prod_by_market: dict[str, CalibrationProfile | None],
    coverage_by_market: dict[str, int],
    *,
    sidecar_valid: bool = True,
) -> tuple[dict[str, OutputMode], dict[str, list[str]]]:
    """Classify output mode per market with the calibration-quality floor.

    Each market is authorized independently from its own production profile (no
    prop->game fallback), so a trustworthy prop market can be ACTIONABLE while
    the game market stays research-only, and vice versa. The single source of the
    rule is omega.ops.output_modes.classify_market_output_mode; this threads each
    market's exact-match production profile and scoped coverage count through it.
    """
    modes: dict[str, OutputMode] = {}
    reasons: dict[str, list[str]] = {}
    for market in _OUTPUT_MODE_MARKETS:
        prof = prod_by_market.get(market)
        metrics = prof.metrics if prof else {}
        mode, why = classify_market_output_mode(
            profile_id=prof.profile_id if prof else None,
            sample_size=prof.sample_size if prof else None,
            calibration_error=metrics.get("calibration_error"),
            trace_count=coverage_by_market.get(market, 0),
            sidecar_valid=sidecar_valid,
        )
        modes[market] = mode
        reasons[market] = why
    return modes, reasons


def _aggregate_scalar_mode(output_modes: dict[str, OutputMode]) -> OutputMode:
    """Conservative backward-compat scalar: ACTIONABLE only when every market is.

    Un-updated consumers that read only the scalar `output_mode` then behave
    safely; updated consumers read the per-market `output_modes` map instead.
    """
    if output_modes and all(m is OutputMode.ACTIONABLE for m in output_modes.values()):
        return OutputMode.ACTIONABLE
    return OutputMode.RESEARCH_CANDIDATE


def _select_market_profiles(
    profiles: list[CalibrationProfile],
) -> dict[str, CalibrationProfile | None]:
    """Pick the most conservative representative production profile per market."""
    def _calibration_error(profile: CalibrationProfile) -> float:
        try:
            return float(profile.metrics.get("calibration_error", 1.0))
        except (TypeError, ValueError):
            return 1.0

    selected: dict[str, CalibrationProfile | None] = {}
    for market in _OUTPUT_MODE_MARKETS:
        market_profiles = [
            p for p in profiles if (p.market or "game") == market and p.context_slice is None
        ]
        if not market_profiles:
            selected[market] = None
            continue
        selected[market] = sorted(
            market_profiles,
            key=lambda p: (
                p.sample_size,
                -_calibration_error(p),
                p.league,
                p.profile_id,
            ),
        )[0]
    return selected


def _render(
    scope_label: str,
    window_days: int,
    counts: dict[str, int],
    portfolio: dict[str, Any],
    realized: dict[str, float] | None,
    realized_prop: dict[str, float] | None,
    distribution_crps: dict[str, Any] | None,
    clv: dict[str, Any],
    sessions: list[dict[str, Any]],
    prod_by_market: dict[str, CalibrationProfile | None],
    production_profiles: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    signal_perf: list[dict[str, Any]],
    output_modes: dict[str, OutputMode],
    output_mode_reasons: dict[str, list[str]],
) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append(f"# Omega Health Report - {scope_label}")
    lines.append("")
    lines.append(f"Generated: `{now}` | Window: last {window_days} days")
    lines.append("")

    # â”€â”€ Agent Directive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Derived from Coverage + Calibration state.  The consuming LLM reads this
    # before acting; it governs what formal output is permitted per market this
    # session. `output_modes`/`output_mode_reasons` are computed once in main()
    # and also written to the report frontmatter, so the machine-readable map and
    # this prose block can never disagree.
    lines.append("## Agent Directive - Output Mode")
    lines.append("")
    lines.append(
        "Output authorization is **per market**: game and prop are classified "
        "independently from their own production profiles. The engine still runs "
        "and the trace still persists in every mode - only the user-facing betting "
        "numbers for a RESEARCH_CANDIDATE market are withheld. See "
        "`prompts/reference/output_modes.md`."
    )
    lines.append("")
    _market_labels = {"game": "Game / spread / total", "prop": "Player props"}
    for _market in ("game", "prop"):
        _mode = output_modes.get(_market, OutputMode.RESEARCH_CANDIDATE)
        _label = _market_labels[_market]
        if _mode is OutputMode.ACTIONABLE:
            _prof = prod_by_market.get(_market)
            _pid = _prof.profile_id if _prof else "?"
            lines.append(
                f"- **{_label} - `ACTIONABLE`**: engine-backed formal output "
                "(Bet Cards, edge%, EV%, Kelly, confidence tiers) is authorized. "
                f"Active calibration profile: `{_pid}`."
            )
        else:
            lines.append(
                f"- **{_label} - `RESEARCH_CANDIDATE`**: formal output (Bet Cards, "
                "edge%, EV%, Kelly, confidence tiers) is **not authorized** for this "
                "market in this window."
            )
            for _r in output_mode_reasons.get(_market, []):
                lines.append(f"    - {_r}")
    lines.append("")
    lines.append(
        "**Permitted in a RESEARCH_CANDIDATE market:** qualitative matchup "
        "narrative, news synthesis, recent form, listed sportsbook lines from a "
        "cited source. **Forbidden language:** \"best bet\", \"Tier A\", "
        "\"Tier B\", \"engine-confirmed\", \"actionable bet\". Stake cap on a "
        "research market: <= 1u."
    )
    lines.append("")

    guidance = _signal_guidance(signal_perf)
    lines.append("## Agent Directive - Evidence Learning")
    lines.append("")
    lines.append(
        "Use this as prompt-time evidence weighting only. It must not change "
        "engine probabilities, EV, Kelly, staking, confidence tiers, or trace IDs. "
        "Rows are derived from `signal_performance`; low sample counts remain "
        "unproven rather than bad."
    )
    lines.append("")
    if not signal_perf:
        lines.append(
            "_No scored evidence signals yet. Run `omega-score-evidence-signals` "
            "after outcomes attach before using evidence-performance warnings._"
        )
    else:
        if guidance["trusted"]:
            lines.append("Trusted signals to preserve:")
            for row in guidance["trusted"][:5]:
                lines.append(
                    f"- {row['league']} `{row['signal_type']}` from `{row['source']}` "
                    f"({row['obs_window']}): n={row['sample_size']}, "
                    f"dir_acc={row['direction_accuracy']:.2f}, "
                    f"cal_gap={row['calibration_gap']:+.2f}"
                )
        if guidance["warnings"]:
            lines.append("Evidence warnings to discount:")
            for row in guidance["warnings"][:5]:
                lines.append(
                    f"- {row['league']} `{row['signal_type']}` from `{row['source']}` "
                    f"({row['obs_window']}): n={row['sample_size']}, "
                    f"dir_acc={row['direction_accuracy']:.2f}, "
                    f"cal_gap={row['calibration_gap']:+.2f}"
                )
        if guidance["insufficient"]:
            lines.append("Insufficient-n signals to treat as unproven:")
            for row in guidance["insufficient"][:5]:
                lines.append(
                    f"- {row['league']} `{row['signal_type']}` from `{row['source']}` "
                    f"({row['obs_window']}): n={row['sample_size']}"
                )
    lines.append("")

    lines.append("## 1. Coverage")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---|")
    lines.append(f"| Traces (all) | {counts['traces']} |")
    lines.append(f"| Traces with model predictions (calibration-eligible) | {counts['with_predictions']} |")
    lines.append(f"| Graded (any outcome) | {counts['graded']} |")
    lines.append(f"| &nbsp;&nbsp;of which game-graded | {counts['graded_game']} |")
    lines.append(f"| &nbsp;&nbsp;of which prop-graded | {counts['graded_prop']} |")
    lines.append(f"| **Graded + calibration-eligible (usable pairs)** | **{counts['graded_calibration']}** |")
    lines.append(f"| With bet_record _(wager tracking only â€” not used for calibration)_ | {counts['with_bet']} |")
    lines.append(f"| With closing_line _(CLV only â€” not required for grading)_ | {counts['with_close']} |")
    lines.append("")

    lines.append("## 1B. Portfolio summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Base bankroll | ${portfolio['base_bankroll']:.2f} |")
    lines.append(f"| Current bankroll | ${portfolio['current_bankroll']:.2f} |")
    lines.append(f"| Realized PnL | ${portfolio['realized_pnl']:+.2f} |")
    lines.append(f"| ROI | {portfolio['roi_pct']:.2f}% |")
    lines.append(f"| Total bets | {portfolio['total_bets']} |")
    lines.append(f"| Decided | {portfolio['decided']} |")
    lines.append(f"| Win rate | {portfolio['win_pct']:.2f}% |")
    lines.append(f"| Open positions | {portfolio['open_positions_count']} |")
    lines.append(f"| Pending exposure | ${portfolio['pending_exposure']:.2f} |")
    lines.append("")
    if portfolio["active_ledgers"]:
        lines.append("| ledger_id | league | market | status | stake | potential_payout | last_updated |")
        lines.append("|---|---|---|---|---:|---:|---|")
        for row in portfolio["active_ledgers"][:20]:
            lines.append(
                f"| `{row.get('ledger_id')}` | {row.get('league') or '?'} | "
                f"{row.get('market_type') or '?'} | {row.get('status') or '?'} | "
                f"${row.get('stake', 0.0):.2f} | ${row.get('potential_payout', 0.0):.2f} | "
                f"{row.get('last_updated') or '?'} |"
            )
        if len(portfolio["active_ledgers"]) > 20:
            lines.append(f"| ... | ... | ... | ... | ... | ... | {len(portfolio['active_ledgers']) - 20} more |")
    else:
        lines.append("_No open ledger positions._")
    lines.append("")

    lines.append("## 2. Production calibration profile")
    lines.append("")
    if not production_profiles:
        lines.append("**None** â€” calibration is using the static fallback policy.")
    else:
        lines.append("| league | market | context_slice | profile_id | method | n | brier | ece | promoted |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---|")
        for p in production_profiles:
            m = p["metrics"]
            promoted = (p.get("promoted_at") or "?")[:10]
            lines.append(
                f"| {p['league']} | {p['market']} | {p.get('context_slice') or 'base'} | `{p['profile_id']}` | "
                f"{p['method']} | {p['sample_size']} | "
                f"{m.get('brier_score', float('nan')):.4f} | "
                f"{m.get('calibration_error', float('nan')):.4f} | {promoted} |"
            )
        lines.append("")
        lines.append("Output mode is resolved per market (see Agent Directive above):")
        for _m in ("game", "prop"):
            _mode = output_modes.get(_m)
            if _mode is not None:
                lines.append(f"- {_m}: `{_mode.value}`")
    lines.append("")

    lines.append("## 3. Realized metrics â€” game plane (graded game traces in window)")
    lines.append("")
    if realized is None:
        lines.append(
            "_Fewer than 10 game-graded traces in window â€” metrics suppressed (noise dominates)._"
        )
    else:
        lines.append(f"- n: {int(realized['n'])}")
        lines.append(f"- Brier: {realized['brier']:.4f}")
        lines.append(f"- ECE (10-bin): {realized['ece']:.4f}")
        lines.append(f"- Log loss: {realized['log_loss']:.4f}")
        if realized["ece"] > 0.05:
            lines.append("")
            lines.append(
                "> **FLAG â€” ECE > 0.05 on game plane.** Investigate which probability quintile is miscalibrated."
            )
    lines.append("")

    lines.append("## 3B. Realized metrics â€” prop plane (graded prop traces in window)")
    lines.append("")
    if realized_prop is None:
        lines.append(
            "_Fewer than 10 prop (prediction, outcome) pairs in window â€” metrics suppressed._"
        )
    else:
        lines.append(f"- n: {int(realized_prop['n'])}")
        lines.append(f"- Brier: {realized_prop['brier']:.4f}")
        lines.append(f"- ECE (10-bin): {realized_prop['ece']:.4f}")
        lines.append(f"- Log loss: {realized_prop['log_loss']:.4f}")
        if realized_prop["ece"] > 0.05:
            lines.append("")
            lines.append(
                "> **FLAG â€” ECE > 0.05 on prop plane.** Prop calibration is separately tunable; consider a prop-specific shrinkage profile."
            )
    lines.append("")

    lines.append("## 3C. Distribution CRPS Ã¢â‚¬â€ prop projection curves")
    lines.append("")
    if distribution_crps is None:
        lines.append(
            "_No V10 distribution rows with realized prop outcomes in window Ã¢â‚¬â€ CRPS suppressed._"
        )
    else:
        lines.append(f"- metric_version: `{distribution_crps['metric_version']}`")
        lines.append(f"- n: {distribution_crps['n']}")
        lines.append(f"- Mean CRPS: {distribution_crps['mean_crps']:.4f}")
        if distribution_crps["skipped"]:
            lines.append(f"- Unsupported/skipped rows: {distribution_crps['skipped']}")
        lines.append("")
        lines.append("| stat_key | n | mean_crps |")
        lines.append("|---|---:|---:|")
        for stat, stats in distribution_crps["by_stat"].items():
            lines.append(f"| {stat} | {stats['n']} | {stats['mean_crps']:.4f} |")
    lines.append("")

    lines.append("## 4. CLV (bets with attached closing lines)")
    lines.append("")
    if clv["n"] == 0:
        lines.append("_No CLV-resolvable bets in window._")
    else:
        lines.append(f"- n: {clv['n']}")
        lines.append(f"- Mean CLV: {clv['mean_clv_cents']:+.2f} cents")
        lines.append(f"- Beat-close rate: {clv['beat_close_pct']:.1f}%")
        if clv["mean_clv_cents"] is not None and clv["mean_clv_cents"] < -0.5:
            lines.append("")
            lines.append(
                "> **FLAG â€” Mean CLV regressing below -0.5 cents.** Review line-sourcing recipes (Â§6.1.5)."
            )
    lines.append("")

    lines.append("## 5. Sessions (most recent across all leagues)")
    lines.append("")
    if not sessions:
        lines.append("_No session_id-tagged traces yet._")
    else:
        lines.append(
            "| session_id | leagues | traces | graded | model | pipeline | next_action | closes | webfetch_fail | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in sessions:
            stats = s["exec_stats"] or {}
            pipeline = s.get("pipeline_status") or {}
            pipeline_label = pipeline.get("overall") or pipeline.get("status") or "?"
            next_action = (s.get("next_required_action") or "?").replace("|", "\\|")
            notes = (s["agent_notes"] or "").replace("|", "\\|").replace("\n", " ")[:80]
            lines.append(
                f"| `{s['session_id']}` | {s.get('leagues') or '?'} | "
                f"{s['trace_count']} | {s['graded_count']} | "
                f"{s.get('model_version') or '?'} | "
                f"{pipeline_label} | "
                f"{next_action[:80]} | "
                f"{stats.get('closing_line_captures', '?')} | "
                f"{stats.get('webfetch_failures', '?')} | {notes} |"
            )
    lines.append("")

    lines.append("## 6. Pending CANDIDATE profiles")
    lines.append("")
    if not candidates:
        lines.append("_No pending candidates._")
    else:
        lines.append("| league | market | profile_id | method | n | brier | ece | log_loss | created |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in candidates:
            m = c["metrics"]
            lines.append(
                f"| {c['league']} | {c['market']} | `{c['profile_id']}` | {c['method']} | {c['sample_size']} | "
                f"{m.get('brier_score', float('nan')):.4f} | "
                f"{m.get('calibration_error', float('nan')):.4f} | "
                f"{m.get('log_loss', float('nan')):.4f} | {c['created_at'][:10]} |"
            )
    lines.append("")

    lines.append("## 6B. Evidence signal performance (retrospective)")
    lines.append("")
    if not signal_perf:
        lines.append(
            "_No scored evidence signals yet â€” run `omega-score-evidence-signals` "
            "after outcomes attach._"
        )
    else:
        lines.append(
            "| signal_type | source | window | n | dir_acc | mean_conf | cal_gap | brier | verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in signal_perf:
            verdict = _signal_verdict(
                r["sample_size"], r["direction_accuracy"], r["calibration_gap"]
            )
            lines.append(
                f"| {r['signal_type']} | {r['source']} | {r['obs_window']} | "
                f"{r['sample_size']} | {r['direction_accuracy']:.2f} | "
                f"{r['mean_confidence']:.2f} | {r['calibration_gap']:+.2f} | "
                f"{r['brier']:.3f} | {verdict} |"
            )
        lines.append("")
        lines.append(
            "> Weight evidence by empirical accuracy: trust `predictive` signal "
            "types/sources, discount `noise`, treat `insufficient_n` as unproven. "
            "A positive `cal_gap` means the agent was overconfident in that signal."
        )
    lines.append("")

    lines.append("## 7. Suggested actions")
    lines.append("")
    lines.append(
        "_This section is intentionally empty. The LLM consumes the data above and emits "
        "an action plan per system_prompt.txt Â§13._"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit a markdown calibration health report.")
    parser.add_argument(
        "--league",
        default=None,
        help=(
            "Deprecated compatibility argument. latest.md is always an overall "
            "cross-league report."
        ),
    )
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument(
        "--out", type=Path, default=None, help="Output path (default: var/reports/latest.md)"
    )
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument(
        "--sessions-inbox",
        type=Path,
        default=session_inbox_dir(),
        help="Directory containing session sidecar JSON files (default: var/inbox/sessions)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    require_sqlite_backend("report_calibration.py")

    out_path: Path = args.out or latest_report_path()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create output directory %s: %s", out_path.parent, exc)
        return 1

    store = TraceStore(db_path=args.db)
    log_effective_db(store, logger)
    registry = CalibrationRegistry()
    sidecars = _load_session_sidecars(args.sessions_inbox)
    cutoff = _window_cutoff(args.window_days)

    report_league: str | None = None
    counts = _section_counts(store, report_league, cutoff)
    portfolio = _section_portfolio(store, cutoff)
    realized = _section_realized_metrics(store, report_league, cutoff)
    realized_prop = _section_prop_realized_metrics(store, report_league, cutoff)
    distribution_crps = _section_distribution_crps(store, report_league, cutoff)
    clv = _section_clv(store, report_league, cutoff)
    sessions = _section_sessions(store, sidecars, report_league)

    # Exact-match production profile per market (no prop->game fallback): output
    # authorization must reflect each market's OWN fitted prior. The runtime
    # calibration path (registry.get_production) still falls back; this lookup
    # deliberately does not.
    prod_profiles_all = registry.list_profiles(
        league=report_league, status=ProfileStatus.PRODUCTION.value
    )
    prod_by_market = _select_market_profiles(prod_profiles_all)
    production_profiles = _section_production_profiles(registry, report_league)
    candidates = _section_pending_candidates(registry, report_league)
    signal_perf = _section_signal_performance(store, report_league)

    # Classify output mode PER MARKET, here, and feed both the machine-readable
    # frontmatter and the prose directive block so they cannot disagree. Sidecar
    # validity is a per-session concern; at the aggregate report level we treat it
    # as valid (individual sessions are filtered upstream).
    coverage_by_market = {
        "game": counts["with_predictions_game"],
        "prop": counts["with_predictions_prop"],
    }
    output_modes, output_mode_reasons = _resolve_output_modes(
        prod_by_market, coverage_by_market, sidecar_valid=True
    )
    # Backward-compat scalar for un-updated consumers: most conservative across
    # markets (ACTIONABLE only when every market is ACTIONABLE). Updated consumers
    # read the per-market `output_modes` map instead.
    scalar_output_mode = _aggregate_scalar_mode(output_modes)

    # Build the derived-artifact front-matter BEFORE closing the store: it reads
    # the effective DB path / source / trace count off the live store so the
    # report names the DB it was generated from (docs/phase6/ARTIFACT_AUTHORITY.md).
    header = header_for_store(
        store,
        ["var/omega_traces.db", "var/inbox/sessions/*.json (sidecars)", "calibration registry"],
        extra_fields={
            "output_mode": scalar_output_mode.value,
            "output_modes": {m: mode.value for m, mode in output_modes.items()},
            "output_mode_reasons": output_mode_reasons,
        },
    )
    store.close()

    rendered = header + _render(
        scope_label="Overall",
        window_days=args.window_days,
        counts=counts,
        portfolio=portfolio,
        realized=realized,
        realized_prop=realized_prop,
        distribution_crps=distribution_crps,
        clv=clv,
        sessions=sessions,
        prod_by_market=prod_by_market,
        production_profiles=production_profiles,
        candidates=candidates,
        signal_perf=signal_perf,
        output_modes=output_modes,
        output_mode_reasons=output_mode_reasons,
    )

    atomic_write_text(out_path, rendered)
    logger.info(
        "Wrote %s (%d traces, %d sessions, %d candidates).",
        out_path,
        counts["traces"],
        len(sessions),
        len(candidates),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
