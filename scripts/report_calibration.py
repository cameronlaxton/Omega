"""
scripts/report_calibration.py — emit a markdown health & feedback report.

Read by the LLM at session start (§13 of system_prompt.txt) to surface
miscalibration trends, line-drift outliers, execution health, and pending
candidate gate status. Writes a single markdown file the operator uploads
to the Claude Project as `reports/latest.md`.

Sources of truth:
- `omega_traces.db` — traces, outcomes, bet_records, closing_lines, session_id
- `inbox/sessions/*.json` — session sidecars with exec_stats
- `CalibrationRegistry` — active production profile + candidate gate status

Determinism: this script makes no judgments. It tallies what is on disk. The
LLM is responsible for distinguishing signal from noise per §13.

Usage:
    python scripts/report_calibration.py --league NBA
    python scripts/report_calibration.py --league NBA --window-days 30 \\
        --out reports/2026-05-15-nba.md
    python scripts/report_calibration.py --league NBA --sessions-inbox inbox/sessions

Exit codes:
    0 — report written
    1 — fatal error (DB missing, --out parent dir cannot be created)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.core.calibration.fitter import CalibrationFitter  # noqa: E402
from omega.core.calibration.profiles import ProfileStatus  # noqa: E402
from omega.core.calibration.registry import CalibrationRegistry  # noqa: E402
from omega.trace.clv import compute_clv  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("report_calibration")


def _load_session_sidecars(inbox: Path) -> List[Dict[str, Any]]:
    """Read every inbox/sessions/*.json sidecar. Bad files are skipped with a warning."""
    if not inbox.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(inbox.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                out.append(json.load(fh))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed session sidecar %s: %s", path.name, exc)
    return out


def _window_cutoff(window_days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()


def _section_counts(store: TraceStore, league: str, cutoff: str) -> Dict[str, int]:
    """Trace / bet / close counts within the window."""
    conn = store.conn
    n_traces = conn.execute(
        "SELECT COUNT(*) FROM traces WHERE league = ? AND timestamp >= ?",
        (league, cutoff),
    ).fetchone()[0]
    n_graded = conn.execute(
        """SELECT COUNT(*) FROM traces t
           JOIN outcomes o ON t.trace_id = o.trace_id
           WHERE t.league = ? AND t.timestamp >= ?""",
        (league, cutoff),
    ).fetchone()[0]
    n_with_bet = conn.execute(
        """SELECT COUNT(DISTINCT t.trace_id) FROM traces t
           JOIN bet_records b ON t.trace_id = b.trace_id
           WHERE t.league = ? AND t.timestamp >= ?""",
        (league, cutoff),
    ).fetchone()[0]
    n_with_close = conn.execute(
        """SELECT COUNT(DISTINCT t.trace_id) FROM traces t
           JOIN closing_lines c ON t.trace_id = c.trace_id
           WHERE t.league = ? AND t.timestamp >= ?""",
        (league, cutoff),
    ).fetchone()[0]
    return {
        "traces": n_traces,
        "graded": n_graded,
        "with_bet": n_with_bet,
        "with_close": n_with_close,
    }


def _section_realized_metrics(
    store: TraceStore, league: str, cutoff: str
) -> Optional[Dict[str, float]]:
    """Brier / ECE / log_loss on graded traces in the window. Returns None if too few."""
    rows = store.query_traces(league=league, start=cutoff, has_outcome=True, limit=100_000)
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
    log_loss = -sum(
        o * log(max(eps, min(1 - eps, p))) + (1 - o) * log(max(eps, min(1 - eps, 1 - p)))
        for p, o in zip(predictions, outcomes)
    ) / n

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
            ece += abs(bin_pred_sum[i] / bin_counts[i] - bin_out_sum[i] / bin_counts[i]) * bin_counts[i] / n

    return {
        "n": float(n),
        "brier": brier,
        "ece": ece,
        "log_loss": log_loss,
    }


def _section_clv(store: TraceStore, league: str, cutoff: str) -> Dict[str, Any]:
    """Mean CLV cents and beat-close rate across bets with attached closes in the window."""
    conn = store.conn
    rows = conn.execute(
        """SELECT b.market, b.selection, b.line_taken, b.odds_taken,
                  c.closing_line, c.closing_odds
           FROM bet_records b
           JOIN closing_lines c ON b.trace_id = c.trace_id
              AND b.market = c.market
              AND b.selection_descriptor = c.selection_descriptor
           JOIN traces t ON b.trace_id = t.trace_id
           WHERE t.league = ? AND t.timestamp >= ?""",
        (league, cutoff),
    ).fetchall()

    if not rows:
        return {"n": 0, "mean_clv_cents": None, "beat_close_pct": None}

    clv_cents: List[float] = []
    beat_count = 0
    for row in rows:
        try:
            r = compute_clv(
                odds_taken=float(row["odds_taken"]),
                closing_odds=float(row["closing_odds"]),
                line_taken=float(row["line_taken"]) if row["line_taken"] is not None else None,
                closing_line=float(row["closing_line"]) if row["closing_line"] is not None else None,
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


def _section_sessions(
    store: TraceStore, sidecars: List[Dict[str, Any]], league: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """Join trace-level session summary with sidecar exec_stats."""
    summaries = store.get_session_summary(league=league, limit=limit)
    sidecar_index = {s.get("session_id"): s for s in sidecars if s.get("session_id")}
    out: List[Dict[str, Any]] = []
    for s in summaries:
        sid = s["session_id"]
        sidecar = sidecar_index.get(sid, {})
        out.append(
            {
                "session_id": sid,
                "trace_count": s["trace_count"],
                "graded_count": s["graded_count"],
                "first_ts": s["first_ts"],
                "last_ts": s["last_ts"],
                "model_version": sidecar.get("model_version"),
                "exec_stats": sidecar.get("exec_stats", {}),
                "agent_notes": (sidecar.get("agent_notes") or "")[:200],
            }
        )
    return out


def _section_pending_candidates(registry: CalibrationRegistry, league: str) -> List[Dict[str, Any]]:
    candidates = registry.list_profiles(league=league, status=ProfileStatus.CANDIDATE.value)
    out = []
    for p in candidates:
        out.append(
            {
                "profile_id": p.profile_id,
                "method": p.method,
                "sample_size": p.sample_size,
                "metrics": p.metrics,
                "created_at": p.created_at,
            }
        )
    return out


def _render(
    league: str,
    window_days: int,
    counts: Dict[str, int],
    realized: Optional[Dict[str, float]],
    clv: Dict[str, Any],
    sessions: List[Dict[str, Any]],
    production_profile_id: Optional[str],
    candidates: List[Dict[str, Any]],
) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append(f"# Omega Calibration Report — {league}")
    lines.append("")
    lines.append(f"Generated: `{now}` | Window: last {window_days} days")
    lines.append("")

    lines.append("## 1. Coverage")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---|")
    lines.append(f"| Traces | {counts['traces']} |")
    lines.append(f"| Graded (outcome attached) | {counts['graded']} |")
    lines.append(f"| With bet_record | {counts['with_bet']} |")
    lines.append(f"| With closing_line | {counts['with_close']} |")
    lines.append("")

    lines.append("## 2. Production calibration profile")
    lines.append("")
    if production_profile_id:
        lines.append(f"Active: `{production_profile_id}`")
    else:
        lines.append("**None** — calibration is using the static fallback policy.")
    lines.append("")

    lines.append("## 3. Realized metrics (graded traces in window)")
    lines.append("")
    if realized is None:
        lines.append("_Fewer than 10 graded traces in window — metrics suppressed (noise dominates)._")
    else:
        lines.append(f"- n: {int(realized['n'])}")
        lines.append(f"- Brier: {realized['brier']:.4f}")
        lines.append(f"- ECE (10-bin): {realized['ece']:.4f}")
        lines.append(f"- Log loss: {realized['log_loss']:.4f}")
        if realized["ece"] > 0.05:
            lines.append("")
            lines.append("> **FLAG — ECE > 0.05.** Investigate which probability quintile is miscalibrated.")
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
            lines.append("> **FLAG — Mean CLV regressing below -0.5 cents.** Review line-sourcing recipes (§6.1.5).")
    lines.append("")

    lines.append("## 5. Sessions (most recent)")
    lines.append("")
    if not sessions:
        lines.append("_No session_id-tagged traces yet._")
    else:
        lines.append("| session_id | traces | graded | model | closes | webfetch_fail | notes |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in sessions:
            stats = s["exec_stats"] or {}
            notes = (s["agent_notes"] or "").replace("|", "\\|").replace("\n", " ")[:80]
            lines.append(
                f"| `{s['session_id']}` | {s['trace_count']} | {s['graded_count']} | "
                f"{s.get('model_version') or '?'} | "
                f"{stats.get('closing_line_captures', '?')} | "
                f"{stats.get('webfetch_failures', '?')} | {notes} |"
            )
    lines.append("")

    lines.append("## 6. Pending CANDIDATE profiles")
    lines.append("")
    if not candidates:
        lines.append("_No pending candidates._")
    else:
        lines.append("| profile_id | method | n | brier | ece | log_loss | created |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in candidates:
            m = c["metrics"]
            lines.append(
                f"| `{c['profile_id']}` | {c['method']} | {c['sample_size']} | "
                f"{m.get('brier_score', float('nan')):.4f} | "
                f"{m.get('calibration_error', float('nan')):.4f} | "
                f"{m.get('log_loss', float('nan')):.4f} | {c['created_at'][:10]} |"
            )
    lines.append("")

    lines.append("## 7. Suggested actions")
    lines.append("")
    lines.append(
        "_This section is intentionally empty. The LLM consumes the data above and emits "
        "an action plan per system_prompt.txt §13._"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit a markdown calibration health report.")
    parser.add_argument("--league", required=True)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None, help="Output path (default: reports/latest.md)")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument(
        "--sessions-inbox",
        type=Path,
        default=_REPO_ROOT / "inbox" / "sessions",
        help="Directory containing session sidecar JSON files",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_path: Path = args.out or (_REPO_ROOT / "reports" / "latest.md")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create output directory %s: %s", out_path.parent, exc)
        return 1

    store = TraceStore(db_path=args.db)
    registry = CalibrationRegistry()
    sidecars = _load_session_sidecars(args.sessions_inbox)
    cutoff = _window_cutoff(args.window_days)

    counts = _section_counts(store, args.league, cutoff)
    realized = _section_realized_metrics(store, args.league, cutoff)
    clv = _section_clv(store, args.league, cutoff)
    sessions = _section_sessions(store, sidecars, args.league)

    prod = registry.get_production(args.league)
    candidates = _section_pending_candidates(registry, args.league)
    store.close()

    rendered = _render(
        league=args.league,
        window_days=args.window_days,
        counts=counts,
        realized=realized,
        clv=clv,
        sessions=sessions,
        production_profile_id=prod.profile_id if prod else None,
        candidates=candidates,
    )

    out_path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote %s (%d traces, %d sessions, %d candidates).",
                out_path, counts["traces"], len(sessions), len(candidates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
