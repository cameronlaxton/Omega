"""Leakage guard: refuse to let post-game information reach a pre-game decision.

Every replayed event passes through :func:`evaluate_leakage` before a request is
ever built. The guard fails closed (or skips with an explicit reason) on any of:

- post-event features (feature ``as_of`` at/after the event start);
- unknown timestamps (missing event start, decision time, or feature ``as_of``);
- a decision made at/after the event (decision_time >= start);
- closing odds masquerading as the decision line (a decision quote timestamped
  at/after the decision cutoff or the event start);
- final-season stats applied earlier in the season / rolling windows whose end
  reaches the current or a future event.

Reasons are machine-stable strings so the leakage report can aggregate them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict

from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalFeatureSnapshot,
    HistoricalMarketSnapshot,
)

UTC = timezone.utc


class LeakageStatus(BaseModel):
    """Verdict for one event's pre-game safety."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["clean", "skipped", "failed"]
    reasons: list[str] = []

    @property
    def is_clean(self) -> bool:
        return self.status == "clean"


def _to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    iso = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def evaluate_leakage(
    event: HistoricalEvent,
    feature_snapshot: HistoricalFeatureSnapshot,
    odds_snapshot: HistoricalMarketSnapshot | None = None,
    *,
    rolling_window_ends: list[str] | None = None,
    policy: Literal["skip", "fail"] = "skip",
) -> LeakageStatus:
    """Return a :class:`LeakageStatus` for one event.

    ``rolling_window_ends`` lets a snapshot declare the end date of every rolling
    window it consumed (e.g. "last 5 games as of D"); any end at/after the event
    start is a leak. ``policy`` selects whether an unsafe row fails closed
    (``"fail"``) or is skipped (``"skip"``).
    """
    reasons: list[str] = []

    start = _to_dt(event.start_time)
    decision = _to_dt(feature_snapshot.decision_time)
    as_of = _to_dt(feature_snapshot.as_of)

    # --- unknown timestamps -------------------------------------------------
    if start is None:
        reasons.append("unknown_event_start")
    if decision is None:
        reasons.append("unknown_decision_time")
    if feature_snapshot.as_of is not None and as_of is None:
        reasons.append("unknown_feature_as_of")

    # --- decision must precede the event -----------------------------------
    if start is not None and decision is not None and decision > start:
        reasons.append("decision_after_event")

    # --- post-event / post-decision features -------------------------------
    if as_of is not None:
        if start is not None and as_of >= start:
            reasons.append("post_event_features")
        if decision is not None and as_of >= decision:
            reasons.append("feature_after_decision_time")

    # --- closing line used as the decision line ----------------------------
    if odds_snapshot is not None:
        for q in odds_snapshot.decision:
            ts = _to_dt(q.timestamp)
            if ts is None:
                continue  # untimestamped source pre-match price; trusted
            if start is not None and ts >= start:
                reasons.append("closing_as_decision")
                break
            if decision is not None and ts >= decision:
                reasons.append("decision_odds_after_decision_time")
                break

    # --- rolling windows / final-season stats reaching the event -----------
    if rolling_window_ends and start is not None:
        for end in rolling_window_ends:
            end_dt = _to_dt(end)
            if end_dt is None:
                reasons.append("unknown_rolling_window_end")
                break
            if end_dt >= start:
                reasons.append("rolling_window_includes_event")
                break

    # de-duplicate while preserving order
    seen: set[str] = set()
    deduped = [r for r in reasons if not (r in seen or seen.add(r))]

    if not deduped:
        return LeakageStatus(status="clean", reasons=[])
    return LeakageStatus(status="failed" if policy == "fail" else "skipped", reasons=deduped)
