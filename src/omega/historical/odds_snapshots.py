"""Assemble per-event odds observations into opening/decision/closing tiers.

Decision odds are chosen by an as-of policy (default ``latest_before_decision``)
so a bet is only ever sized off prices knowable at decision time. Closing odds
are retained for CLV **only** and are never returned as decision odds. An event
with no decision-eligible quotes still yields a snapshot (``missing_odds=True``)
so probability-only replay can proceed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from omega.historical.contracts import (
    HistoricalMarketSnapshot,
    OddsObservation,
    OddsQuote,
)

UTC = timezone.utc


def _to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    iso = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _as_quote(obs: OddsObservation) -> OddsQuote:
    return OddsQuote(
        market=obs.market,
        selection_descriptor=obs.selection_descriptor,
        odds=obs.odds,
        line=obs.line,
        book=obs.book,
        timestamp=obs.timestamp,
    )


def build_odds_snapshot(
    event_id: str,
    observations: list[OddsObservation],
    decision_time: str,
    *,
    event_start: str | None = None,
    policy: str = "latest_before_decision",
) -> HistoricalMarketSnapshot:
    """Build a :class:`HistoricalMarketSnapshot` for one event.

    ``observations`` must already be filtered to this event. Tiering per
    (market, selection):

    - **opening** = a ``tier_hint="opening"`` quote, else the earliest timestamped.
    - **closing** = a ``tier_hint="closing"`` quote, else the latest timestamped
      at/after the event start (CLV only).
    - **decision** = the latest non-closing quote strictly before
      ``decision_time``; if no quote is timestamped, the single source pre-match
      price is trusted as the decision quote.
    """
    decision_dt = _to_dt(decision_time)
    start_dt = _to_dt(event_start)

    groups: dict[tuple[str, str], list[OddsObservation]] = {}
    for obs in observations:
        groups.setdefault((obs.market, obs.selection_descriptor), []).append(obs)

    opening: list[OddsQuote] = []
    decision: list[OddsQuote] = []
    closing: list[OddsQuote] = []

    for _key, obs_list in groups.items():
        timestamped = [o for o in obs_list if _to_dt(o.timestamp) is not None]
        untimestamped = [o for o in obs_list if _to_dt(o.timestamp) is None]

        # --- opening -------------------------------------------------------
        opening_hint = next((o for o in obs_list if o.tier_hint == "opening"), None)
        if opening_hint is not None:
            opening.append(_as_quote(opening_hint))
        elif timestamped:
            opening.append(_as_quote(min(timestamped, key=lambda o: _to_dt(o.timestamp))))  # type: ignore[arg-type]

        # --- closing (CLV only) -------------------------------------------
        closing_hint = next((o for o in obs_list if o.tier_hint == "closing"), None)
        if closing_hint is not None:
            closing.append(_as_quote(closing_hint))
        elif timestamped and start_dt is not None:
            post = [o for o in timestamped if _to_dt(o.timestamp) >= start_dt]  # type: ignore[operator]
            if post:
                closing.append(_as_quote(max(post, key=lambda o: _to_dt(o.timestamp))))  # type: ignore[arg-type]

        # --- decision (as-of safe) ----------------------------------------
        decision_quote: OddsQuote | None = None
        if decision_dt is not None:
            eligible = [
                o
                for o in timestamped
                if o.tier_hint != "closing" and _to_dt(o.timestamp) < decision_dt  # type: ignore[operator]
            ]
            if eligible:
                decision_quote = _as_quote(max(eligible, key=lambda o: _to_dt(o.timestamp)))  # type: ignore[arg-type]
        if decision_quote is None:
            # Source-level pre-match prices with no timestamp (e.g. football-data
            # closing-ish columns) are trusted as the decision quote.
            untimed_non_closing = [o for o in untimestamped if o.tier_hint != "closing"]
            if untimed_non_closing:
                decision_quote = _as_quote(untimed_non_closing[0])
        if decision_quote is not None:
            decision.append(decision_quote)

    snapshot = HistoricalMarketSnapshot(
        event_id=event_id,
        decision_time=decision_time,
        decision_policy=policy,
        opening=opening,
        decision=decision,
        closing=closing,
        missing_odds=len(decision) == 0,
    )
    snapshot.odds_snapshot_hash = snapshot.compute_hash()
    return snapshot
