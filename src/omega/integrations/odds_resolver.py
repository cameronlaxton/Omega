"""Resolve current Odds API markets into Omega engine-ready inputs.

Default behavior is BetMGM-only. Multi-book output is available only when the
caller explicitly requests line shopping or all-book comparison.

This module prepares market inputs and provenance. It does not compute model
probabilities, edge, EV, Kelly, staking, confidence tiers, or trace IDs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from omega.core.betting.odds import american_to_decimal
from omega.integrations._guards import assert_not_replay_mode
from omega.integrations.odds_api import (
    DEFAULT_BOOKMAKER,
    DEFAULT_MARKETS,
    BookOdds,
    EventOdds,
    HistoricalEvent,
    OddsApiBudgetExceeded,
    OddsApiClient,
    OddsApiKeyMissing,
)
from omega.integrations.odds_cache import OddsCache

PROP_MARKET_MAP: dict[str, dict[str, str]] = {
    "NBA": {
        "pts": "player_points",
        "reb": "player_rebounds",
        "ast": "player_assists",
        "3pm": "player_threes",
        "pra": "player_points_rebounds_assists",
        "pts_reb": "player_points_rebounds",
        "pts_ast": "player_points_assists",
        "reb_ast": "player_rebounds_assists",
        "stl": "player_steals",
        "blk": "player_blocks",
    },
    "WNBA": {
        "pts": "player_points",
        "reb": "player_rebounds",
        "ast": "player_assists",
        "3pm": "player_threes",
    },
    "NFL": {
        "pass_yds": "player_pass_yds",
        "rush_yds": "player_rush_yds",
        "rec_yds": "player_reception_yds",
        "receptions": "player_receptions",
        "pass_td": "player_pass_tds",
        "rush_td": "player_rush_tds",
        "rec_td": "player_anytime_td",
        "completions": "player_pass_completions",
        "interceptions": "player_pass_interceptions",
    },
    "MLB": {
        "hits": "batter_hits",
        "total_bases": "batter_total_bases",
        "runs": "batter_runs_scored",
        "rbis": "batter_rbis",
        "hrs": "batter_home_runs",
        "stolen_bases": "batter_stolen_bases",
        "strikeouts_pitched": "pitcher_strikeouts",
        "outs_recorded": "pitcher_outs",
        "earned_runs": "pitcher_earned_runs",
    },
    "NHL": {
        "goals": "player_goals",
        "assists": "player_assists",
        "points": "player_points",
        "shots_on_goal": "player_shots_on_goal",
        "saves": "player_saves",
    },
}


def provider_market_for_prop(league: str, prop_type: str) -> str | None:
    return PROP_MARKET_MAP.get(league.upper(), {}).get(prop_type)


def normalize_book_odds(
    event: EventOdds,
    book: BookOdds,
    *,
    league: str,
    prop_type_by_market: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert one provider outcome into Omega's MarketQuote-compatible shape."""
    prop_type_by_market = prop_type_by_market or {}
    if book.market == "h2h":
        market_type = "moneyline"
        line = None
        player = None
        stat_key = None
    elif book.market == "spreads":
        market_type = "spread"
        line = book.point
        player = None
        stat_key = None
    elif book.market == "totals":
        market_type = "total"
        line = book.point
        player = None
        stat_key = None
    elif book.market in prop_type_by_market:
        market_type = "player_prop"
        line = book.point
        player = book.description
        stat_key = prop_type_by_market[book.market]
    else:
        market_type = book.market
        line = book.point
        player = book.description
        stat_key = None

    return {
        "market_type": market_type,
        "selection": book.selection,
        "price": book.price,
        "line": line,
        "segment": "full_game",
        "player": player,
        "stat_key": stat_key,
        "bookmaker": book.bookmaker,
        "source": f"the-odds-api:{book.bookmaker}",
        "event_id": event.event_id,
        "provider_market_key": book.market,
        "last_update": book.last_update,
        "snapshot_timestamp": book.snapshot_timestamp,
        "league": league.upper(),
        "home_team": event.home_team,
        "away_team": event.away_team,
        "commence_time": event.commence_time,
    }


def normalize_event_odds(
    event: EventOdds,
    *,
    league: str,
    prop_type_by_market: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    return [
        normalize_book_odds(event, book, league=league, prop_type_by_market=prop_type_by_market)
        for book in event.books
    ]


def _norm(s: str | None) -> str:
    return (s or "").strip().casefold()


def _resolve_event_id(
    client: OddsApiClient,
    league: str,
    *,
    event_id: str | None,
    home_team: str | None,
    away_team: str | None,
    commence_time_from: str | None,
    commence_time_to: str | None,
) -> tuple[str | None, list[str]]:
    if event_id:
        return event_id, []
    if not home_team or not away_team:
        return None, ["event_id or exact home_team+away_team is required"]
    events = client.fetch_events(
        league,
        commence_time_from=commence_time_from,
        commence_time_to=commence_time_to,
    )
    for event in events:
        if _norm(event.home_team) == _norm(home_team) and _norm(event.away_team) == _norm(
            away_team
        ):
            return event.event_id, []
    return None, [
        f"no exact event match for {away_team} @ {home_team}",
        f"candidate_events={len(events)}",
    ]


def _bookmakers(bookmaker: str, line_shopping: bool, all_books: bool) -> str | None:
    return None if line_shopping or all_books else bookmaker


def format_budget_exhausted_error(
    exc: OddsApiBudgetExceeded,
    client: OddsApiClient | None = None,
) -> str:
    """Stable CLI stop message for automated agents."""
    usage: int | str = "unknown"
    cap: int | str = "unknown"
    if client is not None:
        try:
            status = client.budget_status()
            usage = status["current_usage"]
            cap = status["monthly_cap"]
        except Exception:
            pass
    if usage == "unknown" or cap == "unknown":
        match = re.search(r"used=(\d+).*cap=(\d+)", str(exc))
        if match:
            usage = match.group(1)
            cap = match.group(2)
    return (
        "[error] Odds API Budget Exhausted. "
        f"Current usage: {usage}, Monthly Cap: {cap}. "
        "To expand, adjust OMEGA_ODDS_API_MONTHLY_BUDGET in .env."
    )


def _event_to_row(event: HistoricalEvent, league: str) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "league": league.upper(),
        "sport_key": event.sport_key,
        "commence_time": event.commence_time,
        "away_team": event.away_team,
        "home_team": event.home_team,
        "resolve_hint": (
            f'--event-id "{event.event_id}" --away-team "{event.away_team}" '
            f'--home-team "{event.home_team}"'
        ),
    }


def list_events(
    *,
    league: str,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
    client: OddsApiClient | None = None,
    cache: OddsCache | None = None,
) -> dict[str, Any]:
    """List current events for a league with a hard local TTL cache."""
    assert_not_replay_mode("Odds event listing")
    cache = cache or OddsCache()
    key = cache.compute_event_list_cache_key(
        league,
        commence_time_from=commence_time_from,
        commence_time_to=commence_time_to,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    client = client or OddsApiClient()
    try:
        events = client.fetch_events(
            league,
            commence_time_from=commence_time_from,
            commence_time_to=commence_time_to,
            request_cost=1,
        )
    except OddsApiKeyMissing as exc:
        return {
            "status": "unavailable",
            "kind": "event_list",
            "league": league.upper(),
            "commence_time_from": commence_time_from,
            "commence_time_to": commence_time_to,
            "events": [],
            "metadata": [],
            "quota": {},
            "skipped_reasons": [str(exc)],
        }

    payload = {
        "status": "success" if events else "empty",
        "kind": "event_list",
        "league": league.upper(),
        "commence_time_from": commence_time_from,
        "commence_time_to": commence_time_to,
        "events": [_event_to_row(event, league) for event in events],
        "metadata": ["source: live_api", "cache_kind: event_list"],
        "quota": dict(client.last_quota_headers),
        "skipped_reasons": [] if events else ["no events returned"],
    }
    cache.set(key, league, "events", payload, entry_type="event_list")
    return payload


def _render_event_summary(result: dict[str, Any]) -> str:
    lines = [
        "Omega Odds Event List",
        "=" * 40,
        f"league       : {result.get('league')}",
        f"status       : {result.get('status')}",
        f"source       : {', '.join(result.get('metadata') or []) or '(none)'}",
        f"events       : {len(result.get('events') or [])}",
    ]
    if result.get("commence_time_from") or result.get("commence_time_to"):
        lines.append(
            "window       : "
            f"{result.get('commence_time_from') or '(open)'} -> "
            f"{result.get('commence_time_to') or '(open)'}"
        )
    if result.get("skipped_reasons"):
        lines.append(f"notes        : {', '.join(result['skipped_reasons'])}")
    lines.append("")
    for row in result.get("events") or []:
        lines.append(
            f"{row['away_team']} @ {row['home_team']} | "
            f"event_id={row['event_id']} | commence_time={row['commence_time']}"
        )
        lines.append(f"  {row['resolve_hint']}")
    if not result.get("events"):
        lines.append("(no events returned)")
    return "\n".join(lines)


def _prop_market_available(
    client: OddsApiClient,
    league: str,
    event_id: str,
    market_key: str,
    bookmaker: str,
    *,
    line_shopping: bool,
    all_books: bool,
) -> tuple[bool, list[str]]:
    availability = client.fetch_event_markets(
        league,
        event_id,
        bookmakers=_bookmakers(bookmaker, line_shopping, all_books),
    )
    if not availability:
        return True, []
    if line_shopping or all_books:
        if any(market_key in row.markets for row in availability):
            return True, []
        return False, [f"market {market_key!r} unavailable across returned bookmakers"]
    for row in availability:
        if row.bookmaker == bookmaker and market_key in row.markets:
            return True, []
    return False, [f"{bookmaker} does not list market {market_key!r} for this event"]


def _select_game_input(
    quotes: Iterable[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for quote in quotes:
        market = quote["market_type"]
        selection = _norm(quote["selection"])
        if market == "moneyline" and selection == _norm(home_team):
            selected.setdefault("moneyline_home", quote["price"])
        elif market == "moneyline" and selection == _norm(away_team):
            selected.setdefault("moneyline_away", quote["price"])
        elif market == "moneyline" and selection == _norm("Draw"):
            # 3-way moneyline (soccer, hockey regulation): the provider's h2h
            # returns Home/Draw/Away; preserve the Draw price for OddsInput.moneyline_draw.
            selected.setdefault("moneyline_draw", quote["price"])
        elif market == "spread" and selection == _norm(home_team):
            selected.setdefault("spread_home", quote["line"])
            selected.setdefault("spread_home_price", quote["price"])
        elif market == "total" and selection == "over":
            selected.setdefault("over_under", quote["line"])
        else:
            _select_exotic_quote(selected, market, selection, quote, home_team, away_team)
    return selected


def _select_exotic_quote(
    selected: dict[str, Any],
    market: str,
    selection: str,
    quote: dict[str, Any],
    home_team: str,
    away_team: str,
) -> None:
    """Map provider exotic-market quotes (soccer) into OddsInput fields.

    Defensive by design: unrecognised selections are ignored. Provider market
    keys follow the-odds-api conventions (double_chance, draw_no_bet, btts,
    correct_score). Selection name matching is normalised and tolerant of the
    common symbolic ('1X'/'12'/'X2') and team-name combo forms.
    """
    price = quote["price"]
    h, a, draw = _norm(home_team), _norm(away_team), _norm("Draw")

    if market == "double_chance":
        if selection in ("1x", "x1") or (h in selection and draw in selection):
            selected.setdefault("dc_home_draw", price)
        elif selection in ("12", "21") or (h in selection and a in selection):
            selected.setdefault("dc_home_away", price)
        elif selection in ("x2", "2x") or (a in selection and draw in selection):
            selected.setdefault("dc_away_draw", price)
    elif market == "draw_no_bet":
        if selection == h:
            selected.setdefault("dnb_home", price)
        elif selection == a:
            selected.setdefault("dnb_away", price)
    elif market in ("btts", "both_teams_to_score"):
        if selection == "yes":
            selected.setdefault("btts_yes", price)
        elif selection == "no":
            selected.setdefault("btts_no", price)
    elif market == "correct_score":
        scoreline = _normalize_scoreline(quote.get("selection"))
        if scoreline is not None:
            selected.setdefault("correct_score", {})
            selected["correct_score"].setdefault(scoreline, price)


def _normalize_scoreline(selection: str | None) -> str | None:
    """Normalise a correct-score selection like '1 - 0' or '1:0' to '1-0'."""
    if not selection:
        return None
    raw = selection.replace(":", "-").replace(" ", "")
    parts = raw.split("-")
    if len(parts) != 2:
        return None
    try:
        h, a = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return f"{h}-{a}"


def _filter_prop_quotes(
    quotes: list[dict[str, Any]],
    *,
    player_name: str,
    prop_type: str,
) -> list[dict[str, Any]]:
    """Return only the quotes matching the requested player and stat type."""
    return [
        q
        for q in quotes
        if q.get("market_type") == "player_prop"
        and q.get("stat_key") == prop_type
        and _norm(q.get("player")) == _norm(player_name)
    ]


def _select_prop_input(
    quotes: Iterable[dict[str, Any]],
    *,
    player_name: str,
    prop_type: str,
    line: float | None,
) -> dict[str, Any]:
    grouped: dict[float, dict[str, float]] = defaultdict(dict)
    for quote in quotes:
        if quote["market_type"] != "player_prop":
            continue
        if quote.get("stat_key") != prop_type:
            continue
        if _norm(quote.get("player")) != _norm(player_name):
            continue
        q_line = quote.get("line")
        if q_line is None:
            continue
        q_line_f = float(q_line)
        if line is not None and q_line_f != line:
            continue
        grouped[q_line_f][quote["selection"].casefold()] = quote["price"]

    for q_line in sorted(grouped):
        sides = grouped[q_line]
        out: dict[str, Any] = {"line": q_line}
        if "over" in sides:
            out["odds_over"] = sides["over"]
        if "under" in sides:
            out["odds_under"] = sides["under"]
        if "odds_over" in out or "odds_under" in out:
            return out
    return {}


def best_price_quotes(quotes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Across all books, return the single best-priced quote per selection.

    Groups quotes by (market_type, normalized selection, line) and keeps, for
    each group, the quote whose price pays the bettor the most — i.e. the
    highest decimal payout. `american_to_decimal` is monotonic in the bettor's
    favor, so comparing decimals ranks +150 above +120 and -110 above -130
    correctly across the sign boundary.

    Each returned row keeps its originating `bookmaker`, so a surfaced best
    price is always attributable to one real, placeable book. This intentionally
    does NOT fabricate a cross-book line (e.g. best over from book A + best under
    from book B): those two sides live in separate selection groups and are
    reported independently, as advisory line-shopping output. De-vig and the
    engine's OddsInput must still be sourced from a single consistent book.

    Ties (equal decimal payout) are broken deterministically: most recent
    `last_update`, then bookmaker name.
    """
    best: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for quote in quotes:
        price = quote.get("price")
        if price is None:
            continue
        try:
            payout = american_to_decimal(float(price))
        except (TypeError, ValueError):
            continue
        key = (
            str(quote.get("market_type") or ""),
            _norm(quote.get("selection")),
            quote.get("line"),
        )
        current = best.get(key)
        if current is None or _is_better_price(quote, payout, current):
            enriched = dict(quote)
            enriched["decimal_payout"] = round(payout, 4)
            best[key] = enriched
    return sorted(
        best.values(),
        key=lambda q: (str(q.get("market_type") or ""), _norm(q.get("selection"))),
    )


def _is_better_price(quote: dict[str, Any], payout: float, current: dict[str, Any]) -> bool:
    """True if `quote` should replace `current` as the best price for its group."""
    cur_payout = current.get("decimal_payout", 0.0)
    if payout != cur_payout:
        return payout > cur_payout
    # Tie on payout: prefer the fresher quote, then a stable bookmaker order.
    q_update = str(quote.get("last_update") or "")
    c_update = str(current.get("last_update") or "")
    if q_update != c_update:
        return q_update > c_update
    return str(quote.get("bookmaker") or "") < str(current.get("bookmaker") or "")


def resolve_odds(
    *,
    kind: str,
    league: str,
    home_team: str | None = None,
    away_team: str | None = None,
    player_name: str | None = None,
    player_id: str | None = None,
    prop_type: str | None = None,
    line: float | None = None,
    event_id: str | None = None,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
    bookmaker: str = DEFAULT_BOOKMAKER,
    line_shopping: bool = False,
    all_books: bool = False,
    client: OddsApiClient | None = None,
    cache: OddsCache | None = None,
) -> dict[str, Any]:
    """Resolve current odds for game or prop analysis with local SQLite caching."""
    assert_not_replay_mode("Odds resolver fetch")

    # 1. Caching Entry Layer
    cache = cache or OddsCache()
    market = (prop_type or "") if kind == "prop" else "game"
    norm_home = (home_team or "").strip().lower()
    norm_away = (away_team or "").strip().lower()
    game_date = (commence_time_from or "").strip().split("T")[0] if commence_time_from else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cached_payload: dict[str, Any] | None = None
    negative_key: str | None = None

    if norm_home and norm_away:
        cache_key = cache.compute_cache_key(
            league, market, norm_home, norm_away, game_date,
            player_name=player_name, player_id=player_id
        )
        negative_key = cache_key
        cached_payload = cache.get(cache_key)
        if not cached_payload:
            cached_payload = cache.find_by_teams(
                league, market, norm_home, norm_away,
                player_name=player_name, player_id=player_id
            )

    if not cached_payload and event_id:
        cached_payload = cache.find_by_event_id(
            league, market, event_id,
            player_name=player_name, player_id=player_id
        )

    if cached_payload:
        return cached_payload

    def _fail(payload: dict[str, Any]) -> dict[str, Any]:
        """Store an unavailable result under the short negative-cache TTL so a
        repeated identical lookup short-circuits before any API call, then return
        it unchanged. No key (teams unknown) means we skip caching this miss."""
        if negative_key is not None:
            cache.set(negative_key, league, market, payload, entry_type="negative")
        return payload

    # 2. Cache Miss - Execute external resolution
    client = client or OddsApiClient()
    skipped: list[str] = []
    resolved_event_id, event_skips = _resolve_event_id(
        client,
        league,
        event_id=event_id,
        home_team=home_team,
        away_team=away_team,
        commence_time_from=commence_time_from,
        commence_time_to=commence_time_to,
    )
    skipped.extend(event_skips)
    if not resolved_event_id:
        return _fail(_unavailable(kind, league, bookmaker, skipped, client))

    prop_type_by_market: dict[str, str] = {}
    if kind == "prop":
        if not player_name or not prop_type:
            return _fail(_unavailable(
                kind,
                league,
                bookmaker,
                ["player_name and prop_type are required for prop odds"],
                client,
            ))
        market_key = provider_market_for_prop(league, prop_type)
        if not market_key:
            return _fail(_unavailable(
                kind,
                league,
                bookmaker,
                [f"no provider market mapping for {league.upper()} prop_type={prop_type!r}"],
                client,
            ))
        available, availability_skips = _prop_market_available(
            client,
            league,
            resolved_event_id,
            market_key,
            bookmaker,
            line_shopping=line_shopping,
            all_books=all_books,
        )
        if not available:
            return _fail(_unavailable(kind, league, bookmaker, availability_skips, client))
        markets = market_key
        prop_type_by_market[market_key] = prop_type
    else:
        markets = DEFAULT_MARKETS

    try:
        event = client.fetch_current_event_odds(
            league,
            resolved_event_id,
            markets=markets,
            bookmakers=_bookmakers(bookmaker, line_shopping, all_books),
        )
    except OddsApiBudgetExceeded:
        raise
    except OddsApiKeyMissing as exc:
        return _fail(_unavailable(kind, league, bookmaker, [str(exc)], client))

    quotes = normalize_event_odds(event, league=league, prop_type_by_market=prop_type_by_market)
    if kind == "prop":
        player_quotes = _filter_prop_quotes(
            quotes,
            player_name=player_name or "",
            prop_type=prop_type or "",
        )
        selected = _select_prop_input(
            player_quotes,
            player_name=player_name or "",
            prop_type=prop_type or "",
            line=line,
        )
        # Record book provenance only when a single book was queried — in that
        # case both sides came from `bookmaker`. Under line shopping the two
        # sides may differ, so leave it unset (ledger records 'consensus') and
        # rely on the best_prices advisory block below.
        if selected and not (line_shopping or all_books):
            selected["bookmaker"] = bookmaker
        request_patch = selected
        output_quotes = player_quotes
    else:
        selected = _select_game_input(
            quotes,
            home_team or event.home_team,
            away_team or event.away_team,
        )
        request_patch = {"odds": {**selected, "markets": quotes}}
        output_quotes = quotes

    if not selected:
        skipped.append(
            "no exact BetMGM market match"
            if not (line_shopping or all_books)
            else "no exact market match"
        )
        return _fail(_unavailable(
            kind, league, bookmaker, skipped, client, quotes=output_quotes, event=event
        ))

    result_payload = {
        "status": "success",
        "kind": kind,
        "league": league.upper(),
        