"""Resolve current Odds API markets into Omega engine-ready inputs.

Default behavior is BetMGM-only. Multi-book output is available only when the
caller explicitly requests line shopping or all-book comparison.

This script prepares market inputs and provenance. It does not compute model
probabilities, edge, EV, Kelly, staking, confidence tiers, or trace IDs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.integrations.odds_api import (  # noqa: E402
    DEFAULT_BOOKMAKER,
    DEFAULT_MARKETS,
    BookOdds,
    EventOdds,
    OddsApiBudgetExceeded,
    OddsApiClient,
    OddsApiKeyMissing,
)

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
        if (
            _norm(event.home_team) == _norm(home_team)
            and _norm(event.away_team) == _norm(away_team)
        ):
            return event.event_id, []
    return None, [
        f"no exact event match for {away_team} @ {home_team}",
        f"candidate_events={len(events)}",
    ]


def _bookmakers(bookmaker: str, line_shopping: bool, all_books: bool) -> str | None:
    return None if line_shopping or all_books else bookmaker


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
        elif market == "spread" and selection == _norm(home_team):
            selected.setdefault("spread_home", quote["line"])
            selected.setdefault("spread_home_price", quote["price"])
        elif market == "total" and selection == "over":
            selected.setdefault("over_under", quote["line"])
    return selected


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
        if line is not None and q_line_f != float(line):
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


def resolve_odds(
    *,
    kind: str,
    league: str,
    home_team: str | None = None,
    away_team: str | None = None,
    player_name: str | None = None,
    prop_type: str | None = None,
    line: float | None = None,
    event_id: str | None = None,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
    bookmaker: str = DEFAULT_BOOKMAKER,
    line_shopping: bool = False,
    all_books: bool = False,
    client: OddsApiClient | None = None,
) -> dict[str, Any]:
    """Resolve current odds for game or prop analysis."""
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
        return _unavailable(kind, league, bookmaker, skipped, client)

    prop_type_by_market: dict[str, str] = {}
    if kind == "prop":
        if not player_name or not prop_type:
            return _unavailable(
                kind,
                league,
                bookmaker,
                ["player_name and prop_type are required for prop odds"],
                client,
            )
        market_key = provider_market_for_prop(league, prop_type)
        if not market_key:
            return _unavailable(
                kind,
                league,
                bookmaker,
                [f"no provider market mapping for {league.upper()} prop_type={prop_type!r}"],
                client,
            )
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
            return _unavailable(kind, league, bookmaker, availability_skips, client)
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
    except (OddsApiKeyMissing, OddsApiBudgetExceeded) as exc:
        return _unavailable(kind, league, bookmaker, [str(exc)], client)

    quotes = normalize_event_odds(event, league=league, prop_type_by_market=prop_type_by_market)
    if kind == "prop":
        selected = _select_prop_input(
            quotes,
            player_name=player_name or "",
            prop_type=prop_type or "",
            line=line,
        )
        request_patch = selected
    else:
        selected = _select_game_input(
            quotes,
            home_team or event.home_team,
            away_team or event.away_team,
        )
        request_patch = {"odds": {**selected, "markets": quotes}}

    if not selected:
        skipped.append(
            "no exact BetMGM market match"
            if not (line_shopping or all_books)
            else "no exact market match"
        )
        return _unavailable(kind, league, bookmaker, skipped, client, quotes=quotes, event=event)

    return {
        "status": "success",
        "kind": kind,
        "league": league.upper(),
        "event_id": event.event_id,
        "home_team": event.home_team,
        "away_team": event.away_team,
        "commence_time": event.commence_time,
        "default_bookmaker": bookmaker,
        "line_shopping": line_shopping,
        "all_books": all_books,
        "request_patch": request_patch,
        "quotes": quotes,
        "skipped_reasons": skipped,
        "quota": dict(client.last_quota_headers),
    }


def _unavailable(
    kind: str,
    league: str,
    bookmaker: str,
    skipped: list[str],
    client: OddsApiClient,
    *,
    quotes: list[dict[str, Any]] | None = None,
    event: EventOdds | None = None,
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "kind": kind,
        "league": league.upper(),
        "event_id": event.event_id if event else None,
        "home_team": event.home_team if event else None,
        "away_team": event.away_team if event else None,
        "default_bookmaker": bookmaker,
        "request_patch": None,
        "quotes": quotes or [],
        "skipped_reasons": skipped,
        "quota": dict(client.last_quota_headers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["game", "prop"], required=True)
    parser.add_argument("--league", required=True)
    parser.add_argument("--home-team")
    parser.add_argument("--away-team")
    parser.add_argument("--player-name")
    parser.add_argument("--prop-type")
    parser.add_argument("--line", type=float)
    parser.add_argument("--event-id")
    parser.add_argument("--commence-time-from")
    parser.add_argument("--commence-time-to")
    parser.add_argument("--bookmaker", default=DEFAULT_BOOKMAKER)
    parser.add_argument("--line-shopping", action="store_true")
    parser.add_argument("--all-books", action="store_true")
    args = parser.parse_args()

    result = resolve_odds(
        kind=args.kind,
        league=args.league,
        home_team=args.home_team,
        away_team=args.away_team,
        player_name=args.player_name,
        prop_type=args.prop_type,
        line=args.line,
        event_id=args.event_id,
        commence_time_from=args.commence_time_from,
        commence_time_to=args.commence_time_to,
        bookmaker=args.bookmaker,
        line_shopping=args.line_shopping,
        all_books=args.all_books,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
