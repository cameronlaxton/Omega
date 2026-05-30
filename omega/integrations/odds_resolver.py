"""Resolve current Odds API markets into Omega engine-ready inputs.

Default behavior is BetMGM-only. Multi-book output is available only when the
caller explicitly requests line shopping or all-book comparison.

This module prepares market inputs and provenance. It does not compute model
probabilities, edge, EV, Kelly, staking, confidence tiers, or trace IDs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from omega.integrations._guards import assert_not_replay_mode
from omega.integrations.odds_cache import OddsCache
from omega.integrations.odds_api import (
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
    cache: OddsCache | None = None,
) -> dict[str, Any]:
    """Resolve current odds for game or prop analysis with local SQLite caching."""
    assert_not_replay_mode("Odds resolver fetch")

    # 1. Caching Entry Layer
    cache = cache or OddsCache()
    market = prop_type if kind == "prop" else "game"
    norm_home = (home_team or "").strip().lower()
    norm_away = (away_team or "").strip().lower()
    game_date = (commence_time_from or "").strip().split("T")[0] if commence_time_from else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cached_payload: dict[str, Any] | None = None
    negative_key: str | None = None

    if norm_home and norm_away:
        cache_key = cache.compute_cache_key(league, market, norm_home, norm_away, game_date)
        negative_key = cache_key
        cached_payload = cache.get(cache_key)
        if not cached_payload:
            cached_payload = cache.find_by_teams(league, market, norm_home, norm_away)

    if not cached_payload and event_id:
        cached_payload = cache.find_by_event_id(league, market, event_id)

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
    except (OddsApiKeyMissing, OddsApiBudgetExceeded) as exc:
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
        "event_id": event.event_id,
        "home_team": event.home_team,
        "away_team": event.away_team,
        "commence_time": event.commence_time,
        "default_bookmaker": bookmaker,
        "line_shopping": line_shopping,
        "all_books": all_books,
        "request_patch": request_patch,
        "quotes": output_quotes,
        "skipped_reasons": skipped,
        "quota": dict(client.last_quota_headers),
        "metadata": [],
    }

    # 3. Store freshly fetched result in cache
    actual_home = event.home_team or home_team or ""
    actual_away = event.away_team or away_team or ""
    actual_date = (event.commence_time or "").split("T")[0] if event.commence_time else game_date

    precise_key = cache.compute_cache_key(
        league=league,
        market=market,
        home_team=actual_home,
        away_team=actual_away,
        game_date=actual_date
    )
    cache.set(precise_key, league, market, result_payload)

    return result_payload


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
    parser.add_argument(
        "--list-prop-types",
        action="store_true",
        help="Print valid prop stat keys for --league (or all leagues) and exit. No API call is made.",
    )
    parser.add_argument("--kind", choices=["game", "prop"])
    parser.add_argument("--league")
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

    if args.list_prop_types:
        league_filter = args.league.upper() if args.league else None
        leagues = [league_filter] if league_filter else sorted(PROP_MARKET_MAP)
        out: dict[str, Any] = {}
        for lg in leagues:
            keys = PROP_MARKET_MAP.get(lg)
            if keys:
                out[lg] = sorted(keys)
        print(json.dumps(out, indent=2))
        return 0

    if not args.kind:
        parser.error("--kind is required unless --list-prop-types is specified")
    if not args.league:
        parser.error("--league is required unless --list-prop-types is specified")

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
