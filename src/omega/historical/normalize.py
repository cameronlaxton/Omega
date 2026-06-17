"""Normalization helpers: raw adapter values → canonical shapes.

Owns date→UTC parsing, league→sport_family mapping, and market-name
canonicalization. Adapters call these helpers after column-mapping + Pydantic
validation; the canonical contracts (HistoricalEvent/Outcome/MarketSnapshot) are
assembled here from already-validated, identity-resolved values.
"""

from __future__ import annotations

from datetime import datetime, timezone

from omega.core.simulation.archetypes import get_archetype_name

UTC = timezone.utc

# Formats tried after ISO 8601 and the integer YYYYMMDD shortcut. Covers the
# public dataset schemas we target (football-data DD/MM/Y, nflverse YYYY-MM-DD,
# Sackmann YYYYMMDD handled separately).
_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d/%m/%y",
    "%m/%d/%Y",
    "%m/%d/%y",
)

# Canonical market names → accepted source spellings (lowercased).
_MARKET_ALIASES = {
    "moneyline": {"moneyline", "ml", "h2h", "money_line", "match_winner"},
    "spread": {"spread", "handicap", "point_spread", "ats", "set_spread"},
    "total": {"total", "totals", "over_under", "ou", "total_games", "total_goals"},
    "home_draw_away": {"home_draw_away", "1x2", "3way", "moneyline_3way", "result"},
    "draw": {"draw", "the_draw"},
    "double_chance": {"double_chance", "dc"},
    "draw_no_bet": {"draw_no_bet", "dnb"},
    "btts": {"btts", "both_teams_to_score", "gg"},
    "puck_line": {"puck_line", "puckline"},
    "run_line": {"run_line", "runline"},
}

_MARKET_LOOKUP = {
    spelling: canonical
    for canonical, spellings in _MARKET_ALIASES.items()
    for spelling in spellings
}


def parse_datetime_utc(raw: str | int | float | datetime) -> str:
    """Parse a heterogeneous date/datetime value into an ISO 8601 UTC string.

    Date-only inputs land at 00:00:00 UTC. Raises ``ValueError`` for
    unparseable input so unknown timestamps fail closed rather than silently
    becoming ``None`` (the leakage guard depends on this).
    """
    if isinstance(raw, datetime):
        dt = raw if raw.tzinfo else raw.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat()

    s = str(raw).strip()
    if not s:
        raise ValueError("empty timestamp")

    # Integer YYYYMMDD (Sackmann tourney_date and similar)
    if s.isdigit() and len(s) == 8:
        dt = datetime.strptime(s, "%Y%m%d").replace(tzinfo=UTC)
        return dt.isoformat()

    # ISO 8601, tolerating a trailing Z
    iso_candidate = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(iso_candidate)
        dt = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat()
    except ValueError:
        pass

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=UTC)
            return dt.isoformat()
        except ValueError:
            continue

    raise ValueError(f"unparseable timestamp: {raw!r}")


def sport_family_for(league: str) -> str:
    """Map a league code to its archetype name, raising if unknown."""
    name = get_archetype_name(league)
    if name is None:
        raise ValueError(f"unknown league {league!r}: no archetype mapping")
    return name


def canonical_market(raw: str) -> str:
    """Canonicalize a market name; unknown names pass through normalized."""
    key = (raw or "").strip().lower().replace(" ", "_")
    return _MARKET_LOOKUP.get(key, key)


def to_bool(raw: object) -> bool:
    """Tolerant truthiness for CSV string flags ('1','true','yes','y')."""
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "t"}


def to_int_or_none(raw: object) -> int | None:
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def to_float_or_none(raw: object) -> float | None:
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def decimal_to_american(decimal_odds: float | None) -> float | None:
    """Convert decimal odds (e.g. football-data B365 columns) to American odds.

    Returns ``None`` for non-positive or even-money-or-below-1 inputs that can't
    be expressed as valid American odds, so they fail closed rather than coerce.
    """
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100)
    return round(-100.0 / (decimal_odds - 1.0))
