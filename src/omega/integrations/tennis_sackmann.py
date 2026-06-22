"""
omega.integrations.tennis_sackmann — Jeff Sackmann ATP/WTA match adapter.

The Sackmann repositories (github.com/JeffSackmann/tennis_atp, tennis_wta) are
the standard open dataset for tour-level tennis. This adapter computes the
surface-segmented rolling serve/return point-win rates (SPW%/RPW%, 12-month
half-life) that feed ``priors_tennis`` and the TennisMarkovBackend
(Phase 7 M3, design Part 5).

Local-first: ``data/tennis/`` already holds ATP singles CSVs and acts as the
pre-seeded snapshot — files found there are read directly with zero network.
Missing files (all WTA years, future ATP updates) are pulled through the ETL
cache (``data/cache/sackmann/``) from raw.githubusercontent.com.

ETL standards: every row validates against ``SackmannMatchRow`` (a renamed or
dropped column fails loud; legitimately blank serve stats — walkovers — are
tolerated and the match is excluded from the rates), and player names resolve
through ``data/aliases/TENNIS.json`` before any priors write. Rows older than
14 days relative to the repo's weekly cadence are handled by ``as_of_date``
staleness checks downstream (design Part 8).
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from omega.integrations._etl import cached_fetch, resolve_entity, validate_records
from omega.integrations._guards import assert_not_replay_mode
from omega.trace.priors import TennisPrior

logger = logging.getLogger("omega.integrations.tennis_sackmann")

_RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{tour}_matches_{year}.csv"
)
_REQUEST_TIMEOUT_SECONDS = 30
_CACHE_TTL_SECONDS = 7 * 24 * 3600  # weekly refresh cadence (design Part 8)
_DEFAULT_LOCAL_ROOT = Path("data/tennis")

_HALF_LIFE_DAYS = 365.25  # 12-month half-life on point weights

TOURS = ("atp", "wta")


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


class SackmannMatchRow(BaseModel):
    """The subset of the 49-column Sackmann match row the rates consume.

    Field *presence* is mandatory (drift fails loud); serve-stat *values* may
    be blank for walkovers/retirements, in which case the match contributes
    nothing to the rates.
    """

    # Required-presence fields: `int | None` WITHOUT a default means the
    # column must exist in the CSV (drift fails loud) while a blank value
    # (walkover) coerces to None and is tolerated.
    tourney_date: int
    surface: str | None
    winner_name: str
    loser_name: str
    score: str | None
    w_svpt: int | None
    w_1stWon: int | None
    w_2ndWon: int | None
    l_svpt: int | None
    l_1stWon: int | None
    l_2ndWon: int | None

    _coerce_blanks = field_validator(
        "surface",
        "score",
        "w_svpt",
        "w_1stWon",
        "w_2ndWon",
        "l_svpt",
        "l_1stWon",
        "l_2ndWon",
        mode="before",
    )(_blank_to_none)


def _download_matches_csv(tour: str, year: int, url_opener: Callable[..., Any]) -> str:
    """Raw network fetch — replay-guarded; wrapped by cached_fetch upstream."""
    assert_not_replay_mode("sackmann match CSV fetch")
    url = _RAW_URL_TEMPLATE.format(tour=tour, year=year)
    logger.info("fetching sackmann CSV: %s", url)
    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_matches_csv(
    tour: str,
    year: int,
    *,
    local_root: str | Path | None = None,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> str:
    """Return one season's match CSV text, local-first then cached fetch."""
    tour = tour.lower()
    if tour not in TOURS:
        raise ValueError(f"tour must be one of {TOURS}, got {tour!r}")

    local = Path(local_root or _DEFAULT_LOCAL_ROOT) / f"{tour}_matches_{year}.csv"
    if local.exists():
        return local.read_text(encoding="utf-8", errors="replace")

    @cached_fetch("sackmann", ttl_seconds=_CACHE_TTL_SECONDS, fmt="text", cache_root=cache_root)
    def _fetch() -> str:
        return _download_matches_csv(tour, year, url_opener)

    return _fetch(cache_key=f"{tour}_matches_{year}")


def parse_matches(csv_text: str, *, session_path: str | None = None) -> list[SackmannMatchRow]:
    """Parse + boundary-validate a Sackmann match CSV."""
    reader = csv.DictReader(io.StringIO(csv_text))
    raw_rows = list(reader)
    return validate_records(
        raw_rows, SackmannMatchRow, source="sackmann", session_path=session_path
    )


def _decay_weight(match_date: int, as_of: date) -> float:
    """12-month half-life weight for a YYYYMMDD tourney_date."""
    played = datetime.strptime(str(match_date), "%Y%m%d").date()
    age_days = max(0.0, (as_of - played).days)
    return 0.5 ** (age_days / _HALF_LIFE_DAYS)


def compute_rolling_rates(
    rows: list[SackmannMatchRow],
    *,
    as_of_date: str,
) -> dict[tuple[str, str], dict[str, float]]:
    """Surface-segmented weighted SPW%/RPW% per player.

    Returns ``{(player, surface): {spw_won, spw_pts, rpw_won, rpw_pts,
    matches}}`` with 12-month half-life weights on points. Matches without
    serve stats (walkovers) or without a surface contribute nothing.
    """
    as_of = date.fromisoformat(as_of_date)
    acc: dict[tuple[str, str], dict[str, float]] = {}

    def _bucket(player: str, surface: str) -> dict[str, float]:
        return acc.setdefault(
            (player, surface),
            {"spw_won": 0.0, "spw_pts": 0.0, "rpw_won": 0.0, "rpw_pts": 0.0, "matches": 0},
        )

    for row in rows:
        if row.surface is None:
            continue
        stats = (row.w_svpt, row.w_1stWon, row.w_2ndWon, row.l_svpt, row.l_1stWon, row.l_2ndWon)
        if any(s is None for s in stats) or row.w_svpt <= 0 or row.l_svpt <= 0:
            continue
        played = datetime.strptime(str(row.tourney_date), "%Y%m%d").date()
        if played > as_of:
            continue
        weight = _decay_weight(row.tourney_date, as_of)
        surface = row.surface.lower()

        w_serve_won = row.w_1stWon + row.w_2ndWon
        l_serve_won = row.l_1stWon + row.l_2ndWon

        winner = _bucket(row.winner_name, surface)
        winner["spw_won"] += weight * w_serve_won
        winner["spw_pts"] += weight * row.w_svpt
        winner["rpw_won"] += weight * (row.l_svpt - l_serve_won)
        winner["rpw_pts"] += weight * row.l_svpt
        winner["matches"] += 1

        loser = _bucket(row.loser_name, surface)
        loser["spw_won"] += weight * l_serve_won
        loser["spw_pts"] += weight * row.l_svpt
        loser["rpw_won"] += weight * (row.w_svpt - w_serve_won)
        loser["rpw_pts"] += weight * row.w_svpt
        loser["matches"] += 1
    return acc


def build_tennis_priors(
    rows: list[SackmannMatchRow],
    *,
    tour: str,
    as_of_date: str,
    alias_table: dict[str, Any] | None = None,
    min_matches: int = 3,
) -> tuple[list[TennisPrior], list[str]]:
    """Aggregate validated match rows into ``TennisPrior`` rows.

    Name handling differs from cross-source adapters: the Sackmann dataset is
    itself the canonical name authority for these priors (its names are the
    lookup keys), so a name that does not resolve through the alias table is
    KEPT verbatim rather than excluded — exclusion is for foreign sources
    whose unresolved names would create ambiguous keys (ETL standard 3). The
    alias table still folds known variants (accents, suffixes) onto one key;
    ``unresolved`` reports them for review. ``min_matches`` guards against
    one-tournament noise on a surface.
    """
    alias_table = alias_table or {"canonical": [], "aliases": {}}
    has_aliases = bool(alias_table.get("canonical") or alias_table.get("aliases"))
    rates = compute_rolling_rates(rows, as_of_date=as_of_date)
    merged_rates: dict[tuple[str, str], dict[str, float]] = {}
    unresolved: set[str] = set()
    for (player, surface), bucket in rates.items():
        canonical = player
        if has_aliases:
            resolved = resolve_entity(player, alias_table)
            if resolved is None:
                unresolved.add(player)
            else:
                canonical = resolved
        merged = merged_rates.setdefault(
            (canonical, surface),
            {"spw_won": 0.0, "spw_pts": 0.0, "rpw_won": 0.0, "rpw_pts": 0.0, "matches": 0},
        )
        for key in ("spw_won", "spw_pts", "rpw_won", "rpw_pts", "matches"):
            merged[key] += bucket[key]

    priors: list[TennisPrior] = []
    for (player, surface), bucket in sorted(merged_rates.items()):
        if bucket["matches"] < min_matches or bucket["spw_pts"] <= 0 or bucket["rpw_pts"] <= 0:
            continue
        priors.append(
            TennisPrior(
                player=player,
                tour=tour.upper(),
                surface=surface,
                spw_pct=round(bucket["spw_won"] / bucket["spw_pts"], 4),
                rpw_pct=round(bucket["rpw_won"] / bucket["rpw_pts"], 4),
                n_matches=int(bucket["matches"]),
                as_of_date=as_of_date,
            )
        )
    if unresolved:
        logger.debug(
            "%d player name(s) kept verbatim (no alias entry): %s",
            len(unresolved),
            sorted(unresolved)[:20],
        )
    return priors, sorted(unresolved)


# ---------------------------------------------------------------------------
# Match Charting Project (point-by-point) — feeds the pressure-coefficient fit
# ---------------------------------------------------------------------------

_CHARTING_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/"
    "master/charting-{sex}-{kind}.csv"
)
_CHARTING_TTL_SECONDS = 30 * 24 * 3600  # MCP updates slowly; ~monthly refresh


class ChartingPointRow(BaseModel):
    """The subset of an MCP point row the pressure fit consumes.

    ``Pts`` is the pre-point game score from the server's perspective
    (e.g. ``30-40`` = break point against the server). ``Svr``/``PtWinner``
    are 1|2 player indices. Column presence is mandatory (drift fails loud).
    """

    match_id: str
    Set1: int
    Set2: int
    Gm1: int
    Gm2: int
    Pts: str
    Svr: int
    PtWinner: int | None

    _coerce_blanks = field_validator("PtWinner", mode="before")(_blank_to_none)


class ChartingMatchRow(BaseModel):
    """MCP match metadata: players + surface (+ best-of when present).

    ``best_of`` is crowdsourced free text in places ("3", "5", occasionally
    annotated) — value-lenient parsing keeps the leading digits or falls back
    to None rather than failing the job on one mistyped cell. Column *absence*
    still fails loud via the explicit key mapping in parse_charting_matches.
    """

    match_id: str
    player_1: str
    player_2: str
    surface: str | None
    best_of: int | None = None

    _coerce_blanks = field_validator("surface", mode="before")(_blank_to_none)

    @field_validator("best_of", mode="before")
    @classmethod
    def _lenient_best_of(cls, value: Any) -> Any:
        if value is None or isinstance(value, int):
            return value
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits in ("3", "5") else None


def fetch_charting_csv(
    sex: str,
    kind: str,
    *,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> str:
    """Fetch one MCP CSV (``kind`` like ``points-2020s`` or ``matches``)."""
    sex = sex.lower()
    if sex not in ("m", "w"):
        raise ValueError(f"sex must be 'm' or 'w', got {sex!r}")

    @cached_fetch("sackmann", ttl_seconds=_CHARTING_TTL_SECONDS, fmt="text", cache_root=cache_root)
    def _fetch() -> str:
        return _download_charting_csv(sex, kind, url_opener)

    return _fetch(cache_key=f"charting_{sex}_{kind}")


def _download_charting_csv(sex: str, kind: str, url_opener: Callable[..., Any]) -> str:
    assert_not_replay_mode("match charting project fetch")
    url = _CHARTING_URL_TEMPLATE.format(sex=sex, kind=kind)
    logger.info("fetching MCP CSV: %s", url)
    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS * 4) as resp:
        return resp.read().decode("utf-8", errors="replace")


_MCP_MAX_DEFECT_RATE = 0.01


def parse_charting_points(
    csv_text: str, *, session_path: str | None = None
) -> list[ChartingPointRow]:
    """Parse MCP point rows, dropping individually defective rows.

    The charting data is crowdsourced: isolated rows carry typos in the
    integer columns. Those rows are dropped and counted rather than failing
    the job — but a defect rate above 1% (or a renamed column, which defects
    every row) still raises ``SourceSchemaDriftError``, preserving the
    fail-loud drift contract.
    """
    from omega.integrations._etl import SourceSchemaDriftError

    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[ChartingPointRow] = []
    dropped = 0
    total = 0
    for r in reader:
        total += 1
        try:
            rows.append(
                ChartingPointRow(
                    match_id=r.get("match_id"),
                    Set1=r.get("Set1"),
                    Set2=r.get("Set2"),
                    Gm1=r.get("Gm1"),
                    Gm2=r.get("Gm2"),
                    Pts=r.get("Pts"),
                    Svr=r.get("Svr"),
                    PtWinner=r.get("PtWinner"),
                )
            )
        except Exception:  # noqa: BLE001 - counted; rate-gated below
            dropped += 1
    if total and dropped / total > _MCP_MAX_DEFECT_RATE:
        raise SourceSchemaDriftError(
            "sackmann_mcp",
            f"{dropped}/{total} point rows failed validation — column drift, not isolated typos",
        )
    if dropped:
        logger.info("dropped %d/%d defective MCP point rows", dropped, total)
    return rows


def parse_charting_matches(
    csv_text: str, *, session_path: str | None = None
) -> list[ChartingMatchRow]:
    reader = csv.DictReader(io.StringIO(csv_text))
    raw = [
        {
            "match_id": r.get("match_id"),
            "player_1": r.get("Player 1", r.get("player_1")),
            "player_2": r.get("Player 2", r.get("player_2")),
            "surface": r.get("Surface", r.get("surface")),
            "best_of": r.get("Best of", r.get("best_of")),
        }
        for r in reader
    ]
    return validate_records(raw, ChartingMatchRow, source="sackmann_mcp", session_path=session_path)


def load_tennis_priors(
    tour: str,
    years: list[int],
    *,
    as_of_date: str,
    local_root: str | Path | None = None,
    cache_root: str | None = None,
    alias_table: dict[str, Any] | None = None,
    session_path: str | None = None,
    min_matches: int = 3,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[list[TennisPrior], list[str]]:
    """Read (local-first) + parse + aggregate one tour's seasons into priors."""
    rows: list[SackmannMatchRow] = []
    for year in years:
        csv_text = fetch_matches_csv(
            tour, year, local_root=local_root, cache_root=cache_root, url_opener=url_opener
        )
        rows.extend(parse_matches(csv_text, session_path=session_path))
    return build_tennis_priors(
        rows,
        tour=tour,
        as_of_date=as_of_date,
        alias_table=alias_table,
        min_matches=min_matches,
    )
