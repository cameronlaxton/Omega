"""Patch 3: closing-line spread/total grading must match the bet's EXACT point.

A -3 spread bet must not be graded against a -5.5 close; an Over 47.5 must not be
graded against an Over 45.5. This mirrors `_match_prop_outcome`, which already
enforces exact `book.point` equality. Moneyline (no point) is unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from omega.integrations.odds_api import BookOdds  # noqa: E402
from omega.ops.fetch_closing_lines import _match_outcome  # noqa: E402

HOME, AWAY = "Lakers", "Celtics"


def _book(market: str, selection: str, point: float | None, price: float = -110.0) -> BookOdds:
    return BookOdds(
        bookmaker="draftkings",
        market=market,
        selection=selection,
        price=price,
        point=point,
        last_update="2026-06-08T00:00:00Z",
    )


def test_spread_rejects_wrong_point():
    books = [_book("spreads", HOME, -5.5), _book("spreads", AWAY, 5.5)]
    assert (
        _match_outcome("spread", "home_spread_-3", HOME, AWAY, books, None, line_taken=-3.0)
        is None
    )


def test_spread_matches_exact_point():
    books = [_book("spreads", HOME, -3.0), _book("spreads", AWAY, 3.0)]
    m = _match_outcome("spread", "home_spread_-3", HOME, AWAY, books, None, line_taken=-3.0)
    assert m is not None and m.point == -3.0 and m.selection == HOME


def test_total_rejects_wrong_point():
    books = [_book("totals", "Over", 45.5), _book("totals", "Under", 45.5)]
    assert (
        _match_outcome("total", "over_total_47.5", HOME, AWAY, books, None, line_taken=47.5)
        is None
    )


def test_total_matches_exact_point():
    books = [_book("totals", "Over", 47.5), _book("totals", "Under", 47.5)]
    m = _match_outcome("total", "over_total_47.5", HOME, AWAY, books, None, line_taken=47.5)
    assert m is not None and m.point == 47.5 and m.selection == "Over"


def test_none_line_falls_back_to_side_only():
    # Legacy bet without a stored line still matches by side (no point filter).
    books = [_book("spreads", HOME, -5.5), _book("spreads", AWAY, 5.5)]
    m = _match_outcome("spread", "home_spread_-3", HOME, AWAY, books, None, line_taken=None)
    assert m is not None and m.selection == HOME


def test_moneyline_unaffected_by_point_check():
    books = [_book("h2h", HOME, None, price=-120.0), _book("h2h", AWAY, None, price=110.0)]
    m = _match_outcome("moneyline", "home_moneyline", HOME, AWAY, books, None, line_taken=None)
    assert m is not None and m.selection == HOME
