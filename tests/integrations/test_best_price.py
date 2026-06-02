"""Tests for best_price_quotes — advisory best-price-across-books selection."""

from __future__ import annotations

from omega.integrations.odds_resolver import best_price_quotes


def _q(market_type, selection, price, *, line=None, bookmaker="bookA",
       last_update="2026-05-17T20:00:00Z"):
    """Minimal normalized-quote dict, matching normalize_book_odds output shape."""
    return {
        "market_type": market_type,
        "selection": selection,
        "price": price,
        "line": line,
        "bookmaker": bookmaker,
        "last_update": last_update,
    }


class TestBestPriceQuotes:
    def test_picks_higher_underdog_payout(self):
        quotes = [
            _q("moneyline", "Boston Celtics", 120, bookmaker="betmgm"),
            _q("moneyline", "Boston Celtics", 145, bookmaker="draftkings"),
            _q("moneyline", "Boston Celtics", 130, bookmaker="fanduel"),
        ]
        best = best_price_quotes(quotes)
        assert len(best) == 1
        assert best[0]["bookmaker"] == "draftkings"
        assert best[0]["price"] == 145

    def test_picks_less_negative_favorite_across_sign_boundary(self):
        # -110 pays the bettor more than -130; +100 more than either.
        quotes = [
            _q("moneyline", "Lakers", -130, bookmaker="betmgm"),
            _q("moneyline", "Lakers", -110, bookmaker="draftkings"),
            _q("moneyline", "Lakers", 100, bookmaker="caesars"),
        ]
        best = best_price_quotes(quotes)
        assert best[0]["bookmaker"] == "caesars"
        assert best[0]["price"] == 100

    def test_separate_selections_and_lines_grouped_independently(self):
        quotes = [
            _q("total", "Over", -115, line=210.5, bookmaker="betmgm"),
            _q("total", "Over", -105, line=210.5, bookmaker="draftkings"),
            _q("total", "Under", -110, line=210.5, bookmaker="betmgm"),
            _q("total", "Over", -120, line=211.5, bookmaker="fanduel"),
        ]
        best = best_price_quotes(quotes)
        # Three groups: Over@210.5, Under@210.5, Over@211.5.
        groups = {(b["selection"], b["line"]): b["bookmaker"] for b in best}
        assert groups[("Over", 210.5)] == "draftkings"  # -105 beats -115
        assert groups[("Under", 210.5)] == "betmgm"
        assert groups[("Over", 211.5)] == "fanduel"

    def test_does_not_fabricate_cross_book_line(self):
        # Best over (book A) and best under (book B) are reported as two rows
        # from their own books — never merged into one synthetic ticket.
        quotes = [
            _q("total", "Over", 105, line=44.5, bookmaker="bookA"),
            _q("total", "Under", 102, line=44.5, bookmaker="bookB"),
        ]
        best = best_price_quotes(quotes)
        by_sel = {b["selection"]: b["bookmaker"] for b in best}
        assert by_sel == {"Over": "bookA", "Under": "bookB"}

    def test_tie_breaks_on_freshness_then_book_name(self):
        quotes = [
            _q("moneyline", "X", 150, bookmaker="zzz", last_update="2026-05-17T20:00:00Z"),
            _q("moneyline", "X", 150, bookmaker="aaa", last_update="2026-05-17T21:00:00Z"),
            _q("moneyline", "X", 150, bookmaker="mmm", last_update="2026-05-17T20:00:00Z"),
        ]
        best = best_price_quotes(quotes)
        # Freshest update wins outright.
        assert best[0]["bookmaker"] == "aaa"

    def test_attaches_decimal_payout_and_skips_unpriced(self):
        quotes = [
            _q("moneyline", "X", 100, bookmaker="bookA"),
            _q("moneyline", "Y", None, bookmaker="bookB"),
        ]
        best = best_price_quotes(quotes)
        assert len(best) == 1
        assert best[0]["selection"] == "X"
        assert best[0]["decimal_payout"] == 2.0

    def test_empty_input(self):
        assert best_price_quotes([]) == []
