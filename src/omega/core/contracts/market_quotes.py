"""Normalized market-quote resolution shared across the contracts/edge layers.

``analyze_game`` (contracts) and the per-sport edge consumers (``omega/core/edge``)
both need to read a quoted line/price out of an ``OddsInput``'s normalized
``markets`` list. Keeping that lookup here — depending only on the schemas — lets
edge consumers resolve their own markets without importing ``service`` (which
would create an import cycle, since ``service`` imports the consumer registry).
"""

from __future__ import annotations

from omega.core.contracts.schemas import MarketQuote, OddsInput


def quote_matches_selection(quote: MarketQuote, *labels: str) -> bool:
    selection = quote.selection.strip().casefold()
    return any(
        selection == (label_text := label.strip().casefold())
        or selection.startswith(f"{label_text} ")
        for label in labels
        if label
    )


def market_quote(
    odds: OddsInput,
    market_type: str,
    *labels: str,
) -> MarketQuote | None:
    for quote in odds.markets or []:
        if quote.market_type != market_type:
            continue
        if quote_matches_selection(quote, *labels):
            return quote
    return None
