"""NFL edge consumer: Wong-teaser legs.

Phase 7 Milestone 4. Prices teaser legs supplied in the normalized
``OddsInput.markets`` list (``market_type="teaser"``) against the backend's
discrete margin/total pmfs. Spread legs (selection Home/Away) read
``margin_counts``; total legs (Over/Under) read ``total_counts``. Each leg's exact
cover EV is bridged into the binary edge framework via the equivalent win
probability, mirroring the soccer consumer. Registered against the
``american_football`` archetype on the EDGE_CONSUMERS registry.

Teaser legs are quoted via the normalized ``markets`` list (the preferred path
for new markets); there are no legacy flat OddsInput fields for teasers. The
per-leg EVs combine multiplicatively into a full teaser-card price, which is a
staking-layer concern handled outside this consumer.
"""

from __future__ import annotations

from typing import Any

from omega.core.contracts.market_quotes import quote_matches_selection
from omega.core.contracts.schemas import EdgeDetail, GameAnalysisRequest
from omega.core.edge.consumers import (
    BuildEdgeFn,
    CalibrateFn,
    register_edge_consumer,
)
from omega.core.edge.nfl_teasers import (
    evaluate_teaser_spread_leg,
    evaluate_teaser_total_leg,
)


class AmericanFootballEdgeConsumer:
    """Prices NFL Wong-teaser legs from the discrete margin/total pmfs."""

    sport = "american_football"

    def consume(
        self,
        sim_result: dict[str, Any],
        request: GameAnalysisRequest,
        bankroll: float,
        calibrate_fn: CalibrateFn,
        build_edge_fn: BuildEdgeFn,
    ) -> list[EdgeDetail]:
        odds = request.odds
        if not odds or not odds.markets:
            return []
        edges: list[EdgeDetail] = []
        gc = request.game_context
        margin_counts = sim_result.get("margin_counts")
        total_counts = sim_result.get("total_counts")

        for quote in odds.markets:
            if quote.market_type != "teaser":
                continue
            line = quote.line
            price = quote.price
            if line is None or price is None:
                continue

            # Classify the leg by selection: Home/Away → spread leg on the margin
            # pmf; Over/Under → total leg on the total pmf.
            if quote_matches_selection(quote, request.home_team, "Home") and margin_counts:
                side, label = "home", f"{request.home_team} {line:+g} (teaser)"
                evaluation = evaluate_teaser_spread_leg(margin_counts, line, "home")
            elif quote_matches_selection(quote, request.away_team, "Away") and margin_counts:
                side, label = "away", f"{request.away_team} {line:+g} (teaser)"
                evaluation = evaluate_teaser_spread_leg(margin_counts, line, "away")
            elif quote_matches_selection(quote, "Over") and total_counts:
                side, label = "over", f"Teaser Over {line:g}"
                evaluation = evaluate_teaser_total_leg(total_counts, line, "over")
            elif quote_matches_selection(quote, "Under") and total_counts:
                side, label = "under", f"Teaser Under {line:g}"
                evaluation = evaluate_teaser_total_leg(total_counts, line, "under")
            else:
                continue

            q_equiv = evaluation.equivalent_win_prob(price)
            if q_equiv <= 0:
                continue
            cal_q, audit = calibrate_fn(
                q_equiv,
                league=request.league,
                context_hints=gc,
                plane="game",
                market="teaser",
            )
            edges.append(
                build_edge_fn(
                    side,
                    label,
                    q_equiv,
                    cal_q,
                    price,
                    bankroll,
                    request.n_iterations,
                    calibration_audit=audit,
                    market="teaser",
                    line=line,
                ).model_copy(update={"confidence_tier": "Pass", "recommended_units": 0.0})
            )

        return edges


register_edge_consumer(AmericanFootballEdgeConsumer.sport, AmericanFootballEdgeConsumer())
