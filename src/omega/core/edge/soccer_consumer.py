"""Soccer edge consumer: Asian handicap + first-half total.

The soccer bivariate-Poisson backend emits empirical pmfs (``margin_counts`` /
``fh_total_counts``); this consumer evaluates the quoted lines — quarter-ball
split, push and half-stake semantics included — and bridges the exact EV back
into the binary edge framework via the equivalent win probability. It is the
first ``EdgeConsumer`` registered against the ``soccer`` archetype; the logic
moved verbatim out of ``service.analyze_game``'s inline derivative blocks.
"""

from __future__ import annotations

from typing import Any

from omega.core.betting.odds import implied_probability
from omega.core.contracts.market_quotes import market_quote
from omega.core.contracts.schemas import EdgeDetail, GameAnalysisRequest, OddsInput
from omega.core.edge.consumers import (
    BuildEdgeFn,
    CalibrateFn,
    register_edge_consumer,
)
from omega.core.edge.soccer_derivatives import (
    evaluate_asian_handicap,
    evaluate_total,
)


def _resolve_game_asian_handicap_market(
    odds: OddsInput,
    home_team: str,
    away_team: str,
) -> tuple[float | None, float | None, float | None]:
    """Resolve home Asian-handicap line and both side prices."""
    home_q = market_quote(odds, "asian_handicap", home_team, "Home")
    away_q = market_quote(odds, "asian_handicap", away_team, "Away")
    line = (
        home_q.line
        if home_q is not None and home_q.line is not None
        else (
            -away_q.line
            if away_q is not None and away_q.line is not None
            else odds.asian_handicap_home
        )
    )
    if line is None:
        return None, None, None
    home_price = home_q.price if home_q is not None else odds.ah_home_price
    away_price = away_q.price if away_q is not None else odds.ah_away_price
    return line, home_price, away_price


def _resolve_game_first_half_total_market(
    odds: OddsInput,
) -> tuple[float | None, float | None, float | None]:
    """Resolve first-half total line and over/under prices."""
    over_q = market_quote(odds, "first_half_total", "Over")
    under_q = market_quote(odds, "first_half_total", "Under")
    line = (
        over_q.line
        if over_q is not None and over_q.line is not None
        else (
            under_q.line
            if under_q is not None and under_q.line is not None
            else odds.first_half_total
        )
    )
    if line is None:
        return None, None, None
    over_price = over_q.price if over_q is not None else odds.fh_over_price
    under_price = under_q.price if under_q is not None else odds.fh_under_price
    return line, over_price, under_price


class SoccerEdgeConsumer:
    """Prices soccer Asian-handicap and first-half-total derivative markets."""

    sport = "soccer"

    def consume(
        self,
        sim_result: dict[str, Any],
        request: GameAnalysisRequest,
        bankroll: float,
        calibrate_fn: CalibrateFn,
        build_edge_fn: BuildEdgeFn,
    ) -> list[EdgeDetail]:
        odds = request.odds
        if not odds:
            return []
        edges: list[EdgeDetail] = []
        gc = request.game_context

        # Asian handicap. The backend emits the margin pmf (margin_counts); the
        # edge module evaluates the quoted line and bridges the exact EV back
        # into the binary edge framework via the equivalent win probability.
        ah_line, ah_home_price, ah_away_price = _resolve_game_asian_handicap_market(
            odds, request.home_team, request.away_team
        )
        margin_counts = sim_result.get("margin_counts")
        if ah_line is not None and margin_counts:
            for ah_side, ah_price, ah_label, ah_signed_line in (
                ("home", ah_home_price, request.home_team, ah_line),
                ("away", ah_away_price, request.away_team, -ah_line),
            ):
                if ah_price is None:
                    continue
                evaluation = evaluate_asian_handicap(margin_counts, ah_line, ah_side)
                q_equiv = evaluation.equivalent_win_prob(ah_price)
                if q_equiv <= 0:
                    continue
                cal_q, ah_audit = calibrate_fn(
                    q_equiv,
                    league=request.league,
                    context_hints=gc,
                    plane="game",
                    market="asian_handicap",
                    market_prob=implied_probability(ah_price),
                )
                edges.append(
                    build_edge_fn(
                        ah_side,
                        f"{ah_label} {ah_signed_line:+g} (AH)",
                        q_equiv,
                        cal_q,
                        ah_price,
                        bankroll,
                        request.n_iterations,
                        calibration_audit=ah_audit,
                        market="asian_handicap",
                        line=ah_signed_line,
                    )
                )

        # First-half total against the thinned first-half pmf (fh_total_counts).
        fh_line, fh_over_price, fh_under_price = _resolve_game_first_half_total_market(odds)
        fh_counts = sim_result.get("fh_total_counts")
        if fh_line is not None and fh_counts:
            for fh_side, fh_price in (
                ("over", fh_over_price),
                ("under", fh_under_price),
            ):
                if fh_price is None:
                    continue
                evaluation = evaluate_total(fh_counts, fh_line, fh_side)
                q_equiv = evaluation.equivalent_win_prob(fh_price)
                if q_equiv <= 0:
                    continue
                cal_q, fh_audit = calibrate_fn(
                    q_equiv,
                    league=request.league,
                    context_hints=gc,
                    plane="game",
                    market="first_half_total",
                    market_prob=implied_probability(fh_price),
                )
                edges.append(
                    build_edge_fn(
                        fh_side,
                        f"1H {fh_side.capitalize()} {fh_line:g}",
                        q_equiv,
                        cal_q,
                        fh_price,
                        bankroll,
                        request.n_iterations,
                        calibration_audit=fh_audit,
                        market="first_half_total",
                        line=fh_line,
                    )
                )

        return edges


register_edge_consumer(SoccerEdgeConsumer.sport, SoccerEdgeConsumer())
