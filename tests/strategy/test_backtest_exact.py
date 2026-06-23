"""Backtest-level exact-evaluation gate.

The exact substrate is only useful if it produces the *same betting decisions* as
the ground-truth (high-n MC) it replaces — that is the acceptance metric in the
exact-eval plan (decision agreement, not probability MAE, because the edge filter
harvests the positive tail of any error). These tests run the full
``BacktestEngine`` and check:

* exact mode is deterministic (no seed dependence in the decisions), and
* exact decisions match a high-iteration MC run on the same artifacts.
"""

from __future__ import annotations

from omega.strategy.backtest.engine import BacktestEngine, HistoricalGame
from omega.strategy.models import StrategyEntry


def _soccer_slate() -> list[HistoricalGame]:
    """A small EPL slate (soccer archetype → exact-eligible) with 3-way odds."""
    specs = [
        ("Arsenal", "Chelsea", 1.7, 1.0, 1.2, 1.3, -140, 360, 280, (2, 1)),
        ("Liverpool", "Everton", 1.9, 0.9, 1.0, 1.4, -210, 520, 330, (1, 1)),
        ("ManCity", "Burnley", 2.2, 0.8, 0.9, 1.6, -320, 700, 420, (3, 0)),
        ("Spurs", "Wolves", 1.5, 1.2, 1.3, 1.2, -120, 300, 250, (0, 1)),
        ("Brighton", "Fulham", 1.4, 1.3, 1.25, 1.25, 110, 240, 230, (1, 2)),
        ("Newcastle", "Brentford", 1.6, 1.1, 1.15, 1.3, -130, 330, 260, (2, 2)),
    ]
    games = []
    for home, away, h_xg, h_xga, a_xg, a_xga, ml_h, ml_a, ml_d, (hs, as_) in specs:
        games.append(
            HistoricalGame(
                {
                    "home_team": home,
                    "away_team": away,
                    "league": "EPL",
                    "home_context": {
                        "off_rating": h_xg,
                        "def_rating": h_xga,
                        "xg_for": h_xg,
                        "xg_against": h_xga,
                    },
                    "away_context": {
                        "off_rating": a_xg,
                        "def_rating": a_xga,
                        "xg_for": a_xg,
                        "xg_against": a_xga,
                    },
                    "odds": {
                        "moneyline_home": ml_h,
                        "moneyline_away": ml_a,
                        "moneyline_draw": ml_d,
                        "over_under": 2.5,
                        "total_over_price": -110,
                        "total_under_price": -110,
                    },
                    "outcome": {"home_score": hs, "away_score": as_},
                }
            )
        )
    return games


def _strategy() -> StrategyEntry:
    return StrategyEntry(
        strategy_id="exact-test",
        name="exact-test",
        leagues=["EPL"],
        edge_threshold=0.02,
        confidence_tiers=["A", "B", "C"],
    )


def test_exact_backtest_is_deterministic():
    games = _soccer_slate()
    strat = _strategy()
    a = BacktestEngine(n_iterations=1000, exact_eval=True).run(strat, games)
    b = BacktestEngine(n_iterations=1000, exact_eval=True).run(strat, games)
    # Decisions and P&L are identical run-to-run (no sampling).
    assert a.total_bets_placed == b.total_bets_placed
    assert a.net_units == b.net_units
    assert a.results_by_market == b.results_by_market


def test_exact_decisions_match_high_n_mc():
    """Exact decisions agree with a 200k-iteration MC run on the same slate."""
    games = _soccer_slate()
    strat = _strategy()

    exact = BacktestEngine(n_iterations=1000, exact_eval=True).run(strat, games)
    mc = BacktestEngine(n_iterations=200_000, exact_eval=False).run(strat, games)

    # Same number of bets and same per-market bet counts (decision agreement).
    assert exact.total_bets_placed == mc.total_bets_placed
    exact_counts = {m: v["bets"] for m, v in exact.results_by_market.items()}
    mc_counts = {m: v["bets"] for m, v in mc.results_by_market.items()}
    assert exact_counts == mc_counts

    # Same win/loss outcome on the placed bets (grading is deterministic given the
    # same selections). Net units agree to the cent — the only residual is the MC
    # path's sampling noise in the Kelly stake, which is exactly what exact removes.
    assert exact.win_count == mc.win_count
    assert abs(exact.net_units - mc.net_units) < 0.05


def _nba_slate() -> list[HistoricalGame]:
    """NBA slate (basketball = Normal archetype, exact via clipped-normal) with
    moneyline, total, and spread markets."""
    specs = [
        ("Celtics", "Pacers", 118.0, 108.0, 112.0, 112.0, -180, 155, -4.5, 224.5, (118, 109)),
        ("Lakers", "Magic", 115.0, 110.0, 110.0, 113.0, -150, 130, -3.5, 222.5, (101, 108)),
        ("Nuggets", "Suns", 117.0, 109.0, 114.0, 110.0, -135, 115, -2.5, 228.5, (120, 117)),
        ("Bucks", "Pistons", 119.0, 107.0, 109.0, 114.0, -260, 215, -7.5, 230.5, (125, 110)),
        ("Heat", "Hawks", 113.0, 111.0, 114.0, 110.0, 105, -125, 1.5, 226.5, (104, 112)),
        ("Wolves", "Jazz", 116.0, 108.0, 111.0, 113.0, -170, 145, -4.5, 219.5, (110, 105)),
    ]
    games = []
    for home, away, h_off, h_def, a_off, a_def, ml_h, ml_a, spread, total, (hs, as_) in specs:
        games.append(
            HistoricalGame(
                {
                    "home_team": home,
                    "away_team": away,
                    "league": "NBA",
                    "home_context": {"off_rating": h_off, "def_rating": h_def, "pace": 100.0},
                    "away_context": {"off_rating": a_off, "def_rating": a_def, "pace": 99.0},
                    "odds": {
                        "moneyline_home": ml_h,
                        "moneyline_away": ml_a,
                        "spread_home": spread,
                        "spread_home_price": -110,
                        "spread_away_price": -110,
                        "over_under": total,
                        "total_over_price": -110,
                        "total_under_price": -110,
                    },
                    "outcome": {"home_score": hs, "away_score": as_},
                }
            )
        )
    return games


def test_nba_exact_decisions_match_high_n_mc():
    """Normal-archetype decision-agreement gate: NBA exact vs 200k MC."""
    games = _nba_slate()
    strat = StrategyEntry(
        strategy_id="nba",
        name="nba",
        leagues=["NBA"],
        edge_threshold=0.02,
        confidence_tiers=["A", "B", "C"],
    )
    exact = BacktestEngine(n_iterations=1000, exact_eval=True).run(strat, games)
    mc = BacktestEngine(n_iterations=200_000, exact_eval=False).run(strat, games)

    assert exact.total_bets_placed == mc.total_bets_placed
    exact_counts = {m: v["bets"] for m, v in exact.results_by_market.items()}
    mc_counts = {m: v["bets"] for m, v in mc.results_by_market.items()}
    assert exact_counts == mc_counts
    assert exact.win_count == mc.win_count
    # Net units agree to within ~2% — the only gap is residual MC noise in the
    # Kelly stakes (a tier flip would move this far more). This is the upward-
    # biased stake noise exact eval removes.
    assert abs(exact.net_units - mc.net_units) < 0.03 * abs(mc.net_units) + 0.05
