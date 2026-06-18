"""omega-historical-live-parity — covariate-shift gate for historical-fit profiles.

A profile fit on historical-replay traces is only valid for live use if the
historical *inputs* resemble live inputs. This compares the two populations and
returns a three-state verdict:

* PASS         — live_n >= min_live_n AND no gated PSI > 0.25 AND mean gated PSI < 0.15
* INCONCLUSIVE — live_n < min_live_n (too little live data to measure shift)
* FAIL         — a gated distribution shifted materially

Gated (decision-critical) distributions bias the probability map: raw win-prob,
favorite/underdog mix, context_source mix. Advisory distributions (book/provider)
differ by nature and are reported but never gate. A historical-only candidate may
be promoted only on PASS; INCONCLUSIVE/FAIL keep it shadow-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from omega.historical.distribution import category_psi, js_divergence, psi
from omega.trace.store import TraceStore

logger = logging.getLogger("omega.ops.historical_live_parity")

GATED_PSI_HARD = 0.25
GATED_PSI_MEAN = 0.15
DEFAULT_MIN_LIVE_N = 200


def _home_prob(trace: dict) -> float | None:
    preds = trace.get("predictions") or {}
    p = preds.get("home_win_prob")
    if p is None:
        return None
    return p / 100.0 if p > 1 else float(p)


def _context_source(trace: dict) -> str:
    return str((trace.get("trace_quality") or {}).get("context_source") or "unknown")


def _book(trace: dict) -> str:
    odds = trace.get("odds_snapshot") or {}
    return str(odds.get("book") or odds.get("bookmaker") or "unknown")


def evaluate_parity(
    historical: list[dict], live: list[dict], *, min_live_n: int = DEFAULT_MIN_LIVE_N
) -> dict:
    """Compute the three-state parity verdict between historical and live traces."""
    hist_probs = [p for p in (_home_prob(t) for t in historical) if p is not None]
    live_probs = [p for p in (_home_prob(t) for t in live) if p is not None]

    gated: dict[str, float] = {}
    advisory: dict[str, float] = {}
    if hist_probs and live_probs:
        gated["raw_prob"] = psi(live_probs, hist_probs)
        gated["raw_prob_js"] = js_divergence(live_probs, hist_probs)
        gated["favorite_rate"] = category_psi(
            ["fav" if p > 0.5 else "dog" for p in live_probs],
            ["fav" if p > 0.5 else "dog" for p in hist_probs],
        )
    gated["context_source"] = category_psi(
        [_context_source(t) for t in live], [_context_source(t) for t in historical]
    )
    # Advisory: book/provider mix differs between sources by nature — reported, not gated.
    advisory["book"] = category_psi([_book(t) for t in live], [_book(t) for t in historical])

    # The JS cross-check is reported, not gated (PSI is the gating metric).
    gated_for_gate = {k: v for k, v in gated.items() if not k.endswith("_js")}
    gated_vals = list(gated_for_gate.values())
    live_n = len(live_probs)

    if live_n < min_live_n:
        state = "INCONCLUSIVE"
    elif any(v > GATED_PSI_HARD for v in gated_vals) or (
        gated_vals and sum(gated_vals) / len(gated_vals) >= GATED_PSI_MEAN
    ):
        state = "FAIL"
    else:
        state = "PASS"

    return {
        "schema_version": 1,
        "state": state,
        "historical_n": len(hist_probs),
        "live_n": live_n,
        "min_live_n": min_live_n,
        "gated": gated,
        "advisory": advisory,
        "promotable_historical_only": state == "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Historical-vs-live distribution parity gate (covariate-shift)."
    )
    parser.add_argument("--league", required=True)
    parser.add_argument("--market", default="game", help="Reporting label only (game/prop/draw).")
    parser.add_argument("--historical-db", required=True)
    parser.add_argument("--live-db", default=None, help="Live trace DB (default: production).")
    parser.add_argument("--min-live-n", type=int, default=DEFAULT_MIN_LIVE_N)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    hstore = TraceStore(db_path=args.historical_db)
    historical = hstore.query_traces(
        league=args.league, execution_mode="historical_replay",
        has_outcome=True, calibration_eligible_only=True, limit=1_000_000,
    )
    hstore.close()

    lstore = TraceStore(db_path=args.live_db)
    live = lstore.get_graded_traces(league=args.league, limit=1_000_000)
    # Live = non-replay graded traces only.
    live = [t for t in live if t.get("execution_mode") != "historical_replay"]
    lstore.close()

    report = evaluate_parity(historical, live, min_live_n=args.min_live_n)
    report["league"] = args.league
    report["market"] = args.market
    print(json.dumps(report, indent=2))
    logger.info("parity verdict: %s (historical_n=%d live_n=%d)",
                report["state"], report["historical_n"], report["live_n"])

    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[report["state"]]


if __name__ == "__main__":
    sys.exit(main())
