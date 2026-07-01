"""Exact per-signal marginal value via counterfactual re-simulation (issue #28 WS1).

The *slow confirmer* behind the fast CLV / model-vs-market measure: for each
evidence-bearing graded trace, re-run the forecast WITHOUT each applied signal and
compare it to the forecast WITH it. Positive deltas mean the signal improved the
forecast (Brier / log-loss fell when it was applied).

Why re-simulation rather than dividing out ``final_applied_factor``: evidence
factors act on simulation **inputs** (team mean/std), not on the output
probability, and no pre-evidence baseline probability is persisted — so the only
*faithful* counterfactual is to run the engine again with the signal removed. That
is exact (no approximation) and bounded to the applied signals on graded traces.

The historical-replay lab itself runs evidence-free, so this operates over
**live, evidence-bearing** graded traces. The counterfactual engine call is
*injected* (``CounterfactualProb``) so the collection + aggregation stay pure and
unit-testable, and so this module carries no hard dependency on the analysis
service. :func:`engine_counterfactual_prob` is the production binding.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from omega.historical.contracts import MarginalValueBlock
from omega.historical.metrics import marginal_value

# Returns the counterfactual home-win probability (0..1) for a trace with one
# ``signal_type`` removed, or None when it cannot be computed faithfully (the
# signal is then honestly omitted — it never contributes an approximated pair).
CounterfactualProb = Callable[[dict[str, Any], str], float | None]


def _home_win_prob(trace: dict[str, Any]) -> float | None:
    p = (trace.get("predictions") or {}).get("home_win_prob")
    if p is None:
        return None
    return p / 100.0 if p > 1 else float(p)


def _applied_signal_types(trace: dict[str, Any]) -> list[str]:
    """Distinct signal_types the engine actually APPLIED on this trace (order-stable)."""
    seen: list[str] = []
    for app in trace.get("evidence_application") or []:
        st = app.get("signal_type")
        if app.get("applied") and st and st not in seen:
            seen.append(st)
    return seen


def collect_signal_pairs(
    traces: Iterable[dict[str, Any]], counterfactual_prob: CounterfactualProb
) -> dict[str, tuple[list[float], list[float], list[int]]]:
    """Group ``(pred_with, pred_without, outcome)`` per applied signal_type.

    Only graded game traces (``_outcome.result`` in home_win/away_win/draw) with a
    home-win prediction contribute. A signal whose counterfactual cannot be computed
    is skipped (no approximated pair is ever invented).
    """
    pairs: dict[str, tuple[list[float], list[float], list[int]]] = defaultdict(
        lambda: ([], [], [])
    )
    for t in traces:
        res = (t.get("_outcome") or {}).get("result")
        if res not in ("home_win", "away_win", "draw"):
            continue
        with_p = _home_win_prob(t)
        if with_p is None:
            continue
        y = 1 if res == "home_win" else 0
        for sig in _applied_signal_types(t):
            without_p = counterfactual_prob(t, sig)
            if without_p is None:
                continue
            w, wo, ys = pairs[sig]
            w.append(with_p)
            wo.append(without_p)
            ys.append(y)
    return pairs


def aggregate_marginal_values(
    traces: Iterable[dict[str, Any]], counterfactual_prob: CounterfactualProb
) -> list[MarginalValueBlock]:
    """Per-signal :class:`MarginalValueBlock` list, most-helpful signal first.

    Empty when no trace carries applied evidence — the honest result for the
    evidence-free historical-replay corpus. Each block is produced by the single
    ``metrics.marginal_value`` path (no metric is redefined here).
    """
    pairs = collect_signal_pairs(traces, counterfactual_prob)
    blocks = [
        marginal_value(sig, with_p, without_p, outs)
        for sig, (with_p, without_p, outs) in pairs.items()
        if with_p
    ]
    return sorted(blocks, key=lambda b: (b.brier_delta if b.brier_delta is not None else 0.0), reverse=True)


def engine_counterfactual_prob(trace: dict[str, Any], signal_type: str) -> float | None:
    """Re-run ``analyze()`` for this trace with ``signal_type`` removed → P(home).

    Reconstructs the original :class:`GameAnalysisRequest` from the trace's
    ``input_snapshot`` (seed included, so the re-sim is deterministic), drops every
    evidence signal of ``signal_type``, and returns the counterfactual home-win
    probability in [0, 1]. Defensive by design: any reconstruction/execution problem
    returns None so a lab/analysis run never breaks and the signal is honestly
    omitted rather than approximated.
    """
    try:
        from omega.core.contracts.schemas import GameAnalysisRequest
        from omega.core.contracts.service import analyze

        snap = trace.get("input_snapshot") or {}
        home_team = snap.get("home_team")
        away_team = snap.get("away_team")
        league = snap.get("league") or trace.get("league")
        if not (home_team and away_team and league):
            return None
        evidence = [
            e
            for e in (snap.get("evidence") or [])
            if (e or {}).get("signal_type") != signal_type
        ]
        req = GameAnalysisRequest(
            home_team=home_team,
            away_team=away_team,
            league=league,
            odds=snap.get("odds"),
            n_iterations=snap.get("n_iterations") or 1000,
            simulation_backend=snap.get("simulation_backend") or "fast_score",
            home_context=snap.get("home_context"),
            away_context=snap.get("away_context"),
            game_context=snap.get("game_context"),
            prior_payload=snap.get("prior_payload"),
            seed=snap.get("seed"),
            evidence=evidence,
        )
        out = analyze(req)
        sim = ((out or {}).get("result") or {}).get("simulation") or {}
        p = sim.get("home_win_prob")
        if p is None:
            return None
        return p / 100.0 if p > 1 else float(p)
    except Exception:
        return None
