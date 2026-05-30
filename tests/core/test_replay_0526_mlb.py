"""Replay harness for the 0526-auto MLB traces.

The sess-20260526-auto session ran against the pre-fix engine and produced
traces with BUG-SIM-2 (draw_prob ~13-14%), BUG-SPREAD-1 (spread true_prob ==
ML true_prob), and BUG-TOTALS-1 (no total edges). Commit 95c8d34 remediated
all three. This test re-runs the original input_snapshot of each corrupt-era
MLB game trace through the patched engine and asserts:

    1. simulation.draw_prob == 0.0
    2. exactly one Over and one Under total edge row
    3. spread true_prob differs from moneyline true_prob

No network calls — the input_snapshot block is sufficient to drive
analyze_game() deterministically with the original seed.

References:
  docs/session_bugs_20260526.md
  commit 95c8d34 ("Remediate 20260525 Omega bugs")
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import analyze_game


# The five game-kind MLB traces from sess-20260526-auto. The two prop traces
# (d8ffc45c, dd8a5763) are excluded — they exercise the player-prop path,
# which the 0526 bugs don't touch.
_TRACE_PREFIXES = [
    "sandbox-bd37a280",
    "sandbox-c27100bd",
    "sandbox-e47ef200",
    "sandbox-e4b14d5a",
    "sandbox-eea964a2",
]


def _trace_path(prefix: str) -> Path:
    # Curated replay fixtures live under fixtures/ (PROJECT_STATE source-of-truth
    # rule: committed test examples belong in fixtures/, not the gitignored
    # inbox/traces/processed/ runtime path).
    matches = list(
        (Path(__file__).resolve().parents[2] / "fixtures" / "replay" / "0526_mlb").glob(
            f"{prefix}-*.json"
        )
    )
    if not matches:
        raise FileNotFoundError(f"no replay fixture matching {prefix}-*.json")
    return matches[0]


def _load_snapshot(prefix: str) -> tuple[dict, dict]:
    """Return (input_snapshot, original_result)."""
    payload = json.loads(_trace_path(prefix).read_text(encoding="utf-8"))
    return payload["trace"]["input_snapshot"], payload["trace"]["result"]


@pytest.mark.parametrize("prefix", _TRACE_PREFIXES)
def test_replay_0526_mlb_trace_has_no_bug_signatures(prefix):
    """Replaying the original snapshot through the patched engine must produce
    a draw-free simulation, both total edge rows, and at least one spread row
    with spread_coverage_prob populated.

    The five fixtures encode the *pre-fix* bug signature: each has exactly two
    edge rows with ``market is None`` and ``spread_coverage_prob is None``,
    and a draw_prob around 13–14%. The patched engine should produce six rows
    (ML home/away + spread home/away + total over/under), draw_prob=0, and
    every spread row carrying a non-null spread_coverage_prob."""
    snapshot, original = _load_snapshot(prefix)

    # Sanity-check the fixture still exhibits the bug signature so this test
    # remains a meaningful regression guard.
    orig_draw = original["simulation"]["draw_prob"]
    assert orig_draw > 10.0, (
        f"{prefix}: original draw_prob {orig_draw} is already 0 — fixture "
        "no longer exercises BUG-SIM-2"
    )
    assert all(e.get("market") is None for e in original["edges"]), (
        f"{prefix}: original edges already have market labels — fixture invalid"
    )
    assert all(e.get("spread_coverage_prob") is None for e in original["edges"]), (
        f"{prefix}: original edges already have spread_coverage_prob — fixture invalid"
    )

    request = GameAnalysisRequest(**snapshot)
    response = analyze_game(request)
    assert response.status == "success", response.skip_reason
    assert response.simulation is not None

    # 1. BUG-SIM-2 — no draw leak for baseball (supports_draw=False)
    draw_prob = response.simulation.draw_prob
    assert draw_prob is not None
    assert draw_prob == 0.0, (
        f"{prefix}: replay draw_prob={draw_prob} should be 0.0 (BUG-SIM-2)"
    )
    assert (
        response.simulation.home_win_prob + response.simulation.away_win_prob
        == pytest.approx(100.0, abs=0.2)
    )

    # 2. BUG-TOTALS-1 — both Over and Under total edge rows emitted
    total_edges = [e for e in response.edges if e.market == "total"]
    total_sides = {e.side for e in total_edges}
    assert total_sides == {"over", "under"}, (
        f"{prefix}: expected Over + Under total edges, got sides={total_sides} "
        "(BUG-TOTALS-1)"
    )

    # 3. BUG-SPREAD-1 — spread rows exist as distinct market='spread' edges
    # with spread_coverage_prob populated (was null in the original trace).
    spread_edges = [e for e in response.edges if e.market == "spread"]
    spread_sides = {e.side for e in spread_edges}
    assert spread_sides == {"home", "away"}, (
        f"{prefix}: expected home + away spread edges, got sides={spread_sides} "
        "(BUG-SPREAD-1)"
    )
    for edge in spread_edges:
        assert edge.spread_coverage_prob is not None, (
            f"{prefix}: spread edge side={edge.side} has null "
            "spread_coverage_prob — BUG-SPREAD-1 regression"
        )
        assert edge.true_prob == pytest.approx(
            edge.spread_coverage_prob, abs=1e-6
        ), (
            f"{prefix}: spread edge true_prob diverges from "
            "spread_coverage_prob — engine is mixing edge probability sources"
        )

    # 4. Total post-fix edge count must be strictly greater than the original
    # (which only ever had 2 rows). Confirms we're not silently regressing to
    # the conflated single-row-per-side shape.
    assert len(response.edges) > len(original["edges"])
