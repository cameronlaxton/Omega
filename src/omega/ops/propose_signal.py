"""
omega-propose-signal — register an LLM-proposed evidence-signal hypothesis.

A proposal is a *typed hypothesis over a whitelisted feature vocabulary* (see
``feature_combo_eval.FEATURE_WHITELIST``) — never arbitrary code. It lands in the
``signal_proposals`` store in ``lifecycle='probation'``, is scored by CLV exactly
like an active signal, and graduates ``probation -> active`` ONLY through the
operator-gated CLV + marginal bar (``omega-promote-adjustment-policy``). "LLM
proposes, data disposes, operator commits."

Because every proposal adds a hypothesis to the run, the Benjamini-Hochberg FDR
correction in the fit automatically tightens for all signals — an expanded
proposal space cannot manufacture spurious edges.

Usage:
    omega-propose-signal --name usage_when_teammate_out --plane player \\
        --direction-rule over --thesis "Usage spikes when the star sits" \\
        --feature-combo '{"kind":"predicate","when":{"op":"AND","terms":[
            {"feature":"usage","op":">","value":0.30},
            {"feature":"teammate_injured","op":"==","value":true}]},
            "true_factor":1.06}'

Exit codes:
    0 — proposal registered (or dry-run validated)
    1 — invalid proposal or fatal error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.contracts.evidence import SIGNAL_REGISTRY  # noqa: E402
from omega.core.simulation.feature_combo_eval import (  # noqa: E402
    FeatureComboError,
    validate_feature_combo,
)
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("propose_signal")

_VALID_PLANES = frozenset({"player", "game", "both"})
_VALID_DIRECTIONS = frozenset({"over", "under", "home", "away", "neutral"})


def propose_signal(
    *,
    name: str,
    feature_combo: dict[str, Any],
    thesis: str = "",
    plane: str = "both",
    direction_rule: str | None = None,
    source: str = "llm",
    db_path: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Validate and (optionally) persist a probation signal proposal.

    Raises ``ValueError`` / ``FeatureComboError`` on invalid input so callers can
    map it to an error response. On success returns a small result dict.
    """
    if not name or not isinstance(name, str):
        raise ValueError("name must be a non-empty string")
    if name in SIGNAL_REGISTRY:
        raise ValueError(f"name {name!r} collides with a built-in signal_type")
    if plane not in _VALID_PLANES:
        raise ValueError(f"plane must be one of {sorted(_VALID_PLANES)}")
    if direction_rule is not None and direction_rule not in _VALID_DIRECTIONS:
        raise ValueError(f"direction_rule must be one of {sorted(_VALID_DIRECTIONS)} or null")
    validate_feature_combo(feature_combo)  # raises FeatureComboError on a bad spec

    if persist:
        store = TraceStore(db_path=db_path)
        try:
            store.upsert_signal_proposal(
                name=name,
                feature_combo=feature_combo,
                thesis=thesis,
                plane=plane,
                direction_rule=direction_rule,
                source=source,
                lifecycle="probation",
            )
        finally:
            store.close()
    return {
        "name": name,
        "lifecycle": "probation",
        "plane": plane,
        "direction_rule": direction_rule,
        "persisted": persist,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register an LLM-proposed signal hypothesis.")
    parser.add_argument("--name", required=True, help="Unique proposal name (snake_case)")
    parser.add_argument("--thesis", default="", help="Plain-text hypothesis")
    parser.add_argument("--plane", default="both", choices=sorted(_VALID_PLANES))
    parser.add_argument("--direction-rule", default=None, help="over|under|home|away|neutral")
    parser.add_argument(
        "--feature-combo", required=True, help="Whitelisted feature_combo spec as a JSON string"
    )
    parser.add_argument("--source", default="llm")
    parser.add_argument("--db", type=str, default=None, help="SQLite path")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not persist")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        combo = json.loads(args.feature_combo)
    except json.JSONDecodeError as exc:
        logger.error("--feature-combo is not valid JSON: %s", exc)
        return 1

    try:
        result = propose_signal(
            name=args.name,
            feature_combo=combo,
            thesis=args.thesis,
            plane=args.plane,
            direction_rule=args.direction_rule,
            source=args.source,
            db_path=args.db,
            persist=not args.dry_run,
        )
    except (ValueError, FeatureComboError) as exc:
        logger.error("Invalid proposal: %s", exc)
        return 1

    logger.info(
        "%s proposal %r (plane=%s, direction_rule=%s) in lifecycle=probation.",
        "Validated" if args.dry_run else "Registered",
        result["name"],
        result["plane"],
        result["direction_rule"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
