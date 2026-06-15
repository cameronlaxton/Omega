"""Per-sport exotic/derivative edge consumers + registry.

Most game markets (moneyline / spread / total / 3-way / double-chance / BTTS /
correct-score) are priced by sport-agnostic blocks in
``omega.core.contracts.service.analyze_game``. A handful of markets are
sport-specific and pmf-driven: soccer Asian-handicap and first-half total today,
NFL Wong teasers next (Milestone 4). Pricing those inline turns the odds section
into a per-sport ``if`` ladder.

This module is the seam that removes the ladder. It mirrors the
``GAME_BACKENDS`` registry in ``omega.core.simulation.backends`` and the
``PRIOR_BUILDERS`` registry in ``omega.trace.priors``: each archetype registers
one ``EdgeConsumer`` at import time, and ``analyze_game`` resolves the consumer
by archetype name and extends ``edges`` with its output via a single dispatch.

Consumers receive ``calibrate_fn`` and ``build_edge_fn`` as arguments rather than
importing them from ``service``; that injection keeps the dependency one-way
(``service`` -> ``edge``) and lets the engine own calibration/edge math while the
consumer owns only the per-sport market evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omega.core.contracts.schemas import (
        CalibrationAudit,
        EdgeDetail,
        GameAnalysisRequest,
    )

# Injected service helpers. Typed loosely (``Callable[..., ...]``) because the
# concrete ``_calibrate_audited`` / ``_build_edge`` are keyword-rich; consumers
# call them exactly as ``analyze_game``'s sport-agnostic blocks do.
CalibrateFn = Callable[..., "tuple[float, CalibrationAudit]"]
BuildEdgeFn = Callable[..., "EdgeDetail"]


class EdgeConsumer(Protocol):
    """A sport's pmf-driven exotic/derivative market pricer.

    ``consume`` evaluates the consumer's markets against the backend's emitted
    pmfs/probabilities in ``sim_result`` and returns the resulting
    ``EdgeDetail`` rows. It resolves its own market lines from ``request.odds``
    and prices them through the injected ``calibrate_fn`` / ``build_edge_fn`` so
    the deterministic calibration/edge math stays owned by the engine layer.
    """

    sport: str

    def consume(
        self,
        sim_result: dict[str, Any],
        request: GameAnalysisRequest,
        bankroll: float,
        calibrate_fn: CalibrateFn,
        build_edge_fn: BuildEdgeFn,
    ) -> list[EdgeDetail]:
        """Return the consumer's edges for one analyzed game (possibly empty)."""


# ---------------------------------------------------------------------------
# Edge-consumer registry (keyed by archetype name, e.g. "soccer", "american_football")
# ---------------------------------------------------------------------------

EDGE_CONSUMERS: dict[str, EdgeConsumer] = {}


def _require_consumer_contract(name: str, consumer: Any) -> None:
    """Validate a consumer satisfies the ``EdgeConsumer`` surface.

    Protocols are structurally typed and not enforced at runtime, so a wiring
    mistake would only surface at dispatch. This makes it fail loudly at
    registration (import) time instead, matching ``register_game_backend``.
    """
    if not callable(getattr(consumer, "consume", None)):
        raise TypeError(f"edge consumer {name!r} must define a callable consume()")


def register_edge_consumer(name: str, consumer: EdgeConsumer) -> None:
    """Register an edge consumer under *name* (its archetype).

    Raises TypeError on an incomplete consumer surface and ValueError on
    duplicate registration, so import-time wiring mistakes fail loudly rather
    than silently shadowing an existing consumer or deferring an attribute error
    to dispatch time.
    """
    _require_consumer_contract(name, consumer)
    if name in EDGE_CONSUMERS:
        raise ValueError(f"edge consumer {name!r} already registered")
    EDGE_CONSUMERS[name] = consumer


def resolve_edge_consumer(name: str | None) -> EdgeConsumer | None:
    """Return the registered consumer for archetype *name*, or None if unknown."""
    if name is None:
        return None
    return EDGE_CONSUMERS.get(name)
