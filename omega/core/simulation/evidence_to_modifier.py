"""Deterministic mapping from EvidenceSignal types to Markov transition modifiers.

Design rules:
- This module contains a strict lookup dict. The LLM decides WHICH signal types
  to emit; this module decides the SCALAR effect of each signal on the Markov
  transition matrix. No LLM inference occurs here.
- Unknown signal types are silently ignored (they remain persisted in the trace
  for audit but do not touch the simulation).
- Multiple signals targeting the same modifier key are multiplied together and
  then clamped so that no single attribute can shift more than ±MAX_CUMULATIVE_SHIFT.
- Directional signals (direction="home" vs direction="away") are applied to
  the appropriate side's modifier key.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omega.core.contracts.evidence import EvidenceSignal

_log = logging.getLogger(__name__)

# Maximum cumulative multiplicative shift allowed for any single modifier key.
# 1.15 means the engine can boost at most +15% or suppress at most ~13% (-1/1.15)
# regardless of how many overlapping signals pile up.
_MAX_CUMULATIVE_SHIFT: float = 1.15

# Strict signal_type → (modifier_key, scalar) mapping.
# All keys must be members of SIGNAL_REGISTRY in omega.core.contracts.evidence.
# Validated at import time via _validate_registry_membership().
_SIGNAL_TO_MODIFIER: dict[str, tuple[str, float]] = {
    # Pace signals
    "pace_up":              ("pace_scalar",              1.06),
    "pace_down":            ("pace_scalar",              0.92),
    # Rest / fatigue
    "rest_advantage":       ("home_score_rate_scalar",   1.04),
    "b2b_fatigue":          ("home_score_rate_scalar",   0.94),
    # Matchup quality signals (applied to the defensive opponent's concession rate)
    "def_matchup_weak":     ("away_score_rate_scalar",   1.05),
    "def_matchup_strong":   ("away_score_rate_scalar",   0.95),
    # Role / injury-driven usage change
    "usage_role_change":    ("home_score_rate_scalar",   0.93),
    # Blowout risk suppresses momentum variance
    "blowout_risk":         ("home_momentum_scalar",     0.98),
}

# Exported for introspection (e.g. champion/challenger reporting).
MAPPED_SIGNAL_TYPES: frozenset[str] = frozenset(_SIGNAL_TO_MODIFIER)

# ---------------------------------------------------------------------------
# Public vocabulary table — single source of truth for prompt generation.
# ---------------------------------------------------------------------------
# Each entry: signal_type -> (modifier_key, scalar, plain-English description)
# Used by omega_markov_evidence_guide() in the MCP server and by the cowork
# prompt section so the LLM's vocabulary is always in sync with this dict.
MARKOV_SIGNAL_VOCABULARY: tuple[tuple[str, str, float, str], ...] = (
    ("pace_up",           "pace_scalar",            1.06, "+6% pace; matchup faster than league baseline"),
    ("pace_down",         "pace_scalar",            0.92, "-8% pace; matchup slower than league baseline"),
    ("rest_advantage",    "home_score_rate_scalar",  1.04, "+4% home scoring rate; directional (home/away)"),
    ("b2b_fatigue",       "home_score_rate_scalar",  0.94, "-6% scoring rate for the fatigued team; directional"),
    ("def_matchup_weak",  "away_score_rate_scalar",  1.05, "+5% offensive scoring vs. weak defender; directional"),
    ("def_matchup_strong","away_score_rate_scalar",  0.95, "-5% offensive scoring vs. strong defender; directional"),
    ("usage_role_change", "home_score_rate_scalar",  0.93, "-7% team scoring rate when key player role is restricted; directional"),
    ("blowout_risk",      "home_momentum_scalar",    0.98, "-2% momentum acceleration; suppresses runaway variance"),
)


def build_markov_vocabulary_table() -> str:
    """Return a formatted text table of Markov-eligible signal types for prompt injection.

    Called by the MCP server's omega_markov_evidence_guide prompt and by
    OMEGA_COWORK.md section generation so the LLM vocabulary is always derived
    from the single source of truth here, never hand-edited in prompts.
    """
    lines = [
        "Markov-eligible signal types (simulation_backend='markov_state' only):",
        "",
        f"  {'signal_type':<24} {'modifier':<26} {'scalar':>6}  description",
        f"  {'-'*24} {'-'*26} {'-'*6}  {'-'*45}",
    ]
    for sig, mod, scalar, desc in MARKOV_SIGNAL_VOCABULARY:
        lines.append(f"  {sig:<24} {mod:<26} {scalar:>6.2f}  {desc}")
    lines += [
        "",
        "Rules:",
        "  - All other signal_types are valid for audit/fast_score paths but have NO",
        "    effect on the Markov transition matrix (silently ignored, still persisted).",
        "  - Cumulative cap: no single modifier attribute can shift by more than +/-15%",
        "    regardless of how many overlapping signals are stacked.",
        "  - Use direction='home' or direction='away' for rest_advantage, b2b_fatigue,",
        "    def_matchup_weak/strong, and usage_role_change to target the correct team.",
    ]
    return "\n".join(lines)


def _validate_registry_membership() -> None:
    """Fail loudly at import time if any key is absent from SIGNAL_REGISTRY."""
    from omega.core.contracts.evidence import SIGNAL_REGISTRY  # noqa: PLC0415

    unknown = MAPPED_SIGNAL_TYPES - frozenset(SIGNAL_REGISTRY)
    if unknown:
        raise ImportError(
            f"evidence_to_modifier: keys not in SIGNAL_REGISTRY: {sorted(unknown)}"
        )


_validate_registry_membership()


def signals_to_transition_modifiers(
    signals: list[EvidenceSignal],
    home_team: str,
) -> dict[str, float]:
    """Map a list of EvidenceSignals to a Markov transition_modifiers dict.

    Args:
        signals: Evidence signals from the orchestrator (may be empty).
        home_team: Home team name, used to resolve home/away directional signals.

    Returns:
        A dict of modifier_key → cumulative scalar, clamped to ±MAX_CUMULATIVE_SHIFT.
        Signals without a known mapping are silently skipped.

    Rules:
        - Multiple signals for the same key are multiplied (compounding effect).
        - The cumulative scalar for each key is then clamped to
          [1/MAX_CUMULATIVE_SHIFT, MAX_CUMULATIVE_SHIFT].
        - Directional signals with direction="away" invert the home/away side.
    """
    if not signals:
        return {}

    raw_accum: dict[str, float] = {}

    for sig in signals:
        entry = _SIGNAL_TO_MODIFIER.get(sig.signal_type)
        if entry is None:
            continue

        modifier_key, base_scalar = entry

        # For directional signals, invert the scalar when signal targets the away side.
        # Example: rest_advantage with direction="away" benefits the away team, so we
        # apply the boost to away_score_rate_scalar instead.
        effective_key = modifier_key
        effective_scalar = base_scalar
        if hasattr(sig, "direction") and sig.direction in ("home", "away"):
            effective_key, effective_scalar = _resolve_direction(
                modifier_key, base_scalar, sig.direction
            )

        if effective_key not in raw_accum:
            raw_accum[effective_key] = 1.0
        raw_accum[effective_key] *= effective_scalar

        _log.debug(
            "signal %r (dir=%r) → %r × %s (running %s)",
            sig.signal_type,
            getattr(sig, "direction", None),
            effective_key,
            effective_scalar,
            raw_accum[effective_key],
        )

    # Clamp each key's cumulative product to [1/MAX, MAX]
    lo = 1.0 / _MAX_CUMULATIVE_SHIFT
    hi = _MAX_CUMULATIVE_SHIFT
    clamped: dict[str, float] = {}
    for key, product in raw_accum.items():
        if not (lo <= product <= hi):
            clamped_val = max(lo, min(hi, product))
            _log.warning(
                "cumulative modifier %r=%r clamped to %r (cap ±%.0f%%)",
                key, product, clamped_val, (_MAX_CUMULATIVE_SHIFT - 1) * 100,
            )
            clamped[key] = clamped_val
        else:
            clamped[key] = product

    return clamped


def _resolve_direction(
    modifier_key: str, scalar: float, direction: str
) -> tuple[str, float]:
    """Swap home/away modifier key when the signal targets the non-default side.

    The default mapping table is written from the home team's perspective.
    When direction="away", we flip the key to target the other team.
    """
    if direction == "away":
        if modifier_key == "home_score_rate_scalar":
            return "away_score_rate_scalar", scalar
        if modifier_key == "away_score_rate_scalar":
            return "home_score_rate_scalar", scalar
        if modifier_key == "home_momentum_scalar":
            return "away_momentum_scalar", scalar
    # For pace_scalar and unknown keys: no directional flip needed.
    return modifier_key, scalar
