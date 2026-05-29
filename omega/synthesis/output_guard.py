"""Output mode classification and enforcement for Omega sessions.

Determines whether a session qualifies for formal/actionable output or must
fall back to the Research Candidate presentation. This keeps presentation
decisions centralized and out of ad hoc agent logic.
"""

from __future__ import annotations

from enum import Enum

# Phrases that must not appear in Research Candidate output.
_BLOCKED_FORMAL_PHRASES: frozenset[str] = frozenset(
    {
        "best bet",
        "Best Bet",
        "Tier A",
        "Tier B",
        "engine-confirmed",
        "actionable bet",
        "Actionable Bet",
    }
)

RESEARCH_CANDIDATE_DISCLAIMER = (
    "*Qualitative review required; uncalibrated baseline variance applies.*"
)

RESEARCH_CANDIDATE_HEADER = "### Research Candidate"


class OutputMode(str, Enum):
    RESEARCH_CANDIDATE = "research_candidate"
    ACTIONABLE = "actionable"


def classify_output_mode(
    *,
    calibration_profile: str | None,
    trace_count: int,
    sidecar_valid: bool,
) -> OutputMode:
    """Determine whether session output qualifies as actionable or research-only.

    Output authorization is a *model-evaluation* decision: it depends on whether
    the engine has a fitted calibration prior and enough calibration-eligible
    coverage to trust the numbers — never on whether the user logged a wager.

    Any of the following triggers Research Candidate mode:
    - calibration_profile is None (static fallback — no fitted prior)
    - trace_count == 0 (no calibration-eligible engine traces in window)
    - sidecar_valid is False (invalid/corrupt session ledger)

    Bet records are deliberately NOT a factor here. A Bet Card is emitted
    *before* any wager exists, so gating actionable output on a logged bet was
    backwards; bet logging is wager-tracking metadata only and has no bearing on
    calibration, grading, or output authorization.
    """
    if calibration_profile is None or trace_count == 0 or not sidecar_valid:
        return OutputMode.RESEARCH_CANDIDATE
    return OutputMode.ACTIONABLE


def cap_stake_for_research(stake_units: float) -> float:
    """Enforce 1u maximum for Research Candidate outputs."""
    return min(stake_units, 1.0)


def contains_blocked_phrase(text: str) -> list[str]:
    """Return any blocked formal phrases found in text.

    Used to audit output blocks before they are emitted in Research Candidate mode.
    """
    found = [phrase for phrase in _BLOCKED_FORMAL_PHRASES if phrase in text]
    return sorted(found)


def format_research_candidate_block(content: str) -> str:
    """Wrap an output block with the Research Candidate header and disclaimer."""
    return f"{RESEARCH_CANDIDATE_HEADER}\n\n{content}\n\n{RESEARCH_CANDIDATE_DISCLAIMER}"
