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


# Calibration-quality floor for per-market ACTIONABLE authorization.
#
# A market is only authorized for formal output when its OWN production profile
# clears these bars. They mirror the standard promotion gate so a force-promoted
# or under-sampled profile (one that slipped past `--min-samples`/`--force`)
# cannot unlock Bet Cards / EV% / Kelly for its market. The floor is read from
# the profile's recorded fit metrics (always present, deterministic) rather than
# in-window realized metrics, so a quiet window never spuriously downgrades a
# sound profile.
MIN_SAMPLES_FOR_ACTIONABLE = 100
MAX_ECE_FOR_ACTIONABLE = 0.05


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


def classify_market_output_mode(
    *,
    profile_id: str | None,
    sample_size: int | None,
    calibration_error: float | None,
    trace_count: int,
    sidecar_valid: bool,
) -> tuple[OutputMode, list[str]]:
    """Per-market output authorization, with a calibration-quality floor.

    Extends :func:`classify_output_mode`: on top of requiring a fitted profile,
    nonzero calibration-eligible coverage, and a valid sidecar, the market's own
    production profile must clear the quality floor
    (``sample_size >= MIN_SAMPLES_FOR_ACTIONABLE`` and
    ``calibration_error <= MAX_ECE_FOR_ACTIONABLE``). This stops a force-promoted
    or under-sampled profile from unlocking formal output for its market.

    Authorization is per market: game and prop are classified independently from
    their own production profiles, so a trustworthy prop market can be ACTIONABLE
    while the game market stays research-only, and vice versa.

    Returns the mode and a list of human-readable downgrade reasons (empty when
    ACTIONABLE) so callers can render the same reasons in the report frontmatter
    and the prose directive without them disagreeing.
    """
    base = classify_output_mode(
        calibration_profile=profile_id,
        trace_count=trace_count,
        sidecar_valid=sidecar_valid,
    )
    reasons: list[str] = []
    if base is OutputMode.RESEARCH_CANDIDATE:
        if profile_id is None:
            reasons.append("No fitted calibration profile for this market - static fallback active.")
        if trace_count == 0:
            reasons.append("0 calibration-eligible traces for this market in window.")
        if not sidecar_valid:
            reasons.append("Session sidecar invalid or corrupt.")
        return OutputMode.RESEARCH_CANDIDATE, reasons

    # Base is ACTIONABLE (profile present, coverage > 0, sidecar valid). Apply
    # the quality floor on the profile's own recorded fit metrics. Missing
    # metrics are treated as failing the floor: trustworthiness cannot be
    # confirmed, so the market stays research-only.
    if sample_size is None or sample_size < MIN_SAMPLES_FOR_ACTIONABLE:
        reasons.append(
            f"Profile sample_size {sample_size} < {MIN_SAMPLES_FOR_ACTIONABLE} floor."
        )
    if calibration_error is None:
        reasons.append("Profile calibration_error missing - cannot verify quality floor.")
    elif calibration_error > MAX_ECE_FOR_ACTIONABLE:
        reasons.append(
            f"Profile ECE {calibration_error:.3f} > {MAX_ECE_FOR_ACTIONABLE} floor."
        )
    if reasons:
        return OutputMode.RESEARCH_CANDIDATE, reasons
    return OutputMode.ACTIONABLE, reasons


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
