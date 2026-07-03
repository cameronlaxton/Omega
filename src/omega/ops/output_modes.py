"""Output mode classification and enforcement for Omega sessions.

Determines whether a session qualifies for formal/actionable output or must
fall back to the Research Candidate presentation. This keeps presentation
decisions centralized and out of ad hoc agent logic.
"""

from __future__ import annotations

import re
from enum import Enum

# Phrases that must not appear in Research Candidate output (numbers fully hidden).
# Stored lowercase; the matchers casefold the text first so a blocked phrase
# cannot slip through with different casing (e.g. "BEST BET", "Engine-Confirmed").
_BLOCKED_FORMAL_PHRASES: frozenset[str] = frozenset(
    {
        "best bet",
        "tier a",
        "tier b",
        "engine-confirmed",
        "actionable bet",
    }
)

# Phrases that must not appear in Research+ output. Research+ DOES surface the
# engine numbers and the (<= B) confidence tier, so tier labels are permitted
# here; only the overclaiming hype phrases stay blocked — a thin/immature profile
# must never be narrated as a settled, engine-confirmed best bet.
_BLOCKED_RESEARCH_PLUS_PHRASES: frozenset[str] = frozenset(
    {
        "best bet",
        "engine-confirmed",
        "actionable bet",
    }
)

# Single-word hype terms banned by AGENTS.md ("lock", "smash") in ANY output
# mode. Matched with word boundaries, not substrings, so "blocked" / "unlock" /
# "locksmith" never false-positive. Applied by both matchers below.
_BLOCKED_HYPE_WORDS: tuple[str, ...] = ("lock", "smash")
_BLOCKED_HYPE_WORDS_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _BLOCKED_HYPE_WORDS) + r")\b"
)

RESEARCH_CANDIDATE_DISCLAIMER = (
    "*Qualitative review required; uncalibrated baseline variance applies.*"
)

RESEARCH_CANDIDATE_HEADER = "### Research Candidate"

RESEARCH_PLUS_DISCLAIMER = (
    "*Thin/provisional calibration: engine numbers shown for transparency but "
    "stake is hard-capped and confidence is held at ≤ B. Size accordingly.*"
)

RESEARCH_PLUS_HEADER = "### Research+ (provisional calibration)"


class OutputMode(str, Enum):
    RESEARCH_CANDIDATE = "research_candidate"
    RESEARCH_PLUS = "research_plus"
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

# Research+ stake ceilings (units) by profile maturity. A real-but-immature
# profile surfaces its engine numbers but is sized far below a production edge;
# production-but-below-floor or unknown maturity falls to the default ceiling.
MAX_STAKE_RESEARCH_PLUS_BY_MATURITY: dict[str, float] = {
    "provisional": 0.5,
    "probation": 1.0,
}
MAX_STAKE_RESEARCH_PLUS_DEFAULT = 0.5


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
    maturity: str | None = None,
) -> tuple[OutputMode, list[str]]:
    """Per-market output authorization across three tiers.

    A *trust dial, not a switch* — only the genuinely uncalibrated case fully
    hides the engine numbers:

    - **RESEARCH_CANDIDATE** (numbers hidden): no fitted profile, zero
      calibration-eligible coverage, an invalid sidecar, or a profile whose
      maturity is ``none``/``retired`` (not trusted to apply). This is the
      genuinely-uncalibrated boundary.
    - **ACTIONABLE** (full): a ``production``-maturity profile that also clears
      the quality floor (``sample_size >= MIN_SAMPLES_FOR_ACTIONABLE`` and
      ``calibration_error <= MAX_ECE_FOR_ACTIONABLE``).
    - **RESEARCH_PLUS** (numbers shown, stake hard-capped, confidence <= B):
      everything in between — a real but immature (``provisional``/``probation``)
      profile, or a ``production`` profile that has not cleared the floor. The
      engine numbers are surfaced under guardrails instead of being withheld.

    ``maturity`` is the operator-assigned trust grade (see ``ProfileMaturity``);
    ``None`` is treated as production-equivalent (legacy profiles whose maturity
    derives from PRODUCTION status). Authorization is per market: game and prop
    are classified independently from their own profiles.

    Returns the mode and a list of human-readable reasons (empty only when
    ACTIONABLE) so callers render the same reasons in the report frontmatter and
    the prose directive without them disagreeing.
    """
    # Tier 1 — genuinely uncalibrated / unusable session: numbers hidden.
    hard_reasons: list[str] = []
    if profile_id is None:
        hard_reasons.append(
            "No fitted calibration profile for this market - static fallback active."
        )
    if trace_count == 0:
        hard_reasons.append("0 calibration-eligible traces for this market in window.")
    if not sidecar_valid:
        hard_reasons.append("Session sidecar invalid or corrupt.")
    if maturity in ("none", "retired"):
        hard_reasons.append(
            f"Profile maturity '{maturity}' is not trusted to apply - no calibration correction."
        )
    if hard_reasons:
        return OutputMode.RESEARCH_CANDIDATE, hard_reasons

    # A real profile exists with coverage and a valid sidecar. Decide ACTIONABLE
    # vs RESEARCH_PLUS on the quality floor + maturity. Any shortfall keeps the
    # numbers visible (research+) rather than hiding them — hard suppression is
    # reserved for Tier 1 above.
    reasons: list[str] = []
    if sample_size is None or sample_size < MIN_SAMPLES_FOR_ACTIONABLE:
        reasons.append(f"Profile sample_size {sample_size} < {MIN_SAMPLES_FOR_ACTIONABLE} floor.")
    if calibration_error is None:
        reasons.append("Profile calibration_error missing - cannot verify quality floor.")
    elif calibration_error > MAX_ECE_FOR_ACTIONABLE:
        reasons.append(f"Profile ECE {calibration_error:.3f} > {MAX_ECE_FOR_ACTIONABLE} floor.")
    if maturity is not None and maturity != "production":
        reasons.append(
            f"Profile maturity '{maturity}' below production - numbers shown, "
            "stake capped, confidence <= B."
        )
    if reasons:
        return OutputMode.RESEARCH_PLUS, reasons
    return OutputMode.ACTIONABLE, reasons


def cap_stake_for_research(stake_units: float) -> float:
    """Enforce 1u maximum for Research Candidate outputs."""
    return min(stake_units, 1.0)


def cap_stake_for_research_plus(stake_units: float, maturity: str | None = None) -> float:
    """Cap stake for Research+ output by profile maturity.

    Real-but-immature profiles surface their numbers but are sized well below a
    production edge: ``provisional`` <= 0.5u, ``probation`` <= 1.0u; any other
    case (production-but-below-floor, unknown) falls to the default ceiling.
    """
    ceiling = MAX_STAKE_RESEARCH_PLUS_BY_MATURITY.get(
        maturity or "", MAX_STAKE_RESEARCH_PLUS_DEFAULT
    )
    return min(stake_units, ceiling)


def contains_blocked_phrase(text: str) -> list[str]:
    """Return any blocked formal phrases found in text.

    Used to audit output blocks before they are emitted in Research Candidate mode.
    Matching is case-insensitive.
    """
    normalized = text.casefold()
    found = [phrase for phrase in _BLOCKED_FORMAL_PHRASES if phrase in normalized]
    found.extend(set(_BLOCKED_HYPE_WORDS_RE.findall(normalized)))
    return sorted(found)


def contains_blocked_phrase_research_plus(text: str) -> list[str]:
    """Return any blocked hype phrases found in Research+ text.

    Research+ permits the engine numbers and the (<= B) confidence tier, so tier
    labels are allowed here; only the overclaiming hype phrases are blocked.
    Matching is case-insensitive.
    """
    normalized = text.casefold()
    found = [phrase for phrase in _BLOCKED_RESEARCH_PLUS_PHRASES if phrase in normalized]
    found.extend(set(_BLOCKED_HYPE_WORDS_RE.findall(normalized)))
    return sorted(found)


def format_research_candidate_block(content: str) -> str:
    """Wrap an output block with the Research Candidate header and disclaimer."""
    return f"{RESEARCH_CANDIDATE_HEADER}\n\n{content}\n\n{RESEARCH_CANDIDATE_DISCLAIMER}"


def format_research_plus_block(content: str) -> str:
    """Wrap an output block with the Research+ header and disclaimer."""
    return f"{RESEARCH_PLUS_HEADER}\n\n{content}\n\n{RESEARCH_PLUS_DISCLAIMER}"
