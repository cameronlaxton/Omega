"""Canonical blocked-language vocabulary shared across output seams.

Single definition of the phrases and hype words that must never appear in
user-facing analytical prose. ``omega.ops.output_modes`` re-exports these for
its mode-specific matchers; the decision-support DTO validators
(``omega.core.contracts.schemas``) apply the strict matcher directly.

Import-light on purpose (stdlib only): contracts-layer modules must be
importable without pulling in ops/trace layers.
"""

from __future__ import annotations

import re

# Phrases that must not appear in Research Candidate / decision-support output
# (numbers fully hidden). Stored lowercase; matchers casefold the text first so
# a blocked phrase cannot slip through with different casing.
BLOCKED_FORMAL_PHRASES: frozenset[str] = frozenset(
    {
        "best bet",
        "tier a",
        "tier b",
        "engine-confirmed",
        "actionable bet",
    }
)

# Phrases that must not appear in Research+ output. Research+ DOES surface the
# engine numbers and the (<= B) confidence tier, so tier labels are permitted;
# only the overclaiming hype phrases stay blocked.
BLOCKED_RESEARCH_PLUS_PHRASES: frozenset[str] = frozenset(
    {
        "best bet",
        "engine-confirmed",
        "actionable bet",
    }
)

# Single-word hype terms banned by AGENTS.md ("lock", "smash") in ANY output
# mode. Matched with word boundaries, not substrings, so "blocked" / "unlock" /
# "locksmith" never false-positive.
BLOCKED_HYPE_WORDS: tuple[str, ...] = ("lock", "smash")
BLOCKED_HYPE_WORDS_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BLOCKED_HYPE_WORDS) + r")\b"
)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Word-bounded phrase match so substrings like "tier a" inside "frontier
    analysis" or "best bet" inside "best beta" don't false-positive."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def blocked_language(text: str) -> list[str]:
    """Return blocked formal phrases / hype words found in ``text`` (strict set).

    This is the matcher applied to decision-support (primary-product) prose:
    recommendation vocabulary ("best bet", tier labels) and hype words are all
    rejected. Matching is case-insensitive; hype words are word-bounded.
    """
    normalized = text.casefold()
    found = [phrase for phrase in BLOCKED_FORMAL_PHRASES if _contains_phrase(normalized, phrase)]
    found.extend(set(BLOCKED_HYPE_WORDS_RE.findall(normalized)))
    return sorted(found)


def blocked_research_plus_language(text: str) -> list[str]:
    """Return blocked hype phrases found in Research+ text (tier labels allowed)."""
    normalized = text.casefold()
    found = [
        phrase for phrase in BLOCKED_RESEARCH_PLUS_PHRASES if _contains_phrase(normalized, phrase)
    ]
    found.extend(set(BLOCKED_HYPE_WORDS_RE.findall(normalized)))
    return sorted(found)
