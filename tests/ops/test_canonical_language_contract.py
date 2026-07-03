"""Canonical-document language contract.

The canonical LLM-facing documents may MENTION banned hype terminology only when
quoting it as banned (inside straight double quotes or backtick code spans).
Any bare-prose use of a banned phrase in these files is drift — the exact
failure mode that let "estimated lean" linger after it was retired.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The canonical interaction-contract documents (AGENTS.md names these).
CANONICAL_DOCS = [
    "AGENTS.md",
    "OMEGA_RUNTIME.md",
    "prompts/system_prompt.txt",
    "prompts/reference/output_modes.md",
    "prompts/reference/presentation_contract.md",
    "docs/LLM_MCP_INTERFACE.md",
]

# Multi-word hype phrases + retired labels. Matched case-insensitively as
# substrings. ("lock" is deliberately NOT doc-scanned: docs legitimately say
# "lockfile" / "lock you out"; the runtime output matcher in
# omega.ops.output_modes enforces it word-bounded where it matters — in
# rendered betting output.)
BANNED_PHRASES = [
    "best bet",
    "estimated lean",
    "engine-confirmed",
    "actionable bet",
    "smash",
]


def _quoted_spans(line: str) -> list[tuple[int, int]]:
    """Character ranges inside straight double quotes or backtick code spans."""
    spans: list[tuple[int, int]] = []
    for pattern in (r'"[^"]*"', r"`[^`]*`", r"'[^']*'"):
        spans.extend(m.span() for m in re.finditer(pattern, line))
    return spans


def _bare_occurrences(text: str, phrase: str) -> list[tuple[int, str]]:
    """(line_number, line) pairs where phrase appears OUTSIDE quoted/code spans."""
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        lowered = line.casefold()
        spans = _quoted_spans(line)
        for m in re.finditer(re.escape(phrase), lowered):
            inside = any(s <= m.start() and m.end() <= e for s, e in spans)
            # "actionable Bet Card" is a legitimate structural term (the payload
            # authorized in ACTIONABLE mode) — only the bare label is banned.
            structural = phrase == "actionable bet" and lowered[m.end() :].startswith(" card")
            if not inside and not structural:
                hits.append((lineno, line.strip()))
                break
    return hits


@pytest.mark.parametrize("doc", CANONICAL_DOCS)
@pytest.mark.parametrize("phrase", BANNED_PHRASES)
def test_banned_phrase_only_appears_quoted(doc: str, phrase: str) -> None:
    path = _REPO_ROOT / doc
    assert path.exists(), f"canonical doc missing: {doc}"
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = _bare_occurrences(text, phrase)
    assert not hits, (
        f"{doc} uses banned phrase {phrase!r} outside a quoted/banned-list context: {hits}"
    )


def test_runtime_matcher_covers_doc_banned_hype() -> None:
    """The phrases AGENTS.md bans must actually be enforced by the runtime
    output matcher — prose bans without a matcher are how drift starts."""
    from omega.ops.output_modes import contains_blocked_phrase

    for sample, expected in [
        ("our best bet tonight", "best bet"),
        ("this is a lock", "lock"),
        ("smash the over", "smash"),
        ("engine-confirmed play", "engine-confirmed"),
        ("an actionable bet here", "actionable bet"),
    ]:
        assert expected in contains_blocked_phrase(sample), sample
