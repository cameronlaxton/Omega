"""Canonical engine-owned (protected) quant field names.

Single definition of the field names the LLM must never supply and
LLM-authored payloads must never carry. The deterministic engine computes
these; every LLM-facing seam rejects them:

- session sidecar audit events (``omega.trace.session_sidecar`` raises
  ``ProtectedValueError``; its set is kept identical to this one by test);
- the RSVG gate (``omega.core.gates.rsvg``, a strict superset);
- caller-supplied ``trace_quality`` on the MCP analyze tools
  (``omega.mcp.server``).

Import-light on purpose: contracts-layer modules and the trace layer can both
depend on it without cycles.
"""

from __future__ import annotations

import re
from typing import Any

# The canonical engine-owned quant fields. Keep in lock-step with
# omega.trace.session_sidecar._PROTECTED_QUANT_FIELDS (enforced by
# tests/core/test_protected_fields.py, not by import — trace already imports
# core, so core must not import trace).
PROTECTED_QUANT_FIELDS: frozenset[str] = frozenset(
    {
        "edge_pct",
        "ev_pct",
        "kelly_fraction",
        "units",
        "confidence_tier",
        "fair_price",
        "no_vig_price",
        "model_probability",
        "over_prob",
        "under_prob",
    }
)


def find_protected_key(value: Any, *, fields: frozenset[str] = PROTECTED_QUANT_FIELDS) -> str | None:
    """Recursively search dict keys for protected engine field names.

    Scans keys only (never values): free prose may legitimately *mention* a
    field name ("Null fields: result.edge_pct"), but a structured payload must
    never carry one as a key.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            if k in fields:
                return str(k)
            found = find_protected_key(v, fields=fields)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = find_protected_key(item, fields=fields)
            if found:
                return found
    return None


# Free-text (prose) can smuggle a protected value as ``field_name=1.2`` /
# ``field_name: -110`` even where structural key scanning (above) never sees a
# dict key. Deliberately narrow — matched against the field name immediately
# followed by ``=``/``:`` and a number — so ordinary qualitative prose that
# merely *discusses* probability/edge/EV/stake concepts (with no attached
# engine value) never false-positives. Mirrors
# ``omega.trace.session_sidecar._PROTECTED_VALUE_LEAK_RE`` (kept in lock-step
# by test; trace must not be imported from core).
_PROTECTED_VALUE_LEAK_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(PROTECTED_QUANT_FIELDS)) + r")\s*[:=]\s*[+-]?\d"
)


def find_protected_value_leak(text: str) -> str | None:
    """Return the matched span if prose embeds a ``field_name=value`` leak."""
    match = _PROTECTED_VALUE_LEAK_RE.search(text)
    return match.group(0) if match else None
