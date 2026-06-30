"""Narrative providers — the provider-agnostic seam for Deep Dive generation.

The deterministic engine owns every number; a provider only turns the disciplined
context pack into prose, counterarguments, and operator warnings. Two
implementations ship:

* :class:`StubProvider` — deterministic and offline. It composes a useful
  enrichment directly from the pack (trust factors, guardrails, market read,
  historical support). All tests use it, so CI needs no network or API key.
* :class:`AnthropicProvider` — calls the Anthropic Messages API with structured
  output validated against :class:`EnrichmentResult`. Lazy-imports the SDK so the
  package imports without ``anthropic`` installed.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from omega.enrichment import PROMPT_VERSION
from omega.enrichment.schemas import (
    EnrichmentResult,
    MarketContext,
    sanitize_raw_result,
)

# Default to the latest, most capable Claude model (overridable via env).
DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are Omega's trace enrichment analyst.

You do NOT create or restate probabilities, edges, EV, grades, or stake sizes —
the deterministic Omega engine owns those, and the context pack deliberately
omits them. You explain the model's existing position using ONLY the provided
context: the trust breakdown, guardrails, market-movement read, signal-conflict
view, evidence audit, and historical-support cohorts.

Your job: synthesize the model case, surface counterarguments, name the missing
context and weak/stale evidence, and give a measured operator recommendation. If
the context is insufficient, say so plainly rather than inventing confidence.

Return JSON matching the requested schema. ``recommendation_type`` must be one of
monitor | lean | avoid_due_to_data_quality | supports_model_position — never a
"bet now" directive. Keep each list item to one sentence."""


class NarrativeProvider(Protocol):
    """Turns a context pack into a validated :class:`EnrichmentResult`."""

    name: str
    model: str | None

    def generate_enrichment(self, context_pack: dict[str, Any]) -> EnrichmentResult: ...


def _risk_from_guardrails(worst_severity: str | None) -> str:
    return {"fail": "high", "warn": "medium", "info": "low", "ok": "low"}.get(
        worst_severity or "", "medium"
    )


def _recommendation_from_pack(pack: dict[str, Any]) -> str:
    """Deterministic, conservative operator recommendation from the pack."""
    gr = pack.get("guardrails") or {}
    if (gr.get("worst_severity") or "") == "fail":
        return "avoid_due_to_data_quality"
    mm = pack.get("market_movement") or {}
    if mm.get("direction") == "against":
        return "monitor"
    if pack.get("historical_support") == "strong" and (gr.get("worst_severity") in (None, "ok", "info")):
        return "supports_model_position"
    return "lean"


class StubProvider:
    """Deterministic, offline provider — composes the pack into an artifact."""

    name = "stub"
    model = None

    def generate_enrichment(self, context_pack: dict[str, Any]) -> EnrichmentResult:
        pack = context_pack or {}
        trust = pack.get("trust") or {}
        gr = pack.get("guardrails") or {}
        mm = pack.get("market_movement") or {}
        sc = pack.get("signal_conflict") or {}
        league = pack.get("league") or "?"
        kind = pack.get("kind") or "trace"

        positives = list(trust.get("positives") or [])
        negatives = list(trust.get("negatives") or [])
        flags = gr.get("flags") or []
        operator_notes = [f["action"] for f in flags if f.get("action")]
        if not operator_notes:
            operator_notes = ["Monitor for new context before acting."]

        headline = f"{league} {kind}: {trust.get('headline') or 'trace review'}"
        if mm.get("direction") == "against":
            headline += " — market disagrees"

        counter_case = list(negatives)
        if sc.get("dominant"):
            counter_case.append(f"Signal conflict: {str(sc['dominant']).replace('_', ' ')}.")
        hist = pack.get("historical_support")
        if hist and hist != "strong":
            counter_case.append(f"Historical support is {hist}.")

        raw = sanitize_raw_result(
            {
                "headline": headline,
                "summary": (
                    f"{trust.get('headline') or 'Trace'} with "
                    f"{len(positives)} supporting and {len(negatives)} cautioning factors. "
                    f"{gr.get('summary') or ''}"
                ).strip(),
                "model_case": positives or ["Model produced a position; supporting factors are limited."],
                "market_context": {
                    "line_movement": mm.get("direction") or "unknown",
                    "interpretation": mm.get("headline") or "No market read available.",
                },
                "counter_case": counter_case or ["No material counterargument surfaced."],
                "risk_rating": _risk_from_guardrails(gr.get("worst_severity")),
                "confidence_explanation": trust.get("headline") or "Trust not recorded.",
                "missing_context": list(pack.get("missing_context") or []),
                "operator_notes": operator_notes,
                "recommendation_type": _recommendation_from_pack(pack),
            }
        )
        return EnrichmentResult.model_validate(raw)


class AnthropicProvider:
    """Anthropic Messages API provider with schema-validated structured output."""

    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OMEGA_ENRICH_MODEL") or DEFAULT_MODEL

    def generate_enrichment(self, context_pack: dict[str, Any]) -> EnrichmentResult:
        import json

        import anthropic  # lazy: only required when this provider actually runs

        client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from env
        message = client.messages.parse(
            model=self.model,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Enrich this trace from its context pack. Use only the "
                        "context provided; do not invent numbers.\n\n"
                        + json.dumps({"context_pack": context_pack}, default=str)
                    ),
                }
            ],
            output_format=EnrichmentResult,
        )
        parsed = message.parsed_output
        if parsed is None:
            raise RuntimeError("Anthropic provider returned no parseable enrichment")
        return parsed


def get_provider(name: str | None, model: str | None = None) -> NarrativeProvider:
    """Resolve a provider by name (``OMEGA_ENRICH_PROVIDER`` if unset)."""
    name = (name or os.environ.get("OMEGA_ENRICH_PROVIDER") or "stub").strip().lower()
    if name in ("anthropic", "claude"):
        return AnthropicProvider(model=model)
    return StubProvider()


__all__ = [
    "NarrativeProvider",
    "StubProvider",
    "AnthropicProvider",
    "get_provider",
    "SYSTEM_PROMPT",
    "DEFAULT_MODEL",
    "PROMPT_VERSION",
]
