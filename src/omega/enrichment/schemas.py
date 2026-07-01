"""Contracts for the enrichment subsystem (result, record, feedback).

The :class:`EnrichmentResult` is the structured artifact the narrative provider
must return. It is intentionally **number-free**: there is no field in which the
LLM could assert a probability/edge/EV/grade/stake, and ``extra="forbid"`` means
it cannot smuggle one in as an unexpected key. The deterministic engine owns
those values; this layer owns prose, counterarguments, and operator warnings.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The operator-recommendation vocabulary is deliberately NOT "bet_this_now".
RECOMMENDATION_TYPES = (
    "monitor",
    "lean",
    "avoid_due_to_data_quality",
    "supports_model_position",
)
RISK_RATINGS = ("low", "medium", "high")
ENRICHMENT_STATUSES = ("queued", "running", "completed", "failed")


class MarketContext(BaseModel):
    """The narrative's market read (qualitative; no protected numbers)."""

    model_config = ConfigDict(extra="forbid")

    line_movement: str = "unknown"  # toward | against | flat | unknown | …
    interpretation: str = ""


class EnrichmentResult(BaseModel):
    """The structured Deep Dive artifact returned by a narrative provider."""

    model_config = ConfigDict(extra="forbid")

    headline: str
    summary: str
    model_case: list[str] = Field(default_factory=list)
    market_context: MarketContext = Field(default_factory=MarketContext)
    counter_case: list[str] = Field(default_factory=list)
    risk_rating: str = "medium"
    confidence_explanation: str = ""
    missing_context: list[str] = Field(default_factory=list)
    operator_notes: list[str] = Field(default_factory=list)
    recommendation_type: str = "monitor"

    @field_validator("recommendation_type")
    @classmethod
    def _check_rec_type(cls, v: str) -> str:
        if v not in RECOMMENDATION_TYPES:
            raise ValueError(
                f"recommendation_type {v!r} not in {RECOMMENDATION_TYPES}; "
                "the enrichment layer must not invent a bet directive"
            )
        return v

    @field_validator("risk_rating")
    @classmethod
    def _check_risk(cls, v: str) -> str:
        if v not in RISK_RATINGS:
            raise ValueError(f"risk_rating {v!r} not in {RISK_RATINGS}")
        return v


# Keys a provider must never return as structured data (the engine owns them).
# Used by :func:`sanitize_raw_result` as defense-in-depth behind ``extra=forbid``.
FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "edge",
        "edge_pct",
        "ev",
        "ev_pct",
        "probability",
        "true_prob",
        "calibrated_prob",
        "fair_value",
        "fair_line",
        "kelly",
        "kelly_fraction",
        "stake",
        "units",
        "grade",
        "bet_this_now",
    }
)


def sanitize_raw_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop any forbidden numeric-authority keys before validation.

    ``EnrichmentResult`` already forbids unknown keys, so a forbidden key would
    raise on validation; stripping first turns a provider slip into a clean
    artifact rather than a hard failure, while still never surfacing the number.
    """
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k not in FORBIDDEN_RESULT_KEYS}


class EnrichmentRecord(BaseModel):
    """A stored enrichment row (read shape for the API)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    trace_id: str
    trace_type: str | None = None
    league: str | None = None
    market: str | None = None
    status: str
    depth: str = "deep"
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    context_pack: dict[str, Any] | None = None
    result: EnrichmentResult | None = None
    narrative_md: str | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class EnrichmentFeedback(BaseModel):
    """Operator feedback on a stored enrichment (👍/👎 + optional note)."""

    model_config = ConfigDict(extra="forbid")

    user_rating: int = Field(ge=-1, le=1)  # -1 down, 0 neutral, +1 up
    feedback_text: str | None = None
