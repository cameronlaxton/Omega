"""Deterministic seed derivation for canonical Omega analyze requests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

_ANALYSIS_IDENTITY_EXCLUDE = {
    "seed",
    "session_id",
    "bankroll",
    "odds",
    "odds_over",
    "odds_under",
    "markets",
    "odds_snapshot",
    "market_snapshots",
    "closing_line",
    "closing_lines",
}


def sanitize_for_analysis_identity(value: Any) -> Any:
    """Drop runtime and market-volatility fields from analysis identity."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(k): sanitize_for_analysis_identity(v)
            for k, v in value.items()
            if str(k) not in _ANALYSIS_IDENTITY_EXCLUDE
        }
    if isinstance(value, list):
        return [sanitize_for_analysis_identity(item) for item in value]
    return value


def canonical_seed_prompt(value: Any) -> str:
    """Canonical UTF-8 text hashed for fallback script and batch seeds."""
    payload = sanitize_for_analysis_identity(value)
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def stable_analysis_hash(value: Any, *, hex_chars: int = 8) -> str:
    """Stable content hash for analysis identity, excluding runtime wrappers."""
    encoded = canonical_seed_prompt(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:hex_chars]


def _extract_date(value: Any) -> str | None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return None
    raw_date = value.get("game_date")
    if raw_date:
        return str(raw_date)[:10]
    game_context = value.get("game_context")
    if isinstance(game_context, dict) and game_context.get("game_date"):
        return str(game_context["game_date"])[:10]
    return None


def derive_seed_from_prompt_date(prompt: str, date: str) -> int:
    """Return ``int.from_bytes(sha256(f"{prompt}|{date}").digest()[:4], "big")``."""
    encoded = f"{prompt}|{date}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def derive_seed_from_request(value: Any, *, date: str | None = None) -> int:
    """Derive a reproducible 32-bit seed from stable request content and date."""
    prompt = canonical_seed_prompt(value)
    effective_date = (date or _extract_date(value) or "undated")[:10]
    return derive_seed_from_prompt_date(prompt, effective_date)
