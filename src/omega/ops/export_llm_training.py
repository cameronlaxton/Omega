"""Export protected-field-safe LLM training examples from persisted traces.

This exporter is intentionally narrow. It trains evidence extraction, source
arbitration, and downgrade/refusal language only; it must not train an LLM to
emit engine-owned probabilities, edge/EV, Kelly, units, tiers, prices, or trace
IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("export_llm_training")

PROTECTED_FIELD_NAMES = frozenset(
    {
        "trace_id",
        "run_id",
        "edge_pct",
        "ev_pct",
        "kelly_fraction",
        "units",
        "recommended_units",
        "confidence_tier",
        "fair_price",
        "no_vig_price",
        "model_probability",
        "model_prob",
        "market_probability",
        "calibrated_prob",
        "over_prob",
        "under_prob",
        "home_win_prob",
        "away_win_prob",
        "draw_prob",
        "home_cover_prob",
        "away_cover_prob",
        "spread_coverage_prob",
        "predictions",
        "recommendations",
        "recommendation",
        "best_bet",
        "edges",
        "edge_over",
        "edge_under",
        "odds_snapshot",
        "market_context",
        "bet_records",
        "bet_ledger",
    }
)

PROTECTED_NAME_FRAGMENTS = (
    "prob",
    "price",
    "odds",
    "edge",
    "ev_",
    "kelly",
    "unit",
    "stake",
    "payout",
)

PROTECTED_TEXT_TOKENS = tuple(sorted(PROTECTED_FIELD_NAMES, key=len, reverse=True))


def _is_protected_key(key: Any) -> bool:
    name = str(key).lower()
    if name in PROTECTED_FIELD_NAMES:
        return True
    return any(fragment in name for fragment in PROTECTED_NAME_FRAGMENTS)


def redact_text(text: str) -> str:
    """Remove explicit protected field names from free-form text."""
    redacted = text
    for token in PROTECTED_TEXT_TOKENS:
        redacted = re.sub(rf"\b{re.escape(token)}\b", "[REDACTED_FIELD]", redacted, flags=re.I)
    return redacted


def redact_protected(value: Any) -> Any:
    """Recursively remove engine-owned or betting-numeric fields."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if _is_protected_key(key):
                continue
            out[key] = redact_protected(item)
        return out
    if isinstance(value, list):
        return [redact_protected(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _trace_ref(trace: dict[str, Any]) -> str:
    raw = str(trace.get("trace_id") or trace.get("run_id") or trace.get("prompt") or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _quality(trace: dict[str, Any]) -> float | None:
    tq = trace.get("trace_quality") or trace.get("quality_gate") or {}
    for source in (tq, trace):
        value = source.get("aggregate_quality") if isinstance(source, dict) else None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _reasoning_text(trace: dict[str, Any]) -> str | None:
    narrative = str(trace.get("reasoning_narrative") or "").strip()
    downgrade = str(trace.get("reasoning_downgrade_rationale") or "").strip()
    tq = trace.get("trace_quality") or {}
    downgrades = tq.get("downgrades") or trace.get("downgrades") or []

    parts: list[str] = []
    if narrative:
        parts.append(f"Reasoning narrative: {redact_text(narrative)}")
    if downgrade:
        parts.append(f"Downgrade rationale: {redact_text(downgrade)}")
    if downgrades:
        redacted = redact_protected(downgrades)
        parts.append(f"Downgrades: {json.dumps(redacted, sort_keys=True)}")
    return "\n".join(parts) if parts else None


def _request_payload(trace: dict[str, Any]) -> dict[str, Any]:
    """Compact, redacted input the LLM can learn source arbitration from."""
    input_snapshot = trace.get("input_snapshot") or {}
    reasoning_inputs = trace.get("reasoning_inputs") or {}
    payload = {
        "prompt": trace.get("prompt"),
        "league": trace.get("league"),
        "kind": trace.get("kind"),
        "matchup": trace.get("matchup"),
        "reasoning_inputs": reasoning_inputs,
        "input_snapshot": {
            "evidence": input_snapshot.get("evidence") or [],
            "home_team": input_snapshot.get("home_team"),
            "away_team": input_snapshot.get("away_team"),
            "player_name": input_snapshot.get("player_name"),
            "prop_type": input_snapshot.get("prop_type"),
            "game_date": input_snapshot.get("game_date"),
        },
    }
    return redact_protected(payload)


def training_record(trace: dict[str, Any]) -> dict[str, Any] | None:
    assistant = _reasoning_text(trace)
    if not assistant:
        return None

    user_payload = _request_payload(trace)
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Omega's evidence and downgrade assistant. Extract "
                    "structured reasoning, source-arbitration notes, and downgrade "
                    "rationales only. Do not emit probabilities, edges, EV, Kelly, "
                    "units, confidence tiers, fair prices, odds, or trace IDs."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, sort_keys=True),
            },
            {
                "role": "assistant",
                "content": assistant,
            },
        ],
        "metadata": {
            "trace_ref": _trace_ref(trace),
            "league": trace.get("league"),
            "kind": trace.get("kind"),
            "quality": _quality(trace),
            "calibration_eligible": (
                (trace.get("trace_quality") or {}).get("calibration_eligible")
            ),
        },
    }


def _contains_protected_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_is_protected_key(key) or _contains_protected_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_protected_key(item) for item in value)
    if isinstance(value, str):
        return any(re.search(rf"\b{re.escape(token)}\b", value, flags=re.I) for token in PROTECTED_TEXT_TOKENS)
    return False


def export_records(
    traces: Iterable[dict[str, Any]],
    *,
    min_quality: float | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trace in traces:
        quality = _quality(trace)
        if min_quality is not None and (quality is None or quality < min_quality):
            continue
        record = training_record(trace)
        if record is None:
            continue
        for message in record["messages"]:
            if message.get("role") == "system":
                continue
            if _contains_protected_key(message):
                raise RuntimeError("protected field leaked into training messages")
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export protected-field-safe Omega LLM training JSONL."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--db", type=str, default=None, help="SQLite path")
    parser.add_argument("--league", default=None, help="Optional league filter")
    parser.add_argument("--limit", type=int, default=1000, help="Max traces to inspect")
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.7,
        help="Minimum trace_quality.aggregate_quality; use -1 to disable",
    )
    parser.add_argument(
        "--include-ungraded",
        action="store_true",
        help="Include traces without attached outcomes",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.limit < 1:
        logger.error("--limit must be positive")
        return 2

    min_quality = None if args.min_quality < 0 else args.min_quality
    store = TraceStore(db_path=args.db, read_only=True)
    log_effective_db(store, logger)
    try:
        traces = store.query_traces(
            league=args.league.upper() if args.league else None,
            has_outcome=None if args.include_ungraded else True,
            limit=args.limit,
        )
    finally:
        store.close()

    records = export_records(traces, min_quality=min_quality)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    logger.info("Wrote %d training record(s) to %s", len(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
