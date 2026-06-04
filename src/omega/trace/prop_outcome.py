"""Shared player-prop outcome result derivation."""

from __future__ import annotations


def normalize_prop_side(side: str) -> str:
    side_norm = side.lower().strip()
    if side_norm not in ("over", "under"):
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")
    return side_norm


def derive_prop_outcome_result(
    *,
    stat_value: float,
    line: float,
    side: str,
    void: bool = False,
) -> tuple[str, str]:
    """Return ``(result, normalized_side)`` for a prop outcome row."""
    side_norm = normalize_prop_side(side)
    if void:
        return "void", side_norm
    if stat_value == line:
        return "push", side_norm
    if (side_norm == "over" and stat_value > line) or (
        side_norm == "under" and stat_value < line
    ):
        return "win", side_norm
    return "loss", side_norm
