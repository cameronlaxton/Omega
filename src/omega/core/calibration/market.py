"""Shared calibration-market routing policy."""

from __future__ import annotations


def calibration_market_for_plane(plane: str, *, market: str | None = None) -> str:
    """Map an evaluation plane to the profile market used for calibration."""
    if plane == "prop":
        return "prop"
    if plane == "draw" or market == "draw":
        return "draw"
    return "game"
