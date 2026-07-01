"""Shared calibration-market routing policy."""

from __future__ import annotations

# Planes whose profile market is the plane name itself (identity routing). game and
# draw keep their explicit handling below for the legacy ``market="draw"`` override.
_IDENTITY_PLANES = {"prop", "cover", "over", "under"}


def calibration_market_for_plane(plane: str, *, market: str | None = None) -> str:
    """Map an evaluation plane to the profile market used for calibration.

    Planes and their markets: game→game, draw→draw, prop→prop, and the
    point-spread/total planes cover→cover, over→over, under→under (issue #28
    Wave 3). Unknown planes fall back to the game market.
    """
    if plane == "draw" or market == "draw":
        return "draw"
    if plane in _IDENTITY_PLANES:
        return plane
    return "game"
