"""Recommendation Lab feature gate.

The recommendation-era surfaces (edge scanner, raw trace detail, similar
historical spots, wager ledger, CLV) are operator lab views, not primary
product. Both their HTML pages and JSON APIs return 404 unless the operator
explicitly enables the lab with ``OMEGA_ENABLE_RECOMMENDATION_LAB=1``.

The flag is read per-request (not cached at import) so tests and operators can
toggle it without restarting the console. Enabling the lab never widens
``output_mode`` — it only restores access to the legacy views.
"""

from __future__ import annotations

import os

from fastapi import HTTPException

ENV_RECOMMENDATION_LAB = "OMEGA_ENABLE_RECOMMENDATION_LAB"


def recommendation_lab_enabled() -> bool:
    return os.environ.get(ENV_RECOMMENDATION_LAB) == "1"


def require_recommendation_lab() -> None:
    """FastAPI dependency: 404 (not 403) so gated routes are indistinguishable
    from absent ones when the lab is off."""
    if not recommendation_lab_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
