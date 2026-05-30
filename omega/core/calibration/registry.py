"""
Calibration profile registry — JSON file storage with promotion workflow.

Stores versioned CalibrationProfiles and provides league-based lookup.
At most ONE production profile exists per (league, context_slice) pair at any time.

context_slice is an optional sub-population label ('playoff', 'regular',
'back_to_back', etc.).  None means the base profile covering all contexts.
get_production(league, context_slice) performs an exact-match lookup first,
then falls back to the base profile (context_slice=None) so callers always
get the best available profile.

Storage: omega/core/calibration/profiles.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omega.core.calibration.profiles import (
    CalibrationProfile,
    ProfileStatus,
)

UTC = timezone.utc

logger = logging.getLogger("omega.core.calibration.registry")

_DEFAULT_PATH = Path(__file__).parent / "profiles.json"
_SCHEMA_VERSION = 1


class CalibrationRegistry:
    """Stores versioned calibration profiles with promotion workflow.

    Storage is a single JSON file. Each call re-reads the file (no
    in-memory caching) to avoid stale-state bugs. This is acceptable
    for the expected ~10-50 profiles.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH

    def _load(self) -> dict[str, Any]:
        """Read the registry file. Returns empty structure if missing."""
        if not self._path.exists():
            return {"schema_version": _SCHEMA_VERSION, "profiles": []}
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read profile registry at %s: %s", self._path, exc)
            return {"schema_version": _SCHEMA_VERSION, "profiles": []}

    def _save(self, data: dict[str, Any]) -> None:
        """Write the registry file atomically (write tmp, rename)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False)
        # Atomic rename (works on POSIX; on Windows, replaces if exists)
        if os.name == "nt":
            # Windows: os.replace is atomic
            os.replace(str(tmp_path), str(self._path))
        else:
            os.rename(str(tmp_path), str(self._path))

    def register(self, profile: CalibrationProfile) -> None:
        """Add a profile to the registry. Validates profile_id uniqueness."""
        data = self._load()
        existing_ids = {p["profile_id"] for p in data["profiles"]}
        if profile.profile_id in existing_ids:
            raise ValueError(f"Profile ID already exists: {profile.profile_id}")
        data["profiles"].append(profile.model_dump())
        self._save(data)
        logger.info(
            "Registered profile %s (league=%s, method=%s)",
            profile.profile_id,
            profile.league,
            profile.method,
        )

    def get_production(
        self,
        league: str,
        context_slice: str | None = None,
        market: str = "game",
    ) -> CalibrationProfile | None:
        """Return the active production profile for (league, context_slice, market).

        Lookup order:
        1. Exact match on (league, market, context_slice, PRODUCTION).
        2. If context_slice is not None and no exact match: fall back to the
           base profile (context_slice=None) for the same (league, market).
        3. If market is not 'game' and no match: fall back to the 'game'-market
           profile for the league (the historical behaviour, where draw probs
           reuse the game profile).
        4. Returns None if no production profile exists at all.

        Legacy profiles stored before the ``market`` field existed are treated
        as ``market == "game"``.
        """
        league_uc = league.upper()
        resolved = self._resolve_market(league_uc, context_slice, market)
        if resolved is None and market != "game":
            resolved = self._resolve_market(league_uc, context_slice, "game")
        return resolved

    def _resolve_market(
        self,
        league_uc: str,
        context_slice: str | None,
        market: str,
    ) -> CalibrationProfile | None:
        """Resolve the production profile for one explicit market, applying the
        context_slice -> base fallback within that market."""
        data = self._load()
        base_profile: CalibrationProfile | None = None
        slice_profile: CalibrationProfile | None = None

        for p in data["profiles"]:
            if (
                p.get("league", "").upper() != league_uc
                or p.get("status") != ProfileStatus.PRODUCTION.value
                or (p.get("market") or "game") != market
            ):
                continue
            p_slice = p.get("context_slice")  # None for base profiles
            if context_slice is not None and p_slice == context_slice:
                slice_profile = CalibrationProfile(**p)
            elif p_slice is None:
                base_profile = CalibrationProfile(**p)

        if context_slice is not None and slice_profile is not None:
            return slice_profile
        return base_profile

    def get_profile(self, profile_id: str) -> CalibrationProfile | None:
        """Retrieve a profile by ID."""
        data = self._load()
        for p in data["profiles"]:
            if p["profile_id"] == profile_id:
                return CalibrationProfile(**p)
        return None

    def promote(self, profile_id: str) -> None:
        """Promote a candidate to production. Archives incumbent for same league."""
        data = self._load()
        target = None
        for p in data["profiles"]:
            if p["profile_id"] == profile_id:
                target = p
                break

        if target is None:
            raise ValueError(f"Profile not found: {profile_id}")
        if target["status"] != ProfileStatus.CANDIDATE.value:
            raise ValueError(
                f"Cannot promote profile with status={target['status']} "
                f"(must be {ProfileStatus.CANDIDATE.value})"
            )

        league = target["league"]
        target_slice = target.get("context_slice")  # None for base profiles
        target_market = target.get("market") or "game"
        now = datetime.now(UTC).isoformat()

        # Archive existing production profile for the same
        # (league, context_slice, market). A playoff profile never archives the
        # base profile, a draw profile never archives the game profile, etc.
        for p in data["profiles"]:
            if (
                p.get("league", "").upper() == league.upper()
                and p.get("status") == ProfileStatus.PRODUCTION.value
                and p.get("context_slice") == target_slice
                and (p.get("market") or "game") == target_market
            ):
                p["status"] = ProfileStatus.ARCHIVED.value
                logger.info(
                    "Archived incumbent profile %s for league=%s slice=%s market=%s",
                    p["profile_id"],
                    league,
                    target_slice,
                    target_market,
                )

        # Promote the target
        target["status"] = ProfileStatus.PRODUCTION.value
        target["promoted_at"] = now
        self._save(data)
        logger.info("Promoted profile %s to production for league %s", profile_id, league)

    def reject(self, profile_id: str, reason: str) -> None:
        """Reject a candidate profile with a documented reason."""
        data = self._load()
        for p in data["profiles"]:
            if p["profile_id"] == profile_id:
                p["status"] = ProfileStatus.REJECTED.value
                p["rejected_at"] = datetime.now(UTC).isoformat()
                p["reject_reason"] = reason
                self._save(data)
                logger.info("Rejected profile %s: %s", profile_id, reason)
                return
        raise ValueError(f"Profile not found: {profile_id}")

    def list_profiles(
        self,
        league: str | None = None,
        status: str | None = None,
        context_slice: str | None = ...,  # type: ignore[assignment]
    ) -> list[CalibrationProfile]:
        """List profiles with optional filters.

        context_slice=... (default sentinel) means no filter — all slices returned.
        context_slice=None means return only base profiles (slice is None).
        context_slice='playoff' means return only profiles with that slice label.
        """
        _UNSET = ...
        data = self._load()
        results = []
        for p in data["profiles"]:
            if league and p.get("league", "").upper() != league.upper():
                continue
            if status and p.get("status") != status:
                continue
            if context_slice is not _UNSET and p.get("context_slice") != context_slice:
                continue
            results.append(CalibrationProfile(**p))
        return results
