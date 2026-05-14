"""
Calibration profile registry — JSON file storage with promotion workflow.

Stores versioned CalibrationProfiles and provides league-based lookup.
At most ONE production profile exists per league at any time.

Storage: omega/core/calibration/profiles.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from omega.core.calibration.profiles import (
    CalibrationProfile,
    ProfileStatus,
)

logger = logging.getLogger("omega.core.calibration.registry")

_DEFAULT_PATH = Path(__file__).parent / "profiles.json"
_SCHEMA_VERSION = 1


class CalibrationRegistry:
    """Stores versioned calibration profiles with promotion workflow.

    Storage is a single JSON file. Each call re-reads the file (no
    in-memory caching) to avoid stale-state bugs. This is acceptable
    for the expected ~10-50 profiles.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH

    def _load(self) -> Dict[str, Any]:
        """Read the registry file. Returns empty structure if missing."""
        if not self._path.exists():
            return {"schema_version": _SCHEMA_VERSION, "profiles": []}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read profile registry at %s: %s", self._path, exc)
            return {"schema_version": _SCHEMA_VERSION, "profiles": []}

    def _save(self, data: Dict[str, Any]) -> None:
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
        logger.info("Registered profile %s (league=%s, method=%s)",
                     profile.profile_id, profile.league, profile.method)

    def get_production(self, league: str) -> Optional[CalibrationProfile]:
        """Return the active production profile for a league, or None."""
        data = self._load()
        for p in data["profiles"]:
            if (p.get("league", "").upper() == league.upper()
                    and p.get("status") == ProfileStatus.PRODUCTION.value):
                return CalibrationProfile(**p)
        return None

    def get_profile(self, profile_id: str) -> Optional[CalibrationProfile]:
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
        now = datetime.now(timezone.utc).isoformat()

        # Archive existing production profile for this league
        for p in data["profiles"]:
            if (p.get("league", "").upper() == league.upper()
                    and p.get("status") == ProfileStatus.PRODUCTION.value):
                p["status"] = ProfileStatus.ARCHIVED.value
                logger.info("Archived incumbent profile %s for league %s",
                            p["profile_id"], league)

        # Promote the target
        target["status"] = ProfileStatus.PRODUCTION.value
        target["promoted_at"] = now
        self._save(data)
        logger.info("Promoted profile %s to production for league %s",
                     profile_id, league)

    def reject(self, profile_id: str, reason: str) -> None:
        """Reject a candidate profile with a documented reason."""
        data = self._load()
        for p in data["profiles"]:
            if p["profile_id"] == profile_id:
                p["status"] = ProfileStatus.REJECTED.value
                p["rejected_at"] = datetime.now(timezone.utc).isoformat()
                p["reject_reason"] = reason
                self._save(data)
                logger.info("Rejected profile %s: %s", profile_id, reason)
                return
        raise ValueError(f"Profile not found: {profile_id}")

    def list_profiles(
        self,
        league: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[CalibrationProfile]:
        """List profiles with optional league and status filters."""
        data = self._load()
        results = []
        for p in data["profiles"]:
            if league and p.get("league", "").upper() != league.upper():
                continue
            if status and p.get("status") != status:
                continue
            results.append(CalibrationProfile(**p))
        return results
