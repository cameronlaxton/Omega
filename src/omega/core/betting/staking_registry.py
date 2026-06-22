"""Versioned registry that selects a :class:`StakingPolicy` per (league, market).

Structurally mirrors :class:`omega.core.calibration.registry.CalibrationRegistry`:
a single JSON file stores versioned entries, at most ONE production entry exists
per ``(league, market)`` at a time, and lookup falls back from the exact slot to
broader wildcards. The decisive difference from calibration: ``get_production``
is **never None** — when nothing is registered it returns the legacy default
policy (:func:`default_policy`), so call sites always have a deterministic policy.

This is the single shared policy-selection path (the staking analogue of
``CalibrationRegistry.get_production``). The fail-closed promotion *gate* for
staking policies lands in a later PR (``staking_gate``); :meth:`activate` here is
the mechanical status transition only, analogous to
``CalibrationRegistry._apply_promotion``.

Bounded-autonomy note (AGENTS.md): policies are deterministic engine code; the
registry only chooses which one applies. No protected value is authored here.

Storage: omega/core/betting/staking_policies.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omega.core.betting.staking_policy import (
    CappedFractionalKelly,
    FlatKelly,
    FractionalKellyByTier,
    StakingPolicy,
)

UTC = timezone.utc

logger = logging.getLogger("omega.core.betting.staking_registry")

_DEFAULT_PATH = Path(__file__).parent / "staking_policies.json"
_SCHEMA_VERSION = 1

#: Wildcard token for the league / market dimensions.
ANY = "*"

STATUS_CANDIDATE = "candidate"
STATUS_PRODUCTION = "production"
STATUS_ARCHIVED = "archived"
STATUS_REJECTED = "rejected"

#: policy_id -> concrete policy class. Explicit (not auto-discovered) so an
#: unknown id fails loudly rather than silently picking a default.
_POLICY_FACTORIES: dict[str, type[StakingPolicy]] = {
    "fractional_kelly_by_tier": FractionalKellyByTier,
    "flat_kelly": FlatKelly,
    "capped_fractional_kelly": CappedFractionalKelly,
}


def default_policy() -> StakingPolicy:
    """The policy used when nothing is registered for a (league, market).

    The legacy tier-scaled fractional Kelly, so behavior with an empty registry
    is identical to the pre-registry ``recommend_stake``.
    """
    return FractionalKellyByTier()


def build_policy(policy_id: str, params: dict[str, Any] | None = None) -> StakingPolicy:
    """Reconstruct a policy instance from its id + persisted constructor params."""
    cls = _POLICY_FACTORIES.get(policy_id)
    if cls is None:
        raise ValueError(
            f"Unknown staking policy_id: {policy_id!r} (known: {sorted(_POLICY_FACTORIES)})"
        )
    return cls(**(params or {}))


@dataclass
class StakingPolicyEntry:
    """One versioned, registered policy bound to a (league, market) slot."""

    entry_id: str
    policy_id: str
    version: int
    league: str  # ANY ("*") matches any league
    market: str  # ANY ("*") matches any market
    status: str = STATUS_CANDIDATE
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    promoted_at: str | None = None
    archived_at: str | None = None
    rejected_at: str | None = None
    reject_reason: str | None = None

    def to_policy(self) -> StakingPolicy:
        return build_policy(self.policy_id, self.params)

    @staticmethod
    def make_entry_id(policy_id: str, league: str, market: str, version: int) -> str:
        return f"{policy_id}_{league}_{market}_v{version}".lower()


class StakingRegistry:
    """Stores versioned staking-policy entries with (league, market) selection.

    Storage is a single JSON file, re-read on each call (no caching) to avoid
    stale state — acceptable for the handful of policies expected.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH

    # ------------------------------------------------------------------ I/O
    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"schema_version": _SCHEMA_VERSION, "entries": []}
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read staking registry at %s: %s", self._path, exc)
            return {"schema_version": _SCHEMA_VERSION, "entries": []}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False)
        os.replace(str(tmp_path), str(self._path))  # atomic on POSIX and Windows

    # --------------------------------------------------------------- writes
    def register(self, entry: StakingPolicyEntry) -> None:
        """Add an entry. Validates entry_id uniqueness and policy_id is known."""
        build_policy(entry.policy_id, entry.params)  # fail loud on bad policy/params
        data = self._load()
        existing = {e["entry_id"] for e in data["entries"]}
        if entry.entry_id in existing:
            raise ValueError(f"Entry ID already exists: {entry.entry_id}")
        data["entries"].append(asdict(entry))
        self._save(data)
        logger.info(
            "Registered staking entry %s (league=%s market=%s status=%s)",
            entry.entry_id,
            entry.league,
            entry.market,
            entry.status,
        )

    def activate(self, entry_id: str) -> None:
        """Mechanically promote an entry to PRODUCTION for its (league, market).

        Archives the existing production entry for the SAME (league, market) and
        flips the target to production. INTERNAL/pre-gate: this performs no gate
        evaluation; the fail-closed staking gate (a later PR) is the safe public
        entry point once it exists.
        """
        data = self._load()
        target = next((e for e in data["entries"] if e["entry_id"] == entry_id), None)
        if target is None:
            raise ValueError(f"Entry not found: {entry_id}")

        now = datetime.now(UTC).isoformat()
        for e in data["entries"]:
            if (
                e["league"] == target["league"]
                and e["market"] == target["market"]
                and e["status"] == STATUS_PRODUCTION
                and e["entry_id"] != entry_id
            ):
                e["status"] = STATUS_ARCHIVED
                e["archived_at"] = now
                logger.info(
                    "Archived staking entry %s for league=%s market=%s",
                    e["entry_id"],
                    e["league"],
                    e["market"],
                )
        target["status"] = STATUS_PRODUCTION
        target["promoted_at"] = now
        self._save(data)
        logger.info("Activated staking entry %s to production", entry_id)

    def reject(self, entry_id: str, reason: str) -> None:
        data = self._load()
        for e in data["entries"]:
            if e["entry_id"] == entry_id:
                e["status"] = STATUS_REJECTED
                e["rejected_at"] = datetime.now(UTC).isoformat()
                e["reject_reason"] = reason
                self._save(data)
                logger.info("Rejected staking entry %s: %s", entry_id, reason)
                return
        raise ValueError(f"Entry not found: {entry_id}")

    # ---------------------------------------------------------------- reads
    def get_entry(self, entry_id: str) -> StakingPolicyEntry | None:
        data = self._load()
        for e in data["entries"]:
            if e["entry_id"] == entry_id:
                return StakingPolicyEntry(**e)
        return None

    def list_entries(
        self,
        league: str | None = None,
        market: str | None = None,
        status: str | None = None,
    ) -> list[StakingPolicyEntry]:
        data = self._load()
        out: list[StakingPolicyEntry] = []
        for e in data["entries"]:
            if league is not None and e["league"].upper() != league.upper():
                continue
            if market is not None and e["market"] != market:
                continue
            if status is not None and e["status"] != status:
                continue
            out.append(StakingPolicyEntry(**e))
        return out

    def get_production(self, league: str, market: str = "game") -> StakingPolicy:
        """Return the active policy for (league, market) — never None.

        League matching is case-insensitive (mirrors ``CalibrationRegistry``);
        market is matched exactly (canonically lowercase: game/prop/draw).

        Lookup order (first production entry wins):
        1. exact (league, market)
        2. (league, ANY market)
        3. (ANY league, market)
        4. (ANY, ANY)
        5. :func:`default_policy` (legacy fractional Kelly).
        """
        data = self._load()
        prod = [e for e in data["entries"] if e["status"] == STATUS_PRODUCTION]

        def find(lg: str, mk: str) -> StakingPolicy | None:
            for e in prod:
                if e["league"].upper() == lg.upper() and e["market"] == mk:
                    return StakingPolicyEntry(**e).to_policy()
            return None

        for lg, mk in ((league, market), (league, ANY), (ANY, market), (ANY, ANY)):
            policy = find(lg, mk)
            if policy is not None:
                return policy
        return default_policy()
