"""
Strategy Registry — register, version, query, promote, reject strategies.

The registry is the single source of truth for all strategy metadata.
Storage is pluggable: starts with in-memory + JSON file, can be swapped
for Postgres/Redis later.

Invariants:
- A strategy_id + version pair is globally unique
- Strategies are immutable once registered (new params = new version)
- Only one version of a strategy can be in PRODUCTION status at a time
- Promotion requires a passing backtest
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from omega.strategy.models import (
    BacktestResult,
    PromotionRecord,
    StrategyEntry,
    StrategyStatus,
    StrategyType,
)

logger = logging.getLogger("omega.strategy.registry")


class StrategyRegistry:
    """In-memory strategy registry with optional JSON file persistence."""

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self._strategies: Dict[str, StrategyEntry] = {}  # key: "{id}:v{version}"
        self._storage_path = storage_path
        if storage_path and os.path.exists(storage_path):
            self._load_from_file(storage_path)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        strategy_id: str,
        name: str,
        description: str = "",
        strategy_type: StrategyType = StrategyType.GAME_EDGE,
        leagues: Optional[List[str]] = None,
        markets: Optional[List[str]] = None,
        edge_threshold: float = 0.03,
        confidence_tiers: Optional[List[str]] = None,
        params: Optional[Dict] = None,
        created_by: str = "system",
    ) -> StrategyEntry:
        """Register a new strategy or a new version of an existing one.

        If strategy_id already exists, auto-increments version.
        Returns the registered StrategyEntry.
        """
        # Determine version
        existing_versions = self._versions_of(strategy_id)
        version = max(existing_versions) + 1 if existing_versions else 1

        entry = StrategyEntry(
            strategy_id=strategy_id,
            version=version,
            name=name,
            description=description,
            strategy_type=strategy_type,
            status=StrategyStatus.CANDIDATE,
            leagues=leagues or [],
            markets=markets or [],
            edge_threshold=edge_threshold,
            confidence_tiers=confidence_tiers or ["A", "B"],
            params=params or {},
            created_by=created_by,
        )

        key = self._key(strategy_id, version)
        self._strategies[key] = entry
        self._persist()
        logger.info("Registered strategy %s v%d", strategy_id, version)
        return entry

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, strategy_id: str, version: Optional[int] = None) -> Optional[StrategyEntry]:
        """Get a specific strategy. If version is None, returns latest."""
        if version is not None:
            return self._strategies.get(self._key(strategy_id, version))

        versions = self._versions_of(strategy_id)
        if not versions:
            return None
        return self._strategies[self._key(strategy_id, max(versions))]

    def get_production(self, strategy_id: str) -> Optional[StrategyEntry]:
        """Get the production version of a strategy (if any)."""
        for entry in self._strategies.values():
            if entry.strategy_id == strategy_id and entry.status == StrategyStatus.PRODUCTION:
                return entry
        return None

    def list_all(
        self,
        status: Optional[StrategyStatus] = None,
        strategy_type: Optional[StrategyType] = None,
        league: Optional[str] = None,
    ) -> List[StrategyEntry]:
        """List strategies with optional filters."""
        results = list(self._strategies.values())

        if status is not None:
            results = [s for s in results if s.status == status]
        if strategy_type is not None:
            results = [s for s in results if s.strategy_type == strategy_type]
        if league is not None:
            league_upper = league.upper()
            results = [s for s in results if league_upper in [l.upper() for l in s.leagues]]

        return sorted(results, key=lambda s: (s.strategy_id, s.version))

    def list_production(self) -> List[StrategyEntry]:
        """List all strategies currently in production."""
        return self.list_all(status=StrategyStatus.PRODUCTION)

    # ------------------------------------------------------------------
    # Backtest recording
    # ------------------------------------------------------------------

    def record_backtest(
        self,
        strategy_id: str,
        version: int,
        result: BacktestResult,
    ) -> StrategyEntry:
        """Record a backtest result against a strategy version."""
        key = self._key(strategy_id, version)
        entry = self._strategies.get(key)
        if entry is None:
            raise ValueError(f"Strategy {strategy_id} v{version} not found")

        entry.backtest_results.append(result)
        entry.latest_roi = result.roi_pct
        entry.latest_win_rate = result.win_rate
        entry.latest_clv = result.avg_closing_line_value

        if result.passed:
            entry.status = StrategyStatus.STAGING
        else:
            entry.status = StrategyStatus.REJECTED

        self._persist()
        logger.info(
            "Backtest recorded for %s v%d: passed=%s roi=%.1f%%",
            strategy_id, version, result.passed, result.roi_pct,
        )
        return entry

    # ------------------------------------------------------------------
    # Promotion / Rejection
    # ------------------------------------------------------------------

    def promote(
        self,
        strategy_id: str,
        version: int,
        reason: str = "Passed backtest criteria",
        decided_by: str = "system",
        backtest_run_id: Optional[str] = None,
    ) -> StrategyEntry:
        """Promote a strategy to production.

        Demotes any existing production version to ARCHIVED.
        Requires strategy to be in STAGING status.
        """
        key = self._key(strategy_id, version)
        entry = self._strategies.get(key)
        if entry is None:
            raise ValueError(f"Strategy {strategy_id} v{version} not found")

        if entry.status != StrategyStatus.STAGING:
            raise ValueError(
                f"Cannot promote {strategy_id} v{version}: "
                f"status is {entry.status.value}, must be 'staging'"
            )

        # Demote current production version
        current_prod = self.get_production(strategy_id)
        if current_prod is not None:
            current_prod.status = StrategyStatus.ARCHIVED
            current_prod.promotion_history.append(PromotionRecord(
                strategy_id=strategy_id,
                strategy_version=current_prod.version,
                action="archive",
                from_status=StrategyStatus.PRODUCTION.value,
                to_status=StrategyStatus.ARCHIVED.value,
                reason=f"Superseded by v{version}",
                decided_by=decided_by,
            ))

        # Promote
        from_status = entry.status.value
        entry.status = StrategyStatus.PRODUCTION
        entry.promotion_history.append(PromotionRecord(
            strategy_id=strategy_id,
            strategy_version=version,
            action="promote",
            from_status=from_status,
            to_status=StrategyStatus.PRODUCTION.value,
            reason=reason,
            decided_by=decided_by,
            backtest_run_id=backtest_run_id,
        ))

        self._persist()
        logger.info("Promoted %s v%d to production", strategy_id, version)
        return entry

    def reject(
        self,
        strategy_id: str,
        version: int,
        reason: str,
        decided_by: str = "system",
    ) -> StrategyEntry:
        """Reject a strategy."""
        key = self._key(strategy_id, version)
        entry = self._strategies.get(key)
        if entry is None:
            raise ValueError(f"Strategy {strategy_id} v{version} not found")

        from_status = entry.status.value
        entry.status = StrategyStatus.REJECTED
        entry.promotion_history.append(PromotionRecord(
            strategy_id=strategy_id,
            strategy_version=version,
            action="reject",
            from_status=from_status,
            to_status=StrategyStatus.REJECTED.value,
            reason=reason,
            decided_by=decided_by,
        ))

        self._persist()
        logger.info("Rejected %s v%d: %s", strategy_id, version, reason)
        return entry

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _key(strategy_id: str, version: int) -> str:
        return f"{strategy_id}:v{version}"

    def _versions_of(self, strategy_id: str) -> List[int]:
        return [
            e.version for e in self._strategies.values()
            if e.strategy_id == strategy_id
        ]

    def _persist(self) -> None:
        if not self._storage_path:
            return
        try:
            path = Path(self._storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                k: v.model_dump() for k, v in self._strategies.items()
            }
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            logger.warning("Failed to persist strategy registry", exc_info=True)

    def _load_from_file(self, path: str) -> None:
        try:
            raw = json.loads(Path(path).read_text())
            for key, entry_data in raw.items():
                self._strategies[key] = StrategyEntry(**entry_data)
            logger.info("Loaded %d strategies from %s", len(self._strategies), path)
        except Exception:
            logger.warning("Failed to load strategy registry from %s", path, exc_info=True)
