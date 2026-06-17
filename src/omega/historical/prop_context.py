"""As-of player-prop context builder for historical replay.

The replay engine already consumes ``ReplayDataset.prop_context`` keyed as
``<event_key>|<player>|<stat_type>``. This module builds that artifact from
historical player-stat rows while enforcing the important invariant: only rows
strictly before the event decision time can enter the context.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median, stdev
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from omega.historical.adapters.csv_player_stats import PlayerStatObservation
from omega.historical.contracts import HistoricalEvent, HistoricalPropMarket

ARTIFACT_SCHEMA_VERSION = 1


def prop_context_key(event_key: str, player_name: str, stat_type: str) -> str:
    """Stable prop-context lookup key used by replay."""
    return f"{event_key}|{player_name}|{stat_type}"


def _dt(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _sample_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 4:
        return "1-4"
    if n <= 9:
        return "5-9"
    return "10+"


def _trend(values: list[float]) -> float | None:
    """Simple recent-vs-earlier trend over the selected chronological window."""
    if len(values) < 2:
        return None
    split = max(1, len(values) // 2)
    early = values[:split]
    late = values[split:]
    if not late:
        return None
    return mean(late) - mean(early)


class PropContextBuildConfig(BaseModel):
    """Policy knobs for historical prop-context generation."""

    model_config = ConfigDict(extra="forbid")

    lookback_games: int = Field(default=10, ge=1)
    min_history_games: int = Field(default=5, ge=1)
    stale_days: int = Field(default=120, ge=1)


@dataclass(frozen=True)
class PropContextTarget:
    event_key: str
    player_name: str
    stat_type: str
    decision_time: str
    season: str | None = None


class PropContextStatCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: int = 0
    contexts_present: int = 0
    missing_contexts: int = 0
    low_sample_contexts: int = 0
    stale_contexts: int = 0
    coverage_rate: float = 0.0


class PropContextAudit(BaseModel):
    """Machine-readable audit emitted beside ``prop_context.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    manifest_id: str
    league: str
    target_count: int
    context_count: int
    missing_context_count: int
    missing_context_rate: float
    stale_context_count: int
    stale_context_rate: float
    low_sample_context_count: int
    player_resolution_failures: int
    player_id_missing_targets: int
    duplicate_player_names: dict[str, list[str]] = Field(default_factory=dict)
    sample_size_buckets: dict[str, int] = Field(default_factory=dict)
    per_stat_coverage: dict[str, PropContextStatCoverage] = Field(default_factory=dict)
    config: PropContextBuildConfig


class PropContextBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: dict[str, dict[str, Any]]
    audit: PropContextAudit


def targets_from_prop_markets(
    events: list[HistoricalEvent],
    prop_markets: dict[str, list[HistoricalPropMarket]],
) -> list[PropContextTarget]:
    """Build unique replay-context targets from decision-time prop markets."""
    by_event = {event.event_id: event for event in events}
    targets: list[PropContextTarget] = []
    seen: set[str] = set()
    for event_id in sorted(prop_markets):
        event = by_event.get(event_id)
        if event is None:
            continue
        for market in prop_markets[event_id]:
            key = prop_context_key(event_id, market.player_name, market.stat_type)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                PropContextTarget(
                    event_key=event_id,
                    player_name=market.player_name,
                    stat_type=market.stat_type,
                    decision_time=event.start_time,
                    season=event.season,
                )
            )
    return sorted(targets, key=lambda t: (t.decision_time, t.event_key, t.player_name, t.stat_type))


def build_prop_context(
    *,
    manifest_id: str,
    league: str,
    targets: list[PropContextTarget],
    observations: list[PlayerStatObservation],
    config: PropContextBuildConfig | None = None,
) -> PropContextBuildResult:
    """Build replay-ready prop context and an audit from historical stat rows."""
    cfg = config or PropContextBuildConfig()
    grouped: dict[tuple[str, str], list[PlayerStatObservation]] = defaultdict(list)
    name_to_ids: dict[str, set[str]] = defaultdict(set)
    names_with_ids: set[str] = set()
    for obs in observations:
        grouped[(obs.player_name, obs.stat_type)].append(obs)
        if obs.player_id:
            name_to_ids[obs.player_name].add(obs.player_id)
            names_with_ids.add(obs.player_name)
    for rows in grouped.values():
        rows.sort(key=lambda r: (r.date, r.event_key))

    duplicate_names = {
        name: sorted(ids) for name, ids in name_to_ids.items() if len(ids) > 1
    }

    context: dict[str, dict[str, Any]] = {}
    buckets = {"0": 0, "1-4": 0, "5-9": 0, "10+": 0}
    per_stat: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "targets": 0,
            "contexts_present": 0,
            "missing_contexts": 0,
            "low_sample_contexts": 0,
            "stale_contexts": 0,
        }
    )
    missing_count = 0
    stale_count = 0
    low_sample_count = 0
    blank_player_targets = 0
    duplicate_target_names = 0
    player_id_missing_targets = 0

    for target in targets:
        stat_counts = per_stat[target.stat_type]
        stat_counts["targets"] += 1
        if not target.player_name.strip():
            blank_player_targets += 1
        if target.player_name in duplicate_names:
            duplicate_target_names += 1
        if target.player_name and target.player_name not in names_with_ids:
            player_id_missing_targets += 1

        decision_dt = _dt(target.decision_time)
        prior_rows = [
            row
            for row in grouped.get((target.player_name, target.stat_type), [])
            if _dt(row.date) < decision_dt
        ]
        selected = prior_rows[-cfg.lookback_games :]
        sample_size = len(selected)
        bucket = _sample_bucket(sample_size)
        buckets[bucket] += 1

        key = prop_context_key(target.event_key, target.player_name, target.stat_type)
        stat_mean_key = f"{target.stat_type}_mean"
        stat_std_key = f"{target.stat_type}_std"
        stat_median_key = f"{target.stat_type}_median"
        stat_trend_key = f"{target.stat_type}_trend"

        base_entry: dict[str, Any] = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "context_source": "historical_player_stats",
            "player_name": target.player_name,
            "stat_type": target.stat_type,
            "sample_size": sample_size,
            "sample_season": target.season,
            "lookback_games": cfg.lookback_games,
            "min_history_games": cfg.min_history_games,
            "missing_context": sample_size == 0,
            "is_low_sample": sample_size < cfg.min_history_games,
            "is_imputed": False,
            "imputed_keys": [],
            "missing_keys": [],
        }

        if not selected:
            missing_count += 1
            stat_counts["missing_contexts"] += 1
            base_entry["sample_season"] = target.season
            base_entry["as_of"] = None
            base_entry["is_stale"] = False
            base_entry["missing_keys"] = [stat_mean_key, stat_std_key]
            context[key] = base_entry
            continue

        stat_counts["contexts_present"] += 1
        values = [row.stat_value for row in selected]
        latest = selected[-1]
        as_of_dt = _dt(latest.date)
        age_days = max(0, (decision_dt - as_of_dt).days)
        is_stale = age_days > cfg.stale_days
        if is_stale:
            stale_count += 1
            stat_counts["stale_contexts"] += 1
        if sample_size < cfg.min_history_games:
            low_sample_count += 1
            stat_counts["low_sample_contexts"] += 1

        entry = dict(base_entry)
        entry.update(
            {
                stat_mean_key: _round(mean(values)),
                stat_median_key: _round(median(values)),
                stat_trend_key: _round(_trend(values)),
                "sample_season": latest.season or target.season or latest.date[:4],
                "as_of": latest.date,
                "context_age_days": age_days,
                "is_stale": is_stale,
            }
        )
        if len(values) >= 2:
            entry[stat_std_key] = _round(stdev(values))
        else:
            entry["missing_keys"] = [stat_std_key]
        if latest.player_id:
            entry["player_id"] = latest.player_id
        teams = {row.team for row in selected if row.team}
        if teams:
            entry["team_change_flag"] = len(teams) > 1
        context[key] = entry

    target_count = len(targets)
    per_stat_coverage = {
        stat: PropContextStatCoverage(
            **counts,
            coverage_rate=round(
                counts["contexts_present"] / counts["targets"], 6
            )
            if counts["targets"]
            else 0.0,
        )
        for stat, counts in sorted(per_stat.items())
    }
    audit = PropContextAudit(
        manifest_id=manifest_id,
        league=league.upper(),
        target_count=target_count,
        context_count=target_count - missing_count,
        missing_context_count=missing_count,
        missing_context_rate=round(missing_count / target_count, 6) if target_count else 0.0,
        stale_context_count=stale_count,
        stale_context_rate=round(stale_count / target_count, 6) if target_count else 0.0,
        low_sample_context_count=low_sample_count,
        player_resolution_failures=blank_player_targets + duplicate_target_names,
        player_id_missing_targets=player_id_missing_targets,
        duplicate_player_names=duplicate_names,
        sample_size_buckets=buckets,
        per_stat_coverage=per_stat_coverage,
        config=cfg,
    )
    return PropContextBuildResult(context=context, audit=audit)
